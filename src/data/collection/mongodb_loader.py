"""
MongoDB Activity Loader

Fetches activities from MongoDB with multi-source support and balanced sampling.
Handles Airbyte data format where documents are wrapped in '_airbyte_data' field.
"""

import os
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pymongo import MongoClient

from .parsers import parser_registry
from .parsers.github import build_user_mapping_from_prs
from .text_cleanup import TextCleaner

logger = logging.getLogger(__name__)


class MongoDBLoader:
    """Load activities from MongoDB with multi-source support."""

    def __init__(self, config: dict):
        """
        Initialize MongoDB loader.

        Args:
            config: Full config dict with collection and paths sections
        """
        self.connection_string = os.environ["MONGODB_CONNECTION_STRING"]
        self.timeout_ms = config["collection"]["mongodb"]["connection_timeout_ms"]
        self.default_databases = config["collection"]["mongodb"]["default_databases"]
        self.max_per_engineer = config["collection"]["mongodb"]["max_activities_per_engineer"]
        self.activities_path = Path(config["paths"]["data"]["collection"]["activities_csv"])

        # User integrations config for cross-source ID mapping
        self.user_integrations_config = config["collection"]["mongodb"]["user_integrations"]

        # Initialize parsers from registry
        parser_configs = config["collection"]["parsers"]
        self.parsers: dict[str, Any] = {}
        for name, cfg in parser_configs.items():
            if name in parser_registry:
                self.parsers[name] = parser_registry.create(name, config=cfg)
                logger.info(f"Initialized {name} parser")

        # Initialize text cleaner if configured
        text_cleanup_config = config["collection"].get("text_cleanup")
        if text_cleanup_config:
            self.text_cleaner = TextCleaner(text_cleanup_config)
        else:
            self.text_cleaner = None

    def fetch_activities(
        self,
        databases: list[str] | None = None,
        append: bool = False,
        max_engineers: int | None = None,
        max_activities_per_engineer: int | None = None
    ) -> dict:
        """
        Fetch activities from MongoDB.

        Args:
            databases: List of "database/collection" strings. Uses default if None.
            append: If True, append to existing activities.csv
            max_engineers: Limit number of engineers
            max_activities_per_engineer: Override config max activities

        Returns:
            Dict with count and engineers info
        """
        sources = databases if databases is not None else self.default_databases
        max_per_eng = max_activities_per_engineer or self.max_per_engineer

        logger.info("Connecting to MongoDB...")
        client = MongoClient(
            self.connection_string,
            serverSelectionTimeoutMS=self.timeout_ms
        )

        # Load global user integrations mapping once
        global_user_mapping = self._load_user_integrations(client)

        all_activities: list[dict] = []

        for source in sources:
            parts = source.split("/")
            if len(parts) != 2:
                logger.warning(f"Invalid source format: {source}. Expected 'database/collection'")
                continue

            db_name, collection_spec = parts
            db = client[db_name]
            logger.info(f"Fetching from {db_name}/{collection_spec}")

            # Build user mapping for this database (includes global mapping)
            user_mapping = self._build_user_mapping(db, global_user_mapping)

            # Determine collections to fetch
            if collection_spec == "all":
                collections = self._get_activity_collections(db)
            else:
                collections = [collection_spec]

            logger.info(f"Processing {len(collections)} collections")

            # Fetch and parse each collection
            for coll_name in collections:
                parser, collection_type = self._get_parser_for_collection(coll_name)
                if parser is None:
                    logger.debug(f"No parser for collection: {coll_name}")
                    continue

                logger.info(f"Parsing collection: {coll_name} ({parser.get_source_name()}/{collection_type})")
                collection = db[coll_name]
                parsed_count = 0
                filtered_count = 0

                for doc in collection.find():
                    try:
                        # Extract data from Airbyte wrapper
                        data = parser.extract_airbyte_data(doc)

                        # Parse document (returns list of activities)
                        activities = parser.parse(data, user_mapping, collection_type)

                        for activity in activities:
                            eng_id = activity.get("engineer_id")
                            if not eng_id or eng_id == "unknown":
                                filtered_count += 1
                                continue

                            text = activity.get("text", "").strip()
                            if not text:
                                filtered_count += 1
                                continue

                            # Apply text cleanup if configured
                            if self.text_cleaner:
                                text = self.text_cleaner.clean(text)
                                if text is None:
                                    filtered_count += 1
                                    continue  # Filtered out by cleanup
                                activity["text"] = text

                            activity["metadata"]["source_database"] = db_name
                            activity["metadata"]["source_collection"] = coll_name
                            all_activities.append(activity)
                            parsed_count += 1

                    except Exception as e:
                        logger.error(f"Error parsing document in {coll_name}: {e}")
                        filtered_count += 1
                        continue

                logger.info(f"  Parsed: {parsed_count}, Filtered: {filtered_count}")

        client.close()

        if not all_activities:
            logger.warning("No activities fetched from MongoDB")
            return {"count": 0, "engineers": 0}

        logger.info(f"Total activities before sampling: {len(all_activities)}")

        # Apply balanced sampling
        sampled = self._balanced_sample(all_activities, max_per_eng, max_engineers)
        logger.info(f"Total activities after sampling: {len(sampled)}")

        # Convert to DataFrame
        df = self._activities_to_dataframe(sampled)

        # Save (append or overwrite)
        self._save_activities(df, append)

        unique_engineers = df["engineer_id"].nunique()
        logger.info(f"Saved {len(df)} activities from {unique_engineers} engineers")

        return {"count": len(sampled), "engineers": unique_engineers}

    def _load_user_integrations(self, client: MongoClient) -> dict[str, str]:
        """
        Load user integrations mapping from Aude-production database.

        Maps activity_user_id (source-specific IDs like GitHub user ID, Slack ID)
        to canonical user_id (UUID).

        Args:
            client: MongoDB client

        Returns:
            Dict mapping activity_user_id -> user_id
        """
        if not self.user_integrations_config["enabled"]:
            logger.info("User integrations mapping disabled")
            return {}

        db_name = self.user_integrations_config["database"]
        coll_name = self.user_integrations_config["collection"]

        logger.info(f"Loading user integrations from {db_name}/{coll_name}")
        db = client[db_name]
        collection = db[coll_name]

        mapping: dict[str, str] = {}
        for doc in collection.find({"archived": {"$ne": True}}):
            activity_user_id = str(doc.get("activity_user_id", ""))
            user_id = doc.get("user_id", "")
            if activity_user_id and user_id:
                mapping[activity_user_id] = user_id

        logger.info(f"Loaded {len(mapping)} user integration mappings")
        return mapping

    def _build_user_mapping(self, db, global_mapping: dict[str, str]) -> dict[str, str]:
        """
        Build mapping from user IDs/logins to canonical engineer IDs.

        Combines:
        1. Global user_integrations mapping (activity_user_id -> user_id)
        2. PR-based mapping (author names -> logins)
        3. GitHub users collection (user ID -> login)

        Args:
            db: MongoDB database object
            global_mapping: Pre-loaded user_integrations mapping

        Returns:
            Dict mapping various user identifiers to canonical ID
        """
        # Start with global user integrations mapping
        user_mapping: dict[str, str] = dict(global_mapping)
        logger.info(f"Starting with {len(user_mapping)} entries from user_integrations")

        # Add mappings from PR data (maps author names to logins)
        pr_coll_name = None
        for name in db.list_collection_names():
            if "GitHub_pull_requests" in name and "commits" not in name and "stats" not in name:
                pr_coll_name = name
                break

        if pr_coll_name:
            pr_docs = list(db[pr_coll_name].find())
            pr_mapping = build_user_mapping_from_prs(pr_docs)
            # Merge PR mapping (don't overwrite global mappings)
            for k, v in pr_mapping.items():
                if k not in user_mapping:
                    user_mapping[k] = v
            logger.info(f"Added {len(pr_mapping)} entries from PRs")

        # Add users from GitHub_users collection
        github_added = 0
        for name in db.list_collection_names():
            if "GitHub_users" in name:
                for doc in db[name].find():
                    data = doc.get("_airbyte_data", doc)
                    login = data.get("login")
                    if login:
                        if login not in user_mapping:
                            user_mapping[login] = login
                            github_added += 1
                        user_id = str(data.get("id", ""))
                        if user_id and user_id not in user_mapping:
                            user_mapping[user_id] = login
                            github_added += 1
        if github_added:
            logger.info(f"Added {github_added} entries from GitHub_users")

        logger.info(f"Total user mapping: {len(user_mapping)} entries")
        return user_mapping

    def _get_activity_collections(self, db) -> list[str]:
        """
        Get list of collections that contain activity data.

        Args:
            db: MongoDB database object

        Returns:
            List of collection names to process
        """
        collections = []
        all_names = db.list_collection_names()

        for name in all_names:
            # Check if any parser handles this collection
            for parser in self.parsers.values():
                if parser.match_collection(name):
                    collections.append(name)
                    break

        return collections

    def _get_parser_for_collection(self, collection_name: str) -> tuple[Any, str] | tuple[None, None]:
        """
        Find parser that handles this collection.

        Args:
            collection_name: MongoDB collection name

        Returns:
            Tuple of (parser, collection_type) or (None, None) if no match
        """
        for parser in self.parsers.values():
            collection_type = parser.match_collection(collection_name)
            if collection_type:
                return parser, collection_type

        return None, None

    def _balanced_sample(
        self,
        activities: list[dict],
        max_per_eng: int,
        max_engineers: int | None
    ) -> list[dict]:
        """
        Apply balanced sampling per engineer.

        Uses round-robin sampling from different sources to maintain diversity.

        Args:
            activities: List of activity dicts
            max_per_eng: Maximum activities per engineer
            max_engineers: Maximum number of engineers (None for unlimited)

        Returns:
            Sampled list of activities
        """
        # Group by engineer
        by_engineer: dict[str, list[dict]] = {}
        for act in activities:
            eng_id = act["engineer_id"]
            if eng_id not in by_engineer:
                by_engineer[eng_id] = []
            by_engineer[eng_id].append(act)

        # Limit engineers if requested
        engineer_ids = list(by_engineer.keys())
        if max_engineers and len(engineer_ids) > max_engineers:
            # Sort by activity count (descending) to keep most active engineers
            engineer_ids.sort(key=lambda e: len(by_engineer[e]), reverse=True)
            engineer_ids = engineer_ids[:max_engineers]
            logger.info(f"Limited to {max_engineers} engineers")

        # Sample activities for each engineer
        sampled: list[dict] = []
        for eng_id in engineer_ids:
            eng_activities = by_engineer[eng_id]
            if len(eng_activities) <= max_per_eng:
                sampled.extend(eng_activities)
            else:
                sampled.extend(self._round_robin_sample(eng_activities, max_per_eng))

        return sampled

    def _round_robin_sample(self, activities: list[dict], max_count: int) -> list[dict]:
        """
        Sample activities evenly from different sources.

        Distributes max_count evenly across sources. If a source has fewer
        activities than its quota, the remainder is redistributed to other sources.

        Example: max_count=300, 3 sources -> 100 each
        If source A has only 50, sources B and C get 125 each.

        Args:
            activities: List of activities from one engineer
            max_count: Maximum to sample

        Returns:
            Sampled activities with even source distribution
        """
        # Group by source
        by_source: dict[str, list[dict]] = {}
        for act in activities:
            source = act["source"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(act)

        if not by_source:
            return []

        # Sort each source by timestamp (most recent first)
        for source in by_source:
            by_source[source].sort(
                key=lambda a: a.get("timestamp", "") or "",
                reverse=True
            )

        # Calculate even distribution with redistribution for sources with fewer items
        sources = list(by_source.keys())
        source_counts = {s: len(by_source[s]) for s in sources}
        source_quotas: dict[str, int] = {}

        remaining = max_count
        active_sources = set(sources)

        # Iteratively allocate quotas, redistributing when sources are exhausted
        while remaining > 0 and active_sources:
            quota_per_source = remaining // len(active_sources)
            if quota_per_source == 0:
                quota_per_source = 1

            newly_exhausted = set()
            allocated_this_round = 0

            for source in list(active_sources):
                available = source_counts[source] - source_quotas.get(source, 0)
                allocation = min(quota_per_source, available, remaining - allocated_this_round)

                if allocation > 0:
                    source_quotas[source] = source_quotas.get(source, 0) + allocation
                    allocated_this_round += allocation

                # Check if source is now exhausted
                if source_quotas.get(source, 0) >= source_counts[source]:
                    newly_exhausted.add(source)

            remaining -= allocated_this_round
            active_sources -= newly_exhausted

            # Safety: break if no progress made
            if allocated_this_round == 0:
                break

        # Sample from each source according to quotas
        sampled: list[dict] = []
        for source, quota in source_quotas.items():
            sampled.extend(by_source[source][:quota])

        return sampled

    def _activities_to_dataframe(self, activities: list[dict]) -> pd.DataFrame:
        """
        Convert activities to DataFrame with proper schema.

        Args:
            activities: List of activity dicts

        Returns:
            DataFrame with standardized columns
        """
        rows = []
        for act in activities:
            row = {
                "engineer_id": act["engineer_id"],
                "text": act["text"],
                "source": act["source"],
                "activity_type": act["activity_type"],
                "timestamp": act["timestamp"],
                "split": "train",  # Default split
            }
            # Flatten metadata (only include non-None values)
            for key, val in act.get("metadata", {}).items():
                if val is not None:
                    row[f"meta_{key}"] = val
            rows.append(row)

        return pd.DataFrame(rows)

    def _save_activities(self, df: pd.DataFrame, append: bool):
        """
        Save activities to CSV.

        Args:
            df: Activities DataFrame
            append: If True, append to existing file
        """
        self.activities_path.parent.mkdir(parents=True, exist_ok=True)

        if append and self.activities_path.exists():
            existing = pd.read_csv(self.activities_path)
            df = pd.concat([existing, df], ignore_index=True)
            logger.info(f"Appended to existing activities ({len(existing)} + {len(df) - len(existing)})")

        df.to_csv(self.activities_path, index=False)
        logger.info(f"Saved {len(df)} activities to {self.activities_path}")
