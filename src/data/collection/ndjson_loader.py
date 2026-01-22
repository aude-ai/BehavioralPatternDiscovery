"""
NDJSON Data Loader

Loads activity data from aggregated NDJSON files containing records from
multiple integrations (Azure DevOps, Slack, Confluence, GitHub, etc.).

Uses the parser registry to delegate parsing to integration-specific parsers
based on the `integration` field in each record.

NDJSON Record Format:
{
    "section": "boards_work_items",
    "bundle_id": "...",
    "ts": "2025-12-16T06:59:59.000Z",
    "integration": "azure_devops",
    "aude_orgId": "org-id",
    "data": {
        "title": "...",
        "description": "...",
        "creator": {"id": "...", "display_name": "...", "email": "..."},
        ...
    }
}
"""

import gzip
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Iterator, Dict, Any, Optional
import pandas as pd

from .parsers import parser_registry
from .text_cleanup import TextCleaner

logger = logging.getLogger(__name__)


# Mapping of integration names in NDJSON to parser registry names
# This handles cases where naive PascalCase->snake_case doesn't work (e.g., DevOps)
INTEGRATION_NAME_MAP = {
    "azuredevops": "azure_devops",
    "azure_devops": "azure_devops",
    "github": "github",
    "slack": "slack",
    "confluence": "confluence",
    "trello": "trello",
}


def _normalize_integration_name(name: str) -> str:
    """
    Normalize integration name to match parser registry.

    Uses explicit mapping for known integrations, falls back to lowercase.

    Examples:
        AzureDevOps -> azure_devops
        GitHub -> github
        Slack -> slack
    """
    # Lowercase and remove spaces/underscores for lookup
    normalized = name.lower().replace(" ", "").replace("_", "")

    # Look up in mapping
    if normalized in INTEGRATION_NAME_MAP:
        return INTEGRATION_NAME_MAP[normalized]

    # Fallback: just lowercase with underscores for spaces
    return name.lower().replace(" ", "_")


class NDJSONLoader:
    """Load activities from aggregated NDJSON files using parser registry."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NDJSON loader.

        Args:
            config: Configuration dict with keys:
                - input_path: Path to folder containing .ndjson.gz data files
                - identity_file: Path to adoIdentities.ndjson for bot detection
                - date_range: Optional dict with start/end dates for file filtering
                - parser_configs: Dict mapping integration name to parser config
                - excluded_sections: List of sections to skip
                - text_cleanup: Config for text cleanup, quality filter, and deduplication
        """
        self.input_path = Path(config["input_path"])
        self.identity_file = Path(config["identity_file"])
        self.parser_configs = config["parser_configs"]
        self.date_range = config.get("date_range")

        # Section filtering
        self.excluded_sections = set(config["excluded_sections"])
        if self.excluded_sections:
            logger.info(f"Excluding sections: {sorted(self.excluded_sections)}")

        # Load identity file for bot detection and user mapping
        self.identity_metadata = self._load_identity_file(config["identity_file"])
        identity_mapping = {k: v["display_name"] for k, v in self.identity_metadata.items() if v.get("display_name")}

        # Build user mapping from users section in NDJSON (first pass)
        self.engineer_mapping = self._build_user_mapping_from_ndjson()

        # Merge with identity file mapping
        self.engineer_mapping.update(identity_mapping)

        # Cache instantiated parsers
        self._parsers: Dict[str, Any] = {}

        # Track which integrations we've seen but don't have parsers for
        self._missing_parsers: set[str] = set()

        # Initialize text cleaner (handles quality filtering and deduplication)
        text_cleanup_config = config["text_cleanup"]
        self.text_cleaner = TextCleaner(text_cleanup_config)

        # Auto-detect internal team CSV file in input folder (if present)
        self.internal_team_ids = self._load_internal_team_file()

        # Statistics for logging
        self._stats = {
            "excluded_by_section": 0,
            "sections_seen": defaultdict(int),
        }

    def _load_identity_file(self, identity_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load identity metadata from adoIdentities.ndjson format.

        Returns dict mapping identifier -> {display_name, is_bot, roles_seen, ...}
        """
        identity_file = Path(identity_path)
        if not identity_file.exists():
            raise FileNotFoundError(f"Identity file not found: {identity_path}")

        metadata: Dict[str, Dict[str, Any]] = {}
        bot_count = 0
        human_count = 0

        with open(identity_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                sample = record.get("sample", {})
                display_name = sample.get("displayName", "")
                is_bot = record.get("type") == "bot"

                if is_bot:
                    bot_count += 1
                else:
                    human_count += 1

                entry = {
                    "display_name": display_name,
                    "is_bot": is_bot,
                    "bot_score": record.get("botScore", 0),
                    "roles_seen": record.get("rolesSeen", []),
                }

                # Map all identifiers to this entry
                for field in ["email", "uniqueName", "id"]:
                    identifier = sample.get(field)
                    if identifier:
                        metadata[identifier] = entry
                        metadata[identifier.lower()] = entry

                if display_name:
                    metadata[display_name] = entry
                    metadata[display_name.lower()] = entry

        logger.info(f"Loaded identity metadata: {human_count} humans, {bot_count} bots")
        return metadata

    def _is_engineer_bot(self, engineer_id: str) -> bool:
        """Check if engineer is a bot based on identity metadata."""
        entry = self.identity_metadata.get(engineer_id) or self.identity_metadata.get(engineer_id.lower())
        if entry:
            return entry.get("is_bot", False)
        return False

    def _load_internal_team_file(self) -> set[str]:
        """
        Auto-detect and load internal team identifiers from CSV file in input folder.

        Looks for any .csv file in the input folder. If found, loads it and
        extracts identifiers from columns like 'email', 'displayName', etc.

        Returns:
            Set of lowercase identifiers for internal team members
        """
        logger.info(f"Checking for internal team CSV in: {self.input_path}")

        # Find CSV files in input folder
        csv_files = list(self.input_path.glob("*.csv"))

        if not csv_files:
            logger.info(f"No internal team CSV found in {self.input_path}")
            return set()

        if len(csv_files) > 1:
            logger.warning(f"Multiple CSV files found in {self.input_path}, using first: {csv_files[0].name}")

        csv_file = csv_files[0]
        internal_ids: set[str] = set()

        try:
            df = pd.read_csv(csv_file)
            # Collect all string values from relevant columns
            for col in df.columns:
                if col.lower() in ('email', 'displayname', 'display_name', 'name', 'uniquename', 'unique_name'):
                    for val in df[col].dropna():
                        internal_ids.add(str(val).lower())

            logger.info(f"Loaded {len(internal_ids)} internal team identifiers from {csv_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load internal team file {csv_file.name}: {e}")

        return internal_ids

    def _is_engineer_internal(self, engineer_id: str) -> bool:
        """Check if engineer is internal based on internal team file."""
        if not self.internal_team_ids:
            return True  # If no file configured, all are considered internal
        return engineer_id.lower() in self.internal_team_ids

    def _build_user_mapping_from_ndjson(self) -> Dict[str, str]:
        """
        Build user mapping from users section in NDJSON.

        Maps id, email, and unique_name to display_name for each user.
        This allows activities to use display_name as the canonical engineer ID.

        Returns:
            Dict mapping various user identifiers to display_name
        """
        mapping: Dict[str, str] = {}
        user_count = 0

        for line in self._iter_lines():
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if record.get("section") != "users":
                continue

            data = record.get("data", {})
            display_name = data.get("display_name", "").strip()

            if not display_name:
                continue

            user_count += 1

            # Map all identifiers to display_name
            user_id = data.get("id", "")
            email = data.get("email", "")
            unique_name = data.get("unique_name", "")

            if user_id and user_id != display_name:
                mapping[user_id] = display_name
            if email and email != display_name:
                mapping[email] = display_name
            if unique_name and unique_name != display_name:
                mapping[unique_name] = display_name

        logger.info(f"Built user mapping from {user_count} users ({len(mapping)} mappings)")
        return mapping

    def _iter_lines(self) -> Iterator[str]:
        """Iterate over NDJSON lines from file or directory, supporting gzip."""
        files_to_load = self._get_files_to_load()

        for file_path in files_to_load:
            logger.info(f"  Loading {file_path.name}")

            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    yield from f
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    yield from f

    def _get_files_to_load(self) -> list[Path]:
        """Get list of NDJSON files to load, applying date range filter if configured."""
        if self.input_path.is_file():
            logger.info(f"Loading NDJSON from file: {self.input_path}")
            return [self.input_path]

        if not self.input_path.is_dir():
            raise FileNotFoundError(f"NDJSON path not found: {self.input_path}")

        # Find all NDJSON files (plain and gzipped), excluding identity file
        ndjson_files = list(self.input_path.glob("*.ndjson")) + list(self.input_path.glob("*.ndjson.gz"))
        ndjson_files = [f for f in ndjson_files if f.resolve() != self.identity_file.resolve()]
        ndjson_files = sorted(ndjson_files)

        # Apply date range filter if configured
        if self.date_range:
            ndjson_files = self._filter_files_by_date(ndjson_files)

        logger.info(f"Loading NDJSON from directory: {self.input_path} ({len(ndjson_files)} files)")
        return ndjson_files

    def _filter_files_by_date(self, files: list[Path]) -> list[Path]:
        """
        Filter files by date range parsed from filename.

        Filename format: AzureDevOps.{timestamp}.{start_date}.{end_date}.ndjson.gz
        Example: AzureDevOps.2025-12-24-00-41-21-309.2025-01-01.2025-01-16.ndjson.gz
        """
        start_filter = self.date_range["start"]
        end_filter = self.date_range["end"]

        start_filter = datetime.strptime(start_filter, "%Y-%m-%d")
        end_filter = datetime.strptime(end_filter, "%Y-%m-%d")

        # Pattern to extract dates from filename
        date_pattern = re.compile(r'\.(\d{4}-\d{2}-\d{2})\.(\d{4}-\d{2}-\d{2})\.ndjson')

        filtered = []
        for file_path in files:
            match = date_pattern.search(file_path.name)
            if not match:
                # Include files without date pattern
                filtered.append(file_path)
                continue

            file_start = datetime.strptime(match.group(1), "%Y-%m-%d")
            file_end = datetime.strptime(match.group(2), "%Y-%m-%d")

            # Include if file overlaps with filter range
            if file_end < start_filter:
                continue
            if file_start > end_filter:
                continue

            filtered.append(file_path)

        logger.info(f"Date filter {start_filter.date()} to {end_filter.date()}: {len(filtered)}/{len(files)} files")
        return filtered

    def _parse_line(self, line: str) -> list[Dict[str, Any]]:
        """
        Parse a single NDJSON line into activity records.

        Args:
            line: Single line from NDJSON file

        Returns:
            List of activity dicts (may be empty if filtered/skipped)
        """
        try:
            record = json.loads(line.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse NDJSON line: {e}")
            return []

        # Get section for filtering
        section = record.get("section")

        # Track section statistics
        self._stats["sections_seen"][section] += 1

        # Skip excluded sections
        if section in self.excluded_sections:
            self._stats["excluded_by_section"] += 1
            return []

        # Get integration to determine parser (normalize casing)
        integration = record.get("integration")
        if not integration:
            logger.warning(f"Record missing 'integration' field, section: {section}")
            return []

        # Normalize integration name: "AzureDevOps" -> "azure_devops"
        integration_normalized = _normalize_integration_name(integration)

        # Get parser for this integration
        if integration_normalized not in parser_registry:
            if integration_normalized not in self._missing_parsers:
                logger.warning(f"No parser registered for integration: {integration_normalized}")
                self._missing_parsers.add(integration_normalized)
            return []

        # Get or create parser instance
        if integration_normalized not in self._parsers:
            parser_config = self.parser_configs.get(integration_normalized)
            if not parser_config:
                if integration_normalized not in self._missing_parsers:
                    logger.warning(f"No config for parser: {integration_normalized}")
                    self._missing_parsers.add(integration_normalized)
                return []
            self._parsers[integration_normalized] = parser_registry.create(integration_normalized, config=parser_config)

        parser = self._parsers[integration_normalized]

        # Parse the record - use record_details (new format)
        data = record["record_details"]
        record_summary = record.get("record_summary", "")

        activities = parser.parse(
            data=data,
            user_mapping=self.engineer_mapping,
            collection_type=section,
            record_summary=record_summary
        )

        # Enrich activities with record-level metadata
        ts = record["ts"]
        bundle_id = record["bundle_id"]
        org_id = record["aude_orgId"]

        # Extract project from record_details
        project_name = ""
        if "project" in data:
            project_name = data["project"]["name"]

        for activity in activities:
            engineer_id = activity["engineer_id"]

            # Use record timestamp if activity doesn't have one
            if not activity.get("timestamp"):
                activity["timestamp"] = ts

            # Add metadata including bot status and project
            activity["metadata"] = {
                "bundle_id": bundle_id,
                "org_id": org_id,
                "integration": integration_normalized,
                "project": project_name,
                "is_bot": self._is_engineer_bot(engineer_id),
            }

        return activities

    def load(self) -> pd.DataFrame:
        """
        Load all activities from NDJSON into DataFrame.

        Returns:
            DataFrame with columns:
                - engineer_id: Canonical engineer identifier
                - timestamp: Activity timestamp
                - text: Text content for embedding
                - source: Source type (azure_devops, slack, etc.)
                - activity_type: Activity type (work_item, message, etc.)
                - split: Train/validation split (default: train)
                - meta_*: Flattened metadata fields
        """
        activities = []
        line_count = 0
        parsed_count = 0

        # Reset text cleaner state for fresh deduplication tracking
        self.text_cleaner.reset_deduplication_state()

        for line in self._iter_lines():
            line_count += 1
            if line_count % 10000 == 0:
                logger.info(f"Processed {line_count} lines, {parsed_count} activities")

            line_activities = self._parse_line(line)
            for activity in line_activities:
                # Validate required fields
                engineer_id = activity.get("engineer_id")
                if not engineer_id:
                    continue

                text = activity.get("text", "").strip()
                if not text:
                    continue

                # Apply text cleanup (includes quality filter and deduplication)
                text = self.text_cleaner.clean(text, engineer_id=engineer_id)
                if text is None:
                    continue  # Filtered out by cleanup
                activity["text"] = text

                activities.append(activity)
                parsed_count += 1

        # Log filtering statistics
        self._log_filtering_stats(line_count, parsed_count)

        if not activities:
            raise ValueError("No activities loaded from NDJSON")

        # Convert to DataFrame
        df = self._activities_to_dataframe(activities)

        # Log summary
        logger.info(f"Activities by source: {df['source'].value_counts().to_dict()}")
        logger.info(f"Unique engineers: {df['engineer_id'].nunique()}")

        return df

    def _log_filtering_stats(self, total_lines: int, final_count: int) -> None:
        """Log statistics about filtering."""
        logger.info(f"Loaded {final_count} activities from {total_lines} NDJSON lines")

        if self._stats["excluded_by_section"] > 0:
            logger.info(f"  Excluded by section filter: {self._stats['excluded_by_section']}")

        # Log text cleaner stats (quality filter and deduplication)
        self.text_cleaner.log_stats()

        # Log section breakdown
        if self._stats["sections_seen"]:
            logger.info("  Sections seen:")
            for section, count in sorted(self._stats["sections_seen"].items(), key=lambda x: -x[1]):
                excluded = " (excluded)" if section in self.excluded_sections else ""
                logger.info(f"    {section}: {count}{excluded}")

    def _activities_to_dataframe(self, activities: list[Dict[str, Any]]) -> pd.DataFrame:
        """Convert activities to DataFrame with proper schema."""
        rows = []
        for act in activities:
            engineer_id = act["engineer_id"]
            row = {
                "engineer_id": engineer_id,
                "text": act["text"],
                "source": act["source"],
                "activity_type": act["activity_type"],
                "timestamp": act["timestamp"],
                "split": "train",
                "is_bot": act["metadata"]["is_bot"],
                "project": act["metadata"]["project"],
            }
            # Add is_internal column only if internal team file is configured
            if self.internal_team_ids:
                row["is_internal"] = self._is_engineer_internal(engineer_id)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Log internal/external breakdown if applicable
        if self.internal_team_ids and "is_internal" in df.columns:
            internal_count = df["is_internal"].sum()
            external_count = len(df) - internal_count
            logger.info(f"Internal/external breakdown: {internal_count} internal, {external_count} external activities")

        return df

    def load_windowed(
        self,
        window_start: datetime,
        window_end: datetime
    ) -> pd.DataFrame:
        """
        Load activities within a specific time window.

        Args:
            window_start: Start of evaluation window
            window_end: End of evaluation window

        Returns:
            DataFrame filtered to window
        """
        df = self.load()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter to window
        mask = (df["timestamp"] >= window_start) & (df["timestamp"] < window_end)
        windowed = df[mask].copy()

        logger.info(f"Window {window_start} to {window_end}: {len(windowed)} activities (from {len(df)} total)")

        return windowed


def load_activities_from_ndjson(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Convenience function to load activities from NDJSON.

    Args:
        config: NDJSON loader configuration with keys:
            - input_path: Path to NDJSON file or directory
            - engineer_id_mapping: Optional path to mapping file

    Returns:
        Activities DataFrame
    """
    loader = NDJSONLoader(config)
    return loader.load()
