"""
Engineer Manager

Handles CRUD operations on activities.csv for engineer management.
Supports the frontend's engineer management UI.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class EngineerManager:
    """Manage engineers in activities.csv."""

    def __init__(self, config: dict):
        """
        Initialize engineer manager.

        Args:
            config: Full config dict with paths.data.collection.activities_csv
        """
        self.activities_path = Path(config["paths"]["data"]["collection"]["activities_csv"])

    def get_summary(self) -> dict:
        """
        Get summary of all engineers.

        Returns:
            Dict with:
            - train_count: Number of engineers in train split
            - validation_count: Number of engineers in validation split
            - bot_count: Number of bot engineers
            - total_engineers: Total unique engineers
            - projects: List of unique projects
            - engineers: List of engineer details with is_bot, projects
        """
        if not self.activities_path.exists():
            return {
                "train_count": 0,
                "validation_count": 0,
                "bot_count": 0,
                "total_engineers": 0,
                "projects": [],
                "engineers": []
            }

        df = pd.read_csv(self.activities_path)

        if len(df) == 0:
            return {
                "train_count": 0,
                "validation_count": 0,
                "bot_count": 0,
                "total_engineers": 0,
                "projects": [],
                "engineers": []
            }

        # Load engineer metadata if available
        meta_path = self.activities_path.parent / "engineer_metadata.csv"
        engineer_meta = {}
        has_internal_column = False
        if meta_path.exists():
            meta_df = pd.read_csv(meta_path)
            has_internal_column = "is_internal" in meta_df.columns
            for _, row in meta_df.iterrows():
                meta_entry = {
                    "is_bot": bool(row["is_bot"]),
                    "projects": row["projects"] if pd.notna(row["projects"]) else ""
                }
                if has_internal_column:
                    meta_entry["is_internal"] = bool(row["is_internal"])
                engineer_meta[row["engineer_id"]] = meta_entry

        engineers = []
        all_projects = set()
        for eng_id, group in df.groupby("engineer_id"):
            split_value = group["split"].iloc[0] if "split" in group.columns else "train"
            meta = engineer_meta.get(eng_id, {"is_bot": False, "projects": ""})

            # Track all projects
            if meta["projects"]:
                for p in meta["projects"].split(","):
                    if p:
                        all_projects.add(p)

            eng_entry = {
                "engineer_id": eng_id,
                "activity_count": len(group),
                "split": split_value,
                "sources": group["source"].unique().tolist(),
                "is_bot": meta["is_bot"],
                "projects": meta["projects"]
            }
            if "is_internal" in meta:
                eng_entry["is_internal"] = meta["is_internal"]
            engineers.append(eng_entry)

        train_count = sum(1 for e in engineers if e["split"] == "train")
        validation_count = sum(1 for e in engineers if e["split"] == "validation")
        bot_count = sum(1 for e in engineers if e["is_bot"])

        logger.info(f"Engineer summary: {len(engineers)} total, {train_count} train, {validation_count} validation, {bot_count} bots")

        return {
            "train_count": train_count,
            "validation_count": validation_count,
            "bot_count": bot_count,
            "total_engineers": len(engineers),
            "projects": sorted(all_projects),
            "engineers": engineers
        }

    def set_split(self, engineer_ids: list[str], split: str) -> dict:
        """
        Set train/validation split for engineers.

        Args:
            engineer_ids: List of engineer IDs to update
            split: Target split ('train' or 'validation')

        Returns:
            Dict with message and rows_updated count
        """
        if not self.activities_path.exists():
            raise FileNotFoundError(f"Activities file not found: {self.activities_path}")

        df = pd.read_csv(self.activities_path)
        mask = df["engineer_id"].isin(engineer_ids)
        rows_updated = mask.sum()

        df.loc[mask, "split"] = split
        df.to_csv(self.activities_path, index=False)

        logger.info(f"Set {len(engineer_ids)} engineers to {split} ({rows_updated} rows)")

        return {
            "message": f"Set {len(engineer_ids)} engineers to {split}",
            "rows_updated": int(rows_updated)
        }

    def remove_engineers(self, engineer_ids: list[str]) -> dict:
        """
        Remove engineers from activities.

        Args:
            engineer_ids: List of engineer IDs to remove

        Returns:
            Dict with message and rows_deleted count
        """
        if not self.activities_path.exists():
            raise FileNotFoundError(f"Activities file not found: {self.activities_path}")

        df = pd.read_csv(self.activities_path)
        original_len = len(df)

        df = df[~df["engineer_id"].isin(engineer_ids)]
        rows_deleted = original_len - len(df)

        df.to_csv(self.activities_path, index=False)

        logger.info(f"Removed {len(engineer_ids)} engineers ({rows_deleted} rows)")

        return {
            "message": f"Removed {len(engineer_ids)} engineers",
            "rows_deleted": rows_deleted
        }

    def merge_engineers(self, source_ids: list[str], target_id: str) -> dict:
        """
        Merge multiple engineers into one canonical ID.

        Args:
            source_ids: List of engineer IDs to merge from
            target_id: Canonical engineer ID to merge into

        Returns:
            Dict with message and rows_updated count
        """
        if not self.activities_path.exists():
            raise FileNotFoundError(f"Activities file not found: {self.activities_path}")

        df = pd.read_csv(self.activities_path)
        mask = df["engineer_id"].isin(source_ids)
        rows_updated = mask.sum()

        df.loc[mask, "engineer_id"] = target_id
        df.to_csv(self.activities_path, index=False)

        logger.info(f"Merged {len(source_ids)} engineers into {target_id} ({rows_updated} rows)")

        return {
            "message": f"Merged {len(source_ids)} engineers into {target_id}",
            "rows_updated": int(rows_updated)
        }
