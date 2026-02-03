"""
Statistical Feature Extractor

Extracts 35 statistical features per engineer from their activity data.
Features are computed at the engineer level but stored per-message.
"""

import logging
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StatisticalFeatureExtractor:
    """Extract 35 statistical features per engineer."""

    FEATURE_NAMES = [
        # Activity Volume (10)
        "commit_frequency",
        "commit_frequency_trend",
        "pr_frequency",
        "pr_frequency_trend",
        "review_frequency",
        "review_frequency_trend",
        "issue_frequency",
        "issue_frequency_trend",
        "burst_ratio",
        "active_weeks_ratio",
        # Code Characteristics (9)
        "avg_files_per_commit",
        "std_files_per_commit",
        "max_files_per_commit",
        "avg_lines_added",
        "avg_lines_deleted",
        "code_churn_ratio",
        "commit_message_length_avg",
        "commit_message_length_std",
        "commit_size_variability",
        # Collaboration (7)
        "pr_comments_avg",
        "pr_comments_std",
        "issue_comments_avg",
        "issue_comments_std",
        "review_count",
        "pr_to_review_ratio",
        "collaboration_breadth",
        # Temporal (4)
        "tenure_weeks",
        "activity_regularity",
        "weekend_activity_ratio",
        "engagement_trajectory",
        # Work Patterns (5)
        "avg_commit_message_words",
        "commit_message_has_issue_ref",
        "commit_message_has_coauthor",
        "pr_title_length_avg",
        "issue_title_length_avg",
    ]

    def __init__(self, config: dict):
        """
        Initialize feature extractor.

        Args:
            config: Full config dict with processing.statistical_features section
        """
        feature_config = config["processing"]["statistical_features"]
        self.feature_count = feature_config["feature_count"]

        activity_volume_config = feature_config.get("activity_volume", {})
        self.trend_window_weeks = activity_volume_config.get("trend_window_weeks", 4)
        self.burst_threshold = activity_volume_config.get("burst_threshold", 3.0)

        if self.feature_count != 35:
            raise ValueError(f"Expected 35 features, config has {self.feature_count}")

        logger.info(f"Initialized StatisticalFeatureExtractor with {self.feature_count} features")
        logger.info(f"  trend_window_weeks={self.trend_window_weeks}, burst_threshold={self.burst_threshold}")

    def extract(self, activities_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Extract features for all engineers.

        Args:
            activities_df: DataFrame with columns:
                engineer_id, text, source, activity_type, timestamp, etc.

        Returns:
            Dict mapping engineer_id to feature array of shape (35,)
        """
        logger.info(f"Extracting features for {activities_df['engineer_id'].nunique()} engineers")

        features_by_engineer: dict[str, np.ndarray] = {}

        for eng_id, group in activities_df.groupby("engineer_id"):
            features = self._extract_engineer_features(group)
            features_by_engineer[str(eng_id)] = features

        logger.info(f"Extracted features for {len(features_by_engineer)} engineers")

        return features_by_engineer

    def normalize_features(
        self,
        features_by_engineer: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Normalize features to [-1, 1] using per-feature max absolute value.

        Each feature is scaled independently using the max absolute value
        across ALL engineers, preserving relative comparisons between engineers.

        Args:
            features_by_engineer: Dict mapping engineer_id to feature array (35,)

        Returns:
            Tuple of:
                - Normalized features dict (same structure, values in [-1, 1])
                - Scale factors array (35,) for denormalization if needed
        """
        if not features_by_engineer:
            logger.warning("No features to normalize")
            return {}, np.ones(35, dtype=np.float32)

        # Stack all features to find global max abs per dimension
        all_features = np.stack(list(features_by_engineer.values()))  # (num_engineers, 35)

        # Compute max absolute value per feature
        max_abs = np.abs(all_features).max(axis=0)  # (35,)

        # Avoid division by zero - if max_abs is 0, feature is constant (use 1.0)
        max_abs = np.where(max_abs == 0, 1.0, max_abs)

        # Normalize each engineer's features
        normalized = {}
        for eng_id, features in features_by_engineer.items():
            normalized[eng_id] = (features / max_abs).astype(np.float32)

        logger.info(f"Normalized {len(normalized)} engineers' features to [-1, 1]")
        logger.info(f"  Scale factors: min={max_abs.min():.4f}, max={max_abs.max():.4f}, mean={max_abs.mean():.4f}")

        # Log any features that were constant (max_abs was 0)
        constant_features = np.sum(np.abs(all_features).max(axis=0) == 0)
        if constant_features > 0:
            logger.info(f"  Constant features (same value for all engineers): {constant_features}/35")

        return normalized, max_abs.astype(np.float32)

    def _extract_engineer_features(self, activities: pd.DataFrame) -> np.ndarray:
        """
        Extract 35 features for one engineer.

        Args:
            activities: DataFrame of activities for one engineer

        Returns:
            Feature array of shape (35,)
        """
        features = np.zeros(35, dtype=np.float32)

        # Activity Volume (indices 0-9)
        features[0:10] = self._extract_activity_volume(activities)

        # Code Characteristics (indices 10-18)
        features[10:19] = self._extract_code_characteristics(activities)

        # Collaboration (indices 19-25)
        features[19:26] = self._extract_collaboration(activities)

        # Temporal (indices 26-29)
        features[26:30] = self._extract_temporal(activities)

        # Work Patterns (indices 30-34)
        features[30:35] = self._extract_work_patterns(activities)

        return features

    def _extract_activity_volume(self, activities: pd.DataFrame) -> np.ndarray:
        """Extract 10 activity volume features."""
        features = np.zeros(10, dtype=np.float32)

        # Count by activity type
        type_counts = activities["activity_type"].value_counts()
        total = len(activities)

        if total == 0:
            return features

        # Parse timestamps for time-based features
        timestamps = self._parse_timestamps(activities)
        weeks = self._group_by_week(timestamps)

        total_weeks = max(len(weeks), 1)

        # Commit frequency and trend (0, 1)
        commit_count = type_counts.get("commit", 0)
        features[0] = commit_count / total_weeks  # commits per week
        features[1] = self._calculate_trend(weeks, "commit")

        # PR frequency and trend (2, 3)
        pr_count = type_counts.get("pull_request", 0)
        features[2] = pr_count / total_weeks
        features[3] = self._calculate_trend(weeks, "pull_request")

        # Review frequency and trend (4, 5)
        review_count = type_counts.get("review", 0)
        features[4] = review_count / total_weeks
        features[5] = self._calculate_trend(weeks, "review")

        # Issue frequency and trend (6, 7)
        issue_count = type_counts.get("issue", 0)
        features[6] = issue_count / total_weeks
        features[7] = self._calculate_trend(weeks, "issue")

        # Burst ratio (8) - ratio of max week to average week, capped at burst_threshold
        weekly_counts = [sum(w.values()) for w in weeks.values()]
        if weekly_counts:
            avg_week = np.mean(weekly_counts)
            max_week = np.max(weekly_counts)
            raw_ratio = max_week / avg_week if avg_week > 0 else 0
            features[8] = min(raw_ratio, self.burst_threshold)

        # Active weeks ratio (9)
        features[9] = len(weeks) / total_weeks if total_weeks > 0 else 0

        return features

    def _extract_code_characteristics(self, activities: pd.DataFrame) -> np.ndarray:
        """Extract 9 code characteristics features."""
        features = np.zeros(9, dtype=np.float32)

        commits = activities[activities["activity_type"] == "commit"]

        if len(commits) == 0:
            return features

        # Files per commit (0, 1, 2)
        files_col = self._find_column(commits, ["meta_files_changed", "files_changed"])
        if files_col:
            files = pd.to_numeric(commits[files_col], errors="coerce").dropna()
            if len(files) > 0:
                features[0] = files.mean()
                features[1] = files.std() if len(files) > 1 else 0
                features[2] = files.max()

        # Lines added/deleted (3, 4, 5)
        add_col = self._find_column(commits, ["meta_additions", "additions"])
        del_col = self._find_column(commits, ["meta_deletions", "deletions"])

        if add_col:
            adds = pd.to_numeric(commits[add_col], errors="coerce").dropna()
            if len(adds) > 0:
                features[3] = adds.mean()

        if del_col:
            dels = pd.to_numeric(commits[del_col], errors="coerce").dropna()
            if len(dels) > 0:
                features[4] = dels.mean()

        # Code churn ratio (5)
        if features[3] + features[4] > 0:
            features[5] = features[4] / (features[3] + features[4])

        # Commit message length (6, 7)
        if "text" in commits.columns:
            msg_lengths = commits["text"].fillna("").str.len()
            features[6] = msg_lengths.mean()
            features[7] = msg_lengths.std() if len(msg_lengths) > 1 else 0

        # Commit size variability (8) - coefficient of variation of files changed
        if files_col and features[0] > 0:
            features[8] = features[1] / features[0]

        return features

    def _extract_collaboration(self, activities: pd.DataFrame) -> np.ndarray:
        """Extract 7 collaboration features."""
        features = np.zeros(7, dtype=np.float32)

        # PR comments (0, 1)
        prs = activities[activities["activity_type"] == "pull_request"]
        if len(prs) > 0 and "text" in prs.columns:
            pr_lengths = prs["text"].fillna("").str.len()
            features[0] = pr_lengths.mean()
            features[1] = pr_lengths.std() if len(pr_lengths) > 1 else 0

        # Issue comments (2, 3)
        issues = activities[activities["activity_type"] == "issue"]
        if len(issues) > 0 and "text" in issues.columns:
            issue_lengths = issues["text"].fillna("").str.len()
            features[2] = issue_lengths.mean()
            features[3] = issue_lengths.std() if len(issue_lengths) > 1 else 0

        # Review count (4)
        features[4] = len(activities[activities["activity_type"] == "review"])

        # PR to review ratio (5)
        pr_count = len(prs)
        review_count = features[4]
        if pr_count > 0:
            features[5] = review_count / pr_count

        # Collaboration breadth (6) - number of unique repos/channels
        repo_col = self._find_column(activities, ["meta_repository", "repository"])
        if repo_col:
            features[6] = activities[repo_col].nunique()

        return features

    def _extract_temporal(self, activities: pd.DataFrame) -> np.ndarray:
        """Extract 4 temporal features."""
        features = np.zeros(4, dtype=np.float32)

        timestamps = self._parse_timestamps(activities)

        if not timestamps:
            return features

        # Tenure weeks (0)
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        features[0] = (max_ts - min_ts).days / 7

        # Activity regularity (1) - inverse of coefficient of variation of weekly activity
        weeks = self._group_by_week(timestamps)
        weekly_counts = [sum(w.values()) for w in weeks.values()]
        if len(weekly_counts) > 1:
            cv = np.std(weekly_counts) / np.mean(weekly_counts) if np.mean(weekly_counts) > 0 else 0
            features[1] = 1 / (1 + cv)

        # Weekend activity ratio (2)
        weekend_count = sum(1 for ts in timestamps if ts.weekday() >= 5)
        features[2] = weekend_count / len(timestamps)

        # Engagement trajectory (3) - trend in activity over time
        if len(weekly_counts) > 1:
            x = np.arange(len(weekly_counts))
            slope, _ = np.polyfit(x, weekly_counts, 1)
            features[3] = slope / np.mean(weekly_counts) if np.mean(weekly_counts) > 0 else 0

        return features

    def _extract_work_patterns(self, activities: pd.DataFrame) -> np.ndarray:
        """Extract 5 work patterns features."""
        features = np.zeros(5, dtype=np.float32)

        commits = activities[activities["activity_type"] == "commit"]

        if len(commits) > 0 and "text" in commits.columns:
            texts = commits["text"].fillna("")

            # Avg commit message words (0)
            word_counts = texts.str.split().str.len()
            features[0] = word_counts.mean()

            # Has issue reference (1) - ratio of commits with #NNN pattern
            has_issue_ref = texts.str.contains(r"#\d+", regex=True)
            features[1] = has_issue_ref.mean()

            # Has co-author (2) - ratio with Co-authored-by
            has_coauthor = texts.str.contains(r"Co-authored-by", case=False, regex=True)
            features[2] = has_coauthor.mean()

        # PR title length avg (3)
        prs = activities[activities["activity_type"] == "pull_request"]
        if len(prs) > 0 and "text" in prs.columns:
            # Assume first line is title
            titles = prs["text"].fillna("").str.split("\n").str[0]
            features[3] = titles.str.len().mean()

        # Issue title length avg (4)
        issues = activities[activities["activity_type"] == "issue"]
        if len(issues) > 0 and "text" in issues.columns:
            titles = issues["text"].fillna("").str.split("\n").str[0]
            features[4] = titles.str.len().mean()

        return features

    def _parse_timestamps(self, activities: pd.DataFrame) -> list[datetime]:
        """Parse timestamp column to datetime objects."""
        timestamps = []

        if "timestamp" not in activities.columns:
            return timestamps

        for ts in activities["timestamp"]:
            if pd.isna(ts):
                continue
            try:
                if isinstance(ts, datetime):
                    timestamps.append(ts)
                elif isinstance(ts, str):
                    # Try ISO format
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    timestamps.append(dt)
            except (ValueError, TypeError):
                continue

        return timestamps

    def _group_by_week(self, timestamps: list[datetime]) -> dict[str, dict[str, int]]:
        """Group timestamps by week and activity type."""
        weeks: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for ts in timestamps:
            week_key = ts.strftime("%Y-W%W")
            weeks[week_key]["total"] += 1

        return dict(weeks)

    def _calculate_trend(self, weeks: dict, activity_type: str) -> float:
        """Calculate trend (slope) for activity type over recent weeks."""
        if len(weeks) < 2:
            return 0.0

        weekly_counts = [w.get(activity_type, 0) for w in weeks.values()]

        # Use only the most recent trend_window_weeks for trend calculation
        if len(weekly_counts) > self.trend_window_weeks:
            weekly_counts = weekly_counts[-self.trend_window_weeks:]

        x = np.arange(len(weekly_counts))
        if len(weekly_counts) > 1:
            slope, _ = np.polyfit(x, weekly_counts, 1)
            return float(slope)

        return 0.0

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Find first existing column from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
