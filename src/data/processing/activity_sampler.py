"""
Activity Sampler

Balanced activity sampling across (source, activity_type) combinations per engineer.
Ensures equal representation of all activity types with redistribution when
some types have fewer items than their quota.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ActivitySampler:
    """Balanced activity sampling across source+activity_type combinations."""

    def __init__(self, max_per_engineer: Optional[int], seed: Optional[int]):
        """
        Initialize activity sampler.

        Args:
            max_per_engineer: Maximum activities per engineer (None = no limit)
            seed: Random seed for reproducibility (None = non-deterministic)
        """
        self.max_per_engineer = max_per_engineer
        self.seed = seed

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample activities with balanced distribution per engineer.

        For each engineer, distributes quota evenly across unique
        (source, activity_type) combinations. If a type has fewer items
        than its quota, the surplus is redistributed to other types.

        Args:
            df: Activities DataFrame with columns: engineer_id, source, activity_type

        Returns:
            Sampled DataFrame with balanced type distribution per engineer
        """
        if self.max_per_engineer is None:
            return df

        rng = np.random.default_rng(self.seed)
        sampled_dfs = []

        for engineer_id, eng_df in df.groupby("engineer_id"):
            sampled = self._sample_engineer(eng_df, rng)
            sampled_dfs.append(sampled)

        if not sampled_dfs:
            return df.iloc[0:0]  # Return empty DataFrame with same schema

        result = pd.concat(sampled_dfs, ignore_index=True)
        return result

    def _sample_engineer(self, eng_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """
        Sample one engineer's activities with balanced type distribution.

        Args:
            eng_df: Single engineer's activities
            rng: NumPy random generator

        Returns:
            Sampled activities for this engineer
        """
        if len(eng_df) <= self.max_per_engineer:
            return eng_df  # Keep all if under limit

        # Group by (source, activity_type) combination
        eng_df = eng_df.copy()
        eng_df["_type_key"] = eng_df["source"].astype(str) + ":" + eng_df["activity_type"].astype(str)
        by_type = {k: g for k, g in eng_df.groupby("_type_key")}

        # Calculate quotas with redistribution
        type_quotas = self._calculate_quotas(by_type)

        # Random sample within each type according to quotas
        sampled_indices = []
        for type_key, quota in type_quotas.items():
            type_df = by_type[type_key]
            if quota >= len(type_df):
                sampled_indices.extend(type_df.index.tolist())
            elif quota > 0:
                chosen = rng.choice(type_df.index.tolist(), size=quota, replace=False)
                sampled_indices.extend(chosen.tolist())

        return eng_df.loc[sampled_indices].drop(columns=["_type_key"])

    def _calculate_quotas(self, by_type: dict[str, pd.DataFrame]) -> dict[str, int]:
        """
        Calculate quota per type with iterative redistribution.

        Distributes max_per_engineer evenly across types. When a type
        has fewer items than its quota, the excess is redistributed
        to remaining types until all quota is allocated or all items used.

        Args:
            by_type: Dict mapping type_key to DataFrame of activities

        Returns:
            Dict mapping type_key to allocated quota
        """
        type_counts = {k: len(v) for k, v in by_type.items()}
        type_quotas = {k: 0 for k in by_type}
        remaining = self.max_per_engineer
        active_types = set(by_type.keys())

        while remaining > 0 and active_types:
            quota_per_type = max(1, remaining // len(active_types))
            exhausted = set()
            allocated_this_round = 0

            for type_key in list(active_types):
                available = type_counts[type_key] - type_quotas[type_key]
                allocation = min(quota_per_type, available, remaining - allocated_this_round)

                if allocation > 0:
                    type_quotas[type_key] += allocation
                    allocated_this_round += allocation

                if type_quotas[type_key] >= type_counts[type_key]:
                    exhausted.add(type_key)

            remaining -= allocated_this_round
            active_types -= exhausted

            if allocated_this_round == 0:
                break

        return type_quotas
