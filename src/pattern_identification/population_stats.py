"""
Population Statistics

Computes population-level statistics needed for Empirical Bayes scoring:
- Per-dimension mean, variance
- Per-engineer aggregated scores
- Between/within engineer variance decomposition
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PopulationStats:
    """Compute and store population-level statistics."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: Full merged config containing paths.pattern_identification.
                    Optional for cloud usage where save() is not called.
        """
        if config and "paths" in config and "pattern_identification" in config["paths"]:
            pi_config = config["paths"]["pattern_identification"]
            self.output_path = Path(pi_config["scoring"]["population_stats"])
        else:
            self.output_path = None

    def compute_engineer_scores(
        self,
        activations: dict[str, np.ndarray],
        message_database: list[dict],
    ) -> dict[str, Any]:
        """
        Compute per-engineer aggregated scores with Empirical Bayes shrinkage.

        Args:
            activations: Output from BatchScorer
            message_database: List of messages with 'engineer_id' field

        Returns:
            Engineer scores with shrinkage applied
        """
        # Group messages by engineer
        engineer_indices = {}
        for idx, msg in enumerate(message_database):
            eng_id = msg["engineer_id"]
            if eng_id not in engineer_indices:
                engineer_indices[eng_id] = []
            engineer_indices[eng_id].append(idx)

        logger.info(f"Computing scores for {len(engineer_indices)} engineers...")

        engineer_scores = {}

        for level_key, act_array in activations.items():
            # Compute per-engineer means
            engineer_means = {}
            engineer_vars = {}
            engineer_counts = {}

            for eng_id, indices in engineer_indices.items():
                eng_acts = act_array[indices]
                engineer_means[eng_id] = eng_acts.mean(axis=0)
                engineer_vars[eng_id] = eng_acts.var(axis=0) if len(indices) > 1 else np.zeros_like(eng_acts[0])
                engineer_counts[eng_id] = len(indices)

            # Compute population parameters
            all_means = np.stack(list(engineer_means.values()))
            pop_mean = all_means.mean(axis=0)
            pop_var = all_means.var(axis=0)  # Between-engineer variance

            # Average within-engineer variance
            within_var = np.mean(list(engineer_vars.values()), axis=0)

            # Apply Empirical Bayes shrinkage
            shrunk_scores = {}
            for eng_id in engineer_indices:
                n = engineer_counts[eng_id]
                eng_mean = engineer_means[eng_id]

                # Shrinkage factor: more messages = less shrinkage
                # Avoid division by zero
                shrinkage = within_var / (within_var + n * pop_var + 1e-8)
                posterior_mean = shrinkage * pop_mean + (1 - shrinkage) * eng_mean

                # Convert to percentiles
                percentiles = np.zeros_like(posterior_mean)
                for dim in range(len(pop_mean)):
                    dim_values = all_means[:, dim]
                    percentiles[dim] = (
                        (dim_values < posterior_mean[dim]).sum() / len(dim_values)
                    ) * 100

                shrunk_scores[eng_id] = {
                    "raw_mean": eng_mean.tolist(),
                    "posterior_mean": posterior_mean.tolist(),
                    "percentiles": percentiles.tolist(),
                    "n_messages": n,
                    "shrinkage": float(shrinkage.mean()),
                }

            # Compute reference distribution from engineer posterior means
            # (used by IndividualScorer for percentile computation)
            all_posterior = np.stack(
                [np.array(s["posterior_mean"]) for s in shrunk_scores.values()]
            )

            engineer_scores[level_key] = {
                "population_mean": pop_mean.tolist(),
                "population_var": pop_var.tolist(),
                "within_var": within_var.tolist(),
                "std": all_posterior.std(axis=0).tolist(),
                "min": all_posterior.min(axis=0).tolist(),
                "max": all_posterior.max(axis=0).tolist(),
                "percentiles": {
                    "25": np.percentile(all_posterior, 25, axis=0).tolist(),
                    "50": np.percentile(all_posterior, 50, axis=0).tolist(),
                    "75": np.percentile(all_posterior, 75, axis=0).tolist(),
                },
                "engineers": shrunk_scores,
            }

            logger.info(f"Computed scores for {level_key}: {len(shrunk_scores)} engineers")

        return engineer_scores

    def save(self, stats: dict[str, Any]) -> None:
        """Save statistics to JSON."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved population stats to {self.output_path}")

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load statistics from JSON."""
        with open(path, "r") as f:
            return json.load(f)
