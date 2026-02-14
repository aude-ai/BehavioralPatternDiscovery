"""
Individual Engineer Scorer

Scores a single engineer's messages through the VAE and computes
percentiles against the existing population from batch scoring.

Designed for remote execution (Modal) - accepts data directly and
returns results without file I/O.
"""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class IndividualScorer:
    """Score individual engineers against the population."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize individual scorer.

        Args:
            config: Config containing scoring section
        """
        scoring_config = config.get("scoring", {})
        self.batch_size = scoring_config.get("batch_size", 64)
        self.device = scoring_config.get("device", "cuda")

    def score_engineer(
        self,
        engineer_id: str,
        vae: torch.nn.Module,
        messages: list[dict],
        population_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score a single engineer against the population.

        Args:
            engineer_id: Engineer identifier
            vae: Trained VAE model
            messages: List of message dicts with 'embedding' key
            population_stats: Pre-computed population statistics from BatchScorer

        Returns:
            Engineer scores with patterns and percentiles
        """
        logger.info(f"Scoring engineer: {engineer_id}")

        n_messages = len(messages)
        if n_messages == 0:
            raise ValueError(f"No messages found for engineer: {engineer_id}")

        logger.info(f"Found {n_messages} messages for {engineer_id}")

        # Extract embeddings
        embeddings = np.stack([m["embedding"] for m in messages])

        # Score through VAE to get activations
        activations = self._score_through_vae(vae, embeddings)

        # Compute EB-shrunk scores using population parameters
        engineer_scores = self._compute_engineer_scores(
            activations, population_stats, n_messages
        )

        return {
            "engineer_id": engineer_id,
            "n_messages": n_messages,
            "scores": engineer_scores,
        }

    def _score_through_vae(
        self,
        vae: torch.nn.Module,
        embeddings: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Pass embeddings through VAE to get activations at all levels.

        Args:
            vae: Trained VAE model
            embeddings: (n_messages, embedding_dim) array

        Returns:
            Dict mapping level keys to (n_messages, n_dims) arrays
        """
        vae.eval()
        device = next(vae.parameters()).device

        n_messages = embeddings.shape[0]
        level_names = vae.get_level_names()
        encoder_names = vae.get_encoder_names()

        activations = {f"{enc}_{lvl}": [] for enc in encoder_names for lvl in level_names}
        activations["unified"] = []

        # Process in batches
        for start_idx in range(0, n_messages, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_messages)
            batch = torch.FloatTensor(embeddings[start_idx:end_idx]).to(device)

            with torch.no_grad():
                latent_codes = vae.get_latent_codes(batch, deterministic=True)

                for enc_name in encoder_names:
                    for level in level_names:
                        key = f"{enc_name}_{level}"
                        activations[key].append(latent_codes[enc_name][level].cpu().float().numpy())

                activations["unified"].append(latent_codes["unified"]["z"].cpu().float().numpy())

        # Concatenate batches
        for key in activations:
            activations[key] = np.concatenate(activations[key], axis=0)

        return activations

    def _compute_engineer_scores(
        self,
        activations: dict[str, np.ndarray],
        population_stats: dict[str, Any],
        n_messages: int,
    ) -> dict[str, dict[str, Any]]:
        """
        Compute EB-shrunk scores using population parameters.

        Args:
            activations: Per-level activation arrays for this engineer
            population_stats: Pre-computed population statistics
            n_messages: Number of messages for this engineer

        Returns:
            Per-level scores with percentiles
        """
        engineer_scores = {}

        for level_key, act_array in activations.items():
            if level_key not in population_stats:
                logger.warning(f"No population stats for {level_key}, skipping")
                continue

            pop_level = population_stats[level_key]

            # Compute engineer's raw mean for this level
            eng_mean = act_array.mean(axis=0)
            eng_var = act_array.var(axis=0) if n_messages > 1 else np.zeros_like(eng_mean)

            # Get population parameters
            pop_mean = np.array(pop_level["population_mean"])
            pop_var = np.array(pop_level["population_var"])

            # Apply EB shrinkage
            within_var = eng_var
            shrinkage = within_var / (within_var + n_messages * pop_var + 1e-8)
            posterior_mean = shrinkage * pop_mean + (1 - shrinkage) * eng_mean

            # Compute percentiles against population
            percentiles = self._compute_percentiles(posterior_mean, pop_level)

            engineer_scores[level_key] = {
                "raw_mean": eng_mean.tolist(),
                "posterior_mean": posterior_mean.tolist(),
                "percentiles": percentiles,
                "shrinkage": float(shrinkage.mean()),
            }

        return engineer_scores

    def _compute_percentiles(
        self,
        posterior_mean: np.ndarray,
        pop_level: dict[str, Any],
    ) -> list[int]:
        """
        Compute percentile for each dimension against population.

        Args:
            posterior_mean: This engineer's posterior mean scores
            pop_level: Population statistics for this level

        Returns:
            List of percentiles (0-100) for each dimension
        """
        percentiles = []

        for dim in range(len(posterior_mean)):
            p25 = pop_level["percentiles"]["25"][dim]
            p50 = pop_level["percentiles"]["50"][dim]
            p75 = pop_level["percentiles"]["75"][dim]
            val = posterior_mean[dim]

            # Interpolate percentile based on quartiles
            if val <= p25:
                # Below 25th percentile
                pop_min = pop_level["min"][dim]
                if p25 > pop_min:
                    pct = int(25 * (val - pop_min) / (p25 - pop_min + 1e-8))
                else:
                    pct = 0
            elif val <= p50:
                # Between 25th and 50th
                pct = 25 + int(25 * (val - p25) / (p50 - p25 + 1e-8))
            elif val <= p75:
                # Between 50th and 75th
                pct = 50 + int(25 * (val - p50) / (p75 - p50 + 1e-8))
            else:
                # Above 75th percentile
                pop_max = pop_level["max"][dim]
                if pop_max > p75:
                    pct = 75 + int(25 * (val - p75) / (pop_max - p75 + 1e-8))
                else:
                    pct = 100

            percentiles.append(max(0, min(100, pct)))

        return percentiles
