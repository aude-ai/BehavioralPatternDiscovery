"""
Individual Engineer Scorer

Scores a single engineer's messages through the VAE and computes
percentiles against the existing population from batch scoring.

Reuses population statistics already computed by PopulationStats
during the batch scoring phase.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.pattern_identification import PopulationStats

logger = logging.getLogger(__name__)


class IndividualScorer:
    """Score individual engineers against the population."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize individual scorer.

        Args:
            config: Full merged configuration
        """
        self.config = config
        self.paths = config["paths"]
        self.scoring_config = config["scoring"]

        self.batch_size = self.scoring_config["batch_size"]
        self.device = self.scoring_config["device"]

        self.output_dir = Path(self.paths["scoring"]["individual_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Path to population stats from batch scoring
        self.population_stats_path = Path(
            self.paths["pattern_identification"]["scoring"]["population_stats"]
        )

    def score_engineer(
        self,
        engineer_id: str,
        vae: torch.nn.Module,
        message_database: list[dict],
        pattern_names: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score a single engineer against the population.

        Args:
            engineer_id: Engineer to score
            vae: Trained VAE model
            message_database: All messages with embeddings
            pattern_names: Pattern names from LLM naming

        Returns:
            Engineer scores with patterns and percentiles
        """
        logger.info(f"Scoring engineer: {engineer_id}")

        # Load population stats (computed during batch scoring)
        population_stats = PopulationStats.load(self.population_stats_path)

        # Filter messages for this engineer
        engineer_messages = [m for m in message_database if m["engineer_id"] == engineer_id]
        n_messages = len(engineer_messages)

        if n_messages == 0:
            raise ValueError(f"No messages found for engineer: {engineer_id}")

        logger.info(f"Found {n_messages} messages for {engineer_id}")

        # Extract embeddings
        embeddings = np.stack([m["embedding"] for m in engineer_messages])

        # Score through VAE to get activations
        activations = self._score_through_vae(vae, embeddings)

        # Compute EB-shrunk scores using population parameters
        engineer_scores = self._compute_engineer_scores(activations, population_stats, n_messages)

        # Build pattern list with names and percentiles
        patterns = self._build_pattern_list(engineer_scores, pattern_names)

        # Build result
        result = {
            "engineer_id": engineer_id,
            "n_messages": n_messages,
            "patterns": patterns,
        }

        # Save to file
        output_path = self.output_dir / f"scores_{engineer_id}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved scores to {output_path}")

        return result

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
                        activations[key].append(latent_codes[enc_name][level].cpu().numpy())

                activations["unified"].append(latent_codes["unified"]["z"].cpu().numpy())

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

        Uses the same shrinkage formula as PopulationStats.compute_engineer_scores().

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
            within_var = np.array(pop_level["within_var"])

            # Apply same EB shrinkage formula as PopulationStats
            shrinkage = within_var / (within_var + n_messages * pop_var + 1e-8)
            posterior_mean = shrinkage * pop_mean + (1 - shrinkage) * eng_mean

            # Compute percentiles against existing engineers
            existing_engineers = pop_level.get("engineers", {})
            percentiles = self._compute_percentiles(posterior_mean, existing_engineers)

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
        existing_engineers: dict[str, Any],
    ) -> list[int]:
        """
        Compute percentile for each dimension against existing population.

        Args:
            posterior_mean: This engineer's posterior mean scores
            existing_engineers: Dict of existing engineer scores

        Returns:
            List of percentiles (0-100) for each dimension
        """
        if not existing_engineers:
            return [50] * len(posterior_mean)

        # Collect all existing posterior means
        all_means = np.array([
            eng_data["posterior_mean"]
            for eng_data in existing_engineers.values()
        ])

        percentiles = []
        for dim in range(len(posterior_mean)):
            dim_values = all_means[:, dim]
            pct = (dim_values < posterior_mean[dim]).sum() / len(dim_values) * 100
            percentiles.append(int(pct))

        return percentiles

    def _build_pattern_list(
        self,
        engineer_scores: dict[str, dict[str, Any]],
        pattern_names: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Build list of patterns with names and percentiles.

        Args:
            engineer_scores: Computed scores per level
            pattern_names: LLM-generated pattern names

        Returns:
            List of pattern dicts for frontend
        """
        patterns = []

        for level_key, level_data in engineer_scores.items():
            # Parse level key (e.g., "enc1_bottom" or "unified")
            if level_key == "unified":
                encoder = "unified"
                level = "unified"
            else:
                parts = level_key.split("_")
                encoder = parts[0]
                level = parts[1]

            posterior_mean = level_data["posterior_mean"]
            percentiles = level_data["percentiles"]

            for dim_idx in range(len(posterior_mean)):
                # Get pattern name
                name = f"{level}_{dim_idx}"
                if level_key in pattern_names:
                    pattern_key = f"{level}_{dim_idx}" if level != "unified" else f"unified_{dim_idx}"
                    if pattern_key in pattern_names[level_key]:
                        name = pattern_names[level_key][pattern_key].get("name", name)

                patterns.append({
                    "id": f"{level_key}_{dim_idx}",
                    "encoder": encoder,
                    "level": level,
                    "dim_idx": dim_idx,
                    "name": name,
                    "score": posterior_mean[dim_idx],
                    "percentile": percentiles[dim_idx],
                })

        return patterns

    def check_score_exists(self, engineer_id: str) -> bool:
        """Check if scores exist for an engineer."""
        output_path = self.output_dir / f"scores_{engineer_id}.json"
        return output_path.exists()

    def load_scores(self, engineer_id: str) -> dict[str, Any] | None:
        """Load existing scores for an engineer."""
        output_path = self.output_dir / f"scores_{engineer_id}.json"
        if not output_path.exists():
            return None

        with open(output_path, "r") as f:
            return json.load(f)
