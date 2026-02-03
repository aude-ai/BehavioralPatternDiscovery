"""
Batch Scorer

Scores all messages through the VAE and returns activations at all
hierarchical levels for each encoder. Level names and encoder count
are read dynamically from the model.

This MUST run before SHAP analysis because message assignment uses
activation scores, not SHAP-based ideal embeddings.

Designed for remote execution (Modal) - accepts data directly and
returns results without file I/O.
"""

import logging
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from src.model.base import BaseVAE

logger = logging.getLogger(__name__)


class BatchScorer:
    """Score all messages and return activations at all levels."""

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Config containing batch_scoring section
        """
        scoring_config = config.get("batch_scoring", {})
        self.batch_size = scoring_config.get("batch_size", 64)
        self.device = scoring_config.get("device", "cuda")

    def score_all(
        self,
        vae: BaseVAE,
        train_input: np.ndarray,
        progress_callback: Callable[[float, int, int], None] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Score all messages and return activations with population statistics.

        Args:
            vae: Trained VAE model implementing BaseVAE interface
            train_input: (n_messages, input_dim) array of training input
            progress_callback: Optional callback(progress, processed, total)

        Returns:
            Tuple of (activations dict, population_stats dict)
        """
        vae.eval()
        vae.to(self.device)

        n_messages = train_input.shape[0]

        # Verify dimensions match model
        if train_input.shape[1] != vae.input_dim:
            raise ValueError(
                f"Training input dim ({train_input.shape[1]}) doesn't match "
                f"model input_dim ({vae.input_dim}). "
                "The model was trained with different data or config."
            )

        dims = vae.latent_dims
        unified_dim = vae.unified_dim

        # Get dynamic level and encoder names from model
        level_names = vae.get_level_names()
        encoder_names = vae.get_encoder_names()

        # Pre-allocate activation arrays
        activations = self._allocate_arrays(
            n_messages, dims, unified_dim, level_names, encoder_names
        )

        logger.info(f"Scoring {n_messages} messages...")
        logger.info(f"  Encoders: {encoder_names}")
        logger.info(f"  Levels: {level_names}")

        with torch.no_grad():
            for start in tqdm(range(0, n_messages, self.batch_size), desc="Batch scoring"):
                end = min(start + self.batch_size, n_messages)
                batch_activations = self._score_batch(
                    vae, train_input, start, end, level_names, encoder_names
                )
                self._store_batch(activations, batch_activations, start, end)

                if progress_callback:
                    progress = min(end / n_messages, 1.0)
                    progress_callback(progress, end, n_messages)

        # Compute population statistics
        population_stats = self._compute_population_stats(activations)

        logger.info(f"Scoring complete: {n_messages} messages")
        return activations, population_stats

    def _allocate_arrays(
        self,
        n_messages: int,
        dims: dict[str, int],
        unified_dim: int,
        level_names: list[str],
        encoder_names: list[str],
    ) -> dict[str, np.ndarray]:
        """Pre-allocate numpy arrays for all activations."""
        arrays = {}

        # Per encoder, per level (N encoders x L levels arrays)
        for enc_name in encoder_names:
            for level in level_names:
                key = f"{enc_name}_{level}"
                arrays[key] = np.zeros((n_messages, dims[level]), dtype=np.float32)

        # Unified
        arrays["unified"] = np.zeros((n_messages, unified_dim), dtype=np.float32)

        return arrays

    def _score_batch(
        self,
        vae: BaseVAE,
        training_input: np.ndarray,
        start: int,
        end: int,
        level_names: list[str],
        encoder_names: list[str],
    ) -> dict[str, np.ndarray]:
        """Score a batch of messages through VAE."""
        # Slice batch from training input
        batch_data = training_input[start:end]
        x_tensor = torch.from_numpy(batch_data).float().to(self.device)

        # Get latent codes (deterministic for scoring)
        latent_codes = vae.get_latent_codes(x_tensor, deterministic=True)

        # Convert to numpy dynamically based on model structure
        batch_activations = {}
        for enc_name in encoder_names:
            for level in level_names:
                key = f"{enc_name}_{level}"
                batch_activations[key] = latent_codes[enc_name][level].cpu().numpy()

        batch_activations["unified"] = latent_codes["unified"]["z"].cpu().numpy()

        return batch_activations

    def _store_batch(
        self,
        activations: dict[str, np.ndarray],
        batch_activations: dict[str, np.ndarray],
        start: int,
        end: int,
    ) -> None:
        """Store batch results into pre-allocated arrays."""
        for key, batch_arr in batch_activations.items():
            activations[key][start:end] = batch_arr

    def _compute_population_stats(
        self,
        activations: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Compute population-level statistics for Empirical Bayes."""
        stats = {}

        for key, arr in activations.items():
            # Per-dimension statistics
            stats[key] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "percentiles": {
                    "25": np.percentile(arr, 25, axis=0).tolist(),
                    "50": np.percentile(arr, 50, axis=0).tolist(),
                    "75": np.percentile(arr, 75, axis=0).tolist(),
                },
            }

        return stats
