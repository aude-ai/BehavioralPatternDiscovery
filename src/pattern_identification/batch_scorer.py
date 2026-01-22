"""
Batch Scorer

Scores all messages through the VAE and stores activations at all
hierarchical levels for each encoder. Level names and encoder count
are read dynamically from the model.

This MUST run before SHAP analysis because message assignment uses
activation scores, not SHAP-based ideal embeddings.

Uses the same pre-built training input files as training to ensure
dimension consistency.
"""

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.model.base import BaseVAE

logger = logging.getLogger(__name__)


class BatchScorer:
    """Score all messages and store activations at all levels."""

    def __init__(self, config: dict[str, Any], device: str | None = None):
        """
        Args:
            config: Full merged config containing paths.pattern_identification
            device: Override device (uses config if None)
        """
        pi_config = config["paths"]["pattern_identification"]
        scoring_config = config["batch_scoring"]
        data_paths = config["paths"]["data"]["processing"]

        self.batch_size = scoring_config["batch_size"]
        self.device = device or scoring_config["device"]
        self.output_path = Path(pi_config["scoring"]["activations"])
        self.stats_path = Path(pi_config["scoring"]["population_stats"])

        # Determine which pre-built training input file to use
        aux_enabled = config["input"]["aux_features"]["enabled"]
        embeddings_dir = Path(data_paths["train_features"]).parent
        if aux_enabled:
            self.training_input_path = embeddings_dir / "train_input_with_aux.npy"
        else:
            self.training_input_path = embeddings_dir / "train_input_embedding_only.npy"

    def score_all(
        self,
        vae: BaseVAE,
        message_database: list[dict],
    ) -> dict[str, np.ndarray]:
        """
        Score all messages and store activations.

        Args:
            vae: Trained VAE model implementing BaseVAE interface
            message_database: List of message dicts (used for count verification)

        Returns:
            Dictionary of activation arrays keyed by level
        """
        vae.eval()
        vae.to(self.device)

        # Load pre-built training input (same as used during training)
        if not self.training_input_path.exists():
            raise FileNotFoundError(
                f"Training input file not found: {self.training_input_path}. "
                "Run training first to create the cached input file."
            )

        training_input = np.load(self.training_input_path)
        n_messages = training_input.shape[0]

        # Verify count matches message database
        if n_messages != len(message_database):
            raise ValueError(
                f"Training input count ({n_messages}) doesn't match "
                f"message database count ({len(message_database)})"
            )

        logger.info(f"Loaded training input: {self.training_input_path}")
        logger.info(f"  Shape: {training_input.shape}")
        logger.info(f"  Model input_dim: {vae.input_dim}")

        # Verify dimensions match model
        if training_input.shape[1] != vae.input_dim:
            raise ValueError(
                f"Training input dim ({training_input.shape[1]}) doesn't match "
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
                    vae, training_input, start, end, level_names, encoder_names
                )
                self._store_batch(activations, batch_activations, start, end)

        # Save to HDF5
        self._save_activations(activations)

        # Compute and save population statistics
        stats = self._compute_population_stats(activations)
        self._save_population_stats(stats)

        logger.info(f"Activations saved to {self.output_path}")
        return activations

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
        # Slice batch from pre-built training input
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

    def _save_activations(self, activations: dict[str, np.ndarray]) -> None:
        """Save activations to HDF5 file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.output_path, "w") as f:
            for key, arr in activations.items():
                f.create_dataset(key, data=arr, compression="gzip")

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

    def _save_population_stats(self, stats: dict[str, Any]) -> None:
        """Save population statistics to JSON."""
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    @staticmethod
    def load_activations(path: str | Path) -> dict[str, np.ndarray]:
        """Load activations from HDF5 file."""
        activations = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                activations[key] = f[key][:]
        return activations
