"""
Training Data Preparer

Prepares final training input data by combining embeddings and aux_features
based on model configuration. Caches results for fast switching between
aux_features enabled/disabled without re-preprocessing.

File naming convention:
- train_input_with_aux.npy: embeddings + aux_features concatenated
- train_input_embedding_only.npy: embeddings only

This allows both formats to coexist for easy config toggling.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class TrainingDataPreparer:
    """
    Prepares training input data based on aux_features config.

    Caches combined files so switching aux_features.enabled doesn't
    require re-running expensive embedding generation.
    """

    def __init__(self, config: dict):
        """
        Initialize preparer with paths from config.

        Args:
            config: Full config dict with paths section
        """
        self.data_paths = config["paths"]["data"]["processing"]

        # Base data files (from preprocessing)
        self.embeddings_path = Path(self.data_paths["train_features"])
        self.aux_features_path = Path(self.data_paths["train_aux_vars"])

        # Cached combined files
        output_dir = self.embeddings_path.parent
        self.combined_path = output_dir / "train_input_with_aux.npy"
        self.embedding_only_path = output_dir / "train_input_embedding_only.npy"

    def prepare(self, aux_features_enabled: bool) -> Path:
        """
        Prepare training input data based on config.

        Creates cached file if it doesn't exist, otherwise loads existing.

        Args:
            aux_features_enabled: Whether to include aux features

        Returns:
            Path to the prepared training input file
        """
        if aux_features_enabled:
            return self._prepare_with_aux()
        else:
            return self._prepare_embedding_only()

    def _prepare_with_aux(self) -> Path:
        """Prepare combined embeddings + aux_features."""
        if self.combined_path.exists():
            logger.info(f"Loading cached training input (with aux): {self.combined_path}")
            return self.combined_path

        logger.info("Creating training input with aux_features...")

        # Load base files
        embeddings = np.load(self.embeddings_path)
        aux_features = np.load(self.aux_features_path)

        logger.info(f"  Embeddings shape: {embeddings.shape}")
        logger.info(f"  Aux features shape: {aux_features.shape}")

        # Validate shapes
        if embeddings.shape[0] != aux_features.shape[0]:
            raise ValueError(
                f"Row count mismatch: embeddings={embeddings.shape[0]}, "
                f"aux_features={aux_features.shape[0]}"
            )

        # Concatenate
        combined = np.concatenate([embeddings, aux_features], axis=1)
        logger.info(f"  Combined shape: {combined.shape}")

        # Save
        np.save(self.combined_path, combined)
        logger.info(f"Saved combined training input to {self.combined_path}")

        return self.combined_path

    def _prepare_embedding_only(self) -> Path:
        """Prepare embeddings-only input."""
        if self.embedding_only_path.exists():
            logger.info(f"Loading cached training input (embedding only): {self.embedding_only_path}")
            return self.embedding_only_path

        logger.info("Creating training input (embedding only)...")

        # Load embeddings
        embeddings = np.load(self.embeddings_path)
        logger.info(f"  Embeddings shape: {embeddings.shape}")

        # Save as the "prepared" format (same data, different path for clarity)
        np.save(self.embedding_only_path, embeddings)
        logger.info(f"Saved embedding-only training input to {self.embedding_only_path}")

        return self.embedding_only_path

    def get_dimensions(self, aux_features_enabled: bool) -> dict:
        """
        Get data dimensions without loading full arrays.

        Args:
            aux_features_enabled: Whether aux features will be included

        Returns:
            Dict with embedding_dim, aux_features_dim, input_dim
        """
        # Load just enough to get shapes
        embeddings = np.load(self.embeddings_path, mmap_mode='r')
        aux_features = np.load(self.aux_features_path, mmap_mode='r')

        embedding_dim = embeddings.shape[1]
        aux_features_dim = aux_features.shape[1]

        if aux_features_enabled:
            input_dim = embedding_dim + aux_features_dim
        else:
            input_dim = embedding_dim

        return {
            "embedding_dim": embedding_dim,
            "aux_features_dim": aux_features_dim,
            "input_dim": input_dim,
            "num_samples": embeddings.shape[0],
        }

    def clear_cache(self):
        """Remove cached combined files to force regeneration."""
        for path in [self.combined_path, self.embedding_only_path]:
            if path.exists():
                path.unlink()
                logger.info(f"Removed cached file: {path}")


def prepare_training_data(config: dict) -> tuple[Path, dict]:
    """
    Convenience function to prepare training data based on config.

    Args:
        config: Full config dict

    Returns:
        Tuple of (path to prepared data, dimensions dict)
    """
    aux_features_enabled = config["input"]["aux_features"]["enabled"]

    preparer = TrainingDataPreparer(config)
    data_path = preparer.prepare(aux_features_enabled)
    dimensions = preparer.get_dimensions(aux_features_enabled)

    return data_path, dimensions
