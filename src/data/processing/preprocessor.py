"""
Data Preprocessor

Orchestrates the full preprocessing pipeline:
1. Load activities from CSV
2. Extract statistical features per engineer
3. Encode all message texts to raw embeddings (no normalization in encoders)
4. Apply normalization pipeline (if enabled in config)
5. Build message database for training

NORMALIZATION HANDLING:
- If a trained model checkpoint exists at the configured path:
  Load the normalization pipeline from the checkpoint and apply it.
  This ensures new data is normalized consistently with training data.
- If no model checkpoint exists and normalization is enabled:
  Fit a new normalization pipeline and apply it.
  The pipeline params will be saved in the checkpoint when training completes.
- If normalization is disabled (enabled: false in config):
  Raw embeddings are used. Normalization can be applied manually via the
  /api/normalization endpoint.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .encoders import create_text_encoder
from .statistical_features import StatisticalFeatureExtractor
from .normalizer import NormalizationPipeline

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Orchestrate the full preprocessing pipeline."""

    def __init__(self, config: dict):
        """
        Initialize preprocessor.

        Args:
            config: Full config dict with processing and paths sections
        """
        self.config = config
        self.data_paths = config["paths"]["data"]

        # Activity prefix config
        self.prefix_config = config["processing"]["activity_prefix"]

        # Model checkpoint path (to check if model exists)
        self.checkpoint_path = Path(config["paths"]["training"]["checkpoint"])

        self.text_encoder = create_text_encoder(config)
        self.feature_extractor = StatisticalFeatureExtractor(config)

        # Will be set during preprocessing
        self.normalization_pipeline: NormalizationPipeline | None = None
        self.normalization_params: dict[str, Any] | None = None

    def _apply_activity_prefix(self, text: str, activity_type: str) -> str:
        """
        Apply activity type prefix to text if enabled.

        Args:
            text: Original text content
            activity_type: Activity type from the data

        Returns:
            Text with prefix applied (or original if disabled)
        """
        if not self.prefix_config["enabled"]:
            return text

        mappings = self.prefix_config["mappings"]
        separator = self.prefix_config["separator"]
        default = self.prefix_config["default"]

        # Get prefix from mappings, fall back to default
        prefix = mappings.get(activity_type, default)

        return f"{prefix}{separator}{text}"

    def _load_normalization_from_checkpoint(self) -> NormalizationPipeline | None:
        """
        Load normalization pipeline from existing model checkpoint.

        Returns:
            NormalizationPipeline if checkpoint exists and has params, None otherwise
        """
        if not self.checkpoint_path.exists():
            logger.info(f"No model checkpoint found at {self.checkpoint_path}")
            return None

        logger.info(f"Loading normalization from checkpoint: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Check if checkpoint has preprocessing info with normalization
        preprocessing = checkpoint.get("preprocessing", {})
        norm_params = preprocessing.get("normalization")

        if norm_params is None:
            logger.info("Checkpoint exists but has no normalization params")
            return None

        pipeline = NormalizationPipeline.from_params(norm_params)
        logger.info(f"Loaded normalization pipeline from checkpoint: {norm_params['pipeline']}")

        return pipeline

    def _get_or_fit_normalization(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """
        Get normalization (from checkpoint or fit new) and apply it.

        Args:
            embeddings: Raw embeddings array

        Returns:
            Tuple of (normalized embeddings, normalization params or None)
        """
        # First, check if we should load from existing checkpoint
        existing_pipeline = self._load_normalization_from_checkpoint()

        if existing_pipeline is not None:
            # Use the pipeline from the checkpoint
            logger.info("Using normalization pipeline from existing model checkpoint")
            self.normalization_pipeline = existing_pipeline
            normalized = existing_pipeline.transform(embeddings)
            self.normalization_params = existing_pipeline.get_params()
            return normalized, self.normalization_params

        # No existing checkpoint - check if normalization is enabled
        norm_config = self.config["processing"]["normalization"]
        if not norm_config["enabled"]:
            logger.info("Normalization disabled in config (can apply manually via API)")
            return embeddings, None

        # Create new pipeline from config
        pipeline = NormalizationPipeline(self.config)

        if pipeline.is_empty:
            logger.info("Normalization pipeline is empty (no normalizations)")
            return embeddings, None

        # Fit new pipeline
        logger.info("Fitting new normalization pipeline")
        normalized = pipeline.fit_transform(embeddings)

        self.normalization_pipeline = pipeline
        self.normalization_params = pipeline.get_params()

        return normalized, self.normalization_params

    def preprocess(self) -> dict:
        """
        Run full preprocessing pipeline.

        Returns:
            Dict with status, statistics, and variance transform params (if any)
        """
        activities_path = Path(self.data_paths["collection"]["activities_csv"])

        logger.info("=" * 80)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 80)

        # Load activities
        logger.info(f"Loading activities from {activities_path}")
        if not activities_path.exists():
            raise FileNotFoundError(f"Activities file not found: {activities_path}")

        activities_df = pd.read_csv(activities_path)
        num_activities = len(activities_df)
        num_engineers = activities_df["engineer_id"].nunique()
        logger.info(f"Loaded {num_activities} activities from {num_engineers} engineers")

        # Apply balanced sampling if configured
        sampling_config = self.config["processing"].get("sampling", {})
        if sampling_config.get("enabled", False):
            from .activity_sampler import ActivitySampler

            max_per_eng = sampling_config.get("max_activities_per_engineer")
            seed = sampling_config.get("seed")

            if max_per_eng is not None:
                logger.info("-" * 80)
                logger.info(f"Applying balanced sampling: max {max_per_eng} per engineer, seed={seed}")

                sampler = ActivitySampler(max_per_engineer=max_per_eng, seed=seed)
                before_count = len(activities_df)
                activities_df = sampler.sample(activities_df)

                logger.info(f"Sampled {len(activities_df)} activities (removed {before_count - len(activities_df)})")

                # Log distribution by type
                type_dist = activities_df.groupby(["source", "activity_type"]).size()
                logger.info(f"Distribution by type: {type_dist.to_dict()}")

        # Extract statistical features (per engineer)
        logger.info("-" * 80)
        logger.info("Extracting statistical features...")
        features_by_engineer_raw = self.feature_extractor.extract(activities_df)

        # Normalize aux features to [-1, 1] for training
        logger.info("-" * 80)
        logger.info("Normalizing aux features...")
        features_by_engineer, aux_scale_factors = self.feature_extractor.normalize_features(
            features_by_engineer_raw
        )

        # Encode all message texts
        logger.info("-" * 80)
        logger.info("Encoding message texts...")

        # Apply activity type prefixes if enabled
        if self.prefix_config["enabled"]:
            logger.info("Applying activity type prefixes...")
            texts = [
                self._apply_activity_prefix(
                    str(row.get("text", "") or ""),
                    str(row.get("activity_type", "") or "")
                )
                for _, row in activities_df.iterrows()
            ]
            logger.info(f"Example prefixed text: {texts[0][:100]}...")
        else:
            texts = activities_df["text"].fillna("").tolist()

        embeddings = self.text_encoder.encode(texts)
        logger.info(f"Encoded {len(texts)} texts to shape {embeddings.shape}")

        # Apply normalization (load from checkpoint or fit new)
        logger.info("-" * 80)
        embeddings_np, normalization_params = self._get_or_fit_normalization(embeddings)

        # Build message database (the format training expects)
        logger.info("-" * 80)
        logger.info("Building message database...")
        message_database = []

        for idx, row in activities_df.iterrows():
            eng_id = str(row["engineer_id"])
            message_database.append({
                "engineer_id": eng_id,
                "embedding": embeddings_np[idx],
                "aux_features": features_by_engineer[eng_id],
                "text": row.get("text", ""),
                "source": row.get("source", ""),
                "activity_type": row.get("activity_type", ""),
                "timestamp": row.get("timestamp"),
            })

        # Build metadata
        metadata = {
            "embedding_dim": embeddings_np.shape[1],
            "aux_features_dim": features_by_engineer[list(features_by_engineer.keys())[0]].shape[0],
            "num_messages": len(message_database),
            "num_engineers": len(features_by_engineer),
            "encoder_type": self.config["processing"]["text_encoder"]["type"],
            "encoder_model": self.text_encoder.model_name,
            # Store FULL normalization params so training can load them
            # This is critical: preprocessing and training are separate script runs,
            # so we need to persist the fitted pipeline for training to save in checkpoint
            "normalization_params": normalization_params,
            "created_at": datetime.now().isoformat(),
        }

        # Create output structure with metadata
        output_data = {
            "messages": message_database,
            "metadata": metadata,
        }

        # Save outputs
        logger.info("-" * 80)
        logger.info("Saving outputs...")

        output_dir = Path(self.data_paths["processing"]["message_database"]).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save message database with metadata
        message_db_path = Path(self.data_paths["processing"]["message_database"])
        with open(message_db_path, "wb") as f:
            pickle.dump(output_data, f)
        logger.info(f"Saved message database to {message_db_path}")

        # Save embeddings (transformed if applicable)
        embeddings_path = Path(self.data_paths["processing"]["train_features"])
        np.save(embeddings_path, embeddings_np)
        logger.info(f"Saved embeddings ({embeddings_np.shape}) to {embeddings_path}")

        # Save aux features (normalized)
        aux_features_np = np.array([
            features_by_engineer[str(row["engineer_id"])]
            for _, row in activities_df.iterrows()
        ])
        aux_path = Path(self.data_paths["processing"]["train_aux_vars"])
        np.save(aux_path, aux_features_np)
        logger.info(f"Saved aux features ({aux_features_np.shape}) to {aux_path}")

        # Save scale factors for denormalization if needed
        scale_path = aux_path.parent / "aux_scale_factors.npy"
        np.save(scale_path, aux_scale_factors)
        logger.info(f"Saved aux scale factors ({aux_scale_factors.shape}) to {scale_path}")

        # Save metadata
        metadata_path = Path(self.data_paths["processing"]["train_metadata"])
        activities_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")

        logger.info("=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"  Messages: {len(message_database)}")
        logger.info(f"  Engineers: {len(features_by_engineer)}")
        logger.info(f"  Embedding dim: {embeddings_np.shape[1]}")
        logger.info(f"  Aux features dim: {aux_features_np.shape[1]}")
        if normalization_params:
            logger.info(f"  Normalization pipeline: {normalization_params['pipeline']} (params saved in message_database)")
        logger.info("=" * 80)

        return {
            "status": "success",
            "num_messages": len(message_database),
            "num_engineers": len(features_by_engineer),
            "embedding_dim": embeddings_np.shape[1],
            "aux_features_dim": aux_features_np.shape[1],
            "normalization_params": normalization_params,
        }

    def get_normalization_params(self) -> dict[str, Any] | None:
        """
        Get the normalization params for saving in model checkpoint.

        Call this after preprocess() to get the pipeline params that were used.

        Returns:
            Pipeline params dict or None if no normalization was used
        """
        return self.normalization_params
