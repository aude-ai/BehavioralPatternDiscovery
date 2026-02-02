"""
Data Module

Contains data collection, synthetic generation, processing, and dataset utilities.

Structure:
- collection/: NDJSON loading and activity parsing
- synthetic/: LLM-based synthetic profile generation
- processing/: Text encoding, feature extraction, normalization, preprocessing
- datasets.py: PyTorch datasets for training

Note: Most imports are lazy to avoid torch dependency in environments that don't need it
(e.g., cloud workers that only use collection utilities).
"""

# These don't require torch
from .collection import EngineerManager

__all__ = [
    # Datasets (lazy loaded - requires torch)
    "EngineerSequenceDataset",
    "MultiEngineerDataset",
    "RandomMessageDataset",
    "engineer_collate_fn",
    "create_dataset",
    "dataset_registry",
    # Collection
    "EngineerManager",
    # Synthetic (lazy loaded)
    "SyntheticProfileGenerator",
    # Processing - Encoders (lazy loaded - requires torch)
    "BaseTextEncoder",
    "JinaEncoder",
    "NVEmbedEncoder",
    "QwenRawEncoder",
    "text_encoder_registry",
    "create_text_encoder",
    # Processing - Features (lazy loaded)
    "StatisticalFeatureExtractor",
    # Processing - Preprocessing (lazy loaded)
    "DataPreprocessor",
    # Processing - Normalization (lazy loaded)
    "BaseNormalization",
    "NormalizationPipeline",
    "normalization_registry",
    "L2Normalization",
    "MaxAbsNormalization",
    "StandardNormalization",
    "ZCANormalization",
    "QuantileNormalization",
    "IsotropicNormalization",
    "WinsorizeNormalization",
]


def __getattr__(name):
    """Lazy import for classes that may require torch or heavy dependencies."""

    # Dataset classes (require torch)
    if name in {
        "EngineerSequenceDataset",
        "MultiEngineerDataset",
        "RandomMessageDataset",
        "engineer_collate_fn",
        "create_dataset",
        "dataset_registry",
    }:
        from .datasets import (
            EngineerSequenceDataset,
            MultiEngineerDataset,
            RandomMessageDataset,
            engineer_collate_fn,
            create_dataset,
            dataset_registry,
        )
        return locals()[name]

    # Synthetic (may have heavy deps)
    if name == "SyntheticProfileGenerator":
        from .synthetic import SyntheticProfileGenerator
        return SyntheticProfileGenerator

    # Processing - Encoders (require torch)
    if name in {
        "BaseTextEncoder",
        "JinaEncoder",
        "NVEmbedEncoder",
        "QwenRawEncoder",
        "text_encoder_registry",
        "create_text_encoder",
    }:
        from .processing import (
            BaseTextEncoder,
            JinaEncoder,
            NVEmbedEncoder,
            QwenRawEncoder,
            text_encoder_registry,
            create_text_encoder,
        )
        return locals()[name]

    # Processing - Features
    if name == "StatisticalFeatureExtractor":
        from .processing import StatisticalFeatureExtractor
        return StatisticalFeatureExtractor

    # Processing - Preprocessing
    if name == "DataPreprocessor":
        from .processing import DataPreprocessor
        return DataPreprocessor

    # Processing - Normalization
    if name in {
        "BaseNormalization",
        "NormalizationPipeline",
        "normalization_registry",
        "L2Normalization",
        "MaxAbsNormalization",
        "StandardNormalization",
        "ZCANormalization",
        "QuantileNormalization",
        "IsotropicNormalization",
        "WinsorizeNormalization",
    }:
        from .processing import (
            BaseNormalization,
            NormalizationPipeline,
            normalization_registry,
            L2Normalization,
            MaxAbsNormalization,
            StandardNormalization,
            ZCANormalization,
            QuantileNormalization,
            IsotropicNormalization,
            WinsorizeNormalization,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
