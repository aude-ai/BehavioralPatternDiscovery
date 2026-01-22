"""
Data Module

Contains data collection, synthetic generation, processing, and dataset utilities.

Structure:
- collection/: MongoDB loading and activity parsing
- synthetic/: LLM-based synthetic profile generation
- processing/: Text encoding, feature extraction, normalization, preprocessing
- datasets.py: PyTorch datasets for training
"""

from .datasets import (
    EngineerSequenceDataset,
    MultiEngineerDataset,
    RandomMessageDataset,
    engineer_collate_fn,
    create_dataset,
    dataset_registry,
)
from .collection import MongoDBLoader, EngineerManager
from .synthetic import SyntheticProfileGenerator
from .processing import (
    BaseTextEncoder,
    JinaEncoder,
    NVEmbedEncoder,
    QwenRawEncoder,
    text_encoder_registry,
    create_text_encoder,
    StatisticalFeatureExtractor,
    DataPreprocessor,
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

__all__ = [
    # Datasets
    "EngineerSequenceDataset",
    "MultiEngineerDataset",
    "RandomMessageDataset",
    "engineer_collate_fn",
    "create_dataset",
    "dataset_registry",
    # Collection
    "MongoDBLoader",
    "EngineerManager",
    # Synthetic
    "SyntheticProfileGenerator",
    # Processing - Encoders
    "BaseTextEncoder",
    "JinaEncoder",
    "NVEmbedEncoder",
    "QwenRawEncoder",
    "text_encoder_registry",
    "create_text_encoder",
    # Processing - Features
    "StatisticalFeatureExtractor",
    # Processing - Preprocessing
    "DataPreprocessor",
    # Processing - Normalization
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
