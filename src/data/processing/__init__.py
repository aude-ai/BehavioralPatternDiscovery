"""
Data Processing Module

Handles text encoding, statistical feature extraction, normalization,
training data preparation, and preprocessing orchestration.
"""

from .encoders import (
    BaseTextEncoder,
    JinaEncoder,
    NVEmbedEncoder,
    QwenRawEncoder,
    text_encoder_registry,
    create_text_encoder,
)
from .statistical_features import StatisticalFeatureExtractor
from .preprocessor import DataPreprocessor
from .activity_sampler import ActivitySampler
from .normalizer import (
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
from .training_data_preparer import (
    TrainingDataPreparer,
    prepare_training_data,
)

__all__ = [
    # Encoders
    "BaseTextEncoder",
    "JinaEncoder",
    "NVEmbedEncoder",
    "QwenRawEncoder",
    "text_encoder_registry",
    "create_text_encoder",
    # Features
    "StatisticalFeatureExtractor",
    # Preprocessing
    "DataPreprocessor",
    # Sampling
    "ActivitySampler",
    # Training data preparation
    "TrainingDataPreparer",
    "prepare_training_data",
    # Normalization
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
