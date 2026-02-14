"""
Data Processing Module

Handles text encoding, statistical feature extraction, normalization,
and activity sampling.
"""

from .encoders import (
    BaseTextEncoder,
    JinaV3Encoder,
    JinaV4Encoder,
    NVEmbedEncoder,
    QwenRawEncoder,
    Qwen3EmbeddingEncoder,
    text_encoder_registry,
    create_text_encoder,
)
from .statistical_features import StatisticalFeatureExtractor
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

__all__ = [
    # Encoders
    "BaseTextEncoder",
    "JinaV3Encoder",
    "JinaV4Encoder",
    "NVEmbedEncoder",
    "QwenRawEncoder",
    "Qwen3EmbeddingEncoder",
    "text_encoder_registry",
    "create_text_encoder",
    # Features
    "StatisticalFeatureExtractor",
    # Sampling
    "ActivitySampler",
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
