"""
Normalization Layers

Provides all normalization layers used in the model. All normalizations
are registered in normalization_registry for configuration-driven selection.

Supported normalizations:
- none: No normalization (identity pass-through)
- rms_norm: Root Mean Square Normalization (recommended)
- layer_norm: Layer Normalization
- dynamic_tanh: Tanh-based normalization (caps values to [-1, 1])
- scaled_rms_norm: RMSNorm with configurable strength
- scaled_layer_norm: LayerNorm with configurable strength
- batch_norm: Batch Normalization (not recommended for variable batch sizes)

SHAP Compatibility: All normalizations use standard PyTorch operations
that maintain gradient flow for SHAP analysis.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from src.core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Registry for normalization layers
normalization_registry = ComponentRegistry[nn.Module]("normalization")


@normalization_registry.register("none")
class NoNorm(nn.Module):
    """
    No normalization (identity pass-through).

    Use when normalization is not desired, e.g., for discriminators.
    """

    def __init__(
        self,
        num_features: int,
        eps: float,
        elementwise_affine: bool,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@normalization_registry.register("rms_norm")
class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).

    Formula: (x / RMS(x)) * gamma
    where RMS(x) = sqrt(eps + mean(x^2))
          gamma = learnable scale parameter (per-feature)

    Advantages:
    - Batch-independent: works with any batch size
    - No value capping: preserves full information range
    - Faster than LayerNorm (7-64% speedup)
    - State-of-the-art: used in Llama 3, Gemma, Mistral
    """

    def __init__(
        self,
        num_features: int,
        eps: float,
        elementwise_affine: bool,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Additional safety: clamp rms to prevent extreme division
        rms = rms.clamp(min=self.eps)
        x_normalized = x / rms

        if self.elementwise_affine:
            return x_normalized * self.weight
        else:
            return x_normalized

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


@normalization_registry.register("layer_norm")
class LayerNorm(nn.Module):
    """
    Layer Normalization (LayerNorm).

    Formula: gamma * ((x - mean(x)) / sqrt(var(x) + eps)) + beta

    Advantages:
    - Batch-independent: works with any batch size
    - No value capping: preserves full information range
    - Better for quantization (centers values around 0)
    """

    def __init__(
        self,
        num_features: int,
        eps: float,
        elementwise_affine: bool,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            num_features,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)

    def extra_repr(self) -> str:
        return self.layer_norm.extra_repr()


@normalization_registry.register("dynamic_tanh")
class DynamicTanh(nn.Module):
    """
    Dynamic Tanh (DyT) normalization.

    Formula: tanh(alpha * x)
    where alpha is a learnable scale parameter per feature.

    Note: Bounds outputs to [-1, 1], which compresses information.
    """

    def __init__(
        self,
        num_features: int,
        eps: float,
        elementwise_affine: bool,
    ):
        super().__init__()
        self.num_features = num_features
        self.alpha = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"


@normalization_registry.register("batch_norm")
class BatchNorm(nn.Module):
    """
    Batch Normalization wrapper.

    Warning: Not recommended for variable batch sizes or small batches.
    Included for completeness but prefer RMSNorm or LayerNorm.
    """

    def __init__(
        self,
        num_features: int,
        eps: float,
        elementwise_affine: bool,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(
            num_features,
            eps=eps,
            affine=elementwise_affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(x)

    def extra_repr(self) -> str:
        return self.batch_norm.extra_repr()


class ScaledNorm(nn.Module):
    """
    Scaled Normalization with configurable strength.

    Applies a base normalization (RMSNorm or LayerNorm) followed by a strength
    multiplier that controls how aggressively the normalization affects activations.

    Formula: strength * normalize(x) + (1 - strength) * x

    - strength = 1.0: Standard normalization
    - strength > 1.0: Stronger normalization effect
    - strength < 1.0: Weaker normalization (blends with original)
    """

    def __init__(
        self,
        num_features: int,
        base_norm_type: str,
        strength: float,
        eps: float,
        elementwise_affine: bool,
    ):
        super().__init__()
        self.num_features = num_features
        self.strength = strength
        self.base_norm_type = base_norm_type

        if base_norm_type == "rms_norm":
            self.norm = RMSNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)
        elif base_norm_type == "layer_norm":
            self.norm = LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(
                f"Unknown base_norm_type: {base_norm_type}. "
                f"Use 'rms_norm' or 'layer_norm'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)

        if self.strength == 1.0:
            return normalized
        else:
            return self.strength * normalized + (1.0 - self.strength) * x

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"base_norm={self.base_norm_type}, "
            f"strength={self.strength}"
        )


@normalization_registry.register("scaled_rms_norm")
class ScaledRMSNorm(ScaledNorm):
    """Scaled RMSNorm with configurable strength."""

    def __init__(
        self,
        num_features: int,
        eps: float,
        elementwise_affine: bool,
        strength: float,
    ):
        super().__init__(
            num_features=num_features,
            base_norm_type="rms_norm",
            strength=strength,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )


@normalization_registry.register("scaled_layer_norm")
class ScaledLayerNorm(ScaledNorm):
    """Scaled LayerNorm with configurable strength."""

    def __init__(
        self,
        num_features: int,
        eps: float,
        elementwise_affine: bool,
        strength: float,
    ):
        super().__init__(
            num_features=num_features,
            base_norm_type="layer_norm",
            strength=strength,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )


def make_norm_layer(
    norm_type: str,
    num_features: int,
    config: dict[str, Any],
) -> nn.Module:
    """
    Factory function to create normalization layers.

    Args:
        norm_type: Type of normalization from normalization_registry
        num_features: Number of features to normalize
        config: Normalization configuration containing:
            - eps: Numerical stability term
            - elementwise_affine: Whether to use learnable affine parameters
            - scaled.strength: Strength for scaled variants

    Returns:
        Configured normalization layer

    Raises:
        KeyError: If norm_type is not registered
    """
    eps = config["eps"]
    elementwise_affine = config["elementwise_affine"]

    if norm_type in ("scaled_rms_norm", "scaled_layer_norm"):
        strength = config["scaled"]["strength"]
        return normalization_registry.create(
            norm_type,
            num_features=num_features,
            eps=eps,
            elementwise_affine=elementwise_affine,
            strength=strength,
        )
    else:
        return normalization_registry.create(
            norm_type,
            num_features=num_features,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
