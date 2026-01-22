"""
Reusable Layer Blocks

Provides composable building blocks for constructing encoder and decoder
layers. Each block follows a consistent pattern and is fully configurable.

SHAP Compatibility: All blocks avoid in-place operations and maintain
gradient flow for SHAP analysis.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from .activations import make_activation, get_swiglu
from .normalizations import make_norm_layer

logger = logging.getLogger(__name__)


class LinearBlock(nn.Module):
    """
    Standard linear layer block with normalization and activation.

    Structure: Linear -> Normalization -> Activation -> Dropout

    For SwiGLU activation, the structure is different:
    SwiGLU (includes linear) -> Normalization -> Dropout
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_type: str,
        activation_config: dict[str, Any],
        norm_type: str,
        norm_config: dict[str, Any],
        dropout: float,
        use_swiglu: bool = False,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            activation_type: Activation type (ignored if use_swiglu=True)
            activation_config: Activation configuration
            norm_type: Normalization type
            norm_config: Normalization configuration
            dropout: Dropout probability
            use_swiglu: If True, use SwiGLU instead of Linear+Activation
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_swiglu = use_swiglu

        if use_swiglu:
            self.swiglu = get_swiglu(in_features, out_features)
            self.linear = None
            self.activation = None
        else:
            self.swiglu = None
            self.linear = nn.Linear(in_features, out_features)
            self.activation = make_activation(activation_type, activation_config)

        self.norm = make_norm_layer(norm_type, out_features, norm_config)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture (more stable for deep networks)
        # Order: Linear → Norm → Activation → Dropout
        if self.use_swiglu:
            x = self.swiglu(x)
        else:
            x = self.linear(x)

        x = self.norm(x)

        if not self.use_swiglu:
            x = self.activation(x)

        x = self.dropout(x)

        return x

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"use_swiglu={self.use_swiglu}"
        )


class ResidualBlock(nn.Module):
    """
    Residual block with optional projection for dimension mismatch.

    Structure: x + Block(x) where Block is a LinearBlock

    If in_features != out_features, a projection layer is used for the residual.

    SHAP Compatibility: Uses x = x + residual (no in-place operations)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_type: str,
        activation_config: dict[str, Any],
        norm_type: str,
        norm_config: dict[str, Any],
        dropout: float,
        use_swiglu: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.block = LinearBlock(
            in_features=in_features,
            out_features=out_features,
            activation_type=activation_type,
            activation_config=activation_config,
            norm_type=norm_type,
            norm_config=norm_config,
            dropout=dropout,
            use_swiglu=use_swiglu,
        )

        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projection is None else self.projection(x)
        out = self.block(x)
        return out + residual

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"has_projection={self.projection is not None}"
        )


def make_layer_block(
    block_type: str,
    in_features: int,
    out_features: int,
    layer_config: dict[str, Any],
) -> nn.Module:
    """
    Factory function to create layer blocks.

    Args:
        block_type: Type of block ("linear" or "residual")
        in_features: Input dimension
        out_features: Output dimension
        layer_config: Configuration containing:
            - activation.type: Activation type
            - activation.[type]: Activation-specific config
            - normalization.type: Normalization type
            - normalization.*: Normalization config
            - dropout: Dropout probability

    Returns:
        Configured layer block

    Raises:
        ValueError: If block_type is unknown
    """
    activation_type = layer_config["activation"]["type"]
    use_swiglu = activation_type == "swiglu"

    # Get activation-specific config if it exists
    if activation_type in layer_config["activation"]:
        activation_config = layer_config["activation"][activation_type]
    else:
        activation_config = {}

    norm_type = layer_config["normalization"]["type"]
    norm_config = layer_config["normalization"]
    dropout = layer_config["dropout"]

    if block_type == "linear":
        return LinearBlock(
            in_features=in_features,
            out_features=out_features,
            activation_type=activation_type,
            activation_config=activation_config,
            norm_type=norm_type,
            norm_config=norm_config,
            dropout=dropout,
            use_swiglu=use_swiglu,
        )
    elif block_type == "residual":
        return ResidualBlock(
            in_features=in_features,
            out_features=out_features,
            activation_type=activation_type,
            activation_config=activation_config,
            norm_type=norm_type,
            norm_config=norm_config,
            dropout=dropout,
            use_swiglu=use_swiglu,
        )
    else:
        raise ValueError(
            f"Unknown block_type: {block_type}. "
            f"Use 'linear' or 'residual'"
        )


def compute_layer_dimensions(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    expansion_factor: float,
) -> list[int]:
    """
    Compute intermediate layer dimensions for an encoder/decoder level.

    The first layer expands the input by expansion_factor, then subsequent
    layers use geometric spacing to reach the target output dimension.

    Args:
        input_dim: Input dimension
        output_dim: Target output dimension
        num_layers: Number of layers
        expansion_factor: How much to expand on first layer

    Returns:
        List of dimensions [input_dim, hidden1, hidden2, ..., output_dim]
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    if num_layers == 1:
        return [input_dim, output_dim]

    # First layer expands
    expanded_dim = int(input_dim * expansion_factor)

    if num_layers == 2:
        return [input_dim, expanded_dim, output_dim]

    # Geometric spacing from expanded to output
    dims = [input_dim, expanded_dim]
    remaining_layers = num_layers - 1

    ratio = (output_dim / expanded_dim) ** (1.0 / remaining_layers)

    current_dim = expanded_dim
    for i in range(remaining_layers - 1):
        current_dim = int(current_dim * ratio)
        dims.append(current_dim)

    dims.append(output_dim)

    return dims
