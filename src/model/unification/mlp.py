"""
MLP Unification Layer

Combines top-level latents from multiple encoders into a single
unified representation. The unification layer learns to synthesize
different aspects discovered by different encoders.

Architecture:
    Concatenated tops (N_encoders * top_dim) -> MLP -> unified_z (unified_dim)

SHAP Compatibility:
- forward() is deterministic when training=False
- Uses distribution.get_mean() instead of sampling
- No in-place operations

Supports optional Mixture Prior for pattern diversity.
Supports optional Scale-VAE for latent utilization.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.core.registry import ComponentRegistry
from src.model.distributions import LatentDistribution, MixturePrior
from src.model.layers import make_layer_block

logger = logging.getLogger(__name__)

# Registry for unification layers
unification_registry = ComponentRegistry[nn.Module]("unification")


def compute_hidden_dims(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    expansion_factor: float,
) -> list[int]:
    """
    Compute hidden dimensions with geometric spacing.

    First hidden = input_dim * expansion_factor
    Remaining layers geometrically spaced down to output_dim.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        expansion_factor: Multiplier for first hidden layer

    Returns:
        List of hidden dimensions
    """
    if num_layers == 0:
        return []

    first_hidden = int(input_dim * expansion_factor)

    if num_layers == 1:
        return [first_hidden]

    # Geometric spacing from first_hidden to output_dim
    ratio = (output_dim / first_hidden) ** (1.0 / (num_layers - 1))
    hidden_dims = []
    for i in range(num_layers):
        dim = int(first_hidden * (ratio ** i))
        dim = max(dim, output_dim)  # Don't go below output_dim
        hidden_dims.append(dim)

    return hidden_dims


@unification_registry.register("mlp")
class MLPUnification(nn.Module):
    """
    MLP-based unification layer.

    Takes concatenated top-level latents from all encoders and produces
    a unified latent representation that synthesizes information from
    all encoders.

    SHAP Compatibility:
        - Set training=False to get deterministic outputs
        - Uses distribution.get_mean() for SHAP analysis
    """

    def __init__(
        self,
        config: dict[str, Any],
        distribution: LatentDistribution,
        mixture_prior_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            config: Unification configuration containing:
                - mlp.input_dim: Input dimension (N_encoders * top_dim)
                - mlp.output_dim: Output unified dimension
                - mlp.num_layers: Number of hidden layers
                - mlp.expansion_factor: Multiplier for first hidden layer
                - mlp.dropout: Dropout rate
                - scale_vae: Optional Scale-VAE config
            distribution: Latent distribution for unified layer
            mixture_prior_config: Optional mixture prior configuration
        """
        super().__init__()

        mlp_config = config["mlp"]
        layer_config = config["layer_config"]
        scale_vae_config = config["scale_vae"]

        self.input_dim = mlp_config["input_dim"]
        self.output_dim = mlp_config["output_dim"]
        num_layers = mlp_config["num_layers"]
        expansion_factor = mlp_config["expansion_factor"]
        dropout = mlp_config["dropout"]

        # Scale-VAE settings
        self.scale_vae_enabled = scale_vae_config["enabled"]
        self.scale_vae_factor = scale_vae_config["scale_factor"]

        # Compute hidden dimensions from num_layers and expansion_factor
        hidden_dims = compute_hidden_dims(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
        )
        logger.info(
            f"MLPUnification: {self.input_dim} -> {hidden_dims} -> {self.output_dim}"
        )

        # Build MLP layers
        layers = []
        current_dim = self.input_dim

        for hidden_dim in hidden_dims:
            # Update layer_config with unification-specific dropout
            unified_layer_config = {
                **layer_config,
                "dropout": dropout,
            }
            block = make_layer_block(
                block_type="linear",
                in_features=current_dim,
                out_features=hidden_dim,
                layer_config=unified_layer_config,
            )
            layers.append(block)
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self._gradient_checkpointing = False
        self.final_hidden_dim = current_dim

        # Distribution parameter layers
        self.param_layers = distribution.create_param_layers(current_dim, self.output_dim)
        self.distribution = distribution

        # Mixture prior (optional) - cluster count tied to output dimension
        self.mixture_prior = None
        if mixture_prior_config is not None and mixture_prior_config["enabled"]:
            mp_config = {
                **mixture_prior_config,
                "num_clusters": self.output_dim,
            }
            self.mixture_prior = MixturePrior(
                base_distribution=distribution,
                config=mp_config,
                latent_dim=self.output_dim,
                hidden_dim=self.final_hidden_dim,
            )
            logger.info(
                f"MLPUnification: Mixture prior enabled with "
                f"{self.output_dim} clusters (tied to output_dim)"
            )

        if self.scale_vae_enabled:
            logger.info(
                f"MLPUnification: Scale-VAE enabled (factor={self.scale_vae_factor})"
            )

        # Get activation type for proper weight initialization
        activation_type = layer_config["activation"]["type"]
        self._init_weights(activation_type)

    def _init_weights(self, activation_type: str = "relu"):
        """
        Initialize weights with He initialization appropriate for the activation.

        Args:
            activation_type: The activation function type being used.
                For XiELU/ELU, uses leaky_relu nonlinearity with appropriate slope.
        """
        # Map activation types to PyTorch's supported nonlinearities for kaiming init
        if activation_type in ("xielu", "elu", "leaky_relu"):
            nonlinearity = "leaky_relu"
            a = 1.0
        elif activation_type in ("swish", "gelu"):
            nonlinearity = "leaky_relu"
            a = 0.5
        else:
            nonlinearity = "relu"
            a = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if nonlinearity == "leaky_relu":
                    nn.init.kaiming_normal_(m.weight, a=a, mode="fan_in", nonlinearity=nonlinearity)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> dict[str, Any]:
        """
        Combine encoder outputs into unified representation.

        Args:
            x: Concatenated top-level latents from all encoders (batch, input_dim)
            training: If True, sample from distribution. If False, use mean.

        Returns:
            Dictionary containing:
            - 'z': Unified latent (batch, output_dim) - may use scaled mu if Scale-VAE enabled
            - 'params': Distribution parameters (original, unscaled - for KL computation)
            - 'cluster_probs': Cluster assignment probs (if mixture prior enabled)
            - 'h': Hidden state (for mixture prior KL computation)
        """
        # Validate inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(
                f"NaN/Inf in unification input. "
                f"NaN: {torch.isnan(x).sum()}, "
                f"Inf: {torch.isinf(x).sum()}"
            )

        # Apply gradient checkpointing if enabled
        if self._gradient_checkpointing and self.training:
            h = checkpoint(self.mlp, x, use_reentrant=False)
        else:
            h = self.mlp(x)

        # Compute distribution parameters (original, unscaled)
        params = self.distribution.forward_params(h, self.param_layers)

        # Sample or use mean, with optional Scale-VAE
        if training:
            if self.scale_vae_enabled and self.scale_vae_factor != 1.0:
                # Scale-VAE: scale mean params for sampling, keep original params for KL
                scaled_params = self.distribution.scale_mean_params(params, self.scale_vae_factor)
                z = self.distribution.reparameterize(scaled_params, training=True)
            else:
                z = self.distribution.reparameterize(params, training=True)
        else:
            # For inference/SHAP, use mean (optionally scaled)
            if self.scale_vae_enabled and self.scale_vae_factor != 1.0:
                z = self.distribution.get_mean(params) * self.scale_vae_factor
            else:
                z = self.distribution.get_mean(params)

        # params stays original (unscaled) for KL computation
        result = {"z": z, "params": params, "h": h}

        # Compute cluster assignment if mixture prior enabled
        if self.mixture_prior is not None:
            cluster_probs = self.mixture_prior.compute_cluster_assignment(h)
            result["cluster_probs"] = cluster_probs

        return result

    def get_input_dim(self) -> int:
        """Return expected input dimension."""
        return self.input_dim

    def get_output_dim(self) -> int:
        """Return unified latent dimensionality."""
        return self.output_dim

    def get_mixture_prior(self) -> MixturePrior | None:
        """Get mixture prior if enabled."""
        return self.mixture_prior

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
