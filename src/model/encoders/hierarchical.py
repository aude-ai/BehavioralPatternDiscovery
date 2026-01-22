"""
Hierarchical Encoder

Configurable hierarchical encoder that produces latent representations
at multiple levels of abstraction. The number and names of levels are
defined in config, not hardcoded.

Each level processes only the output from the previous level (no skip connections),
creating a true hierarchy of representations.

Supports optional Mixture Prior at each level for pattern diversity.

Supports optional Scale-VAE: scales mu before sampling to give decoder
more spread, while keeping original mu for KL computation.

Uses ModelDimensions for computing input dimensions (not stored in config).

SHAP Compatibility:
- forward() is deterministic when training=False
- Uses distribution.get_mean() instead of sampling
- No in-place operations
"""

import logging
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.core.registry import ComponentRegistry
from src.model.distributions import LatentDistribution, MixturePrior
from src.model.layers import make_layer_block, compute_layer_dimensions

if TYPE_CHECKING:
    from src.core.config import ModelDimensions

logger = logging.getLogger(__name__)

# Registry for encoders
encoder_registry = ComponentRegistry[nn.Module]("encoder")


class EncoderLevel(nn.Module):
    """
    Single level of the hierarchical encoder.

    Structure:
    - MLP layers: input_dim -> hidden dims -> final_hidden_dim
    - Distribution params: final_hidden_dim -> latent_dim
    - Optional MixturePrior: cluster assignment for diverse patterns
    - Optional Scale-VAE: scales mu before sampling for decoder spread
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        expansion_factor: float,
        distribution: LatentDistribution,
        layer_config: dict[str, Any],
        mixture_prior_config: dict[str, Any] | None = None,
        pre_distribution_expansion: float = 1.0,
        scale_vae_enabled: bool = False,
        scale_vae_factor: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Compute layer dimensions
        # pre_distribution_expansion: 1.0 (EPP default) or 2.0 (original BPD)
        mlp_output_dim = int(output_dim * pre_distribution_expansion)
        dims = compute_layer_dimensions(
            input_dim=input_dim,
            output_dim=mlp_output_dim,
            num_layers=num_layers,
            expansion_factor=expansion_factor,
        )

        # Build MLP layers
        layers = []
        for i in range(len(dims) - 1):
            block = make_layer_block(
                block_type="linear",
                in_features=dims[i],
                out_features=dims[i + 1],
                layer_config=layer_config,
            )
            layers.append(block)

        self.mlp = nn.Sequential(*layers)
        self._gradient_checkpointing = False

        # Distribution parameter layers
        final_hidden_dim = dims[-1]
        self.param_layers = distribution.create_param_layers(final_hidden_dim, output_dim)
        self.distribution = distribution
        self.final_hidden_dim = final_hidden_dim

        # Mixture prior (optional)
        self.mixture_prior = None
        if mixture_prior_config is not None and mixture_prior_config["enabled"]:
            # Cluster count matches latent dimension
            mp_config = {
                **mixture_prior_config,
                "num_clusters": output_dim,  # Tied to latent dim
            }
            self.mixture_prior = MixturePrior(
                base_distribution=distribution,
                config=mp_config,
                latent_dim=output_dim,
                hidden_dim=final_hidden_dim,
            )

        # Scale-VAE (optional)
        self.scale_vae_enabled = scale_vae_enabled
        self.scale_vae_factor = scale_vae_factor

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through this encoder level.

        Args:
            x: Input tensor (batch, input_dim)
            training: If True, sample from distribution. If False, use mean.

        Returns:
            Dictionary containing:
            - 'z': Latent sample/mean (batch, output_dim) - may use scaled mu if Scale-VAE enabled
            - 'params': Distribution parameters (original, unscaled - for KL computation)
            - 'cluster_probs': Cluster assignment probs (if mixture prior enabled)
            - 'h': Hidden state (for mixture prior KL computation)
        """
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


@encoder_registry.register("hierarchical")
class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder with configurable number of levels.

    Architecture:
        Input -> Level 1 -> Level 2 -> ... -> Level N

    Each level only receives input from the previous level, creating
    a true information bottleneck hierarchy. The number and names of
    levels are defined in config, not hardcoded.

    Supports optional Mixture Prior at each level for pattern diversity.
    Cluster count is tied to the latent dimension of each level.

    SHAP Compatibility:
        - Set training=False to get deterministic outputs
        - Uses distribution.get_mean() for SHAP analysis
    """

    def __init__(
        self,
        config: dict[str, Any],
        distribution: LatentDistribution,
        dims: "ModelDimensions",
    ):
        """
        Args:
            config: Encoder configuration containing:
                - hierarchical.levels: List of level configs (name, output_dim, num_layers, expansion_factor)
                - hierarchical.dropout: Dropout rate
                - layer_config: Encoder layer configuration
                - mixture_prior: Optional mixture prior config
                - scale_vae: Optional Scale-VAE config
            distribution: Latent distribution for all levels (injected dependency)
            dims: ModelDimensions for computing derived values (input_dim per level)
        """
        super().__init__()

        hierarchical_config = config["hierarchical"]
        levels_config = hierarchical_config["levels"]
        base_layer_config = config["layer_config"]
        mixture_prior_config = config["mixture_prior"]
        scale_vae_config = config["scale_vae"]

        # Pre-distribution expansion factor: 1.0 (EPP default) or 2.0 (original BPD)
        pre_dist_expansion = hierarchical_config["pre_distribution_expansion"]

        # Parse scale factors into dict by level name
        scale_vae_enabled = scale_vae_config["enabled"]
        scale_factors = {}
        if scale_vae_enabled:
            for sf in scale_vae_config["scale_factors"]:
                scale_factors[sf["name"]] = sf["value"]

        # Merge encoder-specific dropout into layer_config
        layer_config = {
            **base_layer_config,
            "dropout": hierarchical_config["dropout"],
        }

        # Build levels dynamically using ModuleDict
        self.levels = nn.ModuleDict()
        self.level_names: list[str] = []
        self.level_dims: dict[str, int] = {}

        for level_cfg in levels_config:
            level_name = level_cfg["name"]
            self.level_names.append(level_name)

            input_dim, output_dim = dims.get_level_dims(level_name)
            self.level_dims[level_name] = output_dim

            scale_factor = scale_factors.get(level_name, 1.0) if scale_vae_enabled else 1.0

            self.levels[level_name] = EncoderLevel(
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=level_cfg["num_layers"],
                expansion_factor=level_cfg["expansion_factor"],
                distribution=distribution,
                layer_config=layer_config,
                mixture_prior_config=mixture_prior_config,
                pre_distribution_expansion=pre_dist_expansion,
                scale_vae_enabled=scale_vae_enabled and scale_factor != 1.0,
                scale_vae_factor=scale_factor,
            )

        self.distribution = distribution
        self.mixture_prior_enabled = (
            mixture_prior_config is not None and
            mixture_prior_config["enabled"]
        )

        # Get activation type for proper weight initialization
        activation_type = base_layer_config["activation"]["type"]
        self._init_weights(activation_type)

        # Log architecture
        if self.mixture_prior_enabled:
            logger.info(
                "Mixture prior enabled at all levels: "
                + ", ".join(f"{name}={self.level_dims[name]} clusters" for name in self.level_names)
            )

        if scale_vae_enabled:
            active_levels = {
                name: scale_factors.get(name, 1.0)
                for name in self.level_names
                if scale_factors.get(name, 1.0) != 1.0
            }
            if active_levels:
                logger.info(f"Scale-VAE enabled at levels: {active_levels}")

        logger.info(
            f"HierarchicalEncoder: {len(self.level_names)} levels "
            f"({' -> '.join(self.level_names)}), "
            f"dims: {[self.level_dims[n] for n in self.level_names]}"
        )

    def _init_weights(self, activation_type: str = "relu"):
        """
        Initialize weights with He initialization appropriate for the activation.

        Args:
            activation_type: The activation function type being used.
                For XiELU/ELU, uses leaky_relu nonlinearity with appropriate slope.
        """
        # Map activation types to PyTorch's supported nonlinearities for kaiming init
        # XiELU behaves like leaky_relu with slope=alpha (default 1.0)
        # This prevents the gain mismatch that can cause encoder collapse
        if activation_type in ("xielu", "elu", "leaky_relu"):
            nonlinearity = "leaky_relu"
            # XiELU and ELU have ~linear negative slope, use a=1.0 for gain calculation
            # This gives gain = sqrt(2 / (1 + 1^2)) = 1.0 instead of sqrt(2) for relu
            a = 1.0
        elif activation_type == "swish":
            # Swish is roughly linear for large values, use leaky_relu approximation
            nonlinearity = "leaky_relu"
            a = 0.5  # Approximate slope of swish for negative values
        elif activation_type == "gelu":
            # GELU is similar to swish
            nonlinearity = "leaky_relu"
            a = 0.5
        else:
            # Default to relu for relu, cone variants, etc.
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
        Encode input through all hierarchical levels.

        Args:
            x: Combined input tensor (batch, input_dim) containing embedding + aux features
            training: If True, sample from distributions. If False, use means.

        Returns:
            Dictionary containing per-level outputs keyed by level name.
            Each level contains: {'z': latent, 'params': distribution params, 'cluster_probs': optional}
        """
        # Validate inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(
                f"NaN/Inf in input. "
                f"NaN: {torch.isnan(x).sum()}, Inf: {torch.isinf(x).sum()}"
            )

        result = {}
        current_input = x

        for level_name in self.level_names:
            level_out = self.levels[level_name](current_input, training=training)

            result[level_name] = {
                "z": level_out["z"],
                "params": level_out["params"],
            }

            if self.mixture_prior_enabled:
                result[level_name]["cluster_probs"] = level_out.get("cluster_probs")
                result[level_name]["h"] = level_out["h"]

            # Next level's input is this level's output
            current_input = level_out["z"]

        return result

    def get_latent_dim(self) -> int:
        """Return the final level latent dimensionality."""
        return self.level_dims[self.level_names[-1]]

    def get_level_names(self) -> list[str]:
        """Return ordered list of level names."""
        return self.level_names.copy()

    def get_mixture_prior(self, level: str) -> MixturePrior | None:
        """Get mixture prior for a specific level."""
        if level in self.levels:
            return self.levels[level].mixture_prior
        return None

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        for level in self.levels.values():
            level._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        for level in self.levels.values():
            level._gradient_checkpointing = False
