"""
Gaussian Latent Distribution

Standard Gaussian distribution with N(0,1) prior for VAE latent spaces.

Parameters:
- mu: Mean of the Gaussian
- logvar: Log variance (for numerical stability)

KL Divergence:
- KL(N(mu, sigma) || N(0, 1)) = 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from .base import LatentDistribution, distribution_registry

logger = logging.getLogger(__name__)


@distribution_registry.register("gaussian")
class GaussianDistribution(LatentDistribution):
    """
    Standard Gaussian latent distribution with N(0,1) prior.

    Uses logvar parameterization for numerical stability:
    - Network outputs mu and logvar
    - logvar is clamped to prevent extreme values
    - std = exp(0.5 * logvar)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Must contain:
                - logvar_clamp_min: Minimum logvar value
                - logvar_clamp_max: Maximum logvar value
        """
        self.logvar_clamp_min = config["logvar_clamp_min"]
        self.logvar_clamp_max = config["logvar_clamp_max"]

    @property
    def param_names(self) -> tuple[str, ...]:
        return ("mu", "logvar")

    def create_param_layers(
        self,
        in_dim: int,
        out_dim: int
    ) -> nn.ModuleDict:
        """Create mu and logvar projection layers."""
        return nn.ModuleDict({
            "mu": nn.Linear(in_dim, out_dim),
            "logvar": nn.Linear(in_dim, out_dim),
        })

    def forward_params(
        self,
        h: torch.Tensor,
        layers: nn.ModuleDict
    ) -> dict[str, torch.Tensor]:
        """
        Compute mu and logvar from hidden state.

        logvar is clamped to [logvar_clamp_min, logvar_clamp_max] for stability.
        mu is unbounded - the network should learn proper ranges.
        """
        mu = layers["mu"](h)
        logvar = layers["logvar"](h)

        # Clamp logvar to safe range (preventive, not reactive)
        logvar = logvar.clamp(min=self.logvar_clamp_min, max=self.logvar_clamp_max)

        return {"mu": mu, "logvar": logvar}

    def reparameterize(
        self,
        params: dict[str, torch.Tensor],
        training: bool
    ) -> torch.Tensor:
        """
        Sample using reparameterization trick.

        During training: z = mu + std * epsilon, where epsilon ~ N(0, 1)
        During inference: z = mu (deterministic)
        """
        mu = params["mu"]
        logvar = params["logvar"]

        if training:
            # Clamp immediately before exp for safety (belt and suspenders with forward_params)
            std = torch.exp(0.5 * logvar.clamp(min=self.logvar_clamp_min, max=self.logvar_clamp_max))
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def get_mean(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return mu (deterministic mean for SHAP)."""
        return params["mu"]

    def kl_divergence(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        KL(N(mu, sigma) || N(0, 1))

        Formula: 0.5 * sum_i (mu_i^2 + exp(logvar_i) - 1 - logvar_i)

        Returns:
            Per-sample KL divergence (batch,)
        """
        mu = params["mu"]
        logvar = params["logvar"]

        # KL per dimension, then sum over latent dims
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        return kl_per_dim.sum(dim=-1)

    def scale_mean_params(
        self,
        params: dict[str, torch.Tensor],
        scale_factor: float
    ) -> dict[str, torch.Tensor]:
        """
        Scale mu for Scale-VAE.

        For Gaussian, mean = mu, so we scale mu directly.

        Args:
            params: Distribution parameters containing 'mu' and 'logvar'
            scale_factor: Factor to scale mu by

        Returns:
            New params dict with scaled mu (logvar unchanged)
        """
        return {
            "mu": params["mu"] * scale_factor,
            "logvar": params["logvar"],
        }
