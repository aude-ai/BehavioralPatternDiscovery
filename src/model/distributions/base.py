"""
Latent Distribution Base Class and Registry

Provides the abstract interface for latent distributions and the registry
for configuration-driven distribution selection.

Distributions handle:
- Creating parameter layers (e.g., mu/logvar or concentration/rate)
- Computing distribution parameters from hidden states
- Reparameterization for training
- Deterministic mean for SHAP
- KL divergence computation
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from src.core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Registry for latent distributions
distribution_registry = ComponentRegistry["LatentDistribution"]("distribution")


class LatentDistribution(ABC):
    """
    Abstract base class for latent space distributions.

    All distributions must implement:
    - create_param_layers: Create network layers for distribution parameters
    - forward_params: Compute parameters from hidden state
    - reparameterize: Sample with reparameterization trick (or return mean)
    - get_mean: Deterministic mean for SHAP
    - kl_divergence: KL from prior

    SHAP Compatibility:
        - get_mean() provides deterministic outputs
        - All operations maintain gradient flow
    """

    @abstractmethod
    def create_param_layers(
        self,
        in_dim: int,
        out_dim: int
    ) -> nn.ModuleDict:
        """
        Create distribution parameter layers.

        Args:
            in_dim: Input dimension from encoder hidden state
            out_dim: Latent dimension (number of latent variables)

        Returns:
            ModuleDict containing parameter projection layers
        """
        pass

    @abstractmethod
    def forward_params(
        self,
        h: torch.Tensor,
        layers: nn.ModuleDict
    ) -> dict[str, torch.Tensor]:
        """
        Compute distribution parameters from hidden state.

        Args:
            h: Hidden state tensor (batch, hidden_dim)
            layers: Parameter layers from create_param_layers()

        Returns:
            Dictionary of distribution parameters
        """
        pass

    @abstractmethod
    def reparameterize(
        self,
        params: dict[str, torch.Tensor],
        training: bool
    ) -> torch.Tensor:
        """
        Sample from the distribution using reparameterization trick.

        Args:
            params: Distribution parameters from forward_params()
            training: If True, sample with noise. If False, return mean.

        Returns:
            Sampled latent tensor (batch, latent_dim)
        """
        pass

    @abstractmethod
    def get_mean(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return deterministic mean of the distribution.

        CRITICAL for SHAP: This is used during SHAP analysis.

        Args:
            params: Distribution parameters from forward_params()

        Returns:
            Mean tensor (batch, latent_dim)
        """
        pass

    @abstractmethod
    def kl_divergence(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute KL divergence from the prior distribution.

        Args:
            params: Distribution parameters from forward_params()

        Returns:
            Per-sample KL divergence (batch,)
        """
        pass

    @property
    @abstractmethod
    def param_names(self) -> tuple[str, ...]:
        """Return the names of the distribution parameters."""
        pass

    @abstractmethod
    def scale_mean_params(
        self,
        params: dict[str, torch.Tensor],
        scale_factor: float
    ) -> dict[str, torch.Tensor]:
        """
        Scale distribution parameters to achieve scaled mean for Scale-VAE.

        Scale-VAE scales the mean of the distribution to give the decoder
        more spread, while keeping original params for KL computation.

        Args:
            params: Distribution parameters from forward_params()
            scale_factor: Factor to scale the mean by

        Returns:
            New params dict with scaled parameters (original dict unchanged)
        """
        pass


def create_distribution(
    dist_type: str,
    config: dict[str, Any]
) -> LatentDistribution:
    """
    Factory function to create a latent distribution.

    Args:
        dist_type: Distribution type from distribution_registry
        config: Distribution configuration

    Returns:
        LatentDistribution instance

    Raises:
        KeyError: If dist_type is not registered
    """
    return distribution_registry.create(dist_type, config=config)
