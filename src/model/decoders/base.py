"""
Base Decoder Interface

Defines the contract that all decoders must implement.
Each decoder type defines its own training loss and validation metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderTrainingOutput:
    """
    Output from decoder training loss computation.

    Attributes:
        loss: Primary loss tensor for backpropagation (backward compatible)
        flow_loss: Velocity MSE loss (flow decoders only)
        base_recon_loss: Direct reconstruction loss (MLP MSE or diff ODE recon)
        focal_loss: Additional focal penalty (focal_weighted - uniform)
        metrics: Dictionary of training metrics to log
        reconstruction_approx: Optional approximate reconstruction (for logging only)
    """
    loss: torch.Tensor
    flow_loss: torch.Tensor | None = None
    base_recon_loss: torch.Tensor | None = None
    focal_loss: torch.Tensor | None = None
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    reconstruction_approx: torch.Tensor | None = None


@dataclass
class DecoderValidationOutput:
    """
    Output from decoder validation metrics computation.

    Attributes:
        metrics: Dictionary of all validation metrics
        reconstruction: Actual reconstruction (ODE solved for flow decoders)
    """
    metrics: dict[str, torch.Tensor]
    reconstruction: torch.Tensor


class BaseDecoder(ABC, nn.Module):
    """
    Abstract base class for decoders.

    Each decoder type implements its own training loss and validation metrics.
    This allows different decoder types (MLP, flow matching, diffusion) to
    define what makes sense for their paradigm.

    Key methods:
    - compute_training_loss(): Returns loss + training metrics (gradients flow)
    - compute_validation_metrics(): Returns all metrics + actual reconstruction
    - compute_reconstruction_quality(): Universal quality metrics
    - generate(): Inference-time reconstruction
    - generate_differentiable(): Optional reconstruction with gradients

    Config separation:
    - model_config: Architecture (dimensions, layers, ODE solver)
    - training_config: Training behavior (loss normalization, direction weight)
    """

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by all decoders
    # =========================================================================

    @abstractmethod
    def compute_training_loss(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
        include_reconstruction_loss: bool = False,
    ) -> DecoderTrainingOutput:
        """
        Compute loss for training (gradients flow through this).

        Each decoder type defines its own training loss:
        - Flow matching: velocity MSE (with optional normalization and direction)
        - MLP: reconstruction MSE

        Args:
            z: Conditioning latent tensor (batch, latent_dim)
            x_target: Target tensor for reconstruction (batch, output_dim)
            include_reconstruction_loss: If True, include actual reconstruction loss
                via differentiable ODE (for flow matching) or direct MSE (for MLP).
                Used periodically for direct supervision.

        Returns:
            DecoderTrainingOutput with:
                - loss: Primary loss for backpropagation
                - metrics: Training metrics to log
                - reconstruction_approx: Optional approximate reconstruction
        """
        pass

    @abstractmethod
    def compute_validation_metrics(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
    ) -> DecoderValidationOutput:
        """
        Compute all validation metrics.

        MUST include:
        1. Training-style metrics (for comparison with training loss)
        2. Actual reconstruction quality (after ODE solve for flow)
        3. Baselines (zero, centroid) for context

        Args:
            z: Conditioning latent tensor (batch, latent_dim)
            x_target: Target tensor for reconstruction (batch, output_dim)

        Returns:
            DecoderValidationOutput with:
                - metrics: All validation metrics
                - reconstruction: Actual reconstruction
        """
        pass

    @abstractmethod
    def compute_reconstruction_quality(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Universal reconstruction quality metrics for this decoder's output space.

        For spherical: cosine similarity, geodesic distance
        For Euclidean: MSE, MAE

        Args:
            x_recon: Reconstructed tensor (batch, output_dim)
            x_target: Target tensor (batch, output_dim)

        Returns:
            Dictionary of quality metrics
        """
        pass

    @abstractmethod
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate reconstruction from latent (inference mode).

        For flow matching: ODE solve
        For MLP: forward pass

        Args:
            z: Conditioning latent tensor (batch, latent_dim)

        Returns:
            Reconstructed tensor (batch, output_dim)
        """
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        """Return expected latent input dimension."""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return output dimension."""
        pass

    # =========================================================================
    # OPTIONAL METHODS - Default implementations provided
    # =========================================================================

    def generate_differentiable(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Generate reconstruction WITH gradient support (optional).

        For flow matching: differentiable ODE solve
        For MLP: same as generate()

        Args:
            z: Conditioning latent tensor (batch, latent_dim)
            x_target: Optional target for computing reconstruction loss

        Returns:
            Tuple of (reconstruction, optional_loss)
        """
        # Default: just call generate() without computing loss
        with torch.enable_grad():
            recon = self.generate(z)

        if x_target is not None:
            # Default reconstruction loss is MSE
            loss = ((recon - x_target) ** 2).mean()
            return recon, loss

        return recon, None

    def get_training_loss_config(self) -> dict[str, Any]:
        """Return the loss configuration for logging/documentation."""
        return {}

    def get_validation_metrics_names(self) -> list[str]:
        """Return names of validation metrics this decoder provides."""
        return []

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        pass

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        pass


# =============================================================================
# Utility Functions for Common Metrics
# =============================================================================

def compute_spherical_metrics(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute reconstruction quality metrics for spherical (L2-normalized) data.

    Args:
        x_recon: Reconstructed tensor (batch, dim), should be normalized
        x_target: Target tensor (batch, dim), should be normalized

    Returns:
        Dictionary with cosine_sim, geodesic_dist, euclidean_dist, mse_total, mse_per_dim
    """
    # Cosine similarity (primary metric for L2-normalized)
    cos_sim = F.cosine_similarity(x_recon, x_target, dim=-1)

    # Geodesic distance (arc length on sphere)
    dot_product = (x_recon * x_target).sum(dim=-1).clamp(-1, 1)
    geodesic_dist = torch.acos(dot_product)

    # Euclidean distance (chord length)
    diff = x_recon - x_target
    euclidean_dist = (diff ** 2).sum(dim=-1).sqrt()

    # MSE
    mse_total = (diff ** 2).sum(dim=-1)
    mse_per_dim = mse_total / x_recon.shape[-1]

    return {
        "cosine_sim": cos_sim.mean(),
        "cosine_sim_std": cos_sim.std(),
        "geodesic_dist": geodesic_dist.mean(),
        "geodesic_dist_std": geodesic_dist.std(),
        "euclidean_dist": euclidean_dist.mean(),
        "mse_total": mse_total.mean(),
        "mse_per_dim": mse_per_dim.mean(),
    }


def compute_euclidean_metrics(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute reconstruction quality metrics for Euclidean (non-normalized) data.

    Args:
        x_recon: Reconstructed tensor (batch, dim)
        x_target: Target tensor (batch, dim)

    Returns:
        Dictionary with mse_total, mse_per_dim, mae, max_error
    """
    diff = x_recon - x_target

    # MSE
    mse_total = (diff ** 2).sum(dim=-1)
    mse_per_dim = mse_total / x_recon.shape[-1]

    # MAE
    mae = diff.abs().mean(dim=-1)

    # Max error
    max_error = diff.abs().max(dim=-1).values

    return {
        "mse_total": mse_total.mean(),
        "mse_per_dim": mse_per_dim.mean(),
        "mse_std": mse_total.std(),
        "mae": mae.mean(),
        "max_error": max_error.mean(),
    }
