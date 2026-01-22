"""
Reconstruction Losses

Standard MSE and focal variants for VAE reconstruction.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Registry for losses
loss_registry = ComponentRegistry[nn.Module]("loss")


@loss_registry.register("mse")
class ReconstructionLoss(nn.Module):
    """
    Standard MSE reconstruction loss.

    Computes mean squared error between reconstructed and target embeddings.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Loss configuration (currently unused, for interface consistency)
        """
        super().__init__()
        self.reduction = config["reduction"]

    def forward(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE reconstruction loss.

        Args:
            x_recon: (batch, dim) reconstructed embeddings
            x_target: (batch, dim) target embeddings

        Returns:
            Scalar loss value
        """
        if self.reduction == "none":
            return F.mse_loss(x_recon, x_target, reduction="none")
        elif self.reduction == "sum":
            return F.mse_loss(x_recon, x_target, reduction="sum")
        else:
            return F.mse_loss(x_recon, x_target, reduction="mean")


@loss_registry.register("focal")
class FocalReconstructionLoss(nn.Module):
    """
    Focal loss for VAE reconstruction.

    Down-weights easy examples (low error), focuses on hard/rare patterns.
    Useful for imbalanced pattern distributions.

    Formula:
        focal_weight = (1 - exp(-mse))^gamma
        loss = focal_weight * mse
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Loss configuration containing:
                - gamma: Focal focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.gamma = config["gamma"]

    def forward(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal reconstruction loss.

        Args:
            x_recon: (batch, dim) reconstructed embeddings
            x_target: (batch, dim) target embeddings

        Returns:
            Scalar focal loss value
        """
        # Per-sample MSE
        mse_per_sample = F.mse_loss(x_recon, x_target, reduction="none")
        mse_per_sample = mse_per_sample.mean(dim=1)

        # Focal weighting: higher weight for harder samples
        p_t = torch.exp(-mse_per_sample)
        focal_weight = (1 - p_t) ** self.gamma

        # Weighted loss
        loss = (mse_per_sample * focal_weight).mean()

        return loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}"
