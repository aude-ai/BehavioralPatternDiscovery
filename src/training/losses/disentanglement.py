"""
Disentanglement Losses

Importance-Weighted Orthogonality (IWO) loss for encouraging
orthogonal latent dimensions.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IWOLoss(nn.Module):
    """
    Importance-Weighted Orthogonality Loss.

    Measures non-orthogonality of latent dimensions weighted by their
    importance (variance). Higher loss indicates more correlated dimensions.

    Formula:
        1. Compute variance-based importance weights: w_i = var_i / sum(var)
        2. Compute correlation matrix of z
        3. Weight matrix = outer(w, w)
        4. Loss = weighted sum of |corr_ij| for i < j

    Properties:
        - Scale-invariant (uses correlation, not covariance)
        - Importance-weighted (focuses on high-variance dimensions)
        - 0 = perfect orthogonality
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Loss configuration containing:
                - eps: Numerical stability constant
        """
        super().__init__()
        self.eps = config["eps"]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute IWO loss.

        Args:
            z: (batch, dim) latent representations

        Returns:
            Scalar IWO loss (lower is better)
        """
        batch_size, latent_dim = z.shape

        if batch_size < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        # Compute importance weights (variance-based) with safe division
        variances = z.var(dim=0)
        importance_weights = variances / variances.sum().clamp(min=self.eps)

        # Compute correlation matrix with safe std
        z_centered = z - z.mean(dim=0)
        z_std = z_centered.std(dim=0).clamp(min=self.eps)  # clamp result, not add eps
        z_normalized = z_centered / z_std

        # Correlation matrix: (latent_dim, latent_dim)
        # batch_size >= 2 guaranteed by guard above
        corr_matrix = torch.mm(z_normalized.t(), z_normalized) / (batch_size - 1)

        # Create weight matrix (outer product of importance weights)
        weight_matrix = torch.outer(importance_weights, importance_weights)

        # Mask for upper triangle (off-diagonal only)
        mask = torch.triu(
            torch.ones(latent_dim, latent_dim, device=z.device),
            diagonal=1,
        )

        # Weighted sum of absolute off-diagonal correlations
        weighted_corr = weight_matrix * torch.abs(corr_matrix) * mask
        weighted_sum = weighted_corr.sum()
        weight_total = (weight_matrix * mask).sum()

        iwo_loss = weighted_sum / (weight_total + self.eps)

        return iwo_loss

    def compute_metrics(self, z: torch.Tensor) -> dict[str, float]:
        """
        Compute interpretable disentanglement metrics.

        Args:
            z: (batch, dim) latent representations

        Returns:
            Dictionary of metrics
        """
        batch_size, latent_dim = z.shape

        if batch_size < 2:
            return {
                "iwo_loss": 0.0,
                "mean_abs_corr": 0.0,
                "max_abs_corr": 0.0,
            }

        # Correlation matrix with safe std
        z_centered = z - z.mean(dim=0)
        z_std = z_centered.std(dim=0).clamp(min=self.eps)
        z_normalized = z_centered / z_std
        corr_matrix = torch.mm(z_normalized.t(), z_normalized) / (batch_size - 1)

        # Upper triangle mask
        mask = torch.triu(
            torch.ones(latent_dim, latent_dim, device=z.device),
            diagonal=1,
        )

        off_diag = torch.abs(corr_matrix) * mask

        return {
            "iwo_loss": self.forward(z).item(),
            "mean_abs_corr": (off_diag.sum() / mask.sum()).item(),
            "max_abs_corr": off_diag.max().item(),
        }

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
