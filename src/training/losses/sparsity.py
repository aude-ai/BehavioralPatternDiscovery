"""
Sparsity Losses

Hoyer sparsity regularization for encouraging sparse latent activations.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HoyerLoss(nn.Module):
    """
    Hoyer sparsity regularization for latent vectors.

    Computes squared Hoyer ratio: ||z||_1^2 / ||z||_2^2
    Lower values indicate sparser activations.

    Properties:
        - Minimum (H=1): Only one dimension is non-zero (perfect sparsity)
        - Maximum (H=N): All N dimensions have equal magnitude
        - Scale-Invariant: H(αz) = H(z) for any α ≠ 0

    Can optionally target a specific sparsity level using normalized sparsity:
        S = (sqrt(N) - ||z||_1/||z||_2) / (sqrt(N) - 1)
        S ∈ [0, 1], higher is sparser

    References:
        "Non-negative Matrix Factorization with Sparseness Constraints"
        (Hoyer, 2004) - Journal of Machine Learning Research
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Loss configuration containing:
                - eps: Numerical stability constant
                - target_sparsity: Target normalized sparsity (0-1), None for direct minimization
                - per_sample: Apply sparsity per sample (row-wise)
                - per_dimension: Apply sparsity per dimension (column-wise)
                - absolute_values: Use absolute values (required for Gaussian)
        """
        super().__init__()
        self.eps = config["eps"]
        self.target_sparsity = config["target_sparsity"]
        self.per_sample = config["per_sample"]
        self.per_dimension = config["per_dimension"]
        self.absolute_values = config["absolute_values"]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Hoyer sparsity loss.

        Args:
            z: (batch, dim) latent tensor

        Returns:
            Scalar loss value
        """
        if self.absolute_values:
            z = z.abs()

        losses = []

        if self.per_sample:
            loss_sample = self._hoyer_loss(z, dim=1)
            losses.append(loss_sample)

        if self.per_dimension:
            loss_dim = self._hoyer_loss(z, dim=0)
            losses.append(loss_dim)

        if not losses:
            return self._hoyer_loss(z, dim=1)

        return sum(losses) / len(losses)

    def _hoyer_loss(self, z: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Compute Hoyer loss along specified dimension.

        Args:
            z: Tensor (already absolute if needed)
            dim: Dimension to compute sparsity over

        Returns:
            Scalar loss
        """
        l1 = z.sum(dim=dim)
        l2_sq = (z ** 2).sum(dim=dim)

        if self.target_sparsity is not None:
            # Target a specific normalized sparsity level
            n = z.shape[dim]
            l2 = torch.sqrt(l2_sq.clamp(min=self.eps))
            sqrt_n = torch.sqrt(
                torch.tensor(n, dtype=z.dtype, device=z.device)
            )

            # Normalized sparsity: 0 = dense, 1 = sparse
            normalized_sparsity = (sqrt_n - l1 / l2) / (sqrt_n - 1 + self.eps)
            normalized_sparsity = torch.clamp(normalized_sparsity, 0, 1)

            # Penalize deviation from target
            loss = (normalized_sparsity - self.target_sparsity) ** 2
            return loss.mean()
        else:
            # Minimize Hoyer ratio directly
            hoyer = (l1 ** 2) / (l2_sq + self.eps)
            return hoyer.mean()

    def compute_metrics(self, z: torch.Tensor) -> dict[str, float]:
        """
        Compute interpretable sparsity metrics for logging.

        Args:
            z: (batch, dim) latent tensor

        Returns:
            Dictionary of metrics
        """
        if self.absolute_values:
            z = z.abs()

        batch_size, latent_dim = z.shape

        # Compute normalized sparsity per sample
        l1_sample = z.sum(dim=1)
        l2_sample = torch.sqrt((z ** 2).sum(dim=1) + self.eps)

        sqrt_n = torch.sqrt(
            torch.tensor(latent_dim, dtype=z.dtype, device=z.device)
        )
        normalized_sparsity = (sqrt_n - l1_sample / l2_sample) / (sqrt_n - 1 + self.eps)
        normalized_sparsity = torch.clamp(normalized_sparsity, 0, 1)

        # Effective dimensions via entropy
        z_normalized = z / (z.sum(dim=1, keepdim=True) + self.eps)
        entropy = -(z_normalized * torch.log(z_normalized + self.eps)).sum(dim=1)
        effective_dims = torch.exp(entropy)

        # Count active dimensions
        active_50pct = self._count_active_dims(z, threshold=0.5)
        active_10pct = self._count_active_dims(z, threshold=0.1)

        return {
            "normalized_sparsity_mean": normalized_sparsity.mean().item(),
            "normalized_sparsity_std": normalized_sparsity.std().item(),
            "effective_dims_mean": effective_dims.mean().item(),
            "effective_dims_std": effective_dims.std().item(),
            "active_dims_50pct": active_50pct.float().mean().item(),
            "active_dims_10pct": active_10pct.float().mean().item(),
        }

    def _count_active_dims(
        self,
        z: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """
        Count dimensions with activation above threshold of max.

        Args:
            z: Latent tensor
            threshold: Fraction of max to consider "active"

        Returns:
            Count per sample
        """
        max_vals = z.max(dim=1, keepdim=True)[0]
        active = (z > threshold * max_vals).sum(dim=1)
        return active

    def extra_repr(self) -> str:
        return (
            f"target_sparsity={self.target_sparsity}, "
            f"per_sample={self.per_sample}, "
            f"per_dimension={self.per_dimension}"
        )


class AdaptiveHoyerLoss(HoyerLoss):
    """
    Hoyer regularizer with adaptive weight based on current sparsity.

    Increases regularization when activations are too dense,
    decreases when already sparse enough.

    This is the EPP implementation ported to BPD architecture.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Loss configuration containing:
                - All HoyerLoss config options
                - adaptation_rate: Rate of weight adaptation (default 0.01)
                - min_weight: Minimum adaptive weight (default 0.01)
                - max_weight: Maximum adaptive weight (default 10.0)
        """
        super().__init__(config)
        self.adaptation_rate = config["adaptation_rate"]
        self.min_weight = config["min_weight"]
        self.max_weight = config["max_weight"]
        self.register_buffer("current_weight", torch.tensor(1.0))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Compute loss with adaptive weight.

        Args:
            z: (batch, dim) latent tensor

        Returns:
            Tuple of (weighted_loss, current_weight)
        """
        base_loss = super().forward(z)

        # Compute current sparsity
        metrics = self.compute_metrics(z)
        current_sparsity = metrics["normalized_sparsity_mean"]

        # Adapt weight based on sparsity vs target
        if self.target_sparsity is not None:
            if current_sparsity < self.target_sparsity:
                # Too dense: increase weight
                self.current_weight = torch.clamp(
                    self.current_weight * (1 + self.adaptation_rate),
                    max=self.max_weight
                )
            else:
                # Sparse enough: decrease weight
                self.current_weight = torch.clamp(
                    self.current_weight * (1 - self.adaptation_rate),
                    min=self.min_weight
                )

        return base_loss * self.current_weight, self.current_weight.item()

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, "
            f"adaptation_rate={self.adaptation_rate}, "
            f"weight_range=[{self.min_weight}, {self.max_weight}]"
        )
