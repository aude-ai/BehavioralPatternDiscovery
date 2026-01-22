"""
Entropy Uniformity Loss for Stratification Prevention

Penalizes low-entropy (stratified) distributions in latent space.
Uses differentiable soft histogram for gradient flow.

Computed separately per encoder per level - no cross-boundary comparisons.
Fully vectorized for performance.
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class EntropyUniformityLoss:
    """
    Penalize stratified distributions using soft histogram entropy.

    Computes entropy per dimension, penalizes if below target.
    Trainer calls compute(mu) per encoder/level.

    Fully vectorized - computes all dimensions in parallel.
    """

    def __init__(self, config: dict):
        """
        Initialize EntropyUniformityLoss.

        Args:
            config: entropy_uniformity_loss config section with keys:
                - enabled: bool
                - num_bins: int (histogram resolution)
                - min_entropy_ratio: float (target as fraction of max entropy)
                - temperature: float (penalty sharpness)
                - soft_bin_temperature: float (histogram softness)
                - max_element_penalty: float (max penalty per dimension, default 100)

        Raises:
            KeyError: If required config keys are missing
        """
        self.enabled = config["enabled"]
        if not self.enabled:
            logger.info("EntropyUniformityLoss disabled")
            return

        self.num_bins = config["num_bins"]
        self.min_entropy_ratio = config["min_entropy_ratio"]
        self.temperature = config["temperature"]
        self.soft_bin_temp = config["soft_bin_temperature"]

        # Compute exp_clamp from max_element_penalty
        # exp(clamp) - 1 = max_penalty => clamp = log(max_penalty + 1)
        max_element_penalty = config.get("max_element_penalty", 100.0)
        self.exp_clamp = float(torch.log(torch.tensor(max_element_penalty + 1.0)).item())

        logger.info(
            f"Initialized EntropyUniformityLoss "
            f"(num_bins={self.num_bins}, min_entropy_ratio={self.min_entropy_ratio}, "
            f"temperature={self.temperature}, soft_bin_temp={self.soft_bin_temp}, "
            f"max_element_penalty={max_element_penalty})"
        )

    def compute(self, mu: Tensor) -> Tensor:
        """
        Compute entropy loss for single encoder/level (vectorized).

        Args:
            mu: (batch, latent_dim) mu values

        Returns:
            Scalar loss tensor
        """
        if not self.enabled:
            return torch.tensor(0.0, device=mu.device)

        batch_size, latent_dim = mu.shape

        # Skip if batch too small for meaningful histogram
        if batch_size < self.num_bins:
            return torch.tensor(0.0, device=mu.device)

        # Transpose to (latent_dim, batch) for vectorized operations
        mu_t = mu.t()  # (latent_dim, batch)

        # Get min/max per dimension
        dim_min = mu_t.min(dim=1, keepdim=True).values  # (latent_dim, 1)
        dim_max = mu_t.max(dim=1, keepdim=True).values  # (latent_dim, 1)
        dim_range = dim_max - dim_min  # (latent_dim, 1)

        # Handle collapsed dimensions (all same value) - give them max entropy (no penalty)
        collapsed_mask = (dim_range < 1e-6).squeeze(1)  # (latent_dim,)

        # Normalize mu to [0, 1] range per dimension (avoid division by zero)
        mu_normalized = (mu_t - dim_min) / dim_range.clamp(min=1e-6)  # (latent_dim, batch)

        # Create bin centers in [0, 1]
        bin_centers = torch.linspace(0, 1, self.num_bins, device=mu.device)  # (num_bins,)

        # Compute distances: (latent_dim, batch, num_bins)
        # mu_normalized: (latent_dim, batch) -> (latent_dim, batch, 1)
        # bin_centers: (num_bins,) -> (1, 1, num_bins)
        distances = (mu_normalized.unsqueeze(2) - bin_centers.view(1, 1, -1)) ** 2

        # Soft assignment via softmax over bins: (latent_dim, batch, num_bins)
        assignments = F.softmax(-distances / self.soft_bin_temp, dim=2)

        # Sum over batch to get bin counts: (latent_dim, num_bins)
        bin_counts = assignments.sum(dim=1)

        # Normalize to probability: (latent_dim, num_bins)
        probs = bin_counts / bin_counts.sum(dim=1, keepdim=True).clamp(min=1e-10)

        # Compute entropy per dimension: (latent_dim,)
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=1)  # (latent_dim,)

        # Target entropy
        max_entropy = torch.log(torch.tensor(float(self.num_bins), device=mu.device))
        target_entropy = self.min_entropy_ratio * max_entropy

        # Exponential penalty for low entropy
        # Clamp scaled violations so max per-dimension penalty = max_element_penalty
        violations = F.relu(target_entropy - entropy)  # (latent_dim,)
        scaled_violations = (violations / self.temperature).clamp(max=self.exp_clamp)
        dim_losses = torch.exp(scaled_violations) - 1.0  # (latent_dim,)

        # Set collapsed dimensions to zero loss (they get max entropy = no penalty)
        dim_losses = dim_losses.masked_fill(collapsed_mask, 0.0)

        # Average over dimensions
        return dim_losses.mean()
