"""
Range Regularization Loss for Latent Utilization

Penalizes latent dimensions that have narrow activation range.
Range = max(mu) - min(mu) across the batch.

Narrow range indicates the dimension is constrained to a small
region of the latent space, limiting its representational capacity.

Uses exponential penalty similar to cluster separation loss.

Optionally includes variance component (toggleable).
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class RangeRegularizationLoss:
    """
    Penalize dimensions with range and/or variance below threshold.

    Range Loss = mean(exp(relu(min_range - range) / temperature) - 1)
    Variance Loss = mean(exp(relu(min_variance - variance) / temperature) - 1)

    Exponential penalty provides:
    - Zero loss when value >= threshold
    - Smooth gradient for small violations
    - Strong penalty for large violations
    """

    def __init__(self, config: dict):
        """
        Args:
            config: range_regularization config section containing:
                - enabled: bool
                - min_range: float (target minimum range)
                - temperature: float (penalty sharpness)
                - variance_enabled: bool (whether to include variance penalty)
                - min_variance: float (target minimum variance, if variance_enabled)
                - soft_temperature: float (temperature for soft max/min, default 0.1)
                - max_element_penalty: float (max penalty per dimension, default 100)
        """
        self.enabled = config["enabled"]
        if not self.enabled:
            logger.info("Range Regularization Loss disabled")
            return

        self.min_range = config["min_range"]
        self.temperature = config["temperature"]

        # Soft max/min temperature (for gradient flow at collapse)
        self.soft_temperature = config["soft_temperature"]

        # Compute exp_clamp from max_element_penalty
        # exp(clamp) - 1 = max_penalty => clamp = log(max_penalty + 1)
        self.max_element_penalty = config.get("max_element_penalty", 100.0)
        self.exp_clamp = float(torch.log(torch.tensor(self.max_element_penalty + 1.0)).item())

        # Variance component (optional)
        self.variance_enabled = config["variance_enabled"]
        if self.variance_enabled:
            self.min_variance = config["min_variance"]
            logger.info(
                f"Initialized RangeRegularizationLoss (min_range={self.min_range}, "
                f"min_variance={self.min_variance}, temperature={self.temperature}, "
                f"soft_temperature={self.soft_temperature}, max_element_penalty={self.max_element_penalty})"
            )
        else:
            logger.info(
                f"Initialized RangeRegularizationLoss (min_range={self.min_range}, "
                f"temperature={self.temperature}, soft_temperature={self.soft_temperature}, "
                f"variance_enabled=False, max_element_penalty={self.max_element_penalty})"
            )

    def _soft_range(self, mu: Tensor) -> Tensor:
        """
        Compute soft range using temperature-scaled softmax.

        Unlike hard max/min, soft max/min have non-zero gradients even when
        all values are equal (at collapse). This allows gradient flow to
        push values apart.

        soft_max = sum_i(softmax(mu_i / temp) * mu_i)
        soft_min = sum_i(softmax(-mu_i / temp) * mu_i)
        soft_range = soft_max - soft_min

        Args:
            mu: (batch, latent_dim) tensor

        Returns:
            (latent_dim,) tensor of soft ranges per dimension
        """
        # Transpose to (latent_dim, batch) for per-dimension softmax
        mu_t = mu.t()  # (latent_dim, batch)

        # Soft max: weight by softmax(mu/temp)
        weights_max = F.softmax(mu_t / self.soft_temperature, dim=1)  # (latent_dim, batch)
        soft_max = (weights_max * mu_t).sum(dim=1)  # (latent_dim,)

        # Soft min: weight by softmax(-mu/temp)
        weights_min = F.softmax(-mu_t / self.soft_temperature, dim=1)  # (latent_dim, batch)
        soft_min = (weights_min * mu_t).sum(dim=1)  # (latent_dim,)

        return soft_max - soft_min

    def _soft_variance(self, mu: Tensor) -> Tensor:
        """
        Compute soft variance that has non-zero gradients at collapse.

        Uses weighted variance where weights come from distance to mean.
        At collapse, standard variance has zero gradient, but this soft
        version maintains gradient flow.

        Args:
            mu: (batch, latent_dim) tensor

        Returns:
            (latent_dim,) tensor of soft variances per dimension
        """
        # Standard variance (for baseline)
        dim_mean = mu.mean(dim=0)  # (latent_dim,)
        deviations = mu - dim_mean  # (batch, latent_dim)

        # Standard variance
        variance = (deviations ** 2).mean(dim=0)  # (latent_dim,)

        # Add entropy-like term that has gradient even at collapse
        # -sum(p * log(p)) where p = softmax(mu/temp)
        mu_t = mu.t()  # (latent_dim, batch)
        log_probs = F.log_softmax(mu_t / self.soft_temperature, dim=1)
        probs = F.softmax(mu_t / self.soft_temperature, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)  # (latent_dim,)

        # At collapse: entropy = log(batch_size) (uniform distribution)
        # When spread: entropy < log(batch_size) (concentrated distribution)
        # We want to MAXIMIZE spread, so we add a term that rewards high variance

        # Combine: variance gives magnitude, entropy provides gradient at collapse
        # Scale entropy term to be comparable to variance scale
        batch_size = mu.shape[0]
        max_entropy = torch.log(torch.tensor(float(batch_size), device=mu.device))

        # soft_var = variance + (1 - entropy/max_entropy) * small_constant
        # This doesn't change much when variance is healthy, but provides
        # gradient signal when variance is zero

        return variance + 0.01 * (max_entropy - entropy)

    def compute(self, mu: Tensor) -> Tensor:
        """
        Compute range regularization loss.

        Uses soft range/variance to ensure gradient flow even at collapse.
        Hard max/min have zero gradient when all values are equal, but
        soft versions (using softmax weighting) maintain gradient flow.

        Additionally, small perturbation noise is added to break symmetry
        at perfect collapse. This ensures non-zero gradients flow back
        to push samples apart.

        Args:
            mu: Distribution means (batch, latent_dim) - should be TRUE mu without dropout

        Returns:
            Scalar loss tensor (0 if disabled)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=mu.device)

        # Guard: variance requires batch_size >= 2
        if mu.shape[0] < 2:
            return torch.tensor(0.0, device=mu.device)

        # Add small perturbation to break symmetry at collapse
        # This ensures gradients can flow even when all mu are identical
        # The noise scale is small enough to not affect normal training
        # but provides gradient direction at collapse
        perturbation_scale = 0.01
        perturbation = torch.randn_like(mu) * perturbation_scale
        mu_perturbed = mu + perturbation

        # Compute SOFT range per dimension (has gradients at collapse)
        soft_range = self._soft_range(mu_perturbed)  # (latent_dim,)

        # Exponential penalty for ranges below threshold
        # Clamp scaled violations so max per-dimension penalty = max_element_penalty
        range_violations = F.relu(self.min_range - soft_range)
        scaled_range_violations = (range_violations / self.temperature).clamp(max=self.exp_clamp)
        range_penalty = torch.exp(scaled_range_violations) - 1.0

        # Mean across dimensions
        loss = range_penalty.mean()

        # Add variance component if enabled
        if self.variance_enabled:
            # Use soft variance for gradient flow at collapse
            # Use perturbed mu for consistent gradient flow
            soft_variance = self._soft_variance(mu_perturbed)  # (latent_dim,)

            # Exponential penalty for variance below threshold
            # Clamp scaled violations so max per-dimension penalty = max_element_penalty
            var_violations = F.relu(self.min_variance - soft_variance)
            scaled_var_violations = (var_violations / self.temperature).clamp(max=self.exp_clamp)
            var_penalty = torch.exp(scaled_var_violations) - 1.0

            loss = loss + var_penalty.mean()

            # Additional: Direct penalty proportional to violation
            # This provides linear gradient when exponential saturates
            collapse_threshold = 0.01
            collapse_violations = F.relu(collapse_threshold - soft_variance)
            collapse_penalty = collapse_violations.mean() * 10.0  # Moderate multiplier
            loss = loss + collapse_penalty.clamp(max=self.max_element_penalty)

        return loss
