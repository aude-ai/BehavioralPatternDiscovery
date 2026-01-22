"""
Gamma Latent Distribution

Gamma distribution for positive-valued latent spaces.

Parameters:
- concentration (alpha): Shape parameter
- rate (beta): Rate parameter (inverse scale)
- Mean = alpha / beta
- Variance = alpha / beta^2

Uses log-space parameterization for numerical stability:
- Network outputs log_concentration and log_rate
- These are clamped and then exponentiated

KL Divergence uses analytical formula for Gamma distributions.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Gamma

from .base import LatentDistribution, distribution_registry

logger = logging.getLogger(__name__)


@distribution_registry.register("gamma")
class GammaDistribution(LatentDistribution):
    """
    Gamma latent distribution for positive-valued latent spaces.

    Benefits:
    - Latents are always positive
    - Natural sparsity with appropriate prior
    - Good for mixture models and hierarchical structures

    Uses log-space parameterization (similar to Gaussian's logvar):
    - Network outputs log_concentration and log_rate
    - Clamp in log-space for stability
    - exp() converts to positive values
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Must contain:
                - prior_concentration: Prior alpha (1.0 gives exponential/sparse prior)
                - prior_rate: Prior beta
                - log_concentration_min: Min clamp for log(concentration)
                - log_concentration_max: Max clamp for log(concentration)
                - log_rate_min: Min clamp for log(rate)
                - log_rate_max: Max clamp for log(rate)
                - use_monte_carlo_kl: If True, use MC estimate instead of analytical
                - monte_carlo_samples: Number of samples for MC estimate
        """
        self.prior_concentration = config["prior_concentration"]
        self.prior_rate = config["prior_rate"]
        self.log_concentration_min = config["log_concentration_min"]
        self.log_concentration_max = config["log_concentration_max"]
        self.log_rate_min = config["log_rate_min"]
        self.log_rate_max = config["log_rate_max"]

        # Monte Carlo KL option (avoids digamma/lgamma numerical issues)
        self.use_monte_carlo_kl = config["use_monte_carlo_kl"]
        self.monte_carlo_samples = config["monte_carlo_samples"]

        # Compute concentration/rate bounds from log bounds (for rsample stability)
        # Enforce minimum 0.5 for concentration to ensure digamma/lgamma safety
        import math
        self.concentration_min = max(0.5, math.exp(self.log_concentration_min))
        self.concentration_max = math.exp(self.log_concentration_max)
        self.rate_min = math.exp(self.log_rate_min)
        self.rate_max = math.exp(self.log_rate_max)

        if self.use_monte_carlo_kl:
            logger.info(
                f"GammaDistribution using Monte Carlo KL with "
                f"{self.monte_carlo_samples} samples"
            )

    @property
    def param_names(self) -> tuple[str, ...]:
        return ("concentration", "rate")

    def create_param_layers(
        self,
        in_dim: int,
        out_dim: int
    ) -> nn.ModuleDict:
        """Create log_concentration and log_rate projection layers."""
        return nn.ModuleDict({
            "log_concentration": nn.Linear(in_dim, out_dim),
            "log_rate": nn.Linear(in_dim, out_dim),
        })

    def forward_params(
        self,
        h: torch.Tensor,
        layers: nn.ModuleDict
    ) -> dict[str, torch.Tensor]:
        """
        Compute Gamma parameters from hidden state.

        Uses log-space parameterization:
        1. Network outputs log_concentration and log_rate
        2. Clamp in log-space for numerical stability (preventive)
        3. exp() converts to positive concentration and rate
        """
        log_conc = layers["log_concentration"](h)
        log_rate = layers["log_rate"](h)

        # Clamp in log-space (preventive, not reactive)
        log_conc = log_conc.clamp(min=self.log_concentration_min, max=self.log_concentration_max)
        log_rate = log_rate.clamp(min=self.log_rate_min, max=self.log_rate_max)

        # Convert to positive values
        concentration = torch.exp(log_conc)
        rate = torch.exp(log_rate)

        return {
            "concentration": concentration,
            "rate": rate,
            "log_concentration": log_conc,
            "log_rate": log_rate,
        }

    def reparameterize(
        self,
        params: dict[str, torch.Tensor],
        training: bool
    ) -> torch.Tensor:
        """
        Sample using reparameterization trick.

        During training: rsample() from Gamma distribution
        During inference: mean = concentration / rate (deterministic)

        Uses simple direct rsample() like EPP2 for stability.
        Parameters are clamped to safe range before rsample.
        """
        concentration = params["concentration"]
        rate = params["rate"]

        if training:
            # Use class constants for consistent bounds (0.5 min for digamma safety)
            safe_concentration = concentration.clamp(min=self.concentration_min, max=self.concentration_max)
            safe_rate = rate.clamp(min=self.rate_min, max=self.rate_max)

            # Direct rsample - parameters are clamped to safe range
            gamma_dist = Gamma(safe_concentration, safe_rate)
            samples = gamma_dist.rsample()

            # Ensure samples are positive
            samples = samples.clamp(min=1e-10)

            return samples

        # Inference: return deterministic mean
        return concentration / rate

    def get_mean(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return alpha/beta (deterministic mean for SHAP)."""
        return params["concentration"] / params["rate"]

    def kl_divergence(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        KL(Gamma(alpha_q, beta_q) || Gamma(alpha_p, beta_p))

        Uses Monte Carlo estimation if configured, otherwise analytical formula.

        Returns:
            Per-sample KL divergence (batch,)
        """
        if self.use_monte_carlo_kl:
            return self._kl_divergence_monte_carlo(params)
        else:
            return self._kl_divergence_analytical(params)

    def _kl_divergence_monte_carlo(
        self,
        params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Monte Carlo estimate of KL divergence.

        Avoids digamma/lgamma functions which can be numerically unstable.
        KL = E_q[log q(z) - log p(z)]

        Returns:
            Per-sample KL divergence (batch,)
        """
        concentration = params["concentration"]
        rate = params["rate"]

        # Clamp for stability
        safe_conc = torch.clamp(
            concentration,
            min=self.concentration_min,
            max=self.concentration_max
        )
        safe_rate = torch.clamp(
            rate,
            min=self.rate_min,
            max=self.rate_max
        )

        # Create distributions
        q = Gamma(safe_conc, safe_rate)
        p = Gamma(
            torch.full_like(safe_conc, self.prior_concentration),
            torch.full_like(safe_rate, self.prior_rate)
        )

        # Sample from q: (n_samples, batch, dim)
        samples = q.rsample((self.monte_carlo_samples,))

        # Clamp samples to valid range for log_prob
        samples = samples.clamp(min=1e-10)

        # Compute log probabilities
        # Parameters and samples are clamped to safe ranges, log_prob should not produce NaN
        log_q = q.log_prob(samples)
        log_p = p.log_prob(samples)

        # KL = E_q[log q - log p], sum over dims, mean over samples
        kl_per_sample_per_dim = log_q - log_p  # (n_samples, batch, dim)
        kl_per_dim = kl_per_sample_per_dim.mean(dim=0)  # (batch, dim)
        kl = kl_per_dim.sum(dim=-1)  # (batch,)

        # Clamp to non-negative
        kl = torch.clamp(kl, min=0.0)

        return kl

    def _kl_divergence_analytical(
        self,
        params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Analytical KL divergence for Gamma distributions.

        Uses preventive clamping for numerical stability:
        - Clamps concentration to 0.5+ for digamma/lgamma safety
        - All inputs are clamped before operations

        KL = (alpha_q - alpha_p) * digamma(alpha_q)
             - log(Gamma(alpha_q)) + log(Gamma(alpha_p))
             + alpha_p * (log(beta_q) - log(beta_p))
             + alpha_q * (beta_p/beta_q - 1)

        Returns:
            Per-sample KL divergence (batch,)
        """
        alpha_q = params["concentration"]
        beta_q = params["rate"]
        alpha_p = self.prior_concentration
        beta_p = self.prior_rate

        # Clamp to safe range for digamma/lgamma (0.5 min for stability)
        safe_alpha_q = alpha_q.clamp(min=self.concentration_min, max=self.concentration_max)
        safe_beta_q = beta_q.clamp(min=self.rate_min, max=self.rate_max)

        # Create tensors for prior parameters on correct device
        alpha_p_tensor = torch.tensor(
            alpha_p, device=alpha_q.device, dtype=alpha_q.dtype
        )
        beta_p_tensor = torch.tensor(
            beta_p, device=beta_q.device, dtype=beta_q.dtype
        )

        # Analytical KL divergence for Gamma distributions
        digamma_term = torch.digamma(safe_alpha_q)
        lgamma_q = torch.lgamma(safe_alpha_q)
        lgamma_p = torch.lgamma(alpha_p_tensor)
        log_beta_q = torch.log(safe_beta_q)
        log_beta_p = torch.log(beta_p_tensor)

        kl_per_dim = (
            (safe_alpha_q - alpha_p) * digamma_term
            - lgamma_q
            + lgamma_p
            + alpha_p * (log_beta_q - log_beta_p)
            + safe_alpha_q * (beta_p / safe_beta_q - 1)
        )

        # Clamp to non-negative (KL divergence is always >= 0)
        kl_per_dim = kl_per_dim.clamp(min=0.0)

        return kl_per_dim.sum(dim=-1)

    def scale_mean_params(
        self,
        params: dict[str, torch.Tensor],
        scale_factor: float
    ) -> dict[str, torch.Tensor]:
        """
        Scale concentration for Scale-VAE.

        For Gamma, mean = concentration / rate.
        Scaling concentration by k gives: new_mean = k * concentration / rate = k * original_mean

        Args:
            params: Distribution parameters containing 'concentration', 'rate', etc.
            scale_factor: Factor to scale mean by

        Returns:
            New params dict with scaled concentration (rate unchanged)
        """
        return {
            "concentration": params["concentration"] * scale_factor,
            "rate": params["rate"],
            "log_concentration": params["log_concentration"],
            "log_rate": params["log_rate"],
        }
