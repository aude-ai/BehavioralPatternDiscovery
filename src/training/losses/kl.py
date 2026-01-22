"""
KL Divergence Loss

KL divergence between approximate posterior and prior.
Supports Gaussian, Gamma, and vMF distributions with capacity control.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from src.core.numerical import NumericalConstants

logger = logging.getLogger(__name__)


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss with capacity control.

    Computes KL(q(z|x) || p(z)) with optional capacity constraint:
        loss = |KL - C|

    Supports:
    - Gaussian: KL(N(mu, sigma) || N(0, 1))
    - Gamma: KL(Gamma(alpha, beta) || Gamma(prior_alpha, prior_beta))
    - vMF: KL(vMF(mu, kappa) || Uniform) or KL(vMF(mu, kappa) || vMF(mu_0, kappa_0))
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Loss configuration containing:
                - distribution_type: "gaussian", "gamma", or "vmf"
                - use_capacity: Whether to use capacity constraint
                - For Gamma prior: prior_alpha, prior_beta
                - For vMF: prior_type, prior_kappa (optional)
        """
        super().__init__()
        self.distribution_type = config["distribution_type"]
        self.use_capacity = config["use_capacity"]

        if self.distribution_type == "gamma":
            self.prior_alpha = config["prior_alpha"]
            self.prior_beta = config["prior_beta"]
        elif self.distribution_type == "vmf":
            self.vmf_prior_type = config["prior_type"]
            self.vmf_prior_kappa = config["prior_kappa"]

    def forward(
        self,
        params: dict[str, torch.Tensor],
        capacity: float = 0.0,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute KL divergence loss.

        Args:
            params: Distribution parameters
                - Gaussian: {"mu": tensor, "logvar": tensor}
                - Gamma: {"concentration": tensor, "rate": tensor, "log_concentration": tensor, "log_rate": tensor}
            capacity: Current capacity target (if using capacity control)
            return_components: If True, return (loss, raw_kl, capacity) tuple

        Returns:
            If return_components=False: Scalar KL loss value
            If return_components=True: Tuple of (loss, raw_kl, capacity_tensor)
        """
        if self.distribution_type == "gaussian":
            kl = self._gaussian_kl(params["mu"], params["logvar"])
        elif self.distribution_type == "gamma":
            kl = self._gamma_kl(params["log_concentration"], params["log_rate"])
        elif self.distribution_type == "vmf":
            kl = self._vmf_kl(params["mu"], params["kappa"])
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        if self.use_capacity:
            loss = torch.abs(kl - capacity)
        else:
            loss = kl

        if return_components:
            capacity_tensor = torch.tensor(capacity, device=kl.device, dtype=kl.dtype)
            return loss, kl, capacity_tensor
        else:
            return loss

    def forward_with_dim_stats(
        self,
        params: dict[str, torch.Tensor],
        capacity: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute KL loss and per-dimension statistics.

        Args:
            params: Distribution parameters
            capacity: Current capacity target

        Returns:
            Tuple of (loss, stats_dict) where stats_dict contains:
                - kl_dim_mean: Mean KL per dimension (averaged over batch)
                - kl_dim_std: Std of KL across dimensions
        """
        if self.distribution_type == "gaussian":
            kl_per_dim = self._gaussian_kl_per_dim(params["mu"], params["logvar"])
        elif self.distribution_type == "gamma":
            kl_per_dim = self._gamma_kl_per_dim(
                params["log_concentration"], params["log_rate"]
            )
        elif self.distribution_type == "vmf":
            kl_per_dim = self._vmf_kl_per_dim(params["mu"], params["kappa"])
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        # Per-sample KL (sum over dims)
        kl_per_sample = kl_per_dim.sum(dim=1)
        kl = kl_per_sample.mean()

        # Per-dimension stats (mean over batch first, then stats over dims)
        kl_dim_mean_per_dim = kl_per_dim.mean(dim=0)  # (dim,)
        dim_stats = {
            "kl_dim_mean": kl_dim_mean_per_dim.mean().item(),
            "kl_dim_std": kl_dim_mean_per_dim.std().item(),
        }

        if self.use_capacity:
            loss = torch.abs(kl - capacity)
        else:
            loss = kl

        return loss, dim_stats

    def _gaussian_kl_per_dim(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return per-dimension KL (batch, dim) without summing.

        Args:
            mu: (batch, dim) mean
            logvar: (batch, dim) log variance

        Returns:
            (batch, dim) KL per dimension
        """
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    def _gamma_kl_per_dim(
        self,
        log_concentration: torch.Tensor,
        log_rate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return per-dimension KL (batch, dim) without summing.

        Args:
            log_concentration: (batch, dim) log shape parameter
            log_rate: (batch, dim) log rate parameter

        Returns:
            (batch, dim) KL per dimension
        """
        concentration = log_concentration.exp()
        rate = log_rate.exp()

        # Clamp for digamma/lgamma stability
        safe_concentration = concentration.clamp(
            min=NumericalConstants.GAMMA_CONC_MIN,
            max=NumericalConstants.GAMMA_CONC_MAX
        )
        safe_rate = rate.clamp(
            min=NumericalConstants.GAMMA_RATE_MIN,
            max=NumericalConstants.GAMMA_RATE_MAX
        )

        prior_conc = torch.tensor(
            self.prior_alpha, device=concentration.device, dtype=concentration.dtype
        )
        prior_rate = torch.tensor(
            self.prior_beta, device=rate.device, dtype=rate.dtype
        )

        digamma_term = torch.digamma(safe_concentration)
        lgamma_conc = torch.lgamma(safe_concentration)
        lgamma_prior = torch.lgamma(prior_conc)
        log_safe_rate = torch.log(safe_rate)
        log_prior_rate = torch.log(prior_rate)

        kl_per_dim = (
            (safe_concentration - prior_conc) * digamma_term
            - lgamma_conc
            + lgamma_prior
            + prior_conc * (log_safe_rate - log_prior_rate)
            + safe_concentration * (prior_rate / safe_rate - 1)
        )

        # Clamp to non-negative (KL divergence is always >= 0)
        kl_per_dim = kl_per_dim.clamp(min=0.0)

        return kl_per_dim

    def _gaussian_kl(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence for Gaussian: KL(N(mu, sigma) || N(0, 1))

        Args:
            mu: (batch, dim) mean
            logvar: (batch, dim) log variance

        Returns:
            Scalar KL divergence (mean over batch)
        """
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_sample = kl_per_dim.sum(dim=1)
        return kl_per_sample.mean()

    def _gamma_kl(
        self,
        log_concentration: torch.Tensor,
        log_rate: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence for Gamma: KL(Gamma(conc, rate) || Gamma(prior_conc, prior_rate))

        Uses preventive clamping for numerical stability:
        - Clamps concentration to 0.5+ for digamma/lgamma safety
        - All inputs are clamped before operations

        Args:
            log_concentration: (batch, dim) log shape parameter
            log_rate: (batch, dim) log rate parameter

        Returns:
            Scalar KL divergence (mean over batch)
        """
        concentration = log_concentration.exp()
        rate = log_rate.exp()

        # Clamp for digamma/lgamma stability
        safe_concentration = concentration.clamp(
            min=NumericalConstants.GAMMA_CONC_MIN,
            max=NumericalConstants.GAMMA_CONC_MAX
        )
        safe_rate = rate.clamp(
            min=NumericalConstants.GAMMA_RATE_MIN,
            max=NumericalConstants.GAMMA_RATE_MAX
        )

        prior_conc = torch.tensor(
            self.prior_alpha, device=concentration.device, dtype=concentration.dtype
        )
        prior_rate = torch.tensor(
            self.prior_beta, device=rate.device, dtype=rate.dtype
        )

        # KL for Gamma distributions with safe values
        digamma_term = torch.digamma(safe_concentration)
        lgamma_conc = torch.lgamma(safe_concentration)
        lgamma_prior = torch.lgamma(prior_conc)
        log_safe_rate = torch.log(safe_rate)
        log_prior_rate = torch.log(prior_rate)

        kl_per_dim = (
            (safe_concentration - prior_conc) * digamma_term
            - lgamma_conc
            + lgamma_prior
            + prior_conc * (log_safe_rate - log_prior_rate)
            + safe_concentration * (prior_rate / safe_rate - 1)
        )

        # Clamp to non-negative (KL divergence is always >= 0)
        kl_per_dim = kl_per_dim.clamp(min=0.0)

        kl_per_sample = kl_per_dim.sum(dim=1)
        return kl_per_sample.mean()

    def _vmf_kl(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence for vMF: KL(vMF(mu, kappa) || Uniform) or KL(vMF(mu, kappa) || vMF(mu_0, kappa_0))

        For uniform prior (kappa_0 = 0):
            KL = kappa + log(C_d(kappa)) - log(C_d(0))

        Args:
            mu: (batch, dim) mean direction (unit normalized)
            kappa: (batch, 1) or (batch,) concentration parameter

        Returns:
            Scalar KL divergence (mean over batch)
        """
        from src.model.distributions.vmf_utils import log_vmf_normalizer, log_uniform_density

        if kappa.dim() == 2:
            kappa = kappa.squeeze(-1)  # (batch,)

        dim = mu.shape[-1]

        # Clamp kappa for numerical stability
        kappa = kappa.clamp(min=NumericalConstants.VMF_KAPPA_MIN)

        if self.vmf_prior_type == "uniform":
            # KL to uniform distribution on sphere
            log_c_kappa = log_vmf_normalizer(kappa, dim)
            log_c_0 = log_uniform_density(dim)

            # KL = kappa + log(C_d(kappa)) - log(C_d(0))
            kl_per_sample = kappa + log_c_kappa - log_c_0
        else:
            # KL to vMF prior with kappa_0
            # Simplified: assuming mu aligns with prior_mu for now
            prior_kappa = torch.tensor(
                self.vmf_prior_kappa, device=kappa.device, dtype=kappa.dtype
            ).clamp(min=NumericalConstants.VMF_KAPPA_MIN)

            log_c_kappa = log_vmf_normalizer(kappa, dim)
            log_c_prior = log_vmf_normalizer(prior_kappa.expand_as(kappa), dim)

            kl_per_sample = log_c_kappa - log_c_prior + (kappa - prior_kappa)

        # Clamp to non-negative
        kl_per_sample = kl_per_sample.clamp(min=0.0)

        return kl_per_sample.mean()

    def _vmf_kl_per_dim(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return per-dimension KL (batch, dim) for vMF.

        Note: vMF KL is not naturally decomposable per-dimension since it operates
        on the full direction vector. We distribute the scalar KL equally across
        dimensions for compatibility with the per-dim stats interface.

        Args:
            mu: (batch, dim) mean direction
            kappa: (batch, 1) or (batch,) concentration

        Returns:
            (batch, dim) KL distributed across dimensions
        """
        from src.model.distributions.vmf_utils import log_vmf_normalizer, log_uniform_density

        if kappa.dim() == 2:
            kappa = kappa.squeeze(-1)

        batch_size, dim = mu.shape

        # Clamp kappa
        kappa = kappa.clamp(min=NumericalConstants.VMF_KAPPA_MIN)

        if self.vmf_prior_type == "uniform":
            log_c_kappa = log_vmf_normalizer(kappa, dim)
            log_c_0 = log_uniform_density(dim)
            kl_per_sample = kappa + log_c_kappa - log_c_0
        else:
            prior_kappa = torch.tensor(
                self.vmf_prior_kappa, device=kappa.device, dtype=kappa.dtype
            ).clamp(min=NumericalConstants.VMF_KAPPA_MIN)

            log_c_kappa = log_vmf_normalizer(kappa, dim)
            log_c_prior = log_vmf_normalizer(prior_kappa.expand_as(kappa), dim)

            kl_per_sample = log_c_kappa - log_c_prior + (kappa - prior_kappa)

        kl_per_sample = kl_per_sample.clamp(min=0.0)

        # Distribute KL equally across dimensions (for stats interface compatibility)
        kl_per_dim = kl_per_sample.unsqueeze(1) / dim  # (batch, 1)
        kl_per_dim = kl_per_dim.expand(batch_size, dim)  # (batch, dim)

        return kl_per_dim

    def extra_repr(self) -> str:
        return f"distribution_type={self.distribution_type}, use_capacity={self.use_capacity}"
