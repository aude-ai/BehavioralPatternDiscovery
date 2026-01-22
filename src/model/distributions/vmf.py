"""
Von Mises-Fisher Distribution for Hyperspherical Latent Spaces.

The vMF distribution is defined on the unit hypersphere S^{d-1}.
Parameters:
- mu: Mean direction (unit vector on sphere)
- kappa: Concentration parameter (kappa >= 0, kappa=0 is uniform)

All operations use centralized numerical stability utilities from src/core/numerical.py.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from src.core.numerical import (
    NumericalConstants,
    safe_exp,
    safe_log,
    safe_normalize,
)
from .base import LatentDistribution, distribution_registry
from .vmf_utils import (
    log_vmf_normalizer,
    log_uniform_density,
    sample_vmf,
)

logger = logging.getLogger(__name__)


@distribution_registry.register("vmf")
class VonMisesFisherDistribution(LatentDistribution):
    """
    Von Mises-Fisher distribution for hyperspherical latent space.

    The vMF distribution has density:
        f(x; mu, kappa) = C_d(kappa) * exp(kappa * mu^T x)

    where:
        - x is on the unit sphere S^{d-1}
        - mu is the mean direction (unit vector)
        - kappa >= 0 is the concentration parameter
        - C_d(kappa) is the normalization constant

    For kappa = 0, the distribution is uniform on the sphere.
    As kappa -> infinity, the distribution concentrates at mu.

    All outputs are on the unit sphere, making this suitable for:
        - Angular/directional data
        - Text embeddings (often normalized)
        - Any representation where direction matters more than magnitude
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize vMF distribution from config.

        Args:
            config: Must contain:
                - kappa_min: Minimum concentration (0 = uniform)
                - kappa_max: Maximum concentration
                - prior_type: "uniform" or "learnable"
                - prior_kappa: Prior concentration if prior_type="learnable"
                - use_monte_carlo_kl: Whether to use MC KL estimation
                - monte_carlo_samples: Number of MC samples for KL
        """
        self.kappa_min = config["kappa_min"]
        self.kappa_max = config["kappa_max"]
        self.prior_type = config["prior_type"]
        self.prior_kappa = config["prior_kappa"]
        self.use_monte_carlo_kl = config["use_monte_carlo_kl"]
        self.monte_carlo_samples = config["monte_carlo_samples"]

        # Compute log bounds for network output clamping
        self.log_kappa_min = math.log(max(self.kappa_min, NumericalConstants.VMF_KAPPA_MIN))
        self.log_kappa_max = math.log(min(self.kappa_max, NumericalConstants.VMF_KAPPA_MAX))

        # For learnable prior (initialized lazily based on latent dim)
        self._prior_mu: torch.Tensor | None = None

        logger.info(
            f"VonMisesFisherDistribution: kappa=[{self.kappa_min:.2f}, {self.kappa_max:.2f}], "
            f"prior={self.prior_type}, mc_kl={self.use_monte_carlo_kl}"
        )

    @property
    def param_names(self) -> tuple[str, ...]:
        return ("mu", "kappa", "log_kappa")

    def create_param_layers(
        self,
        in_dim: int,
        out_dim: int
    ) -> nn.ModuleDict:
        """
        Create layers that output vMF parameters.

        Args:
            in_dim: Input dimension from encoder hidden state
            out_dim: Latent dimension (dimension of mu on sphere)

        Returns:
            ModuleDict with 'mu' and 'log_kappa' layers
        """
        return nn.ModuleDict({
            # mu: direction vector (will be L2-normalized)
            "mu": nn.Linear(in_dim, out_dim),
            # log_kappa: log of concentration (single scalar per sample)
            "log_kappa": nn.Linear(in_dim, 1),
        })

    def forward_params(
        self,
        h: torch.Tensor,
        layers: nn.ModuleDict
    ) -> dict[str, torch.Tensor]:
        """
        Compute vMF parameters from hidden state.

        mu is L2-normalized to unit sphere.
        log_kappa is clamped to safe range before exp.

        Args:
            h: Hidden state (batch, in_dim)
            layers: ModuleDict from create_param_layers

        Returns:
            Dictionary with:
                - 'mu': Unit-normalized direction (batch, out_dim)
                - 'kappa': Concentration (batch, 1)
                - 'log_kappa': Log concentration (batch, 1)
        """
        # Compute mu and normalize to unit sphere (using safe_normalize to handle zero vectors)
        mu_raw = layers["mu"](h)
        mu = safe_normalize(mu_raw, dim=-1)

        # Compute log_kappa and clamp to safe range
        log_kappa_raw = layers["log_kappa"](h)
        log_kappa = log_kappa_raw.clamp(min=self.log_kappa_min, max=self.log_kappa_max)

        # Convert to kappa (safe_exp already clamps, but we've pre-clamped log_kappa)
        kappa = safe_exp(log_kappa)

        return {
            "mu": mu,
            "kappa": kappa,
            "log_kappa": log_kappa,
        }

    def reparameterize(
        self,
        params: dict[str, torch.Tensor],
        training: bool
    ) -> torch.Tensor:
        """
        Sample from vMF distribution.

        During training: Use rejection sampling (reparameterized)
        During inference: Return mean direction mu

        Args:
            params: Dictionary with 'mu' and 'kappa'
            training: Whether in training mode

        Returns:
            Samples on unit sphere (batch, dim)
        """
        mu = params["mu"]
        kappa = params["kappa"]

        if not training:
            # Return mean direction at inference time
            return mu

        # Sample using rejection sampling
        samples = sample_vmf(mu, kappa.squeeze(-1))

        return samples

    def get_mean(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return the mean direction (for SHAP, inference).

        For vMF, the mode (most likely direction) is mu.

        Args:
            params: Dictionary containing 'mu'

        Returns:
            Mean direction (batch, dim)
        """
        return params["mu"]

    def kl_divergence(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute KL divergence from prior.

        For uniform prior (kappa_0 = 0):
            KL(vMF(mu, kappa) || Uniform) = log(C_d(kappa)) - log(1/surface_area) + kappa

        For learnable prior:
            KL(vMF(mu, kappa) || vMF(mu_0, kappa_0))

        Args:
            params: Dictionary with 'mu', 'kappa'

        Returns:
            Per-sample KL divergence (batch,)
        """
        mu = params["mu"]
        kappa = params["kappa"].squeeze(-1)  # (batch,)
        dim = mu.shape[-1]

        if self.use_monte_carlo_kl:
            return self._monte_carlo_kl(params, dim)

        if self.prior_type == "uniform":
            return self._kl_to_uniform(kappa, dim)
        else:
            return self._kl_to_vmf_prior(mu, kappa, dim)

    def _kl_to_uniform(self, kappa: torch.Tensor, dim: int) -> torch.Tensor:
        """
        KL divergence to uniform distribution on sphere.

        KL = log(C_d(kappa)) - log(uniform_density) + kappa * E[mu^T z]

        For vMF, E[mu^T z] = A_d(kappa) where A_d is the mean resultant length.
        For simplicity, we use: E[mu^T z] ≈ 1 - d/(2*kappa) for large kappa, ≈ kappa/(d-1) for small.

        But actually, for KL to uniform:
            KL = log(C_d(kappa)) + kappa - log(C_d(0))

        where C_d(0) = 1/surface_area.

        Args:
            kappa: Concentration (batch,)
            dim: Ambient dimension

        Returns:
            KL divergence (batch,)
        """
        # Clamp kappa for numerical stability
        kappa = kappa.clamp(min=NumericalConstants.VMF_KAPPA_MIN)

        # log(C_d(kappa))
        log_c_kappa = log_vmf_normalizer(kappa, dim)

        # log(C_d(0)) = log(1/surface_area) = log(uniform_density)
        log_c_0 = log_uniform_density(dim)

        # KL = kappa + log(C_d(kappa)) - log(C_d(0))
        # Note: This uses the fact that for vMF, the entropy is related to the normalizer
        kl = kappa + log_c_kappa - log_c_0

        # KL must be non-negative
        return kl.clamp(min=0.0)

    def _kl_to_vmf_prior(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        dim: int
    ) -> torch.Tensor:
        """
        KL divergence to a learnable vMF prior.

        KL(vMF(mu, kappa) || vMF(mu_0, kappa_0)) =
            log(C_d(kappa)) - log(C_d(kappa_0))
            + kappa_0 * (1 - mu^T mu_0)
            + (kappa - kappa_0) * E_q[mu^T z]

        For simplicity, we approximate E_q[mu^T z] ≈ 1 (tight for high kappa).

        Args:
            mu: Mean direction (batch, dim)
            kappa: Concentration (batch,)
            dim: Ambient dimension

        Returns:
            KL divergence (batch,)
        """
        # Initialize prior mu lazily
        if self._prior_mu is None or self._prior_mu.shape[-1] != dim:
            # Random unit vector as prior mean
            prior_mu = torch.randn(dim, device=mu.device, dtype=mu.dtype)
            self._prior_mu = safe_normalize(prior_mu.unsqueeze(0), dim=-1).squeeze(0)

        prior_mu = self._prior_mu.to(mu.device)
        prior_kappa = torch.tensor(
            self.prior_kappa, device=kappa.device, dtype=kappa.dtype
        )

        # Clamp kappas
        kappa = kappa.clamp(min=NumericalConstants.VMF_KAPPA_MIN)
        prior_kappa = prior_kappa.clamp(min=NumericalConstants.VMF_KAPPA_MIN)

        # Cosine similarity between mu and prior_mu
        mu_dot_prior = (mu * prior_mu.unsqueeze(0)).sum(dim=-1)

        # Log normalizers
        log_c_q = log_vmf_normalizer(kappa, dim)
        log_c_p = log_vmf_normalizer(prior_kappa.expand_as(kappa), dim)

        # KL computation (simplified, assuming E[mu^T z] ≈ 1)
        kl = (
            log_c_q - log_c_p
            + prior_kappa * (1.0 - mu_dot_prior)
            + (kappa - prior_kappa) * mu_dot_prior
        )

        return kl.clamp(min=0.0)

    def _monte_carlo_kl(
        self,
        params: dict[str, torch.Tensor],
        dim: int
    ) -> torch.Tensor:
        """
        Estimate KL divergence using Monte Carlo sampling.

        More stable in high dimensions than analytic computation.

        KL = E_q[log q(z) - log p(z)]

        Args:
            params: Distribution parameters
            dim: Ambient dimension

        Returns:
            KL estimate (batch,)
        """
        mu = params["mu"]
        kappa = params["kappa"].squeeze(-1)
        batch_size = mu.shape[0]

        # Sample from q(z|x) = vMF(mu, kappa)
        log_q_sum = torch.zeros(batch_size, device=mu.device, dtype=mu.dtype)
        log_p_sum = torch.zeros(batch_size, device=mu.device, dtype=mu.dtype)

        for _ in range(self.monte_carlo_samples):
            z = sample_vmf(mu, kappa)

            # log q(z; mu, kappa) = log(C_d(kappa)) + kappa * mu^T z
            kappa_clamped = kappa.clamp(min=NumericalConstants.VMF_KAPPA_MIN)
            log_c_kappa = log_vmf_normalizer(kappa_clamped, dim)
            mu_dot_z = (mu * z).sum(dim=-1)
            log_q = log_c_kappa + kappa_clamped * mu_dot_z

            # log p(z) depends on prior type
            if self.prior_type == "uniform":
                log_p = log_uniform_density(dim)
            else:
                # log p(z; mu_0, kappa_0)
                if self._prior_mu is None or self._prior_mu.shape[-1] != dim:
                    prior_mu = torch.randn(dim, device=mu.device, dtype=mu.dtype)
                    self._prior_mu = safe_normalize(prior_mu.unsqueeze(0), dim=-1).squeeze(0)

                prior_mu = self._prior_mu.to(mu.device)
                prior_kappa = torch.tensor(
                    self.prior_kappa, device=kappa.device, dtype=kappa.dtype
                ).clamp(min=NumericalConstants.VMF_KAPPA_MIN)

                log_c_prior = log_vmf_normalizer(prior_kappa.expand(batch_size), dim)
                prior_dot_z = (prior_mu.unsqueeze(0) * z).sum(dim=-1)
                log_p = log_c_prior + prior_kappa * prior_dot_z

            log_q_sum = log_q_sum + log_q
            log_p_sum = log_p_sum + log_p

        # Average over samples
        kl = (log_q_sum - log_p_sum) / self.monte_carlo_samples

        return kl.clamp(min=0.0)

    def scale_mean_params(
        self,
        params: dict[str, torch.Tensor],
        scale_factor: float
    ) -> dict[str, torch.Tensor]:
        """
        Scale distribution parameters for Scale-VAE.

        For vMF, we increase kappa to make the distribution more concentrated.
        This effectively "scales" the mean in the sense of making samples
        closer to the mean direction.

        Note: Unlike Gaussian, we cannot truly scale mu since it must remain
        on the unit sphere. Instead, we scale kappa.

        Args:
            params: Distribution parameters containing 'mu', 'kappa', 'log_kappa'
            scale_factor: Factor to scale concentration by

        Returns:
            New params dict with scaled kappa (mu unchanged)
        """
        device = params["kappa"].device
        dtype = params["kappa"].dtype

        new_kappa = params["kappa"] * scale_factor
        new_kappa = new_kappa.clamp(
            min=safe_exp(torch.tensor(self.log_kappa_min, device=device, dtype=dtype)),
            max=safe_exp(torch.tensor(self.log_kappa_max, device=device, dtype=dtype))
        )
        new_log_kappa = safe_log(new_kappa)

        return {
            "mu": params["mu"],
            "kappa": new_kappa,
            "log_kappa": new_log_kappa,
        }

    def log_prob(self, z: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute log probability of z under vMF(mu, kappa).

        log p(z; mu, kappa) = log(C_d(kappa)) + kappa * mu^T z

        Args:
            z: Points on sphere (batch, dim), must be unit normalized
            params: Distribution parameters

        Returns:
            Log probabilities (batch,)
        """
        mu = params["mu"]
        kappa = params["kappa"].squeeze(-1)
        dim = mu.shape[-1]

        kappa = kappa.clamp(min=NumericalConstants.VMF_KAPPA_MIN)
        log_c = log_vmf_normalizer(kappa, dim)
        mu_dot_z = (mu * z).sum(dim=-1)

        return log_c + kappa * mu_dot_z
