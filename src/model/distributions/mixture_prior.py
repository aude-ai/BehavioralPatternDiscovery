"""
Mixture Prior Distribution

Wraps a base distribution with a mixture prior to prevent the "average trap"
where the VAE learns a single average pattern instead of diverse patterns.

Each cluster has its own prior parameters, and samples are soft-assigned to clusters.
The KL divergence becomes:
    KL = E_q(c|x)[KL(q(z|x) || p(z|c))] + KL(q(c|x) || p(c))

This encourages diverse patterns by having multiple "islands" in latent space.

Supports both Gaussian and Gamma distributions.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.numerical import NumericalConstants, safe_log, safe_normalize
from .base import LatentDistribution

logger = logging.getLogger(__name__)


class MixturePrior(nn.Module):
    """
    Mixture prior wrapper for latent distributions.

    Wraps a base distribution (Gaussian or Gamma) with K cluster centers.
    Each cluster has its own prior parameters.

    During training:
    - Computes q(c|x) = softmax of assignment logits
    - KL is computed against the weighted mixture of cluster priors

    This prevents mode collapse by encouraging diverse pattern clusters.
    """

    def __init__(
        self,
        base_distribution: LatentDistribution,
        config: dict[str, Any],
        latent_dim: int,
        hidden_dim: int,
    ):
        """
        Args:
            base_distribution: The underlying distribution (Gaussian or Gamma)
            config: Mixture prior config containing:
                - num_clusters: Number of clusters (or "match_latent_dim")
                - learnable_cluster_params: Whether cluster priors are learnable
                - learnable_mixture_weights: Whether mixture weights are learnable
            latent_dim: Dimension of the latent space
            hidden_dim: Dimension of the hidden state for cluster assignment
        """
        super().__init__()

        self.base_distribution = base_distribution
        self.latent_dim = latent_dim

        # Detect distribution type from param_names
        param_names = base_distribution.param_names
        if "mu" in param_names and "logvar" in param_names:
            self.dist_type = "gaussian"
        elif "concentration" in param_names and "rate" in param_names:
            self.dist_type = "gamma"
        elif "mu" in param_names and "kappa" in param_names:
            self.dist_type = "vmf"
        else:
            raise ValueError(
                f"MixturePrior requires Gaussian (mu, logvar), Gamma (concentration, rate), "
                f"or vMF (mu, kappa) distribution. Got param_names: {param_names}"
            )

        # Parse num_clusters
        num_clusters = config["num_clusters"]
        if num_clusters == "match_latent_dim":
            self.num_clusters = latent_dim
        else:
            self.num_clusters = int(num_clusters)

        self.learnable_cluster_params = config["learnable_cluster_params"]
        self.learnable_mixture_weights = config["learnable_mixture_weights"]

        # Config options for EPP parity
        # cluster_head_layers: 1 (EPP default) or 2 (original BPD)
        cluster_head_layers = config["cluster_head_layers"]
        # cluster_init_scale: 2.0 (EPP default) or 0.5 (original BPD)
        self.cluster_init_scale = config["cluster_init_scale"]

        # Cluster assignment head: h -> logits for q(c|x)
        if cluster_head_layers == 1:
            # EPP: single linear layer
            self.cluster_head = nn.Linear(hidden_dim, self.num_clusters)
        else:
            # BPD: 2-layer MLP
            self.cluster_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, self.num_clusters),
            )

        # Initialize cluster parameters based on distribution type
        self._init_cluster_params(latent_dim)

        # Mixture weights (prior p(c))
        if self.learnable_mixture_weights:
            self.mixture_logits = nn.Parameter(torch.zeros(self.num_clusters))
        else:
            self.register_buffer(
                "mixture_logits",
                torch.zeros(self.num_clusters)  # Uniform
            )

        logger.info(
            f"MixturePrior ({self.dist_type}): {self.num_clusters} clusters, "
            f"learnable_params={self.learnable_cluster_params}, "
            f"learnable_weights={self.learnable_mixture_weights}"
        )

    def _init_cluster_params(self, latent_dim: int):
        """Initialize cluster parameters based on distribution type."""
        if self.dist_type == "gaussian":
            self._init_gaussian_clusters(latent_dim)
        elif self.dist_type == "gamma":
            self._init_gamma_clusters(latent_dim)
        else:  # vmf
            self._init_vmf_clusters(latent_dim)

    def _init_gaussian_clusters(self, latent_dim: int):
        """Initialize Gaussian cluster parameters (mu, logvar)."""
        if self.learnable_cluster_params:
            self.cluster_mu = nn.Parameter(
                torch.randn(self.num_clusters, latent_dim) * self.cluster_init_scale
            )
            self.cluster_logvar = nn.Parameter(
                torch.zeros(self.num_clusters, latent_dim)
            )
        else:
            self.register_buffer(
                "cluster_mu",
                self._init_cluster_centers(self.num_clusters, latent_dim)
            )
            self.register_buffer(
                "cluster_logvar",
                torch.zeros(self.num_clusters, latent_dim)
            )

    def _init_gamma_clusters(self, latent_dim: int):
        """Initialize Gamma cluster parameters (log_concentration, log_rate).

        Uses linspace initialization like EPP2 for numerical stability:
        - log_concentration in [-1.0, 1.5] → concentration in [0.37, 4.48]
        - This range is safe for digamma/lgamma computations
        - Deterministic spread ensures all clusters start in valid range
        """
        # EPP2-style initialization: linspace for deterministic, safe spread
        # Each cluster gets a different log_concentration, broadcast to all dims
        init_log_conc = torch.linspace(-1.0, 1.5, self.num_clusters).unsqueeze(1).expand(-1, latent_dim).clone()
        init_log_rate = torch.zeros(self.num_clusters, latent_dim)

        if self.learnable_cluster_params:
            self.cluster_log_concentration = nn.Parameter(init_log_conc)
            self.cluster_log_rate = nn.Parameter(init_log_rate)
        else:
            self.register_buffer("cluster_log_concentration", init_log_conc)
            self.register_buffer("cluster_log_rate", init_log_rate)

    def _init_vmf_clusters(self, latent_dim: int):
        """Initialize vMF cluster parameters (mu on unit sphere, optional kappa).

        Cluster centers are random unit vectors spread on the sphere.
        Each cluster can optionally have its own concentration parameter.
        """
        # Initialize cluster means as random unit vectors
        cluster_mu = torch.randn(self.num_clusters, latent_dim)
        cluster_mu = safe_normalize(cluster_mu, dim=-1)

        if self.learnable_cluster_params:
            self.cluster_mu = nn.Parameter(cluster_mu)
            # Optional per-cluster kappa (log-space for stability)
            # Initialize to log(10) ≈ 2.3, a moderate concentration
            self.cluster_log_kappa = nn.Parameter(
                torch.full((self.num_clusters,), 2.3)
            )
        else:
            self.register_buffer("cluster_mu", cluster_mu)
            self.register_buffer(
                "cluster_log_kappa",
                torch.full((self.num_clusters,), 2.3)
            )

    def _init_cluster_centers(self, num_clusters: int, latent_dim: int) -> torch.Tensor:
        """Initialize cluster centers spread on unit sphere (for Gaussian)."""
        centers = torch.randn(num_clusters, latent_dim)
        centers = F.normalize(centers, dim=-1)
        return centers

    def compute_cluster_assignment(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cluster assignment probabilities q(c|x).

        Args:
            h: Hidden state from encoder (batch, hidden_dim)

        Returns:
            (batch, num_clusters) cluster assignment probabilities
        """
        logits = self.cluster_head(h)
        return F.softmax(logits, dim=-1)

    def kl_divergence_mixture(
        self,
        params: dict[str, torch.Tensor],
        cluster_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mixture KL divergence.

        KL = E_q(c|x)[KL(q(z|x) || p(z|c))] + KL(q(c|x) || p(c))

        Dispatches to appropriate method based on distribution type.

        Args:
            params: Distribution parameters (Gaussian: mu/logvar, Gamma: concentration/rate)
            cluster_probs: q(c|x) probabilities (B, K)

        Returns:
            Per-sample KL divergence (batch,)
        """
        if self.dist_type == "gaussian":
            return self._kl_divergence_gaussian(params, cluster_probs)
        elif self.dist_type == "gamma":
            return self._kl_divergence_gamma(params, cluster_probs)
        else:  # vmf
            return self._kl_divergence_vmf(params, cluster_probs)

    def _kl_divergence_gaussian(
        self,
        params: dict[str, torch.Tensor],
        cluster_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mixture KL for Gaussian distribution.

        KL(N(mu_x, sigma_x) || N(mu_c, sigma_c)) = 0.5 * sum_i [
            (mu_x - mu_c)^2 / sigma_c^2 +
            sigma_x^2 / sigma_c^2 -
            log(sigma_x^2 / sigma_c^2) - 1
        ]
        """
        mu_x = params["mu"]  # (B, D)
        logvar_x = params["logvar"]  # (B, D)

        # Cluster prior parameters
        mu_c = self.cluster_mu  # (K, D)
        logvar_c = self.cluster_logvar  # (K, D)

        # Expand for broadcasting: (B, 1, D) vs (1, K, D)
        mu_x_exp = mu_x.unsqueeze(1)
        logvar_x_exp = logvar_x.unsqueeze(1)
        mu_c_exp = mu_c.unsqueeze(0)
        logvar_c_exp = logvar_c.unsqueeze(0)

        # Clamp logvar before exp to prevent overflow
        var_x = torch.exp(logvar_x_exp.clamp(
            min=NumericalConstants.LOGVAR_CLAMP_MIN,
            max=NumericalConstants.LOGVAR_CLAMP_MAX
        ))
        var_c = torch.exp(logvar_c_exp.clamp(
            min=NumericalConstants.LOGVAR_CLAMP_MIN,
            max=NumericalConstants.LOGVAR_CLAMP_MAX
        ))

        # KL per cluster, sum over dimensions
        kl_per_cluster = 0.5 * (
            (mu_x_exp - mu_c_exp).pow(2) / var_c +
            var_x / var_c -
            logvar_x_exp + logvar_c_exp - 1
        ).sum(dim=-1)  # (B, K)

        return self._compute_weighted_kl(kl_per_cluster, cluster_probs)

    def _kl_divergence_gamma(
        self,
        params: dict[str, torch.Tensor],
        cluster_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mixture KL for Gamma distribution.

        KL(Gamma(α_x, β_x) || Gamma(α_c, β_c)) =
            (α_x - α_c) * ψ(α_x) - log Γ(α_x) + log Γ(α_c)
            + α_c * (log β_x - log β_c) + α_x * (β_c - β_x) / β_x

        Where ψ is the digamma function.

        Includes numerical stability improvements for small concentration values.
        """
        conc_x = params["concentration"]  # (B, D)
        rate_x = params["rate"]  # (B, D)

        # Cluster prior parameters (stored in log-space for stability)
        # Clamp log values before exp to prevent overflow
        conc_c = torch.exp(self.cluster_log_concentration.clamp(
            min=NumericalConstants.LOG_GAMMA_CONC_MIN,
            max=NumericalConstants.LOG_GAMMA_CONC_MAX
        ))  # (K, D)
        rate_c = torch.exp(self.cluster_log_rate.clamp(
            min=NumericalConstants.LOG_GAMMA_RATE_MIN,
            max=NumericalConstants.LOG_GAMMA_RATE_MAX
        ))  # (K, D)

        # Clamp to safe range for digamma/lgamma
        safe_conc_x = conc_x.clamp(
            min=NumericalConstants.GAMMA_CONC_MIN,
            max=NumericalConstants.GAMMA_CONC_MAX
        )
        safe_rate_x = rate_x.clamp(
            min=NumericalConstants.GAMMA_RATE_MIN,
            max=NumericalConstants.GAMMA_RATE_MAX
        )
        safe_conc_c = conc_c.clamp(
            min=NumericalConstants.GAMMA_CONC_MIN,
            max=NumericalConstants.GAMMA_CONC_MAX
        )
        safe_rate_c = rate_c.clamp(
            min=NumericalConstants.GAMMA_RATE_MIN,
            max=NumericalConstants.GAMMA_RATE_MAX
        )

        # Expand for broadcasting: (B, 1, D) vs (1, K, D)
        conc_x_exp = safe_conc_x.unsqueeze(1)
        rate_x_exp = safe_rate_x.unsqueeze(1)
        conc_c_exp = safe_conc_c.unsqueeze(0)
        rate_c_exp = safe_rate_c.unsqueeze(0)

        # KL divergence for Gamma distributions
        # Using PyTorch's lgamma and digamma functions with safe values
        digamma_term = torch.digamma(conc_x_exp)
        lgamma_x = torch.lgamma(conc_x_exp)
        lgamma_c = torch.lgamma(conc_c_exp)
        log_rate_x = torch.log(rate_x_exp)
        log_rate_c = torch.log(rate_c_exp)

        kl_per_dim = (
            (conc_x_exp - conc_c_exp) * digamma_term
            - lgamma_x + lgamma_c
            + conc_c_exp * (log_rate_x - log_rate_c)
            + conc_x_exp * (rate_c_exp - rate_x_exp) / rate_x_exp
        )

        # Clamp to non-negative (KL divergence is always >= 0)
        kl_per_dim = kl_per_dim.clamp(min=0.0)

        # Sum over dimensions to get per-cluster KL
        kl_per_cluster = kl_per_dim.sum(dim=-1)  # (B, K)

        return self._compute_weighted_kl(kl_per_cluster, cluster_probs)

    def _kl_divergence_vmf(
        self,
        params: dict[str, torch.Tensor],
        cluster_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mixture KL for von Mises-Fisher distribution.

        KL(vMF(mu_x, kappa_x) || vMF(mu_c, kappa_c)) is computed using
        the normalized cosine similarity and concentration parameters.

        Uses numerically stable Bessel function computations.
        """
        # Import here to avoid circular imports
        from .vmf_utils import log_vmf_normalizer

        mu_x = params["mu"]  # (B, D)
        kappa_x = params["kappa"].squeeze(-1)  # (B,)
        dim = mu_x.shape[-1]

        # Cluster prior parameters
        mu_c = self.cluster_mu  # (K, D)
        # Clamp log_kappa before exp to prevent overflow
        kappa_c = torch.exp(self.cluster_log_kappa.clamp(
            min=NumericalConstants.VMF_LOG_KAPPA_MIN,
            max=NumericalConstants.VMF_LOG_KAPPA_MAX
        ))  # (K,)

        # Clamp kappa_x to safe range
        kappa_x = kappa_x.clamp(
            min=NumericalConstants.VMF_KAPPA_MIN,
            max=NumericalConstants.VMF_KAPPA_MAX
        )

        # Ensure cluster_mu is normalized (safety check)
        mu_c = safe_normalize(mu_c, dim=-1)

        # Compute cosine similarity between each sample and each cluster center
        # mu_x: (B, D), mu_c: (K, D) -> similarities: (B, K)
        similarities = torch.mm(mu_x, mu_c.t())

        # For each sample-cluster pair, compute KL divergence
        # KL(vMF(mu_x, kappa_x) || vMF(mu_c, kappa_c))
        #
        # Full formula involves Bessel functions. For numerical stability,
        # we use the approximation valid for high kappa:
        #   KL ≈ kappa_x - kappa_c * cos(angle) + log(C(kappa_x)) - log(C(kappa_c))
        #
        # where cos(angle) = mu_x · mu_c

        # Expand for broadcasting: (B, 1) and (1, K)
        kappa_x_exp = kappa_x.unsqueeze(1)  # (B, 1)
        kappa_c_exp = kappa_c.unsqueeze(0)  # (1, K)

        # Log normalizers
        log_c_x = log_vmf_normalizer(kappa_x, dim).unsqueeze(1)  # (B, 1)
        log_c_c = log_vmf_normalizer(kappa_c, dim).unsqueeze(0)  # (1, K)

        # KL divergence per sample-cluster pair
        # KL = log(C_x) - log(C_c) + kappa_c * (1 - cos(angle)) + (kappa_x - kappa_c) * E[mu·z]
        # For simplicity, we approximate E[mu·z] ≈ 1 for high kappa
        kl_per_cluster = (
            log_c_x - log_c_c
            + kappa_c_exp * (1.0 - similarities)
            + (kappa_x_exp - kappa_c_exp) * similarities
        )  # (B, K)

        # KL must be non-negative
        kl_per_cluster = kl_per_cluster.clamp(min=0.0)

        return self._compute_weighted_kl(kl_per_cluster, cluster_probs)

    def normalize_vmf_clusters(self):
        """
        Re-normalize vMF cluster centers to unit sphere.

        Should be called after optimizer.step() when using vMF distribution
        to ensure cluster centers remain valid unit vectors.
        """
        if self.dist_type == "vmf":
            with torch.no_grad():
                self.cluster_mu.data = safe_normalize(self.cluster_mu.data, dim=-1)

    def _compute_weighted_kl(
        self,
        kl_per_cluster: torch.Tensor,
        cluster_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted KL and add categorical KL.

        Args:
            kl_per_cluster: KL to each cluster (B, K)
            cluster_probs: q(c|x) probabilities (B, K)

        Returns:
            Total KL divergence (B,)
        """
        # Weighted KL: E_q(c|x)[KL(q(z|x) || p(z|c))]
        weighted_kl = (cluster_probs * kl_per_cluster).sum(dim=-1)  # (B,)

        # Prior mixture weights: p(c)
        mixture_probs = F.softmax(self.mixture_logits, dim=0)  # (K,)

        # KL(q(c|x) || p(c)) = sum_c q(c|x) * log(q(c|x) / p(c))
        kl_categorical = (
            cluster_probs * (
                torch.log(cluster_probs + 1e-10) -
                torch.log(mixture_probs + 1e-10).unsqueeze(0)
            )
        ).sum(dim=-1)  # (B,)

        return weighted_kl + kl_categorical

    def get_assigned_cluster(
        self,
        cluster_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get hard cluster assignment (argmax).

        Args:
            cluster_probs: (batch, num_clusters) probabilities

        Returns:
            (batch,) cluster indices
        """
        return cluster_probs.argmax(dim=-1)


def create_mixture_prior_if_enabled(
    base_distribution: LatentDistribution,
    config: dict[str, Any],
    latent_dim: int,
    hidden_dim: int,
) -> MixturePrior | None:
    """
    Create MixturePrior if enabled in config.

    Args:
        base_distribution: Base distribution to wrap
        config: Mixture prior config with "enabled" flag
        latent_dim: Latent dimension
        hidden_dim: Hidden dimension for cluster head

    Returns:
        MixturePrior if enabled, else None
    """
    if not config["enabled"]:
        return None

    return MixturePrior(
        base_distribution=base_distribution,
        config=config,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
