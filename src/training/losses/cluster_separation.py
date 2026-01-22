"""
Overlap-Aware Cluster Separation Loss

Penalizes cluster overlap based on distribution-specific geometry and spread.
Uses effective overlap computation that considers cluster concentration/variance.

For vMF: Uses cosine similarity weighted by concentration
For Gaussian: Uses distance normalized by standard deviations
For Gamma: Uses distance normalized by distribution spread
"""

import logging
import math

import torch
import torch.nn as nn

from src.core.numerical import NumericalConstants, safe_log, safe_exp, safe_div

logger = logging.getLogger(__name__)


class ClusterSeparationLoss(nn.Module):
    """
    Overlap-aware cluster separation loss.

    Computes effective overlap between cluster pairs based on their distribution
    parameters (centers and spread). Only penalizes when clusters have high
    effective overlap, allowing clusters to be close if they are tightly concentrated.

    Supports:
    - vMF: cosine similarity with kappa-weighted overlap
    - Gaussian: distance normalized by combined standard deviations
    - Gamma: distance normalized by distribution spread

    Formula:
        overlap = compute_pairwise_overlap(clusters)  # distribution-specific
        penalty = mean(exp(relu(overlap - threshold) / temperature) - 1)

    When overlap <= threshold: penalty = 0
    When overlap > threshold: exponentially increasing penalty
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Must contain:
                - enabled: Whether loss is active
                - overlap_threshold: Maximum acceptable overlap before penalty (0.0-1.0)
                - temperature: Penalty sharpness (lower = sharper)
                - max_element_penalty: (optional) Max penalty per cluster pair, default 100
        """
        super().__init__()
        self.enabled = config["enabled"]
        self.overlap_threshold = config["overlap_threshold"]
        self.temperature = config["temperature"]

        # Compute exp_clamp from max_element_penalty
        # exp(clamp) - 1 = max_penalty => clamp = log(max_penalty + 1)
        max_element_penalty = config.get("max_element_penalty", 100.0)
        self.exp_clamp = float(torch.log(torch.tensor(max_element_penalty + 1.0)).item())

        if self.enabled:
            logger.info(
                f"ClusterSeparationLoss enabled: overlap_threshold={self.overlap_threshold}, "
                f"temperature={self.temperature}, max_element_penalty={max_element_penalty}"
            )

    def forward(
        self,
        mixture_prior: nn.Module,
    ) -> torch.Tensor:
        """
        Compute cluster separation loss for a mixture prior.

        Args:
            mixture_prior: MixturePrior module containing cluster parameters

        Returns:
            Scalar loss value (0 if disabled or < 2 clusters)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=self._get_device(mixture_prior))

        dist_type = mixture_prior.dist_type
        n_clusters = mixture_prior.num_clusters

        if n_clusters < 2:
            return torch.tensor(0.0, device=self._get_device(mixture_prior))

        if dist_type == "vmf":
            overlap = self._compute_vmf_overlap(mixture_prior)
        elif dist_type == "gaussian":
            overlap = self._compute_gaussian_overlap(mixture_prior)
        elif dist_type == "gamma":
            overlap = self._compute_gamma_overlap(mixture_prior)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        return self._compute_penalty(overlap)

    def _get_device(self, mixture_prior: nn.Module) -> torch.device:
        """Get device from mixture prior parameters."""
        return next(mixture_prior.parameters()).device

    def _compute_vmf_overlap(self, mixture_prior: nn.Module) -> torch.Tensor:
        """
        Compute pairwise overlap for vMF clusters.

        Overlap is based on cosine similarity weighted by minimum kappa.
        Two clusters with high kappa can be angularly closer (they're tight).
        Two clusters with low kappa need more angular separation.

        Effective overlap formula:
            cos_sim = mu_i · mu_j  (in range [-1, 1])
            kappa_min = min(kappa_i, kappa_j)

            For high kappa, clusters are tight, so angular proximity is less dangerous.
            For low kappa, clusters are diffuse, so even moderate angular proximity = overlap.

            We use a sigmoid-like weighting:
                overlap = cos_sim * sigmoid(kappa_scale * (1 - 1/kappa_min))

            But simpler: when cos_sim is high AND kappa is low, overlap is high.
            When cos_sim is high but kappa is also high, clusters are distinct.

            Simplified formula:
                overlap = (1 + cos_sim) / 2 * exp(-kappa_min / kappa_reference)

            This gives overlap in [0, 1] where:
            - cos_sim = -1 (opposite): overlap approaches 0
            - cos_sim = 1 (same), low kappa: overlap approaches 1
            - cos_sim = 1 (same), high kappa: overlap reduced by concentration

        Returns:
            Pairwise overlap values for upper triangle (num_pairs,)
        """
        cluster_mu = mixture_prior.cluster_mu  # (K, D)
        cluster_kappa = safe_exp(mixture_prior.cluster_log_kappa.clamp(
            min=NumericalConstants.VMF_LOG_KAPPA_MIN,
            max=NumericalConstants.VMF_LOG_KAPPA_MAX
        ))  # (K,)

        n_clusters = cluster_mu.shape[0]

        # Pairwise cosine similarity (since mu are unit vectors, this is just dot product)
        # cos_sim: (K, K)
        cos_sim = torch.mm(cluster_mu, cluster_mu.t())

        # Pairwise minimum kappa
        # kappa_i, kappa_j -> min(kappa_i, kappa_j) for each pair
        kappa_expanded_i = cluster_kappa.unsqueeze(1)  # (K, 1)
        kappa_expanded_j = cluster_kappa.unsqueeze(0)  # (1, K)
        kappa_min = torch.minimum(kappa_expanded_i, kappa_expanded_j)  # (K, K)

        # Reference kappa for scaling (clusters with kappa > this are considered "tight")
        kappa_reference = 10.0

        # Overlap formula:
        # - Transform cos_sim from [-1, 1] to [0, 1]: (1 + cos_sim) / 2
        # - Weight by how diffuse clusters are: higher kappa = less overlap concern
        # - exp(-kappa_min / kappa_ref) gives ~1 for low kappa, ~0 for high kappa
        cos_sim_normalized = (1.0 + cos_sim) / 2.0
        diffuse_factor = safe_exp(-kappa_min / kappa_reference)
        overlap_matrix = cos_sim_normalized * diffuse_factor

        # Extract upper triangle (avoid double-counting and diagonal)
        triu_indices = torch.triu_indices(
            n_clusters, n_clusters, offset=1, device=cluster_mu.device
        )
        pairwise_overlap = overlap_matrix[triu_indices[0], triu_indices[1]]

        return pairwise_overlap

    def _compute_gaussian_overlap(self, mixture_prior: nn.Module) -> torch.Tensor:
        """
        Compute pairwise overlap for Gaussian clusters.

        Overlap is based on Euclidean distance normalized by combined standard deviations.
        Two clusters overlap significantly when their distance is small relative to their spread.

        Formula:
            distance = ||mu_i - mu_j||
            combined_std = sqrt(std_i^2 + std_j^2)  (average std per dimension)
            normalized_dist = distance / (combined_std + eps)
            overlap = exp(-normalized_dist^2 / 2)  # Gaussian-like decay

        This gives overlap in [0, 1] where:
        - Distance >> combined_std: overlap approaches 0
        - Distance << combined_std: overlap approaches 1

        Returns:
            Pairwise overlap values for upper triangle (num_pairs,)
        """
        cluster_mu = mixture_prior.cluster_mu  # (K, D)
        cluster_logvar = mixture_prior.cluster_logvar  # (K, D)

        n_clusters = cluster_mu.shape[0]
        latent_dim = cluster_mu.shape[1]

        # Compute per-cluster average std (average across dimensions)
        cluster_var = safe_exp(cluster_logvar.clamp(
            min=NumericalConstants.LOGVAR_CLAMP_MIN,
            max=NumericalConstants.LOGVAR_CLAMP_MAX
        ))
        cluster_std = torch.sqrt(cluster_var.mean(dim=1))  # (K,)

        # Pairwise Euclidean distances between cluster centers
        dists = torch.cdist(cluster_mu, cluster_mu, p=2)  # (K, K)

        # Combined std for each pair: sqrt(std_i^2 + std_j^2)
        std_sq_i = cluster_std.pow(2).unsqueeze(1)  # (K, 1)
        std_sq_j = cluster_std.pow(2).unsqueeze(0)  # (1, K)
        combined_std = torch.sqrt(std_sq_i + std_sq_j)  # (K, K)

        # Normalize distance by combined std
        # Scale by sqrt(latent_dim) to account for high-dimensional distance scaling
        normalized_dist = safe_div(
            dists,
            combined_std * math.sqrt(latent_dim),
        )

        # Gaussian-like overlap decay
        overlap_matrix = safe_exp(-0.5 * normalized_dist.pow(2))

        # Extract upper triangle
        triu_indices = torch.triu_indices(
            n_clusters, n_clusters, offset=1, device=cluster_mu.device
        )
        pairwise_overlap = overlap_matrix[triu_indices[0], triu_indices[1]]

        return pairwise_overlap

    def _compute_gamma_overlap(self, mixture_prior: nn.Module) -> torch.Tensor:
        """
        Compute pairwise overlap for Gamma clusters.

        For Gamma distribution, the spread is characterized by:
            mean = concentration / rate
            variance = concentration / rate^2
            std = sqrt(concentration) / rate

        Overlap is based on distance between means normalized by combined spread.

        Formula:
            mean_i = conc_i / rate_i (per dimension, then average)
            std_i = sqrt(conc_i) / rate_i (per dimension, then average)
            distance = |mean_i - mean_j|  (average across dimensions)
            combined_std = sqrt(std_i^2 + std_j^2)
            normalized_dist = distance / (combined_std + eps)
            overlap = exp(-normalized_dist^2 / 2)

        Returns:
            Pairwise overlap values for upper triangle (num_pairs,)
        """
        cluster_log_conc = mixture_prior.cluster_log_concentration  # (K, D)
        cluster_log_rate = mixture_prior.cluster_log_rate  # (K, D)

        n_clusters = cluster_log_conc.shape[0]

        # Get concentration and rate with safe clamping
        cluster_conc = safe_exp(cluster_log_conc.clamp(
            min=NumericalConstants.LOG_GAMMA_CONC_MIN,
            max=NumericalConstants.LOG_GAMMA_CONC_MAX
        ))
        cluster_rate = safe_exp(cluster_log_rate.clamp(
            min=NumericalConstants.LOG_GAMMA_RATE_MIN,
            max=NumericalConstants.LOG_GAMMA_RATE_MAX
        ))

        # Per-cluster mean and std (averaged across dimensions)
        cluster_mean = (cluster_conc / cluster_rate).mean(dim=1)  # (K,)
        cluster_std = (torch.sqrt(cluster_conc) / cluster_rate).mean(dim=1)  # (K,)

        # Pairwise distances between cluster means
        mean_i = cluster_mean.unsqueeze(1)  # (K, 1)
        mean_j = cluster_mean.unsqueeze(0)  # (1, K)
        dists = torch.abs(mean_i - mean_j)  # (K, K)

        # Combined std for each pair
        std_sq_i = cluster_std.pow(2).unsqueeze(1)  # (K, 1)
        std_sq_j = cluster_std.pow(2).unsqueeze(0)  # (1, K)
        combined_std = torch.sqrt(std_sq_i + std_sq_j)  # (K, K)

        # Normalize distance by combined std
        normalized_dist = safe_div(dists, combined_std)

        # Gaussian-like overlap decay
        overlap_matrix = safe_exp(-0.5 * normalized_dist.pow(2))

        # Extract upper triangle
        triu_indices = torch.triu_indices(
            n_clusters, n_clusters, offset=1, device=cluster_log_conc.device
        )
        pairwise_overlap = overlap_matrix[triu_indices[0], triu_indices[1]]

        return pairwise_overlap

    def _compute_penalty(self, overlap: torch.Tensor) -> torch.Tensor:
        """
        Compute exponential penalty for overlap values exceeding threshold.

        Args:
            overlap: Pairwise overlap values in [0, 1]

        Returns:
            Scalar penalty value
        """
        # Penalty only when overlap exceeds threshold
        violations = torch.relu(overlap - self.overlap_threshold)

        # Exponential penalty with overflow protection
        # Clamp scaled violations so max per-pair penalty = max_element_penalty
        scaled_violations = (violations / self.temperature).clamp(max=self.exp_clamp)
        penalty = torch.exp(scaled_violations) - 1.0

        return penalty.mean()

    def compute_stats(self, mixture_prior: nn.Module) -> dict[str, float]:
        """
        Compute cluster overlap statistics for logging.

        Args:
            mixture_prior: MixturePrior module containing cluster parameters

        Returns:
            Dict with max_overlap, mean_overlap, num_violations
        """
        if mixture_prior.num_clusters < 2:
            return {"max_overlap": 0.0, "mean_overlap": 0.0, "num_violations": 0}

        with torch.no_grad():
            dist_type = mixture_prior.dist_type

            if dist_type == "vmf":
                overlap = self._compute_vmf_overlap(mixture_prior)
            elif dist_type == "gaussian":
                overlap = self._compute_gaussian_overlap(mixture_prior)
            elif dist_type == "gamma":
                overlap = self._compute_gamma_overlap(mixture_prior)
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")

            return {
                "max_overlap": overlap.max().item(),
                "mean_overlap": overlap.mean().item(),
                "num_violations": (overlap > self.overlap_threshold).sum().item(),
            }
