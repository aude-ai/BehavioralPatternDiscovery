"""
Composite Loss

Combines all loss components with configurable weights.
Manages the full VAE training objective.

Supports dynamic hierarchy levels - level names are derived from
encoder_params keys rather than being hardcoded.

Supports Mixture Prior KL computation when enabled.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from .reconstruction import ReconstructionLoss, FocalReconstructionLoss, loss_registry
from .kl import KLDivergenceLoss
from .disentanglement import IWOLoss
from .sparsity import HoyerLoss, AdaptiveHoyerLoss
from src.model.distributions import MixturePrior

logger = logging.getLogger(__name__)


@loss_registry.register("composite")
class CompositeLoss(nn.Module):
    """
    Composite VAE loss combining all components.

    Components:
        - Reconstruction: MSE or Focal
        - KL Divergence: Per-level with capacity control
        - TC/PC: From discriminators (passed as inputs)
        - IWO: Importance-weighted orthogonality
        - Hoyer: Sparsity regularization

    The total loss is a weighted sum:
        L = γ_recon * L_recon
            + Σ γ_kl_level * L_kl_level
            + γ_tc_intra * L_tc_intra
            + γ_pc_inter * L_pc_inter
            + γ_tc_unified * L_tc_unified
            + γ_iwo_intra * L_iwo_intra
            + γ_iwo_unified * L_iwo_unified
            + γ_kl_balance * L_kl_balance
            + γ_hoyer * L_hoyer
    """

    def __init__(
        self,
        config: dict[str, Any],
        mixture_priors: dict[str, MixturePrior] | None = None,
        level_names: list[str] | None = None,
    ):
        """
        Args:
            config: Composite loss configuration containing:
                - weights: Dict of loss weights (gamma values, with kl as list format)
                - reconstruction: Reconstruction loss config
                - kl: KL divergence config
                - iwo: IWO loss config
                - hoyer: Hoyer loss config
                - capacity: Capacity schedule config
            mixture_priors: Optional dict of MixturePrior objects with structure:
                {
                    "enc1": {"level_name": MP, ...},
                    "enc2": {"level_name": MP, ...},
                    ...
                    "unified": MP
                }
                Each encoder gets its OWN mixture priors to ensure cluster_probs
                match the correct mixture prior parameters during KL computation.
            level_names: Ordered list of encoder level names (e.g., ["bottom", "mid", "top"]).
                If None, defaults to ["bottom", "mid", "top"] for backward compatibility.
        """
        super().__init__()

        self.weights = config["weights"]

        # Parse reconstruction weights (new structure: recon.base, recon.flow, recon.focal)
        recon_weights = self.weights["recon"]
        self.recon_base_weight = recon_weights["base"]
        self.recon_flow_weight = recon_weights["flow"]
        self.recon_focal_weight = recon_weights["focal"]

        # Parse KL weights (single value for encoder levels, separate for unified)
        kl_config = self.weights["kl"]
        self.kl_weight_encoder_levels = kl_config["encoder_levels"]
        self.kl_weight_unified = kl_config["unified"]

        logger.info(f"CompositeLoss weights: {self.weights}")

        # Store mixture priors for KL computation
        self.mixture_priors = mixture_priors or {}

        # Store level names (defaults for backward compatibility)
        self.level_names = level_names or ["bottom", "mid", "top"]

        # KL losses (one per level + unified)
        kl_config = config["kl"]
        self.kl_losses = nn.ModuleDict()
        for level in self.level_names + ["unified"]:
            kl_loss_config = {
                "distribution_type": kl_config["distribution_type"],
                "use_capacity": kl_config["use_capacity"],
            }
            # Add distribution-specific prior params (passed by trainer based on distribution type)
            if kl_config["distribution_type"] == "gamma":
                kl_loss_config["prior_alpha"] = kl_config["prior_alpha"]
                kl_loss_config["prior_beta"] = kl_config["prior_beta"]
            elif kl_config["distribution_type"] == "vmf":
                kl_loss_config["prior_type"] = kl_config["prior_type"]
                kl_loss_config["prior_kappa"] = kl_config["prior_kappa"]
            self.kl_losses[level] = KLDivergenceLoss(kl_loss_config)

        # IWO loss
        iwo_config = config["iwo"]
        self.iwo_enabled = iwo_config["enabled"]
        if self.iwo_enabled:
            self.iwo_loss = IWOLoss({"eps": iwo_config["eps"]})
        else:
            self.iwo_loss = None

        # Hoyer loss
        hoyer_config = config["hoyer"]
        self.hoyer_enabled = hoyer_config["enabled"]
        self.hoyer_adaptive = hoyer_config["adaptive"]
        if self.hoyer_enabled:
            if self.hoyer_adaptive:
                self.hoyer_loss = AdaptiveHoyerLoss(hoyer_config)
            else:
                self.hoyer_loss = HoyerLoss(hoyer_config)
        else:
            self.hoyer_loss = None

    def forward(
        self,
        x_recon: torch.Tensor | None,
        x_target: torch.Tensor,
        encoder_params: dict[str, dict[str, torch.Tensor]],
        unified_params: dict[str, torch.Tensor],
        encoder_samples: dict[str, dict[str, torch.Tensor]],
        unified_sample: torch.Tensor,
        tc_losses: dict[str, torch.Tensor],
        pc_losses: dict[str, torch.Tensor],
        capacities: dict[str, float],
        encoder_betas: dict[str, float],
        unified_beta: float,
        skip_kl_loss: bool,
        encoder_cluster_probs: dict[str, dict[str, torch.Tensor]] | None = None,
        unified_cluster_probs: torch.Tensor | None = None,
        flow_loss: torch.Tensor | None = None,
        base_recon_loss: torch.Tensor | None = None,
        focal_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss components and weighted total.

        Args:
            x_recon: (batch, dim) reconstructed embeddings (unused, for compatibility)
            x_target: (batch, dim) target embeddings
            encoder_params: Per-encoder, per-level distribution params
                {encoder_name: {level: {"mu": ..., "logvar": ...}}}
            unified_params: Unified distribution params {"mu": ..., "logvar": ...}
            encoder_samples: Per-encoder, per-level samples
                {encoder_name: {level: z_tensor}}
            unified_sample: (batch, unified_dim) unified latent
            tc_losses: TC losses from discriminators
                {"tc_intra": tensor, "tc_unified": tensor}
            pc_losses: PC losses from discriminators
                {"pc_inter": tensor}
            capacities: Per-encoder-level capacity targets
                {"enc1_bottom": float, ..., "unified": float}
            encoder_betas: Per-encoder, per-level beta values from PI controllers
                {"enc1_bottom": float, "enc1_mid": float, ...}
            unified_beta: Beta value for unified KL
            skip_kl_loss: If True, set KL loss to 0 (warmup phase) but still compute raw_kl
            encoder_cluster_probs: Per-encoder, per-level cluster probs (for mixture prior)
                {encoder_name: {level: cluster_probs_tensor}}
            unified_cluster_probs: Unified cluster probs (for mixture prior)
            flow_loss: Velocity MSE loss from flow decoders
            base_recon_loss: Direct reconstruction loss (MLP MSE or diff ODE recon)
            focal_loss: Additional focal penalty (focal_weighted - uniform)

        Returns:
            Dictionary with all loss components and total
        """
        losses = {}
        device = x_target.device

        # Reconstruction losses (from decoder, with separate weights)
        # Flow loss (velocity MSE for flow decoders)
        if flow_loss is not None:
            losses["flow"] = flow_loss
        else:
            losses["flow"] = torch.tensor(0.0, device=device)

        # Base reconstruction loss (MLP MSE or Flow diff ODE recon)
        if base_recon_loss is not None:
            losses["base_recon"] = base_recon_loss
        else:
            losses["base_recon"] = torch.tensor(0.0, device=device)

        # Focal loss (additional penalty for hard samples)
        if focal_loss is not None:
            losses["focal"] = focal_loss
        else:
            losses["focal"] = torch.tensor(0.0, device=device)

        # KL losses per encoder per level (individual tracking)
        kl_total = torch.tensor(0.0, device=device)

        # Check if mixture prior is enabled (cluster probs available)
        # mixture_priors now has per-encoder structure: {"enc1": {"bottom": MP, ...}, "unified": MP}
        has_encoder_mixture_priors = any(
            k.startswith("enc") for k in self.mixture_priors.keys()
        )
        use_mixture_kl = (
            encoder_cluster_probs is not None and
            has_encoder_mixture_priors
        )

        # Track per-level totals for logging (sum across encoders)
        level_kl_totals = {level: 0.0 for level in self.level_names}
        level_raw_kl_totals = {level: 0.0 for level in self.level_names}

        # Track per-level dim stats (for non-mixture path only)
        level_dim_stats = {level: [] for level in self.level_names}

        for enc_name, enc_params in encoder_params.items():
            for level in self.level_names:
                if level not in enc_params:
                    continue

                # Get capacity from external capacities dict
                capacity_key = f"{enc_name}_{level}"
                capacity = capacities[capacity_key]

                # Compute raw KL for this encoder-level
                # Use per-encoder mixture prior to ensure cluster_probs match cluster parameters
                if (use_mixture_kl and
                    enc_name in self.mixture_priors and
                    level in self.mixture_priors[enc_name] and
                    enc_name in encoder_cluster_probs and
                    level in encoder_cluster_probs[enc_name]):
                    cluster_probs = encoder_cluster_probs[enc_name][level]
                    mixture_prior = self.mixture_priors[enc_name][level]  # Correct encoder's MP!
                    raw_kl = mixture_prior.kl_divergence_mixture(
                        enc_params[level], cluster_probs
                    ).mean()
                else:
                    _, raw_kl, _ = self.kl_losses[level](
                        enc_params[level], capacity, return_components=True
                    )

                # Always collect per-dimension stats for monitoring
                _, dim_stats = self.kl_losses[level].forward_with_dim_stats(
                    enc_params[level], capacity
                )
                level_dim_stats[level].append(dim_stats)

                # KL loss: |KL - C| or 0 during warmup
                if skip_kl_loss:
                    kl_loss = torch.tensor(0.0, device=device)
                else:
                    kl_loss = torch.abs(raw_kl - capacity)

                # Store individual values
                kl_key = f"{enc_name}_kl_{level}"
                losses[kl_key] = kl_loss
                losses[f"{enc_name}_raw_kl_{level}"] = raw_kl
                losses[f"{enc_name}_cap_{level}"] = torch.tensor(capacity, device=device)

                # Get beta for this encoder-level (from PI controller)
                beta_key = f"{enc_name}_{level}"
                beta = encoder_betas[beta_key]

                # Add to total with per-encoder-level beta and encoder-level weight
                kl_total = kl_total + self.kl_weight_encoder_levels * beta * kl_loss

                # Track level totals for summary logging
                level_kl_totals[level] += kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss
                level_raw_kl_totals[level] += raw_kl.item() if torch.is_tensor(raw_kl) else raw_kl

        # Store level totals for backward compatibility in logging
        for level in self.level_names:
            losses[f"kl_{level}"] = torch.tensor(level_kl_totals[level], device=device)
            losses[f"raw_kl_{level}"] = torch.tensor(level_raw_kl_totals[level], device=device)

        # Average dim stats across encoders and store
        for level in self.level_names:
            if level_dim_stats[level]:
                avg_mean = sum(
                    s["kl_dim_mean"] for s in level_dim_stats[level]
                ) / len(level_dim_stats[level])
                avg_std = sum(
                    s["kl_dim_std"] for s in level_dim_stats[level]
                ) / len(level_dim_stats[level])
                losses[f"kl_dim_mean_{level}"] = torch.tensor(avg_mean, device=device)
                losses[f"kl_dim_std_{level}"] = torch.tensor(avg_std, device=device)

        # Unified KL - get capacity from external dict
        unified_capacity = capacities["unified"]

        # Compute raw unified KL
        if (use_mixture_kl and
            "unified" in self.mixture_priors and
            unified_cluster_probs is not None):
            mixture_prior = self.mixture_priors["unified"]
            raw_kl_unified = mixture_prior.kl_divergence_mixture(
                unified_params, unified_cluster_probs
            ).mean()
        else:
            _, raw_kl_unified, _ = self.kl_losses["unified"](
                unified_params, unified_capacity, return_components=True
            )

        # Always collect unified dim stats for monitoring (regardless of mixture KL)
        _, uni_dim_stats = self.kl_losses["unified"].forward_with_dim_stats(
            unified_params, unified_capacity
        )
        losses["kl_dim_mean_unified"] = torch.tensor(
            uni_dim_stats["kl_dim_mean"], device=device
        )
        losses["kl_dim_std_unified"] = torch.tensor(
            uni_dim_stats["kl_dim_std"], device=device
        )

        # Unified KL loss: |KL - C| or 0 during warmup
        if skip_kl_loss:
            kl_loss_unified = torch.tensor(0.0, device=device)
        else:
            kl_loss_unified = torch.abs(raw_kl_unified - unified_capacity)

        losses["kl_unified"] = kl_loss_unified
        losses["raw_kl_unified"] = raw_kl_unified
        losses["cap_unified"] = torch.tensor(unified_capacity, device=device)
        kl_total = kl_total + self.kl_weight_unified * unified_beta * kl_loss_unified

        losses["kl_total"] = kl_total

        # KL balance loss (encourages similar KL across ENCODERS)
        # Sum KL across all levels for each encoder, then take variance
        encoder_kl_sums = []
        for enc_name in encoder_params.keys():
            enc_kl = torch.tensor(0.0, device=device)
            for level in self.level_names:
                kl_key = f"{enc_name}_kl_{level}"
                if kl_key in losses:
                    enc_kl = enc_kl + losses[kl_key]
            encoder_kl_sums.append(enc_kl)

        if len(encoder_kl_sums) > 1:
            kl_stack = torch.stack(encoder_kl_sums)
            losses["kl_balance"] = kl_stack.var()
        else:
            losses["kl_balance"] = torch.tensor(0.0, device=device)

        # TC/PC losses (from discriminators)
        losses["tc_intra"] = tc_losses.get(
            "tc_intra", torch.tensor(0.0, device=device)
        )
        losses["tc_unified"] = tc_losses.get(
            "tc_unified", torch.tensor(0.0, device=device)
        )
        losses["pc_inter"] = pc_losses.get(
            "pc_inter", torch.tensor(0.0, device=device)
        )

        # IWO losses
        if self.iwo_enabled and self.iwo_loss is not None:
            iwo_intra_total = torch.tensor(0.0, device=device)
            for enc_name, enc_samples in encoder_samples.items():
                for level, z in enc_samples.items():
                    iwo_intra_total = iwo_intra_total + self.iwo_loss(z)

            losses["iwo_intra"] = iwo_intra_total
            losses["iwo_unified"] = self.iwo_loss(unified_sample)
        else:
            losses["iwo_intra"] = torch.tensor(0.0, device=device)
            losses["iwo_unified"] = torch.tensor(0.0, device=device)

        # Hoyer sparsity
        if self.hoyer_enabled and self.hoyer_loss is not None:
            hoyer_total = torch.tensor(0.0, device=device)
            hoyer_weight = 1.0  # For non-adaptive mode
            for enc_name, enc_samples in encoder_samples.items():
                for level, z in enc_samples.items():
                    result = self.hoyer_loss(z)
                    if self.hoyer_adaptive:
                        loss, weight = result
                        hoyer_total = hoyer_total + loss
                        hoyer_weight = weight  # Track latest weight
                    else:
                        hoyer_total = hoyer_total + result
            result = self.hoyer_loss(unified_sample)
            if self.hoyer_adaptive:
                loss, weight = result
                hoyer_total = hoyer_total + loss
                hoyer_weight = weight
            else:
                hoyer_total = hoyer_total + result
            losses["hoyer"] = hoyer_total
            if self.hoyer_adaptive:
                losses["hoyer_weight"] = torch.tensor(hoyer_weight, device=device)
        else:
            losses["hoyer"] = torch.tensor(0.0, device=device)

        # Weighted total
        # Reconstruction contributions (new structure)
        flow_contrib = self.recon_flow_weight * losses["flow"]
        base_recon_contrib = self.recon_base_weight * losses["base_recon"]
        focal_contrib = self.recon_focal_weight * losses["focal"]

        # Other contributions
        tc_intra_contrib = self.weights["tc_intra"] * losses["tc_intra"]
        pc_inter_contrib = self.weights["pc_inter"] * losses["pc_inter"]
        tc_unified_contrib = self.weights["tc_unified"] * losses["tc_unified"]
        iwo_intra_contrib = self.weights["iwo_intra"] * losses["iwo_intra"]
        iwo_unified_contrib = self.weights["iwo_unified"] * losses["iwo_unified"]
        kl_balance_contrib = self.weights["kl_balance"] * losses["kl_balance"]
        hoyer_contrib = self.weights["hoyer"] * losses["hoyer"]

        total = (
            flow_contrib
            + base_recon_contrib
            + focal_contrib
            + kl_total
            + tc_intra_contrib
            + pc_inter_contrib
            + tc_unified_contrib
            + iwo_intra_contrib
            + iwo_unified_contrib
            + kl_balance_contrib
            + hoyer_contrib
        )

        losses["total"] = total

        return losses
