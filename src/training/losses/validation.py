"""
Validation Loss Computer

Computes validation loss with configurable components and weights.
Allows validation to focus on different metrics than training.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ValidationLossComputer(nn.Module):
    """
    Computes validation loss with configurable weights and components.

    Unlike training's CompositeLoss which includes all components for gradient
    computation, ValidationLossComputer allows selecting which components matter
    for validation quality assessment.

    Supports dynamic hierarchy levels - level names are passed at construction
    time rather than being hardcoded.

    Supports two modes:
        - "mirror_training": Uses same weights as training (backward compatible)
        - "custom": Uses validation-specific weights from config
    """

    def __init__(
        self,
        config: dict[str, Any],
        training_weights: dict[str, float],
        level_names: list[str] | None = None,
    ):
        """
        Args:
            config: Validation loss configuration from training.yaml["validation_loss"]
            training_weights: Training loss weights from training.yaml["loss_weights"]
                Used when mode is "mirror_training". Contains "kl" as list format.
            level_names: Ordered list of encoder level names.
                If None, defaults to ["bottom", "mid", "top"] for backward compatibility.
        """
        super().__init__()

        self.mode = config["mode"]
        self.training_weights = training_weights

        # Dynamic level names (defaults for backward compatibility)
        self.level_names = level_names or ["bottom", "mid", "top"]

        # Parse KL weights (single value for encoder levels, separate for unified)
        kl_config = training_weights["kl"]
        self.kl_weight_encoder_levels = kl_config["encoder_levels"]
        self.kl_weight_unified = kl_config["unified"]

        if self.mode == "mirror_training":
            self._init_mirror_mode()
        elif self.mode == "custom":
            self._init_custom_mode(config["components"])
        else:
            raise ValueError(
                f"[ValidationLossComputer] Unknown mode: '{self.mode}'. "
                f"Available: ['mirror_training', 'custom']"
            )

        self.early_stopping_metric = config["early_stopping_metric"]

        logger.info(f"ValidationLossComputer initialized: mode={self.mode}")
        if self.mode == "custom":
            enabled = [k for k, v in self.component_config.items() if v["enabled"]]
            logger.info(f"  Enabled components: {enabled}")
            logger.info(f"  Early stopping metric: {self.early_stopping_metric}")

    def _init_mirror_mode(self):
        """Initialize to mirror training weights exactly."""
        # Build KL config with single encoder_levels weight and unified weight
        kl_config = {
            "enabled": True,
            "use_raw_kl": False,
            "weight_encoder_levels": self.kl_weight_encoder_levels,
            "weight_unified": self.kl_weight_unified,
        }

        # Get recon sub-weights (new structure)
        recon_weights = self.training_weights["recon"]

        self.component_config = {
            # New: separate recon components
            "flow": {"enabled": True, "weight": recon_weights["flow"]},
            "base_recon": {"enabled": True, "weight": recon_weights["base"]},
            "focal": {"enabled": True, "weight": recon_weights["focal"]},
            # Unchanged
            "kl": kl_config,
            "tc_intra": {"enabled": True, "weight": self.training_weights["tc_intra"]},
            "tc_unified": {"enabled": True, "weight": self.training_weights["tc_unified"]},
            "pc_inter": {"enabled": True, "weight": self.training_weights["pc_inter"]},
            "iwo_intra": {"enabled": True, "weight": self.training_weights["iwo_intra"]},
            "iwo_unified": {"enabled": True, "weight": self.training_weights["iwo_unified"]},
            "kl_balance": {"enabled": True, "weight": self.training_weights["kl_balance"]},
            "hoyer": {"enabled": True, "weight": self.training_weights["hoyer"]},
        }

    def _init_custom_mode(self, components_config: dict[str, Any]):
        """Initialize with custom validation weights."""
        self.component_config = components_config

    def compute(
        self,
        losses: dict[str, torch.Tensor],
        encoder_betas: dict[str, float],
        unified_beta: float,
        num_encoders: int,
    ) -> dict[str, torch.Tensor]:
        """
        Compute validation loss from raw loss components.

        Args:
            losses: Raw loss components from _forward_and_compute_loss()
                Contains: flow, base_recon, focal, raw_kl_*, kl_*, tc_*, pc_*, iwo_*, etc.
            encoder_betas: Per-encoder-level beta values (used for KL if mirroring training)
            unified_beta: Beta for unified KL
            num_encoders: Number of encoders in the model

        Returns:
            Dictionary with:
                - val_total: Total validation loss
                - val_<component>: Individual component values
                - val_early_stopping: Value used for early stopping comparison
        """
        # Get device from any available loss tensor
        device = next(iter(losses.values())).device
        val_losses = {}
        total = torch.tensor(0.0, device=device)

        # Flow loss (flow decoders only)
        if self.component_config["flow"]["enabled"]:
            flow = losses.get("flow", torch.tensor(0.0, device=device))
            weight = self.component_config["flow"]["weight"]
            val_losses["val_flow"] = flow
            total = total + weight * flow

        # Base reconstruction loss (MLP MSE or diff ODE recon)
        if self.component_config["base_recon"]["enabled"]:
            base_recon = losses.get("base_recon", torch.tensor(0.0, device=device))
            weight = self.component_config["base_recon"]["weight"]
            val_losses["val_base_recon"] = base_recon
            total = total + weight * base_recon

        # Focal loss (additional hard sample penalty)
        if self.component_config["focal"]["enabled"]:
            focal = losses.get("focal", torch.tensor(0.0, device=device))
            weight = self.component_config["focal"]["weight"]
            val_losses["val_focal"] = focal
            total = total + weight * focal

        # KL losses
        kl_config = self.component_config["kl"]
        if kl_config["enabled"]:
            kl_total = torch.tensor(0.0, device=device)
            use_raw = kl_config["use_raw_kl"]

            encoder_level_weight = kl_config["weight_encoder_levels"]
            for i in range(num_encoders):
                enc_name = f"enc{i + 1}"
                for level in self.level_names:
                    if use_raw:
                        kl_key = f"{enc_name}_raw_kl_{level}"
                    else:
                        kl_key = f"{enc_name}_kl_{level}"

                    if kl_key in losses:
                        kl_val = losses[kl_key]
                        if self.mode == "mirror_training":
                            beta_key = f"{enc_name}_{level}"
                            beta = encoder_betas[beta_key]
                            kl_total = kl_total + encoder_level_weight * beta * kl_val
                        else:
                            kl_total = kl_total + encoder_level_weight * kl_val

            # Unified KL
            unified_weight = kl_config["weight_unified"]
            if use_raw:
                unified_kl = losses.get("raw_kl_unified", torch.tensor(0.0, device=device))
            else:
                unified_kl = losses.get("kl_unified", torch.tensor(0.0, device=device))

            if self.mode == "mirror_training":
                kl_total = kl_total + unified_weight * unified_beta * unified_kl
            else:
                kl_total = kl_total + unified_weight * unified_kl

            val_losses["val_kl_total"] = kl_total
            total = total + kl_total

        # TC Intra
        if self.component_config["tc_intra"]["enabled"]:
            tc_intra = losses.get("tc_intra", torch.tensor(0.0, device=device))
            weight = self.component_config["tc_intra"]["weight"]
            val_losses["val_tc_intra"] = tc_intra
            total = total + weight * tc_intra

        # TC Unified
        if self.component_config["tc_unified"]["enabled"]:
            tc_unified = losses.get("tc_unified", torch.tensor(0.0, device=device))
            weight = self.component_config["tc_unified"]["weight"]
            val_losses["val_tc_unified"] = tc_unified
            total = total + weight * tc_unified

        # PC Inter
        if self.component_config["pc_inter"]["enabled"]:
            pc_inter = losses.get("pc_inter", torch.tensor(0.0, device=device))
            weight = self.component_config["pc_inter"]["weight"]
            val_losses["val_pc_inter"] = pc_inter
            total = total + weight * pc_inter

        # IWO Intra
        if self.component_config["iwo_intra"]["enabled"]:
            iwo_intra = losses.get("iwo_intra", torch.tensor(0.0, device=device))
            weight = self.component_config["iwo_intra"]["weight"]
            val_losses["val_iwo_intra"] = iwo_intra
            total = total + weight * iwo_intra

        # IWO Unified
        if self.component_config["iwo_unified"]["enabled"]:
            iwo_unified = losses.get("iwo_unified", torch.tensor(0.0, device=device))
            weight = self.component_config["iwo_unified"]["weight"]
            val_losses["val_iwo_unified"] = iwo_unified
            total = total + weight * iwo_unified

        # KL Balance
        if self.component_config["kl_balance"]["enabled"]:
            kl_balance = losses.get("kl_balance", torch.tensor(0.0, device=device))
            weight = self.component_config["kl_balance"]["weight"]
            val_losses["val_kl_balance"] = kl_balance
            total = total + weight * kl_balance

        # Hoyer
        if self.component_config["hoyer"]["enabled"]:
            hoyer = losses.get("hoyer", torch.tensor(0.0, device=device))
            weight = self.component_config["hoyer"]["weight"]
            val_losses["val_hoyer"] = hoyer
            total = total + weight * hoyer

        val_losses["val_total"] = total

        # Determine early stopping value
        if self.early_stopping_metric == "total":
            val_losses["val_early_stopping"] = total
        elif self.early_stopping_metric == "base_recon":
            val_losses["val_early_stopping"] = val_losses.get(
                "val_base_recon", total
            )
        elif self.early_stopping_metric == "kl_total":
            val_losses["val_early_stopping"] = val_losses.get(
                "val_kl_total", total
            )
        else:
            metric_key = f"val_{self.early_stopping_metric}"
            if metric_key in val_losses:
                val_losses["val_early_stopping"] = val_losses[metric_key]
            else:
                logger.warning(
                    f"Early stopping metric '{self.early_stopping_metric}' not found, using total"
                )
                val_losses["val_early_stopping"] = total

        return val_losses
