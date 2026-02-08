"""
MLP Decoder

Simple MLP decoder as a baseline. Directly maps latent z to reconstruction.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.model.layers import make_layer_block
from .base import (
    BaseDecoder,
    DecoderTrainingOutput,
    DecoderValidationOutput,
    compute_euclidean_metrics,
    compute_spherical_metrics,
)
from .flow_matching import decoder_registry

logger = logging.getLogger(__name__)


@decoder_registry.register("mlp")
class MLPDecoder(BaseDecoder):
    """
    Simple MLP decoder.

    Directly reconstructs embeddings from latent representation.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            model_config: Architecture configuration
            training_config: Training behavior configuration
        """
        super().__init__()

        mlp_config = model_config["mlp"]
        layer_config = model_config["layer_config"]

        self.input_dim = mlp_config["input_dim"]
        self.output_dim = mlp_config["output_dim"]
        hidden_dims = mlp_config["hidden_dims"]
        dropout = mlp_config["dropout"]

        # Training config
        if training_config:
            train_cfg = training_config["decoder_training"]["mlp"]
            loss_cfg = train_cfg["loss"]

            self.normalize_loss = loss_cfg.get("normalize", True)
            self.is_spherical = loss_cfg.get("spherical_output", False)
            # Additive loss weights
            self.mse_weight = loss_cfg.get("mse_weight", 1.0)
            self.cosine_weight = loss_cfg.get("cosine_weight", 0.0)

            # Focal is controlled by global focal_loss config
            focal_config = training_config["focal_loss"]
            self.focal_enabled = focal_config["enabled"]
            self.focal_gamma = focal_config["gamma"]
        else:
            self.normalize_loss = True
            self.is_spherical = False
            self.mse_weight = 1.0
            self.cosine_weight = 0.0
            self.focal_enabled = False
            self.focal_gamma = 2.0

        # Build MLP
        layers = []
        current_dim = self.input_dim
        decoder_layer_config = {**layer_config, "dropout": dropout}

        for hidden_dim in hidden_dims:
            block = make_layer_block(
                block_type="linear",
                in_features=current_dim,
                out_features=hidden_dim,
                layer_config=decoder_layer_config,
            )
            layers.append(block)
            current_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_proj = nn.Linear(current_dim, self.output_dim)

        self._gradient_checkpointing = False

        activation_type = layer_config["activation"]["type"]
        self._init_weights(activation_type)

    def _init_weights(self, activation_type: str = "relu"):
        if activation_type in ("xielu", "elu", "leaky_relu"):
            nonlinearity = "leaky_relu"
            a = 1.0
        elif activation_type in ("swish", "gelu"):
            nonlinearity = "leaky_relu"
            a = 0.5
        else:
            nonlinearity = "relu"
            a = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if nonlinearity == "leaky_relu":
                    nn.init.kaiming_normal_(m.weight, a=a, mode="fan_in", nonlinearity=nonlinearity)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self._gradient_checkpointing and self.training:
            h = checkpoint(self.mlp, z, use_reentrant=False)
        else:
            h = self.mlp(z)

        output = self.output_proj(h)

        if self.is_spherical:
            output = F.normalize(output, dim=-1)

        return output

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    # =========================================================================
    # Training Loss
    # =========================================================================

    def compute_training_loss(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
        include_reconstruction_loss: bool = False,  # For API compatibility
    ) -> DecoderTrainingOutput:
        """
        Compute reconstruction loss with separate components.

        Returns:
        - base_recon_loss: Standard MSE reconstruction
        - focal_loss: Additional focal penalty (Option B: focal_weighted - uniform)
        """
        x_recon = self.forward(z)

        if self.is_spherical:
            x_target = F.normalize(x_target, dim=-1)

        metrics: dict[str, torch.Tensor] = {}

        # Base reconstruction loss (MSE)
        mse_per_dim = ((x_recon - x_target) ** 2).mean()
        mse_total = ((x_recon - x_target) ** 2).sum(dim=-1).mean()
        metrics["mse_raw"] = mse_total
        metrics["mse_per_dim"] = mse_per_dim

        if self.normalize_loss:
            base_recon_loss = mse_per_dim
            metrics["mse_normalized"] = base_recon_loss
        else:
            base_recon_loss = mse_total

        # Cosine similarity for metrics
        cos_sim = F.cosine_similarity(x_recon, x_target, dim=-1).mean()
        metrics["cosine_sim"] = cos_sim

        if self.cosine_weight > 0:
            cosine_loss = 1 - cos_sim
            metrics["cosine_loss"] = cosine_loss
            base_recon_loss = base_recon_loss + self.cosine_weight * cosine_loss

        # Focal loss component (Option B: additional penalty from hard samples)
        focal_loss = None
        if self.focal_enabled and self.focal_gamma > 0:
            with torch.no_grad():
                per_sample_error = ((x_recon - x_target) ** 2).sum(dim=-1)
                weights = (1 + per_sample_error) ** self.focal_gamma
                weights = weights / weights.mean()

            per_sample_loss = ((x_recon - x_target) ** 2).sum(dim=-1)
            focal_weighted_loss = (weights * per_sample_loss).mean()
            if self.normalize_loss:
                focal_weighted_loss = focal_weighted_loss / self.output_dim
                uniform_loss = mse_per_dim
            else:
                uniform_loss = mse_total

            # Focal loss = focal_weighted - uniform (the ADDITIONAL penalty)
            focal_loss = focal_weighted_loss - uniform_loss
            metrics["focal_loss_raw"] = focal_loss.detach()
            metrics["focal_weights_mean"] = weights.mean().detach()
            metrics["focal_weights_std"] = weights.std().detach()

        # MLP decoder has no flow loss
        return DecoderTrainingOutput(
            loss=base_recon_loss,
            flow_loss=None,
            base_recon_loss=base_recon_loss,
            focal_loss=focal_loss,
            metrics=metrics,
            reconstruction_approx=x_recon.detach(),
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def compute_validation_metrics(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
    ) -> DecoderValidationOutput:
        """Compute validation metrics."""
        with torch.no_grad():
            x_recon = self.generate(z)

            if self.is_spherical:
                x_target = F.normalize(x_target, dim=-1)

        recon_metrics = self.compute_reconstruction_quality(x_recon, x_target)
        metrics = {f"reconstruction_{k}": v for k, v in recon_metrics.items()}

        return DecoderValidationOutput(
            metrics=metrics,
            reconstruction=x_recon,
        )

    def compute_reconstruction_quality(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.is_spherical:
            return compute_spherical_metrics(x_recon, x_target)
        else:
            return compute_euclidean_metrics(x_recon, x_target)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)

    def get_training_loss_config(self) -> dict[str, Any]:
        return {
            "normalize": self.normalize_loss,
            "focal_enabled": self.focal_enabled,
            "focal_gamma": self.focal_gamma,
        }

    def get_validation_metrics_names(self) -> list[str]:
        if self.is_spherical:
            return ["reconstruction_cosine_sim", "reconstruction_geodesic_dist"]
        else:
            return ["reconstruction_mse_total", "reconstruction_mae"]

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False
