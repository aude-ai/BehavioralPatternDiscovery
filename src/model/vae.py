"""
Multi-Encoder Hierarchical VAE

Assembles all model components into a complete VAE:
- N parallel hierarchical encoders (configurable number of levels)
- Unification layer (concatenated final latents → unified latent)
- Flow matching or MLP decoder
- SHAP-compatible interface for interpretability

Supports dynamic hierarchy levels - the number and names of levels
are read from config rather than being hardcoded.

Uses ModelDimensions to compute derived values (input_dim, output_dim)
that are not stored in config because they must match other values.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.core.config import ModelDimensions
from src.model.base import BaseVAE
from src.model.encoders import encoder_registry
from src.model.unification import unification_registry
from src.model.decoders import decoder_registry
from src.model.distributions import distribution_registry

logger = logging.getLogger(__name__)


class MultiEncoderVAE(BaseVAE):
    """
    Multi-encoder hierarchical VAE with flow matching decoder.

    Implements the BaseVAE interface, allowing the Trainer to work
    with this model without knowing its specific implementation details.

    Architecture:
        Input → N Encoders (each produces configurable hierarchical levels)
              → Concatenate final level outputs → Unification
              → Decoder → Reconstruction

    The number of levels and their names/dimensions are configurable
    via config, not hardcoded. Supports any number of hierarchical levels.

    SHAP Compatibility:
        - Deterministic get_mean() methods for reproducible attributions
        - No in-place operations
        - Encoder wrappers for individual encoder analysis
    """

    def __init__(
        self,
        config: dict[str, Any],
        dims: ModelDimensions,
        training_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            config: Model configuration containing:
                - encoder: Encoder configuration (with its own layer_config)
                - unification: Unification layer configuration (with its own layer_config)
                - decoder: Decoder configuration (with its own layer_config)
                - distribution: Latent distribution configuration
            dims: Pre-computed ModelDimensions instance
            training_config: Training configuration (optional, for decoder training settings)
        """
        super().__init__()

        self.config = config
        self.training_config = training_config
        self.dims = dims

        encoder_config = config["encoder"]
        unification_config = config["unification"]
        decoder_config = config["decoder"]
        distribution_config = config["distribution"]

        self.num_encoders = dims.num_encoders
        self.input_dim = dims.encoder_input_dim
        self.output_dim = dims.decoder_output_dim  # Embedding dim for reconstruction

        # Create distribution for encoders
        distribution_type = distribution_config["type"]
        self.distribution = distribution_registry.create(
            distribution_type,
            config=distribution_config[distribution_type],
        )

        # Create separate distribution for unified layer with tighter bounds (like EPP)
        # This prevents gradient explosion from large logvar values
        dist_type_config = distribution_config[distribution_type]
        unified_dist_config = dist_type_config.copy()
        if distribution_type == "gaussian":
            # Use unified-specific bounds from config
            unified_dist_config["logvar_clamp_min"] = dist_type_config["unified_logvar_clamp_min"]
            unified_dist_config["logvar_clamp_max"] = dist_type_config["unified_logvar_clamp_max"]
        self.unified_distribution = distribution_registry.create(
            distribution_type,
            config=unified_dist_config,
        )

        # Mixture prior config (applies to all levels)
        mixture_prior_config = config["mixture_prior"]

        # Get per-section layer configs (each section has its own)
        encoder_layer_config = encoder_config["layer_config"]
        unification_layer_config = unification_config["layer_config"]

        # Create encoders (each with different seed for diversity)
        # Master seed can be null for random initialization (non-reproducible)
        self.encoders = nn.ModuleList()
        master_seed = encoder_config["seed"]

        for i in range(self.num_encoders):
            # Set seed for initialization diversity (derived from master seed)
            if master_seed is not None:
                torch.manual_seed(master_seed + i)

            encoder = encoder_registry.create(
                encoder_config["type"],
                config={
                    **encoder_config,
                    "layer_config": encoder_layer_config,
                    "mixture_prior": mixture_prior_config,
                },
                distribution=self.distribution,
                dims=dims,
            )
            self.encoders.append(encoder)

        # Set seed for unification layer (derived from master seed)
        if master_seed is not None:
            torch.manual_seed(master_seed + self.num_encoders)

        # Build unification config with computed input_dim
        unification_type = unification_config["type"]
        merged_unification_config = {
            **unification_config,
            unification_type: {
                **unification_config[unification_type],
                "input_dim": dims.unification_input_dim,
            },
            "layer_config": unification_layer_config,
        }

        self.unification = unification_registry.create(
            unification_type,
            config=merged_unification_config,
            distribution=self.unified_distribution,  # Use tighter bounds for unified
            mixture_prior_config=mixture_prior_config,
        )

        # Create decoder with computed dimensions
        decoder_type = decoder_config["type"]
        decoder_layer_config = decoder_config["layer_config"]

        # Build model_config for decoder (architecture only)
        decoder_model_config = {
            **decoder_config,
            decoder_type: {
                **decoder_config[decoder_type],
                "input_dim": dims.decoder_input_dim,
                "output_dim": dims.decoder_output_dim,
            },
            "layer_config": decoder_layer_config,
        }

        self.decoder = decoder_registry.create(
            decoder_type,
            model_config=decoder_model_config,
            training_config=training_config,
        )

        # Store dimensions for reference
        self.latent_dims = dims.latent_dims
        self.unified_dim = dims.unified_output_dim

        # Track mixture prior status
        self.mixture_prior_enabled = mixture_prior_config["enabled"]

        # Store decoder type for logging
        self.decoder_type = decoder_type

        self._gradient_checkpointing = False

        # Log model architecture
        self._log_architecture()

    def _log_architecture(self):
        """Log model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Format latent dims dynamically
        level_names = list(self.dims.level_names)
        latent_str = ", ".join(f"{name}={self.latent_dims[name]}" for name in level_names)

        logger.info("MODEL ARCHITECTURE")
        logger.info(f"  Input dim: {self.input_dim}")
        logger.info(f"  Num encoders: {self.num_encoders}")
        logger.info(f"  Hierarchy levels: {len(level_names)} ({' -> '.join(level_names)})")
        logger.info(f"  Latent dims: {latent_str}")
        logger.info(f"  Unified dim: {self.unified_dim}")
        logger.info(f"  Decoder: {self.decoder_type}")
        if self.decoder_type in ("flow_matching", "spherical_flow_matching"):
            logger.info(f"    ODE solver: {self.decoder.ode_solver.method}, steps={self.decoder.ode_num_steps}")
            logger.info(f"    Timestep sampling: {self.decoder.timestep_sampler.distribution}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

    def encode(
        self,
        x: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Encode input through all encoders and unification layer.

        Args:
            x: (batch, input_dim) input embeddings

        Returns:
            Dictionary containing:
                - enc1, enc2, ...: Per-encoder outputs with z and params per level
                - unified: Unified latent and params
        """
        encodings = {}

        # Encode through each encoder
        for i, encoder in enumerate(self.encoders):
            enc_output = encoder(x)
            encodings[f"enc{i + 1}"] = enc_output

        # Concatenate final level latents for unification
        final_level = self.dims.last_level
        final_latents = [
            encodings[f"enc{i + 1}"][final_level]["z"]
            for i in range(self.num_encoders)
        ]
        final_concat = torch.cat(final_latents, dim=1)

        # Unification layer
        unified_output = self.unification(final_concat)
        encodings["unified"] = unified_output

        return encodings

    def decode(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Decode from unified latent to reconstruction.

        Delegates all decoder-specific logic (flow matching, ODE solving, etc.)
        to the decoder itself via the BaseDecoder interface.

        Args:
            z: (batch, unified_dim) unified latent
            x_target: (batch, output_dim) target tensor
                      If provided, returns (reconstruction, loss)

        Returns:
            If x_target is None: (batch, output_dim) reconstruction only
            If x_target provided: ((batch, output_dim), loss) tuple
        """
        if x_target is not None:
            # Compute loss mode: use decoder's compute_training_loss method
            output = self.decoder.compute_training_loss(z, x_target)
            # Return approximate reconstruction and loss
            recon = output.reconstruction_approx if output.reconstruction_approx is not None else self.decoder.generate(z)
            return recon, output.loss
        else:
            # Generation mode: use decoder's generate method
            return self.decoder.generate(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Full forward pass: encode → decode.

        Args:
            x: (batch, input_dim) input embeddings

        Returns:
            Tuple of:
                - (batch, input_dim) reconstruction
                - encodings dictionary
        """
        encodings = self.encode(x)
        unified_z = encodings["unified"]["z"]

        if self.training:
            # Training mode: compute loss via decoder's compute_loss method
            # Flow matching reconstructs only embedding, not aux features
            x_embedding = x[:, :self.output_dim]
            x_recon, flow_loss = self.decode(unified_z, x_embedding)
            encodings["flow_loss"] = flow_loss
        else:
            # Inference mode: generate via decoder's generate method
            x_recon = self.decode(unified_z)

        return x_recon, encodings

    def get_latent_codes(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Get all latent codes for interpretation (SHAP-compatible).

        Args:
            x: (batch, input_dim) input embeddings
            deterministic: If True, use mean instead of sampling

        Returns:
            Dictionary of latent codes per encoder and unified
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            encodings = self.encode(x)

        if was_training:
            self.train()

        result = {}
        for i in range(self.num_encoders):
            enc_key = f"enc{i + 1}"
            enc_data = encodings[enc_key]
            result[enc_key] = {}
            for level_name in self.get_level_names():
                if deterministic:
                    result[enc_key][level_name] = self.distribution.get_mean(enc_data[level_name]["params"])
                else:
                    result[enc_key][level_name] = enc_data[level_name]["z"]

        result["unified"] = {
            "z": encodings["unified"]["z"] if not deterministic
                 else self.unified_distribution.get_mean(encodings["unified"]["params"]),
        }

        return result

    def get_level_names(self) -> list[str]:
        """Return ordered list of encoder level names."""
        return list(self.dims.level_names)

    def get_encoder_names(self) -> list[str]:
        """Return list of encoder names (enc1, enc2, etc.)."""
        return [f"enc{i + 1}" for i in range(self.num_encoders)]

    def get_encoder_view(self, encoder_index: int) -> "EncoderView":
        """
        Get a view of a single encoder for SHAP analysis.

        Args:
            encoder_index: Index of encoder (0, 1, or 2)

        Returns:
            EncoderView wrapping the specified encoder
        """
        return EncoderView(self, encoder_index)

    def get_all_encoder_views(self) -> list["EncoderView"]:
        """
        Get views for all encoders.

        Returns:
            List of EncoderView objects
        """
        return [self.get_encoder_view(i) for i in range(self.num_encoders)]

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True
        for encoder in self.encoders:
            if hasattr(encoder, "enable_gradient_checkpointing"):
                encoder.enable_gradient_checkpointing()
        if hasattr(self.decoder, "enable_gradient_checkpointing"):
            self.decoder.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        for encoder in self.encoders:
            if hasattr(encoder, "disable_gradient_checkpointing"):
                encoder.disable_gradient_checkpointing()
        if hasattr(self.decoder, "disable_gradient_checkpointing"):
            self.decoder.disable_gradient_checkpointing()


class EncoderView(nn.Module):
    """
    View of a single encoder from MultiEncoderVAE.

    Provides SHAP-compatible interface for analyzing individual encoders.
    """

    def __init__(self, parent: MultiEncoderVAE, encoder_index: int):
        """
        Args:
            parent: Parent MultiEncoderVAE
            encoder_index: Index of the encoder (0, 1, 2, etc.)
        """
        super().__init__()
        self.parent = parent
        self.encoder_index = encoder_index
        self.encoder_key = f"enc{encoder_index + 1}"

        # Expose encoder for direct access
        self.encoder = parent.encoders[encoder_index]
        self.distribution = parent.distribution

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Forward pass through single encoder.

        Args:
            x: (batch, input_dim) input embeddings

        Returns:
            Encoder output dictionary with z and params per level
        """
        return self.encoder(x)

    def get_latent_codes(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Get latent codes from this encoder (SHAP-compatible).

        Args:
            x: (batch, input_dim) input embeddings
            deterministic: If True, use mean instead of sampling

        Returns:
            Dictionary with latent codes per level (keyed by level name)
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            enc_output = self.encoder(x)

        if was_training:
            self.train()

        result = {}
        for level_name in self.parent.get_level_names():
            if deterministic:
                result[level_name] = self.distribution.get_mean(enc_output[level_name]["params"])
            else:
                result[level_name] = enc_output[level_name]["z"]

        return result
