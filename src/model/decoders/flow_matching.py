"""
Flow Matching Decoder

Learns a velocity field v(x_t, t, z) for ODE-based reconstruction:
    dx/dt = v(x, t, z)

Training:
1. Sample t ~ [0, 1]
2. Sample noise x_0 ~ N(0, I)
3. Interpolate: x_t = t * x_target + (1-t) * x_0
4. Predict velocity: v = network(x_t, t, z)
5. Loss: ||v - (x_target - x_0)||²

Inference:
1. Sample x_0 ~ N(0, I)
2. Solve ODE from t=0 to t=1
3. Return x_1 as reconstruction

This decoder is self-contained: it owns its ODE solver, timestep sampler,
and optimal transport coupling. The VAE simply calls compute_loss() during
training and generate() during inference.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import torch.nn.functional as F

from src.core.registry import ComponentRegistry
from src.model.layers import make_activation, get_swiglu, make_norm_layer
from .base import (
    BaseDecoder,
    DecoderTrainingOutput,
    DecoderValidationOutput,
    compute_euclidean_metrics,
)

logger = logging.getLogger(__name__)

# Registry for decoders
decoder_registry = ComponentRegistry[nn.Module]("decoder")


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding (Transformer-style positional encoding).

    Maps scalar timestep t to a high-dimensional embedding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size,) tensor of timesteps in [0, 1]

        Returns:
            (batch_size, dim) tensor of time embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual MLP block for the flow decoder."""

    def __init__(
        self,
        hidden_dim: int,
        activation_type: str,
        activation_config: dict[str, Any],
        norm_type: str,
        norm_config: dict[str, Any],
        dropout: float,
    ):
        super().__init__()

        use_swiglu = activation_type == "swiglu"

        # First sub-layer
        if norm_type != "none":
            self.norm1 = make_norm_layer(norm_type, hidden_dim, norm_config)
        else:
            self.norm1 = None

        if use_swiglu:
            self.fc1 = get_swiglu(hidden_dim, hidden_dim)
            self.activation1 = None
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.activation1 = make_activation(activation_type, activation_config)

        # Second sub-layer
        if norm_type != "none":
            self.norm2 = make_norm_layer(norm_type, hidden_dim, norm_config)
        else:
            self.norm2 = None

        if use_swiglu:
            self.fc2 = get_swiglu(hidden_dim, hidden_dim)
            self.activation2 = None
        else:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.activation2 = make_activation(activation_type, activation_config)

        self.dropout = nn.Dropout(dropout)
        self._gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # First sub-layer
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.fc1(x)
        if self.activation1 is not None:
            x = self.activation1(x)
        x = self.dropout(x)

        # Second sub-layer
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.fc2(x)
        if self.activation2 is not None:
            x = self.activation2(x)
        x = self.dropout(x)

        # Residual connection (no in-place for SHAP)
        return x + residual


@decoder_registry.register("flow_matching")
class FlowMatchingDecoder(BaseDecoder):
    """
    Conditional Flow Matching Decoder for Euclidean space.

    Learns velocity field v(x_t, t, z) for ODE-based generation.
    Training loss is velocity MSE with optional normalization.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any] | None = None,
    ):
        """
        Args:
            model_config: Architecture configuration (decoder section from model.yaml)
            training_config: Training behavior configuration (from training.yaml)
        """
        super().__init__()

        # =====================================================================
        # Architecture from model config
        # =====================================================================
        flow_config = model_config["flow_matching"]
        layer_config = model_config["layer_config"]

        self.input_dim = flow_config["input_dim"]
        self.output_dim = flow_config["output_dim"]
        self.hidden_dim = flow_config["hidden_dim"]
        num_blocks = flow_config["num_blocks"]
        time_embedding_dim = flow_config["time_embedding_dim"]
        latent_embedding_dim = flow_config["latent_embedding_dim"]
        dropout = flow_config["dropout"]

        from src.model.decoders.ode_solver import ODESolver, TimestepSampler

        self.ode_solver = ODESolver(method=flow_config["ode_solver"])
        self.ode_num_steps = flow_config["num_steps"]

        # Optimal transport config
        ot_config = flow_config["optimal_transport"]
        self.ot_enabled = ot_config["enabled"]
        self.ot_eps = ot_config["eps"]
        self.ot_n_iters = ot_config["n_iters"]

        # Noise scaling config
        noise_config = flow_config["noise_scaling"]
        self.noise_scaling_type = noise_config["type"]
        self.noise_fixed_std = noise_config["fixed_std"]

        # =====================================================================
        # Training behavior from training config
        # =====================================================================
        if training_config:
            train_cfg = training_config["decoder_training"]["flow_matching"]
            loss_cfg = train_cfg["loss"]

            # Training mode: "ideal" (linear interp) or "trajectory" (actual ODE path)
            self.training_mode = train_cfg.get("training_mode", "ideal")

            self.normalize_loss = loss_cfg["normalize"]
            self.direction_loss_ratio = loss_cfg["direction_loss_ratio"]
            self.track_baselines = loss_cfg["track_baselines"]

            # Timestep sampling (for ideal mode)
            self.timestep_sampler = TimestepSampler(
                distribution=train_cfg["timestep_sampling"],
                logit_normal_mean=train_cfg["logit_normal_mean"],
                logit_normal_std=train_cfg["logit_normal_std"],
                u_shaped_concentration=train_cfg["u_shaped_concentration"],
            )

            # Differentiable ODE training
            diff_cfg = train_cfg["differentiable_ode"]
            self.diff_ode_enabled = diff_cfg["enabled"]

            # Focal loss (applied to differentiable ODE reconstruction)
            self.focal_enabled = training_config["focal_loss"]["enabled"] and self.diff_ode_enabled
            self.focal_gamma = training_config["focal_loss"]["gamma"]
        else:
            # Defaults for inference-only mode
            self.training_mode = "ideal"
            self.normalize_loss = True
            self.direction_loss_ratio = 0.0
            self.track_baselines = False
            self.timestep_sampler = TimestepSampler(
                distribution="uniform",
                logit_normal_mean=0.0,
                logit_normal_std=1.0,
                u_shaped_concentration=0.5,
            )
            self.diff_ode_enabled = False
            self.focal_enabled = False
            self.focal_gamma = 2.0

        logger.info(f"Flow matching training mode: {self.training_mode} (direction_ratio={self.direction_loss_ratio}, mse_ratio={1 - self.direction_loss_ratio})")

        # =====================================================================
        # Build network layers
        # =====================================================================
        activation_type = layer_config["activation"]["type"]
        if activation_type in layer_config["activation"]:
            activation_config = layer_config["activation"][activation_type]
        else:
            activation_config = {}

        norm_type = layer_config["normalization"]["type"]
        norm_config = layer_config["normalization"]
        use_swiglu = activation_type == "swiglu"

        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        if use_swiglu:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embedding_dim, time_embedding_dim),
                get_swiglu(time_embedding_dim, time_embedding_dim),
                nn.Linear(time_embedding_dim, time_embedding_dim),
            )
        else:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embedding_dim, time_embedding_dim),
                make_activation(activation_type, activation_config),
                nn.Linear(time_embedding_dim, time_embedding_dim),
            )

        if use_swiglu:
            self.latent_mlp = nn.Sequential(
                nn.Linear(self.input_dim, latent_embedding_dim),
                get_swiglu(latent_embedding_dim, latent_embedding_dim),
                nn.Linear(latent_embedding_dim, latent_embedding_dim),
            )
        else:
            self.latent_mlp = nn.Sequential(
                nn.Linear(self.input_dim, latent_embedding_dim),
                make_activation(activation_type, activation_config),
                nn.Linear(latent_embedding_dim, latent_embedding_dim),
            )

        input_concat_dim = self.output_dim + time_embedding_dim + latent_embedding_dim
        self.input_proj = nn.Linear(input_concat_dim, self.hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dim=self.hidden_dim,
                activation_type=activation_type,
                activation_config=activation_config,
                norm_type=norm_type,
                norm_config=norm_config,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._gradient_checkpointing = False

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field v(x_t, t, z)."""
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        z_emb = self.latent_mlp(z)
        combined = torch.cat([x_t, t_emb, z_emb], dim=-1)
        h = self.input_proj(combined)

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        velocity = self.output_proj(h)
        return velocity

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def _sample_scaled_noise(self, x_target: torch.Tensor) -> torch.Tensor:
        """Sample noise with appropriate scaling."""
        x_0 = torch.randn_like(x_target)

        if self.noise_scaling_type == "fixed":
            return x_0 * self.noise_fixed_std
        elif self.noise_scaling_type == "data_driven":
            with torch.no_grad():
                data_std = x_target.std(dim=0, keepdim=True).clamp(min=1e-6)
            return x_0 * data_std
        elif self.noise_scaling_type == "data_driven_l2":
            with torch.no_grad():
                data_norms = torch.norm(x_target, dim=1)
                avg_norm = data_norms.mean().clamp(min=1e-6)
                noise_norms = torch.norm(x_0, dim=1, keepdim=True).clamp(min=1e-6)
            return x_0 * (avg_norm / noise_norms)
        else:
            raise ValueError(f"Unknown noise_scaling type: {self.noise_scaling_type}")

    # =========================================================================
    # Training Loss
    # =========================================================================

    def compute_trajectory_flow_loss(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute flow loss along the model's actual integration trajectory.

        Instead of computing loss at ideal linear interpolation points, this runs
        the actual ODE integration and computes velocity error at each step.

        Args:
            z: Latent codes (batch, latent_dim)
            x_target: Target embeddings (batch, output_dim)

        Returns:
            flow_loss: Combined MSE + direction loss along trajectory
            x_final: Final position after integration
            metrics: Dictionary of trajectory metrics
        """
        batch_size = z.shape[0]
        device = z.device

        # Start from scaled noise
        x_0 = self._sample_scaled_noise(x_target)

        # Optimal transport pairing
        if self.ot_enabled and batch_size > 1:
            from src.model.decoders.ode_solver import sinkhorn_coupling
            with torch.no_grad():
                indices = sinkhorn_coupling(x_0, x_target, self.ot_eps, self.ot_n_iters)
            x_0 = x_0[indices]

        x = x_0
        dt = 1.0 / self.ode_num_steps

        # Accumulators for trajectory losses
        total_mse_loss = 0.0
        total_direction_loss = 0.0
        total_v_pred_norm = 0.0
        total_v_target_norm = 0.0
        total_cos_sim = 0.0

        # Integration loop - compute loss at each step along actual trajectory
        for step in range(self.ode_num_steps):
            t_val = step * dt
            t = torch.full((batch_size,), t_val, device=device)

            # Predict velocity at current position
            v_pred = self.forward(x, t, z)

            # Target velocity: direction from current position to target
            # In Euclidean flow matching, this is (x_target - x) / (1 - t)
            # But at trajectory positions, we use constant velocity: x_target - x_0
            # Since we're following the actual trajectory, use direct vector to target
            remaining_time = 1.0 - t_val
            if remaining_time > 1e-6:
                v_target = (x_target - x) / remaining_time
            else:
                v_target = x_target - x

            # Compute norms for metrics
            v_pred_norm = torch.norm(v_pred, dim=-1, keepdim=True)
            v_target_norm = torch.norm(v_target, dim=-1, keepdim=True)

            # MSE at this step
            step_mse = ((v_pred - v_target) ** 2).mean()

            # Direction loss at this step
            cos_sim = F.cosine_similarity(v_pred, v_target, dim=-1)
            step_direction_loss = (1 - cos_sim).mean()

            # Accumulate
            total_mse_loss = total_mse_loss + step_mse
            total_direction_loss = total_direction_loss + step_direction_loss
            total_v_pred_norm = total_v_pred_norm + v_pred_norm.mean()
            total_v_target_norm = total_v_target_norm + v_target_norm.mean()
            total_cos_sim = total_cos_sim + cos_sim.mean()

            # Take step using predicted velocity
            x = x + v_pred * dt

        # Average over steps
        num_steps = self.ode_num_steps
        avg_mse_loss = total_mse_loss / num_steps
        avg_direction_loss = total_direction_loss / num_steps
        avg_v_target_norm = total_v_target_norm / num_steps

        # Zero baseline
        with torch.no_grad():
            zero_baseline = avg_v_target_norm ** 2
            improvement_vs_zero = 1.0 - (avg_mse_loss / (zero_baseline + 1e-8))

        # Combine using true ratio
        ratio = self.direction_loss_ratio
        flow_loss = (1 - ratio) * avg_mse_loss + ratio * avg_direction_loss

        # Compute metrics
        metrics = {
            "trajectory_mse": avg_mse_loss.detach(),
            "trajectory_direction_loss": avg_direction_loss.detach(),
            "trajectory_v_pred_norm": (total_v_pred_norm / num_steps).detach(),
            "trajectory_v_target_norm": avg_v_target_norm.detach(),
            "trajectory_cos_sim": (total_cos_sim / num_steps).detach(),
            "zero_baseline": zero_baseline.detach(),
            "improvement_vs_zero": improvement_vs_zero.detach(),
            "flow_loss_mse_component": ((1 - ratio) * avg_mse_loss).detach(),
            "flow_loss_direction_component": (ratio * avg_direction_loss).detach(),
        }

        return flow_loss, x, metrics

    def compute_ideal_flow_loss(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute flow loss at ideal linearly interpolated points.

        Traditional flow matching approach: sample random timesteps, interpolate
        between noise and target, and compute velocity error at those points.

        Args:
            z: Latent codes (batch, latent_dim)
            x_target: Target embeddings (batch, output_dim)

        Returns:
            flow_loss: Combined MSE + direction loss
            x_recon_approx: Approximate reconstruction for logging
            metrics: Dictionary of training metrics
        """
        batch_size = z.shape[0]
        device = z.device

        t = self.timestep_sampler.sample(batch_size, device)
        x_0 = self._sample_scaled_noise(x_target)

        # Optimal transport coupling
        if self.ot_enabled and batch_size > 1:
            from src.model.decoders.ode_solver import sinkhorn_coupling
            with torch.no_grad():
                indices = sinkhorn_coupling(x_0, x_target, self.ot_eps, self.ot_n_iters)
            x_0 = x_0[indices]

        # Linear interpolation
        t_expanded = t.unsqueeze(1)
        x_t = t_expanded * x_target + (1 - t_expanded) * x_0

        # Predict and compute loss
        v_pred = self.forward(x_t, t, z)
        v_target = x_target - x_0

        # Stability: clamp velocity magnitudes to prevent explosion
        v_pred_norm = torch.norm(v_pred, dim=-1, keepdim=True)
        v_target_norm = torch.norm(v_target, dim=-1, keepdim=True)
        max_velocity = 10.0
        if (v_pred_norm > max_velocity).any():
            v_pred = v_pred * (max_velocity / v_pred_norm.clamp(min=1e-8)).clamp(max=1.0)

        raw_mse = ((v_pred - v_target) ** 2).mean()
        raw_mse = raw_mse.clamp(max=100.0)

        # Compute metrics
        metrics: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            metrics["velocity_mse_raw"] = raw_mse
            metrics["velocity_mse_per_dim"] = ((v_pred - v_target) ** 2).mean()

            cos_sim = F.cosine_similarity(v_pred, v_target, dim=-1).mean()
            metrics["velocity_cosine_sim"] = cos_sim
            metrics["velocity_pred_norm"] = v_pred_norm.mean()
            metrics["velocity_target_norm"] = v_target_norm.mean()

            zero_baseline = (v_target ** 2).mean()
            metrics["zero_baseline"] = zero_baseline
            metrics["improvement_vs_zero"] = 1.0 - (raw_mse / (zero_baseline + 1e-8))

        # Compute FLOW LOSS with TRUE RATIO
        # 1. MSE component (optionally normalized)
        if self.normalize_loss:
            with torch.no_grad():
                baseline_detached = zero_baseline.detach()
            mse_loss = raw_mse / (baseline_detached + 1e-8)
            metrics["velocity_mse_normalized"] = mse_loss.detach()
        else:
            mse_loss = raw_mse

        # 2. Direction loss component
        cos_sim_train = F.cosine_similarity(v_pred, v_target, dim=-1)
        direction_loss = (1 - cos_sim_train).mean()
        metrics["direction_loss"] = direction_loss.detach()

        # 3. Combine using TRUE RATIO (not additive!)
        # direction_loss_ratio=0: pure MSE
        # direction_loss_ratio=0.5: 50% MSE, 50% direction
        # direction_loss_ratio=1: pure direction
        ratio = self.direction_loss_ratio
        flow_loss = (1 - ratio) * mse_loss + ratio * direction_loss
        metrics["flow_loss_mse_component"] = ((1 - ratio) * mse_loss).detach()
        metrics["flow_loss_direction_component"] = (ratio * direction_loss).detach()

        # Approximate reconstruction for logging
        with torch.no_grad():
            x_recon_approx = x_t + v_pred

        return flow_loss, x_recon_approx, metrics

    def compute_training_loss(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
        include_reconstruction_loss: bool = False,
    ) -> DecoderTrainingOutput:
        """
        Compute flow matching training loss.

        Mode selection:
        - "ideal": Train at linearly interpolated points (traditional flow matching)
        - "trajectory": Train along actual ODE integration path (learns correct magnitudes)

        Args:
            z: Latent codes (batch, latent_dim)
            x_target: Target embeddings (batch, output_dim)
            include_reconstruction_loss: If True, compute reconstruction loss.
        """
        if self.training_mode == "trajectory":
            # Compute flow loss along actual integration trajectory
            flow_loss, x_final, traj_metrics = self.compute_trajectory_flow_loss(z, x_target)

            # Map trajectory metrics to standard names
            metrics: dict[str, torch.Tensor] = {}
            metrics["velocity_mse_raw"] = traj_metrics["trajectory_mse"]
            metrics["velocity_cosine_sim"] = traj_metrics["trajectory_cos_sim"]
            metrics["velocity_pred_norm"] = traj_metrics["trajectory_v_pred_norm"]
            metrics["velocity_target_norm"] = traj_metrics["trajectory_v_target_norm"]
            metrics["direction_loss"] = traj_metrics["trajectory_direction_loss"]
            metrics["zero_baseline"] = traj_metrics["zero_baseline"]
            metrics["improvement_vs_zero"] = traj_metrics["improvement_vs_zero"]
            metrics["flow_loss_mse_component"] = traj_metrics["flow_loss_mse_component"]
            metrics["flow_loss_direction_component"] = traj_metrics["flow_loss_direction_component"]

            # Compute reconstruction metrics from trajectory endpoint
            with torch.no_grad():
                per_sample_loss = ((x_final - x_target) ** 2).sum(dim=-1)
                recon_mse = per_sample_loss.mean()
                metrics["reconstruction_loss"] = recon_mse
                metrics["reconstruction_mse"] = recon_mse

                # Centroid baseline: how much better is recon than predicting centroid
                x_centroid = x_target.mean(dim=0)
                target_centroid_mse = ((x_target - x_centroid) ** 2).sum(dim=-1).mean()
                metrics["target_centroid_mse"] = target_centroid_mse
                metrics["improvement_vs_centroid"] = target_centroid_mse - recon_mse

            x_recon_approx = x_final.detach()

        else:  # "ideal" mode
            # Compute flow loss at linearly interpolated points
            flow_loss, x_recon_approx, metrics = self.compute_ideal_flow_loss(z, x_target)
            x_final = None

        # Initialize separate loss components
        base_recon_loss = None
        focal_loss = None

        # Compute BASE RECON LOSS and FOCAL LOSS
        if include_reconstruction_loss:
            if self.training_mode == "trajectory" and x_final is not None:
                # Use trajectory endpoint for reconstruction loss
                per_sample_loss_grad = ((x_final - x_target) ** 2).sum(dim=-1)
            else:
                # Run differentiable ODE for ideal mode
                x_recon, per_sample_loss_grad = self.generate_differentiable(
                    z, x_target, return_per_sample=True
                )
                if per_sample_loss_grad is not None:
                    x_recon_approx = x_recon.detach()
                    recon_mse = per_sample_loss_grad.mean().detach()
                    metrics["reconstruction_mse"] = recon_mse

                    # Centroid baseline for ideal mode
                    with torch.no_grad():
                        x_centroid = x_target.mean(dim=0)
                        target_centroid_mse = ((x_target - x_centroid) ** 2).sum(dim=-1).mean()
                        metrics["target_centroid_mse"] = target_centroid_mse
                        metrics["improvement_vs_centroid"] = target_centroid_mse - recon_mse
                else:
                    per_sample_loss_grad = None

            if per_sample_loss_grad is not None:
                base_recon_loss = per_sample_loss_grad.mean()
                metrics["reconstruction_loss"] = base_recon_loss.detach()

                # Focal loss component
                if self.focal_enabled:
                    with torch.no_grad():
                        weights = (1 + per_sample_loss_grad) ** self.focal_gamma
                        weights = weights / weights.mean()
                    focal_weighted_loss = (weights * per_sample_loss_grad).mean()
                    focal_loss = focal_weighted_loss - base_recon_loss
                    metrics["focal_weights_mean"] = weights.mean().detach()
                    metrics["focal_weights_std"] = weights.std().detach()
                    metrics["focal_loss_raw"] = focal_loss.detach()

        # Return separate loss components
        return DecoderTrainingOutput(
            loss=flow_loss,
            flow_loss=flow_loss,
            base_recon_loss=base_recon_loss,
            focal_loss=focal_loss,
            metrics=metrics,
            reconstruction_approx=x_recon_approx,
        )

    # =========================================================================
    # Validation Metrics
    # =========================================================================

    def compute_validation_metrics(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
    ) -> DecoderValidationOutput:
        """Compute all validation metrics."""
        metrics: dict[str, torch.Tensor] = {}

        # Velocity metrics (always computed for parity with training)
        with torch.no_grad():
            training_output = self.compute_training_loss(z, x_target)
            for k, v in training_output.metrics.items():
                metrics[k] = v

        # Get reconstruction - reuse from training if available (trajectory mode runs full ODE)
        if self.training_mode == "trajectory":
            # In trajectory mode, compute_training_loss already ran full ODE
            x_recon = training_output.reconstruction_approx
        else:
            # In ideal mode, need to run ODE for actual reconstruction
            with torch.no_grad():
                x_recon = self.generate(z)

        # Reconstruction quality
        recon_metrics = self.compute_reconstruction_quality(x_recon, x_target)
        for k, v in recon_metrics.items():
            metrics[f"reconstruction_{k}"] = v

        # Baselines (always computed)
        with torch.no_grad():
            x_centroid = x_target.mean(dim=0)
            centroid_mse = ((x_recon - x_centroid) ** 2).sum(dim=-1).mean()
            target_centroid_mse = ((x_target - x_centroid) ** 2).sum(dim=-1).mean()
            metrics["centroid_mse"] = centroid_mse
            metrics["target_centroid_mse"] = target_centroid_mse

            # Improvement: how much better is recon than predicting centroid
            # Lower MSE is better, so improvement = centroid_mse - recon_mse
            recon_mse = metrics["reconstruction_mse_total"]
            metrics["improvement_vs_centroid"] = target_centroid_mse - recon_mse

        return DecoderValidationOutput(
            metrics=metrics,
            reconstruction=x_recon,
        )

    def compute_reconstruction_quality(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute Euclidean reconstruction quality metrics."""
        return compute_euclidean_metrics(x_recon, x_target)

    # =========================================================================
    # Generation
    # =========================================================================

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate reconstruction using ODE solver."""
        batch_size = z.shape[0]
        device = z.device

        x_0 = torch.randn(batch_size, self.output_dim, device=device)

        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.forward(x_t, t, z)

        x_1 = self.ode_solver.solve(velocity_fn, x_0, self.ode_num_steps)
        return x_1

    def generate_differentiable(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor | None = None,
        return_per_sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Generate with gradient support.

        Args:
            z: Latent codes
            x_target: Target embeddings for loss computation
            return_per_sample: If True, return per-sample loss instead of mean
        """
        try:
            from torchdiffeq import odeint_adjoint as odeint
        except ImportError:
            x_recon = self.generate(z)
            if x_target is not None:
                per_sample_loss = ((x_recon - x_target) ** 2).sum(dim=-1)
                if return_per_sample:
                    return x_recon, per_sample_loss
                return x_recon, per_sample_loss.mean()
            return x_recon, None

        batch_size = z.shape[0]
        device = z.device

        x_0 = torch.randn(batch_size, self.output_dim, device=device)

        def velocity_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            t_batch = torch.full((batch_size,), t.item(), device=device)
            return self.forward(x, t_batch, z)

        t_span = torch.linspace(0, 1, self.ode_num_steps + 1, device=device)
        trajectory = odeint(velocity_fn, x_0, t_span, method='midpoint')
        x_final = trajectory[-1]

        if x_target is not None:
            per_sample_loss = ((x_final - x_target) ** 2).sum(dim=-1)
            if return_per_sample:
                return x_final, per_sample_loss
            return x_final, per_sample_loss.mean()

        return x_final, None

    def get_training_loss_config(self) -> dict[str, Any]:
        return {
            "mode": self.training_mode,
            "normalize": self.normalize_loss,
            "direction_loss_ratio": self.direction_loss_ratio,
            "mse_ratio": 1 - self.direction_loss_ratio,
            "ode_num_steps": self.ode_num_steps,
        }

    def get_validation_metrics_names(self) -> list[str]:
        return [
            "reconstruction_mse_total",
            "reconstruction_mae",
            "velocity_mse_raw",
            "velocity_cosine_sim",
            "direction_loss",
            "improvement_vs_zero",
            "centroid_mse",
            "improvement_vs_centroid",
        ]

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True
        for block in self.blocks:
            block._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False
        for block in self.blocks:
            block._gradient_checkpointing = False
