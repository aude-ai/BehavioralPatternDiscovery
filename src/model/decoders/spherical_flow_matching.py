"""
Spherical Flow Matching Decoder

Flow matching for unit-normalized embeddings using geodesic paths on the sphere.
Uses SLERP interpolation and optional optimal transport for noise-target pairing.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.core.registry import ComponentRegistry
from src.model.layers import make_activation, get_swiglu, make_norm_layer
from .base import (
    BaseDecoder,
    DecoderTrainingOutput,
    DecoderValidationOutput,
    compute_spherical_metrics,
)
from .flow_matching import (
    decoder_registry,
    SinusoidalTimeEmbedding,
    ResidualBlock,
)

logger = logging.getLogger(__name__)


def project_to_sphere(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project vectors to unit sphere via L2 normalization."""
    norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    return x / norms


def slerp(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP).

    Interpolates along the great circle connecting x_0 and x_1 on the unit sphere.
    """
    t = t.unsqueeze(1)

    cos_omega = (x_0 * x_1).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_omega)

    sin_omega = torch.sin(omega).clamp(min=eps)

    small_angle = omega.abs() < 1e-4
    if small_angle.any():
        lerp_result = (1 - t) * x_0 + t * x_1
        lerp_result = project_to_sphere(lerp_result)
    else:
        lerp_result = None

    coeff_0 = torch.sin((1 - t) * omega) / sin_omega
    coeff_1 = torch.sin(t * omega) / sin_omega
    slerp_result = coeff_0 * x_0 + coeff_1 * x_1

    if lerp_result is not None:
        result = torch.where(small_angle, lerp_result, slerp_result)
    else:
        result = slerp_result

    return project_to_sphere(result)


def geodesic_velocity(x_t: torch.Tensor, x_1: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the geodesic velocity at x_t pointing toward x_1.

    This is the logarithmic map: log_{x_t}(x_1), giving the tangent vector
    at x_t pointing along the great circle toward x_1.
    """
    cos_angle = (x_t * x_1).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cos_angle)

    tangent = x_1 - cos_angle * x_t
    tangent_norm = tangent.norm(dim=-1, keepdim=True).clamp(min=eps)

    small_angle = omega.abs() < 1e-4
    large_angle_v = omega * tangent / tangent_norm
    small_angle_v = x_1 - x_t

    return torch.where(small_angle, small_angle_v, large_angle_v)


def exponential_map(x: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the exponential map on the unit sphere.

    Given a point x on the sphere and a tangent vector v at x,
    returns the point reached by traveling along the geodesic
    in direction v for distance |v|.

    Formula: exp_x(v) = cos(|v|) * x + sin(|v|) * (v / |v|)

    Args:
        x: Points on unit sphere (batch, dim)
        v: Tangent vectors at x (batch, dim) - should be orthogonal to x
        eps: Small value for numerical stability

    Returns:
        Points on unit sphere after geodesic travel (batch, dim)
    """
    v_norm = torch.norm(v, dim=-1, keepdim=True)

    # Handle zero velocity case (no movement)
    is_zero = v_norm < eps

    # Compute unit direction in tangent space
    v_dir = v / v_norm.clamp(min=eps)

    # Exponential map: cos(|v|) * x + sin(|v|) * v_dir
    cos_norm = torch.cos(v_norm)
    sin_norm = torch.sin(v_norm)

    result = cos_norm * x + sin_norm * v_dir

    # For zero velocity, return original point
    result = torch.where(is_zero, x, result)

    # Ensure output is on sphere (numerical safety)
    return project_to_sphere(result)


@decoder_registry.register("spherical_flow_matching")
class SphericalFlowMatchingDecoder(BaseDecoder):
    """
    Flow Matching Decoder for unit-normalized embeddings.

    Uses geodesic paths (SLERP) on the unit sphere.
    Training loss is velocity MSE with optional normalization and direction component.
    Validation includes actual ODE reconstruction quality.
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
        flow_config = model_config["spherical_flow_matching"]
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

        # =====================================================================
        # Training behavior from training config
        # =====================================================================
        if training_config:
            train_cfg = training_config["decoder_training"]["spherical_flow_matching"]
            loss_cfg = train_cfg["loss"]

            # Training mode: "ideal" (SLERP points) or "trajectory" (actual ODE path)
            self.training_mode = train_cfg.get("training_mode", "ideal")

            self.normalize_loss = loss_cfg["normalize"]
            self.direction_loss_ratio = loss_cfg["direction_loss_ratio"]
            self.track_baselines = loss_cfg["track_baselines"]
            self.weight_endpoints = loss_cfg["weight_endpoints"]
            self.weight_clamp = loss_cfg["weight_clamp"]

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
            self.weight_endpoints = True
            self.weight_clamp = 10.0
            self.timestep_sampler = TimestepSampler(
                distribution="logit_normal",
                logit_normal_mean=0.0,
                logit_normal_std=1.0,
            )
            self.diff_ode_enabled = False
            self.focal_enabled = False
            self.focal_gamma = 2.0

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

        if self.ot_enabled:
            logger.info(f"Spherical flow matching using optimal transport (eps={self.ot_eps})")
        logger.info(f"Training mode: {self.training_mode} (direction_ratio={self.direction_loss_ratio}, mse_ratio={1 - self.direction_loss_ratio})")
        logger.info("Using exponential map for geodesic ODE integration")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict tangent velocity field on the sphere."""
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

        Instead of computing loss at ideal SLERP points, this runs the actual
        ODE integration and computes velocity error at each step. This ensures
        the model learns to predict correct velocities at positions it actually
        visits during generation.

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

        # Start from random noise on sphere
        x_0 = torch.randn(batch_size, self.output_dim, device=device)
        x_0 = project_to_sphere(x_0)
        x_target = project_to_sphere(x_target)

        # Optimal transport pairing for noise-target alignment
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

            # Predict velocity at current (actual) position
            v_pred = self.forward(x, t, z)

            # Target velocity: geodesic from current position to target
            v_target = geodesic_velocity(x, x_target)

            # Compute norms for metrics
            v_pred_norm = torch.norm(v_pred, dim=-1, keepdim=True)
            v_target_norm = torch.norm(v_target, dim=-1, keepdim=True)

            # MSE at this step (unweighted - all steps equally important)
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

            # Take step using predicted velocity (this is where we actually go)
            x = exponential_map(x, v_pred * dt)

        # Average over steps
        num_steps = self.ode_num_steps
        avg_mse_loss = total_mse_loss / num_steps
        avg_direction_loss = total_direction_loss / num_steps
        avg_v_target_norm = total_v_target_norm / num_steps

        # Zero baseline: MSE if we predicted zero velocity
        # This is avg(|v_target|²) which equals avg_v_target_norm² approximately
        with torch.no_grad():
            # Zero baseline is computed as if we predicted v=0 at each step
            zero_baseline = avg_v_target_norm ** 2
            improvement_vs_zero = 1.0 - (avg_mse_loss / (zero_baseline + 1e-8))

        # Combine using ratio
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

        x_final = project_to_sphere(x)
        return flow_loss, x_final, metrics

    def compute_ideal_flow_loss(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute flow loss at ideal SLERP-interpolated points.

        Traditional flow matching approach: sample random timesteps, interpolate
        between noise and target using SLERP, and compute velocity error at those points.

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

        x_target = project_to_sphere(x_target)
        t = self.timestep_sampler.sample(batch_size, device)

        # Sample noise on sphere
        x_0 = torch.randn_like(x_target)
        x_0 = project_to_sphere(x_0)

        # Optimal transport pairing
        if self.ot_enabled and batch_size > 1:
            from src.model.decoders.ode_solver import sinkhorn_coupling
            with torch.no_grad():
                indices = sinkhorn_coupling(x_0, x_target, self.ot_eps, self.ot_n_iters)
            x_0 = x_0[indices]

        # Interpolate on sphere using SLERP
        x_t = slerp(x_0, x_target, t)

        # Predict velocity
        v_pred = self.forward(x_t, t, z)

        # Target velocity (geodesic direction from x_t to x_target)
        v_target = geodesic_velocity(x_t, x_target)

        # Stability: clamp velocity magnitudes to prevent explosion
        v_pred_norm = torch.norm(v_pred, dim=-1, keepdim=True)
        v_target_norm = torch.norm(v_target, dim=-1, keepdim=True)
        max_velocity = 10.0
        if (v_pred_norm > max_velocity).any():
            v_pred = v_pred * (max_velocity / v_pred_norm.clamp(min=1e-8)).clamp(max=1.0)

        # Compute weighted MSE
        if self.weight_endpoints:
            t_expanded = t.unsqueeze(1)
            weight = 1.0 / (1e-4 + t_expanded * (1 - t_expanded))
            weight = weight.clamp(max=self.weight_clamp)
        else:
            weight = torch.ones_like(t).unsqueeze(1)

        raw_mse = (weight * (v_pred - v_target) ** 2).mean()
        raw_mse = raw_mse.clamp(max=100.0)

        # Compute metrics
        metrics: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            mse_per_dim = ((v_pred - v_target) ** 2).mean()
            metrics["velocity_mse_raw"] = raw_mse
            metrics["velocity_mse_per_dim"] = mse_per_dim

            cos_sim = F.cosine_similarity(v_pred, v_target, dim=-1).mean()
            metrics["velocity_cosine_sim"] = cos_sim
            metrics["velocity_pred_norm"] = v_pred_norm.mean()
            metrics["velocity_target_norm"] = v_target_norm.mean()

            zero_baseline = (weight * v_target ** 2).mean()
            metrics["zero_baseline"] = zero_baseline
            metrics["improvement_vs_zero"] = 1.0 - (raw_mse / (zero_baseline + 1e-8))

        # Compute FLOW LOSS with true ratio
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

        # 3. Combine using TRUE RATIO
        ratio = self.direction_loss_ratio
        flow_loss = (1 - ratio) * mse_loss + ratio * direction_loss
        metrics["flow_loss_mse_component"] = ((1 - ratio) * mse_loss).detach()
        metrics["flow_loss_direction_component"] = (ratio * direction_loss).detach()

        # Approximate reconstruction for logging
        with torch.no_grad():
            x_recon_approx = project_to_sphere(x_t + v_pred * 0.1)

        return flow_loss, x_recon_approx, metrics

    def compute_training_loss(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor,
        include_reconstruction_loss: bool = False,
    ) -> DecoderTrainingOutput:
        """
        Compute spherical flow matching training loss.

        Mode selection:
        - "ideal": Train at SLERP-interpolated points (traditional flow matching)
        - "trajectory": Train along actual ODE integration path (learns correct magnitudes)

        Args:
            z: Latent codes (batch, latent_dim)
            x_target: Target embeddings (batch, output_dim)
            include_reconstruction_loss: If True, compute reconstruction loss.
        """
        x_target = project_to_sphere(x_target)

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
                per_sample_loss = 1 - F.cosine_similarity(x_final, x_target, dim=-1)
                base_recon = per_sample_loss.mean()
                metrics["reconstruction_loss"] = base_recon
                metrics["reconstruction_cosine_sim"] = 1 - base_recon

                # Centroid baseline: how much better is recon→target vs centroid→target?
                x_centroid = x_target.mean(dim=0)
                x_centroid_norm = F.normalize(x_centroid, dim=0)
                target_centroid_cos_sim = F.cosine_similarity(
                    x_target,
                    x_centroid_norm.unsqueeze(0).expand_as(x_target),
                    dim=-1
                ).mean()
                metrics["target_centroid_cos_sim"] = target_centroid_cos_sim
                metrics["improvement_vs_centroid"] = (1 - base_recon) - target_centroid_cos_sim

            x_recon_approx = x_final.detach()

        else:  # "ideal" mode
            # Compute flow loss at SLERP-interpolated points
            flow_loss, x_recon_approx, metrics = self.compute_ideal_flow_loss(z, x_target)
            x_final = None  # Not computed in ideal mode

        # Initialize separate loss components
        base_recon_loss = None
        focal_loss = None

        # Compute BASE RECON LOSS and FOCAL LOSS
        if include_reconstruction_loss:
            if self.training_mode == "trajectory" and x_final is not None:
                # Use trajectory endpoint for reconstruction loss
                per_sample_loss_grad = 1 - F.cosine_similarity(x_final, x_target, dim=-1)
            else:
                # Run differentiable ODE for ideal mode
                x_recon, per_sample_loss_grad = self.generate_differentiable(
                    z, x_target, return_per_sample=True
                )
                if per_sample_loss_grad is not None:
                    x_recon_approx = x_recon.detach()
                    recon_cos_sim = (1 - per_sample_loss_grad.mean()).detach()
                    metrics["reconstruction_cosine_sim"] = recon_cos_sim

                    # Centroid baseline for ideal mode
                    with torch.no_grad():
                        x_centroid = x_target.mean(dim=0)
                        x_centroid_norm = F.normalize(x_centroid, dim=0)
                        target_centroid_cos_sim = F.cosine_similarity(
                            x_target,
                            x_centroid_norm.unsqueeze(0).expand_as(x_target),
                            dim=-1
                        ).mean()
                        metrics["target_centroid_cos_sim"] = target_centroid_cos_sim
                        metrics["improvement_vs_centroid"] = recon_cos_sim - target_centroid_cos_sim
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
        """
        Compute all validation metrics including actual reconstruction.
        """
        x_target = project_to_sphere(x_target)
        metrics: dict[str, torch.Tensor] = {}

        # 1. Training-style velocity metrics (always computed for parity with training)
        with torch.no_grad():
            training_output = self.compute_training_loss(z, x_target)
            for k, v in training_output.metrics.items():
                metrics[k] = v

        # 2. Get reconstruction - reuse from training if available (trajectory mode runs full ODE)
        if self.training_mode == "trajectory":
            # In trajectory mode, compute_training_loss already ran full ODE
            x_recon = training_output.reconstruction_approx
        else:
            # In ideal mode, need to run ODE for actual reconstruction
            with torch.no_grad():
                x_recon = self.generate(z)

        # 3. Reconstruction quality metrics
        recon_metrics = self.compute_reconstruction_quality(x_recon, x_target)
        for k, v in recon_metrics.items():
            metrics[f"reconstruction_{k}"] = v

        # 4. Baseline comparisons (always computed)
        with torch.no_grad():
            # Centroid of targets
            x_centroid = x_target.mean(dim=0)
            x_centroid_norm = F.normalize(x_centroid, dim=0)

            # How similar is reconstruction to centroid?
            centroid_cos_sim = F.cosine_similarity(
                x_recon,
                x_centroid_norm.unsqueeze(0).expand_as(x_recon),
                dim=-1
            ).mean()
            metrics["centroid_cos_sim"] = centroid_cos_sim

            # How similar are targets to their centroid?
            target_centroid_cos_sim = F.cosine_similarity(
                x_target,
                x_centroid_norm.unsqueeze(0).expand_as(x_target),
                dim=-1
            ).mean()
            metrics["target_centroid_cos_sim"] = target_centroid_cos_sim

            # Improvement: how much better is recon→target vs centroid→target?
            recon_target_sim = metrics["reconstruction_cosine_sim"]
            improvement = recon_target_sim - target_centroid_cos_sim
            metrics["improvement_vs_centroid"] = improvement

        return DecoderValidationOutput(
            metrics=metrics,
            reconstruction=x_recon,
        )

    def compute_reconstruction_quality(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute spherical reconstruction quality metrics."""
        return compute_spherical_metrics(x_recon, x_target)

    # =========================================================================
    # Generation
    # =========================================================================

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate reconstruction using geodesic ODE solver on the sphere.

        Uses the exponential map for proper geodesic integration, ensuring
        that velocity magnitude matters for the trajectory (not just direction).
        """
        batch_size = z.shape[0]
        device = z.device

        x_0 = torch.randn(batch_size, self.output_dim, device=device)
        x_0 = project_to_sphere(x_0)

        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.forward(x_t, t, z)

        def geodesic_step(x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
            """Take a geodesic step using the exponential map.

            exp_x(v*dt) travels distance |v|*dt along the geodesic
            in direction v from point x.
            """
            return exponential_map(x, v * dt)

        x = x_0
        dt = 1.0 / self.ode_num_steps
        for step in range(self.ode_num_steps):
            t_val = step * dt
            t = torch.full((batch_size,), t_val, device=device)
            v = velocity_fn(x, t)
            x = geodesic_step(x, v, dt)

        return project_to_sphere(x)

    def generate_differentiable(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor | None = None,
        return_per_sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Generate reconstruction with gradient support using differentiable geodesic ODE.

        Uses manual geodesic integration with exponential map instead of torchdiffeq,
        ensuring proper spherical geometry is maintained throughout.

        Args:
            z: Latent codes
            x_target: Target embeddings for loss computation
            return_per_sample: If True, return per-sample loss instead of mean
        """
        batch_size = z.shape[0]
        device = z.device

        x_0 = torch.randn(batch_size, self.output_dim, device=device)
        x_0 = project_to_sphere(x_0)

        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.forward(x_t, t, z)

        def geodesic_step(x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
            """Take a geodesic step using the exponential map."""
            return exponential_map(x, v * dt)

        # Manual integration loop (fully differentiable)
        x = x_0
        dt = 1.0 / self.ode_num_steps
        for step in range(self.ode_num_steps):
            t_val = step * dt
            t = torch.full((batch_size,), t_val, device=device)
            v = velocity_fn(x, t)
            x = geodesic_step(x, v, dt)

        x_final = project_to_sphere(x)

        if x_target is not None:
            x_target = project_to_sphere(x_target)
            per_sample_loss = 1 - F.cosine_similarity(x_final, x_target, dim=-1)
            if return_per_sample:
                return x_final, per_sample_loss
            return x_final, per_sample_loss.mean()

        return x_final, None

    # =========================================================================
    # Configuration
    # =========================================================================

    def get_training_loss_config(self) -> dict[str, Any]:
        return {
            "mode": self.training_mode,
            "direction_loss_ratio": self.direction_loss_ratio,
            "mse_ratio": 1 - self.direction_loss_ratio,
            "integration": "exponential_map",
            "ode_num_steps": self.ode_num_steps,
        }

    def get_validation_metrics_names(self) -> list[str]:
        return [
            "reconstruction_cosine_sim",
            "reconstruction_geodesic_dist",
            "reconstruction_mse_total",
            "velocity_mse_raw",
            "velocity_cosine_sim",
            "direction_loss",
            "improvement_vs_zero",
            "centroid_cos_sim",
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
