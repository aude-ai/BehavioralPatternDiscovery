"""
VAE Trainer

Handles the complete training loop for the Multi-Encoder VAE:
- Model forward pass and loss computation
- Discriminator training (TC intra, PC inter, TC unified)
- Beta controller updates for dynamic β adjustment (Dual Gradient Descent or legacy PI)
- Latent utilization losses (Batch Variance, Range Regularization)
- Gradient accumulation and mixed precision
- Checkpointing and logging
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.config import ModelDimensions
from src.model.base import BaseVAE
from src.model.discriminators import TCDiscriminator, PCDiscriminator, DiversityDiscriminator, DiversityDiscriminatorLoss
from src.training.losses import (
    CompositeLoss,
    ClusterSeparationLoss,
    RangeRegularizationLoss,
    ContrastiveMemoryLoss,
    EntropyUniformityLoss,
)
from src.training.optimizers import create_optimizer
from src.training.beta_controller import BetaControllerManager
from src.training.capacity_scheduler import CapacitySchedulerManager
from src.training.performance import AMPManager, enable_gradient_checkpointing
from src.training.scheduler import create_scheduler
from src.training.monitoring import LatentMonitor
from src.training.ema import EMAWrapper

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for Multi-Encoder VAE.

    Handles:
        - Training loop with discriminator updates
        - Mixed precision training (AMP)
        - Gradient clipping
        - Beta controller updates for dynamic β (Dual Gradient Descent or legacy PI)
        - Latent utilization losses (Batch Variance, Range Regularization)
        - Checkpointing
        - Logging
    """

    def __init__(
        self,
        model: BaseVAE,
        config: dict[str, Any],
        device: str = "cuda",
        num_engineers: int | None = None,
        normalization_params: dict[str, Any] | None = None,
        stop_signal: Any | None = None,
        dims: Any | None = None,
        on_epoch_end: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Args:
            model: MultiEncoderVAE instance
            config: Training configuration
            device: Device to train on
            num_engineers: Number of unique engineers (required if diversity discriminator enabled)
            normalization_params: Normalization params from preprocessing to save in checkpoint
            stop_signal: Optional threading.Event for early termination
            dims: ModelDimensions instance (optional, used for metadata in checkpoint)
            on_epoch_end: Optional callback for epoch completion (epoch, metrics, is_best)
            metadata: Optional metadata dict to save in checkpoint (e.g., embedder info)
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.num_engineers = num_engineers
        self.normalization_params = normalization_params
        self.stop_signal = stop_signal
        self.dims = dims if dims is not None else model.dims
        self.on_epoch_end_callback = on_epoch_end
        self.metadata = metadata or {}

        training_config = config["training"]
        loss_config = config["loss_weights"]
        capacity_config = config["capacity"]

        # Validate config combinations
        self._validate_config(config)

        # Use dimensions from model (already computed there)
        dims = model.dims

        # Get level names from model for dynamic iteration
        self.level_names = model.get_level_names()
        self.encoder_names = model.get_encoder_names()

        # Build discriminators
        self._build_discriminators(config, dims)

        # Build mixture priors dict if enabled
        mixture_priors = self._build_mixture_priors()

        # Build loss function
        dist_type = config["model"]["distribution"]["type"]
        distribution_config = config["model"]["distribution"]
        hoyer_config = config["hoyer"]
        iwo_config = config["iwo"]

        # Build KL config based on distribution type
        kl_config = {
            "distribution_type": dist_type,
            "use_capacity": True,
        }
        if dist_type == "gamma":
            gamma_config = distribution_config["gamma"]
            kl_config["prior_alpha"] = gamma_config["prior_concentration"]
            kl_config["prior_beta"] = gamma_config["prior_rate"]
        elif dist_type == "vmf":
            vmf_config = distribution_config["vmf"]
            kl_config["prior_type"] = vmf_config["prior_type"]
            kl_config["prior_kappa"] = vmf_config["prior_kappa"]

        self.loss_fn = CompositeLoss(
            config={
                "weights": loss_config,
                "kl": kl_config,
                "iwo": {
                    "enabled": iwo_config["enabled"],
                    "eps": iwo_config["eps"],
                },
                "hoyer": {
                    "enabled": hoyer_config["enabled"],
                    "target_sparsity": hoyer_config["target_sparsity"],
                    "per_sample": hoyer_config["per_sample"],
                    "per_dimension": hoyer_config["per_dimension"],
                    "absolute_values": hoyer_config["absolute_values"],
                    "eps": hoyer_config["eps"],
                    "adaptive": hoyer_config["adaptive"],
                    "adaptation_rate": hoyer_config["adaptation_rate"],
                    "min_weight": hoyer_config["min_weight"],
                    "max_weight": hoyer_config["max_weight"],
                },
                "capacity": capacity_config,
            },
            mixture_priors=mixture_priors,
            level_names=self.level_names,
        )

        # Build optimizers
        self._build_optimizers(config)

        # Build scheduler
        total_epochs = training_config["epochs"]
        scheduler_config = config["scheduler"]
        self.model_scheduler = create_scheduler(
            self.model_optimizer,
            scheduler_config,
            total_epochs,
        )
        # Only create disc scheduler if discriminator optimizer exists
        if self.disc_optimizer is not None:
            self.disc_scheduler = create_scheduler(
                self.disc_optimizer,
                scheduler_config,
                total_epochs,
            )
        else:
            self.disc_scheduler = None

        # Beta controller (Dual Gradient Descent or legacy PI)
        self.beta_controller = BetaControllerManager(config)

        # Capacity scheduler (per-encoder-level adaptive scheduling)
        self.capacity_scheduler = CapacitySchedulerManager(config)

        # Latent monitoring (uses logging config)
        logging_config = config["logging"]
        latent_stats_enabled = logging_config["latent_stats"]["enabled"]
        self.latent_monitor = LatentMonitor(
            enabled=latent_stats_enabled,
            level_names=self.level_names,
            encoder_names=self.encoder_names,
        )
        # Store logging flags for KL and Range stats
        self.kl_stats_enabled = logging_config["kl_stats"]["enabled"]
        self.range_stats_enabled = logging_config["range_stats"]["enabled"]

        # Cluster separation loss (mixture prior only)
        cluster_sep_config = config["cluster_separation"]
        self.cluster_separation_loss = ClusterSeparationLoss(cluster_sep_config)
        self.cluster_separation_weight = loss_config["cluster_separation"]

        # Range regularization loss (latent utilization)
        range_reg_config = config["range_regularization"]
        self.range_regularization_loss = RangeRegularizationLoss(range_reg_config)
        self.range_regularization_weight = loss_config["range_regularization"]

        # Contrastive memory loss (inter-batch diversity)
        contrastive_config = config["contrastive_memory_loss"]
        self.contrastive_memory_loss = ContrastiveMemoryLoss(contrastive_config)
        self.contrastive_memory_weight = loss_config["contrastive_memory"]

        # Entropy uniformity loss (stratification prevention)
        entropy_config = config["entropy_uniformity_loss"]
        self.entropy_uniformity_loss = EntropyUniformityLoss(entropy_config)
        self.entropy_uniformity_weight = loss_config["entropy_uniformity"]

        # Decoder EMA for stable inference
        decoder_type = config["model"]["decoder"]["type"]
        decoder_config = config["model"]["decoder"][decoder_type]
        self.decoder_ema = EMAWrapper(self.model.decoder, decoder_config["ema"])

        # Differentiable ODE training config (for flow matching decoders)
        self.diff_ode_step_counter = 0
        if decoder_type in ("flow_matching", "spherical_flow_matching"):
            train_cfg = config["decoder_training"][decoder_type]
            diff_cfg = train_cfg["differentiable_ode"]
            self.diff_ode_enabled = diff_cfg["enabled"]
            self.diff_ode_interval = diff_cfg["interval"]
            if self.diff_ode_enabled:
                logger.info(f"Differentiable ODE training enabled (interval={self.diff_ode_interval})")
        else:
            self.diff_ode_enabled = False
            self.diff_ode_interval = 0

        # Mixed precision
        amp_config = config["performance"]["amp"]
        self.amp_manager = AMPManager(
            enabled=amp_config["enabled"],
            dtype=amp_config["dtype"],
        )

        # Gradient checkpointing
        gc_config = config["performance"]["gradient_checkpointing"]
        if gc_config["enabled"]:
            enable_gradient_checkpointing(self.model)

        # torch.compile for kernel fusion and optimization
        # Store reference to original model for checkpoint saving/loading
        # (compiled models have different state_dict keys)
        self._original_model = self.model
        compile_config = config["performance"]["torch_compile"]
        if compile_config["enabled"]:
            compile_mode = compile_config["mode"]
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                logger.info(f"Model compiled with torch.compile (mode={compile_mode})")
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without compilation: {e}")

        # Gradient clipping
        clip_config = training_config["gradient_clipping"]
        self.clip_enabled = clip_config["enabled"]
        self.clip_max_norm = clip_config["max_norm"]

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Early stopping state
        early_stopping_config = training_config["early_stopping"]
        self.early_stopping_enabled = early_stopping_config["enabled"]
        self.patience = early_stopping_config["patience"]
        self.save_best = early_stopping_config["save_best"]
        self.patience_counter = 0
        self.best_val_loss = float("inf")
        self.best_model_state = None

        # Debug logging config
        logging_config = config["logging"]
        self.log_every_n_batches = logging_config["log_every_n_batches"]
        self.eval_mode_variance_enabled = logging_config["eval_mode_variance"]

        # Weights & Biases logging
        wandb_config = logging_config["wandb"]
        self.wandb_enabled = wandb_config["enabled"]
        self.wandb_run = None
        if self.wandb_enabled:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_config["project"],
                    entity=wandb_config["entity"],
                    config={
                        "model": config["model"],
                        "training": config["training"],
                        "loss_weights": config["loss_weights"],
                        "capacity": config["capacity"],
                    },
                )
                logger.info(f"W&B logging enabled: {self.wandb_run.url}")
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging")
                self.wandb_enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                self.wandb_enabled = False

        # Store config for logging
        self._training_config = training_config
        self._loss_config = loss_config
        self._capacity_config = capacity_config

        # Log training settings
        self._log_training_settings()

    def _validate_config(self, config: dict[str, Any]):
        """Validate config combinations that are incompatible."""
        # Focal loss only works with MLP decoder OR flow matching with differentiable ODE
        focal_enabled = config["focal_loss"]["enabled"]
        decoder_type = config["model"]["decoder"]["type"]

        if focal_enabled and decoder_type in ("flow_matching", "spherical_flow_matching"):
            # Check if differentiable ODE is enabled - if so, focal can apply to reconstruction
            train_cfg = config["decoder_training"][decoder_type]
            diff_ode_enabled = train_cfg["differentiable_ode"]["enabled"]

            if not diff_ode_enabled:
                raise ValueError(
                    f"Focal reconstruction loss is not compatible with {decoder_type} decoder "
                    "unless differentiable_ode is enabled. "
                    "Focal loss requires direct reconstruction comparison, but "
                    "flow matching uses velocity MSE which bypasses the reconstruction loss. "
                    "Either set focal_loss.enabled=false, use decoder.type='mlp', "
                    "or enable decoder_training.{decoder_type}.differentiable_ode.enabled=true."
                )

    def _log_training_settings(self):
        """Log training configuration summary."""
        logger.info("TRAINING SETTINGS")
        logger.info(f"  Device: {self.device}")

        # AMP status
        if self.amp_manager.enabled:
            logger.info(f"  AMP: enabled ({self.amp_manager.dtype})")
        else:
            logger.info("  AMP: disabled")

        # Gradient clipping
        if self.clip_enabled:
            logger.info(f"  Gradient clipping: enabled (max_norm={self.clip_max_norm})")
        else:
            logger.info("  Gradient clipping: disabled")

        # Loss weights
        logger.info("  Loss weights:")
        kl_config = self._loss_config["kl"]
        kl_enc = kl_config["encoder_levels"]
        kl_uni = kl_config["unified"]
        recon_weights = self._loss_config["recon"]
        logger.info(f"    recon=[base:{recon_weights['base']}, flow:{recon_weights['flow']}, focal:{recon_weights['focal']}], kl=[enc:{kl_enc}, uni:{kl_uni}]")
        logger.info(f"    tc_intra={self._loss_config['tc_intra']}, pc_inter={self._loss_config['pc_inter']}, "
                    f"tc_unified={self._loss_config['tc_unified']}")
        logger.info(f"    iwo_intra={self._loss_config['iwo_intra']}, iwo_unified={self._loss_config['iwo_unified']}, "
                    f"hoyer={self._loss_config['hoyer']}")

        # Focal loss
        if self.config["focal_loss"]["enabled"]:
            focal_gamma = self.config["focal_loss"]["gamma"]
            logger.info(f"  Focal loss: enabled (gamma={focal_gamma})")
        else:
            logger.info("  Focal loss: disabled")

        # Hoyer sparsity
        hoyer_config = self.config["hoyer"]
        if hoyer_config["enabled"]:
            logger.info(f"  Hoyer sparsity: enabled (target={hoyer_config['target_sparsity']})")
        else:
            logger.info("  Hoyer sparsity: disabled")

        # Discriminators (TC/PC)
        logger.info(f"  Discriminators: TC_intra={self.tc_intra_enabled}, "
                    f"PC_inter={self.pc_inter_enabled}, TC_unified={self.tc_unified_enabled}")

        # IWO loss
        iwo_config = self.config["iwo"]
        if iwo_config["enabled"]:
            logger.info(f"  IWO loss: enabled (eps={iwo_config['eps']})")
        else:
            logger.info("  IWO loss: disabled")

        # Capacity schedule
        floor_list = self._capacity_config["floor_targets"]
        final_list = self._capacity_config["final_targets"]
        warmup = self._capacity_config["warmup_epochs"]
        ramp_down = self._capacity_config["ramp_down_epochs"]
        ramp_up = self._capacity_config["ramp_up_epochs"]
        logger.info(f"  Capacity schedule: warmup={warmup}, ramp_down={ramp_down}, ramp_up={ramp_up}")
        floor_str = ", ".join(f"{t['name']}={t['value']}" for t in floor_list)
        final_str = ", ".join(f"{t['name']}={t['value']}" for t in final_list)
        logger.info(f"    Floor: {floor_str}")
        logger.info(f"    Final: {final_str}")

        # Beta controller
        beta_config = self.config["beta_controller"]
        beta_type = beta_config["type"]
        if beta_type == "dual_gradient":
            dg_config = beta_config["dual_gradient"]
            logger.info(f"  Beta Controller: {beta_type} (lr_dual={dg_config['lr_dual']}, "
                        f"constraint_ema={dg_config['use_constraint_ema']}, "
                        f"beta_range=[{beta_config['beta_min']}, {beta_config['beta_max']}])")
        else:
            pi_config = beta_config["pi"]
            logger.info(f"  Beta Controller: {beta_type} (kp={pi_config['kp']}, ki={pi_config['ki']}, "
                        f"beta_range=[{beta_config['beta_min']}, {beta_config['beta_max']}])")

        # Early stopping (patience starts after capacity warmup)
        if self.early_stopping_enabled:
            warmup_epochs = self.config["capacity"]["warmup_epochs"]
            logger.info(f"  Early stopping: enabled (patience={self.patience}, starts after epoch {warmup_epochs})")
        else:
            logger.info("  Early stopping: disabled")

        # Optimizers
        vae_opt = self.config["vae_optimizer"]
        disc_opt = self.config["discriminator_optimizer"]
        logger.info(f"  VAE Optimizer: {vae_opt['type']}, lr={vae_opt['learning_rate']}, "
                    f"weight_decay={vae_opt['weight_decay']}")
        logger.info(f"  Disc Optimizer: {disc_opt['type']}, lr={disc_opt['learning_rate']}, "
                    f"weight_decay={disc_opt['weight_decay']}")

        # Scheduler
        sched_config = self.config["scheduler"]
        logger.info(f"  Scheduler: {sched_config['type']}, warmup={sched_config['warmup_epochs']} epochs, "
                    f"min_lr_ratio={sched_config['min_lr_ratio']}")

    def _build_mixture_priors(self) -> dict:
        """
        Build dict of MixturePrior objects from model if enabled.

        Returns per-encoder mixture priors to ensure each encoder's cluster
        probabilities match the correct mixture prior parameters during KL computation.

        Returns:
            Dict with structure:
            {
                "enc1": {"level_name": MP, ...},
                "enc2": {"level_name": MP, ...},
                ...
                "unified": MP
            }
        """
        mixture_priors = {}

        if not self.model.mixture_prior_enabled:
            return mixture_priors

        # Get mixture priors from EACH encoder (not just enc1!)
        for i, encoder in enumerate(self.model.encoders):
            enc_name = f"enc{i + 1}"
            enc_priors = {}
            for level in self.level_names:
                mp = encoder.get_mixture_prior(level)
                if mp is not None:
                    enc_priors[level] = mp
            if enc_priors:
                mixture_priors[enc_name] = enc_priors

        # Get mixture prior from unification
        if hasattr(self.model.unification, "get_mixture_prior"):
            mp = self.model.unification.get_mixture_prior()
            if mp is not None:
                mixture_priors["unified"] = mp

        if mixture_priors:
            enc_levels = [k for k in mixture_priors.keys() if k.startswith("enc")]
            has_unified = "unified" in mixture_priors
            logger.info(
                f"Mixture prior KL enabled: {len(enc_levels)} encoders, unified={has_unified}"
            )

        return mixture_priors

    def _build_discriminators(self, config: dict[str, Any], dims: ModelDimensions):
        """Build TC and PC discriminators if enabled."""
        disc_config = config["model"]["discriminators"]

        # Loss formulation config from training.yaml
        loss_form_config = config["discriminator"]
        tc_loss_formulation = loss_form_config["tc_loss_formulation"]
        pc_loss_formulation = loss_form_config["pc_loss_formulation"]

        # Training frequency optimization
        self.disc_train_every_n_batches = disc_config.get("train_every_n_batches", 1)
        self.disc_steps_per_round = disc_config.get("discriminator_steps", 1)
        self._disc_batch_counter = 0

        # Discriminator-specific layer config (each section has its own)
        disc_layer_config = disc_config["layer_config"]
        activation_config = disc_layer_config["activation"]

        # Track which discriminators are enabled
        self.tc_intra_enabled = disc_config["tc_intra"]["enabled"]
        self.pc_inter_enabled = disc_config["pc_inter"]["enabled"]
        self.tc_unified_enabled = disc_config["tc_unified"]["enabled"]

        # TC Intra discriminators (one per encoder level)
        self.tc_intra_discriminators = nn.ModuleDict()
        if self.tc_intra_enabled:
            for level in dims.level_names:
                _, dim = dims.get_level_dims(level)
                tc_config = disc_config["tc_intra"]
                self.tc_intra_discriminators[level] = TCDiscriminator({
                    "input_dim": dim,
                    "hidden_dims": tc_config["hidden_dims"],
                    "dropout": tc_config["dropout"],
                    "activation": activation_config,
                    "tc_loss_formulation": tc_loss_formulation,
                }).to(self.device)

        # PC Inter discriminators (one per level, operates on all encoders)
        self.pc_inter_discriminators = nn.ModuleDict()
        if self.pc_inter_enabled:
            for level in dims.level_names:
                _, dim = dims.get_level_dims(level)
                pc_config = disc_config["pc_inter"]
                self.pc_inter_discriminators[level] = PCDiscriminator({
                    "latent_dim_per_encoder": dim,
                    "num_encoders": dims.num_encoders,
                    "hidden_dims": pc_config["hidden_dims"],
                    "dropout": pc_config["dropout"],
                    "activation": activation_config,
                    "pc_loss_formulation": pc_loss_formulation,
                }).to(self.device)

        # TC Unified discriminator (one for unified latent)
        self.tc_unified_discriminator = None
        if self.tc_unified_enabled:
            tc_unified_config = disc_config["tc_unified"]
            self.tc_unified_discriminator = TCDiscriminator({
                "input_dim": dims.unified_output_dim,
                "hidden_dims": tc_unified_config["hidden_dims"],
                "dropout": tc_unified_config["dropout"],
                "activation": activation_config,
                "tc_loss_formulation": tc_loss_formulation,
            }).to(self.device)

        # Diversity discriminator (engineer classification from unified latent)
        # Only works with single-engineer batch mode
        diversity_config = disc_config["diversity"]
        self.diversity_enabled = diversity_config["enabled"]
        self.diversity_discriminator = None
        self.diversity_disc_loss = None
        if self.diversity_enabled:
            # Validate batch mode
            batch_mode = config["batching"]["mode"]
            if batch_mode != "engineer":
                raise ValueError(
                    f"DiversityDiscriminator requires batching.mode='engineer', "
                    f"got '{batch_mode}'. Each batch must contain messages from a single engineer."
                )

            # Validate num_engineers
            if self.num_engineers is None:
                raise ValueError(
                    "DiversityDiscriminator requires num_engineers parameter. "
                    "Pass the number of unique engineers when creating the Trainer."
                )

            self.diversity_discriminator = DiversityDiscriminator(
                config=diversity_config,
                input_dim=dims.unified_output_dim,
                num_engineers=self.num_engineers,
            ).to(self.device)

            self.diversity_disc_loss = DiversityDiscriminatorLoss(
                diversity_config,
                self.diversity_discriminator,
            )

            self.diversity_weight = config["loss_weights"]["diversity_discriminator"]

            # Build engineer_id to index mapping (will be populated during training)
            self.engineer_id_to_idx: dict[str, int] = {}

    def _build_optimizers(self, config: dict[str, Any]):
        """Build separate optimizers for VAE model and discriminators."""
        vae_opt_config = config["vae_optimizer"]
        disc_opt_config = config["discriminator_optimizer"]

        # Model optimizer (encoders, unification, decoder)
        self.model_optimizer = create_optimizer(
            self.model.parameters(),
            vae_opt_config,
        )

        # Discriminator optimizer (only if any discriminators enabled)
        disc_params = []
        if self.tc_intra_enabled:
            for disc in self.tc_intra_discriminators.values():
                disc_params.extend(disc.parameters())
        if self.pc_inter_enabled:
            for disc in self.pc_inter_discriminators.values():
                disc_params.extend(disc.parameters())
        if self.tc_unified_enabled and self.tc_unified_discriminator is not None:
            disc_params.extend(self.tc_unified_discriminator.parameters())
        if self.diversity_enabled and self.diversity_discriminator is not None:
            disc_params.extend(self.diversity_discriminator.parameters())

        # Only create optimizer if there are parameters
        if disc_params:
            self.disc_optimizer = create_optimizer(
                disc_params,
                disc_opt_config,
            )
        else:
            self.disc_optimizer = None

    # =========================================================================
    # Shared Helper Methods (used by both train_step and validate)
    # =========================================================================

    def _extract_encoder_data(
        self,
        encodings: dict[str, Any],
    ) -> tuple[dict, dict, dict]:
        """
        Extract encoder params, samples, and cluster probs from encodings.

        Returns:
            Tuple of (encoder_params, encoder_samples, encoder_cluster_probs)
        """
        encoder_params = {}
        encoder_samples = {}
        encoder_cluster_probs = {}

        for enc_key in self.encoder_names:
            enc_data = encodings[enc_key]

            encoder_params[enc_key] = {}
            encoder_samples[enc_key] = {}

            for level in self.level_names:
                encoder_params[enc_key][level] = enc_data[level]["params"]
                encoder_samples[enc_key][level] = enc_data[level]["z"]

            if self.model.mixture_prior_enabled:
                encoder_cluster_probs[enc_key] = {}
                for level in self.level_names:
                    if "cluster_probs" in enc_data[level]:
                        encoder_cluster_probs[enc_key][level] = enc_data[level]["cluster_probs"]

        return encoder_params, encoder_samples, encoder_cluster_probs

    def _get_mu_without_dropout(self, x: torch.Tensor, enable_grad: bool = True) -> dict[str, Any]:
        """
        Get mu values with dropout disabled to see true encoder state.

        This is critical for anti-collapse losses - they must operate on
        the actual mu values, not mu + dropout noise.

        Args:
            x: Input tensor (batch, input_dim)
            enable_grad: If True, keep gradients for loss computation.
                        If False, use no_grad for monitoring only.

        Returns:
            Dict mapping enc_key -> level -> mu tensor, plus "unified" -> mu tensor
        """
        # Store original dropout states
        dropout_modules = []
        original_training = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                dropout_modules.append(module)
                original_training.append(module.training)
                module.eval()  # Disable dropout

        try:
            # Forward pass - with or without gradients
            if enable_grad:
                encodings = self.model.encode(x)
            else:
                with torch.no_grad():
                    encodings = self.model.encode(x)

            # Extract mu from params
            result = {}
            for enc_key in self.encoder_names:
                result[enc_key] = {}
                for level in self.level_names:
                    params = encodings[enc_key][level]["params"]
                    result[enc_key][level] = self.model.distribution.get_mean(params)

            # Add unified
            unified_params = encodings["unified"]["params"]
            result["unified"] = self.model.unified_distribution.get_mean(unified_params)

            return result

        finally:
            # Restore original dropout states
            for module, training in zip(dropout_modules, original_training):
                module.train(training)

    def _log_eval_mode_variance(self, x: torch.Tensor) -> None:
        """
        Log mu variance with dropout disabled to detect collapse.

        Only logs when eval_mode_variance is enabled in config.
        Uses log_every_n_batches as the logging frequency.
        """
        if not self.eval_mode_variance_enabled:
            return

        # Use enable_grad=False since this is monitoring only
        true_mu_dict = self._get_mu_without_dropout(x, enable_grad=False)

        logger.info("-" * 60)
        logger.info(f"[EvalMode Check @ step {self.global_step}] True mu variance (no dropout):")

        collapsed_total = 0
        total_dims = 0

        for enc_key in self.encoder_names:
            for level in self.level_names:
                mu = true_mu_dict[enc_key][level]
                dim_var = mu.var(dim=0)
                collapsed = (dim_var < 0.01).sum().item()
                collapsed_total += collapsed
                total_dims += mu.shape[1]

                if collapsed > 0:
                    logger.warning(
                        f"  {enc_key}_{level}: var[min={dim_var.min():.4f} max={dim_var.max():.4f}] "
                        f"COLLAPSED={collapsed}/{mu.shape[1]}"
                    )
                else:
                    logger.info(
                        f"  {enc_key}_{level}: var[min={dim_var.min():.4f} max={dim_var.max():.4f}] OK"
                    )

        # Unified
        unified_mu = true_mu_dict["unified"]
        dim_var = unified_mu.var(dim=0)
        collapsed = (dim_var < 0.01).sum().item()
        collapsed_total += collapsed
        total_dims += unified_mu.shape[1]

        if collapsed > 0:
            logger.warning(
                f"  unified: var[min={dim_var.min():.4f} max={dim_var.max():.4f}] "
                f"COLLAPSED={collapsed}/{unified_mu.shape[1]}"
            )
        else:
            logger.info(
                f"  unified: var[min={dim_var.min():.4f} max={dim_var.max():.4f}] OK"
            )

        # Overall summary
        collapse_ratio = collapsed_total / total_dims
        if collapse_ratio > 0.5:
            logger.error(
                f"  CRITICAL: {collapsed_total}/{total_dims} ({collapse_ratio:.1%}) dimensions collapsed!"
            )
        elif collapse_ratio > 0.1:
            logger.warning(
                f"  WARNING: {collapsed_total}/{total_dims} ({collapse_ratio:.1%}) dimensions collapsed"
            )

        logger.info("-" * 60)

    def _get_all_betas(self) -> tuple[dict[str, float], float]:
        """
        Get all beta values from beta controllers.

        Returns:
            Tuple of (encoder_betas dict, unified_beta)
        """
        encoder_betas = {}
        for enc_name in self.encoder_names:
            for level in self.level_names:
                key = f"{enc_name}_{level}"
                encoder_betas[key] = self.beta_controller.get_beta(enc_name, level)

        unified_beta = self.beta_controller.get_unified_beta()
        return encoder_betas, unified_beta

    def _forward_and_compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        flow_loss_mode: str = "training",
        ode_steps: int = 10,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Shared forward pass and loss computation for training and validation.

        Args:
            batch: Batch containing "embeddings" tensor
            flow_loss_mode: "training" uses decoder's compute_loss, "inference" uses generate
            ode_steps: Unused (kept for API compatibility)

        Returns:
            Tuple of (losses dict, encodings dict with additional metadata)
        """
        x = batch["embeddings"].to(self.device)
        x_embedding = x[:, :self.model.output_dim]

        # Forward pass through encoder
        encodings = self.model.encode(x)
        unified_z = encodings["unified"]["z"]

        # Store for external access (needed by validate())
        encodings["unified_z"] = unified_z
        encodings["x_embedding"] = x_embedding

        # Compute reconstruction/flow loss via decoder interface
        # The decoder handles all decoder-specific logic (flow matching, ODE, etc.)
        if flow_loss_mode == "training":
            # Check if we should include differentiable ODE reconstruction loss
            include_recon_loss = False
            if self.diff_ode_enabled:
                self.diff_ode_step_counter += 1
                if self.diff_ode_step_counter >= self.diff_ode_interval:
                    include_recon_loss = True
                    self.diff_ode_step_counter = 0

            # Training mode: use decoder's compute_training_loss
            decoder_output = self.model.decoder.compute_training_loss(
                unified_z, x_embedding, include_reconstruction_loss=include_recon_loss
            )
            x_recon = decoder_output.reconstruction_approx

            # Store separate loss components
            encodings["flow_loss"] = decoder_output.flow_loss
            encodings["base_recon_loss"] = decoder_output.base_recon_loss
            encodings["focal_loss"] = decoder_output.focal_loss
            encodings["decoder_metrics"] = decoder_output.metrics
        else:
            # Validation mode: use decoder's compute_validation_metrics
            decoder_val_output = self.model.decoder.compute_validation_metrics(unified_z, x_embedding)
            x_recon = decoder_val_output.reconstruction
            recon_metrics = decoder_val_output.metrics

            # For validation, compute base_recon_loss from metrics
            if "reconstruction_mse_total" in recon_metrics:
                encodings["base_recon_loss"] = recon_metrics["reconstruction_mse_total"]
            elif "reconstruction_cosine_sim" in recon_metrics:
                # For spherical: use 1 - cosine_sim as loss
                encodings["base_recon_loss"] = 1 - recon_metrics["reconstruction_cosine_sim"]
            else:
                encodings["base_recon_loss"] = ((x_recon - x_embedding) ** 2).mean()

            # Flow loss from validation metrics if available
            if "velocity_velocity_mse_normalized" in recon_metrics:
                encodings["flow_loss"] = recon_metrics["velocity_velocity_mse_normalized"]
            else:
                encodings["flow_loss"] = None

            encodings["focal_loss"] = None  # Not computed during validation
            encodings["decoder_val_metrics"] = recon_metrics

        # Extract encoder data
        encoder_params, encoder_samples, encoder_cluster_probs = (
            self._extract_encoder_data(encodings)
        )

        # Compute adversarial losses
        tc_losses, pc_losses = self._compute_adversarial_losses(encodings)

        # Get betas and capacities
        encoder_betas, unified_beta = self._get_all_betas()
        in_warmup = self.capacity_scheduler.in_warmup(self.current_epoch)
        capacities = self.capacity_scheduler.get_all_capacities(
            self.current_epoch, self.model.num_encoders
        )

        # Compute composite loss with separate recon components
        losses = self.loss_fn(
            x_recon=x_recon,
            x_target=x_embedding,
            encoder_params=encoder_params,
            unified_params=encodings["unified"]["params"],
            encoder_samples=encoder_samples,
            unified_sample=unified_z,
            tc_losses=tc_losses,
            pc_losses=pc_losses,
            capacities=capacities,
            encoder_betas=encoder_betas,
            unified_beta=unified_beta,
            skip_kl_loss=in_warmup,
            encoder_cluster_probs=encoder_cluster_probs if self.model.mixture_prior_enabled else None,
            unified_cluster_probs=encodings["unified"].get("cluster_probs"),
            flow_loss=encodings.get("flow_loss"),
            base_recon_loss=encodings.get("base_recon_loss"),
            focal_loss=encodings.get("focal_loss"),
        )

        # Add useful metadata to encodings for external use
        encodings["x_embedding"] = x_embedding
        encodings["x_recon"] = x_recon
        encodings["encoder_params"] = encoder_params

        return losses, encodings

    def _compute_cluster_separation_loss(self) -> torch.Tensor:
        """
        Compute cluster separation loss across all mixture priors.

        Returns:
            Total cluster separation loss
        """
        if not self.cluster_separation_loss.enabled or not self.model.mixture_prior_enabled:
            return torch.tensor(0.0, device=self.device)

        cluster_sep_total = torch.tensor(0.0, device=self.device)

        for encoder in self.model.encoders:
            for level in encoder.get_level_names():
                mp = encoder.get_mixture_prior(level)
                if mp is not None:
                    cluster_sep_total = cluster_sep_total + self.cluster_separation_loss(mp)

        # Unified mixture prior
        if hasattr(self.model.unification, 'mixture_prior'):
            uni_mp = self.model.unification.mixture_prior
            if uni_mp is not None:
                cluster_sep_total = cluster_sep_total + self.cluster_separation_loss(uni_mp)

        return cluster_sep_total

    def _compute_regularization_losses(
        self,
        encodings: dict[str, Any],
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute regularization losses (range_reg, entropy, cluster_sep).

        Used by both train_step and validate for consistent loss computation.

        Args:
            encodings: Output from model.encode()
            x: Input tensor (needed for _get_mu_without_dropout)

        Returns:
            Dictionary of regularization loss tensors
        """
        losses = {}

        # Check if any regularization losses are enabled
        needs_true_mu = (
            self.range_regularization_loss.enabled
            or self.entropy_uniformity_loss.enabled
        )

        if not needs_true_mu:
            return losses

        # Get mu without dropout for accurate collapse detection
        true_mu_dict = self._get_mu_without_dropout(x)

        # Cluster separation loss
        cluster_sep_loss = self._compute_cluster_separation_loss()
        if cluster_sep_loss > 0:
            losses["cluster_separation"] = cluster_sep_loss

        # Range regularization loss
        if self.range_regularization_loss.enabled:
            range_reg_total = torch.tensor(0.0, device=self.device)

            for level in self.level_names:
                for enc_key in self.encoder_names:
                    mu = true_mu_dict[enc_key][level]
                    level_loss = self.range_regularization_loss.compute(mu)
                    range_reg_total = range_reg_total + level_loss

            # Unified level
            unified_mu = true_mu_dict["unified"]
            unified_loss = self.range_regularization_loss.compute(unified_mu)
            range_reg_total = range_reg_total + unified_loss

            losses["range_regularization"] = range_reg_total

        # Entropy uniformity loss
        if self.entropy_uniformity_loss.enabled:
            entropy_total = torch.tensor(0.0, device=self.device)

            for level in self.level_names:
                for enc_key in self.encoder_names:
                    mu = true_mu_dict[enc_key][level]
                    level_loss = self.entropy_uniformity_loss.compute(mu)
                    entropy_total = entropy_total + level_loss

            # Unified level
            unified_mu = true_mu_dict["unified"]
            unified_loss = self.entropy_uniformity_loss.compute(unified_mu)
            entropy_total = entropy_total + unified_loss

            losses["entropy_uniformity"] = entropy_total

        return losses

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Single training step.

        Uses shared _forward_and_compute_loss for consistency with validation.

        Args:
            batch: Dictionary with "embeddings" tensor

        Returns:
            Dictionary of loss values and metrics
        """
        x = batch["embeddings"].to(self.device)

        # === Phase 1: Train discriminators (with frequency control) ===
        self._disc_batch_counter += 1
        disc_losses = {}
        if self._disc_batch_counter >= self.disc_train_every_n_batches:
            self._disc_batch_counter = 0
            for _ in range(self.disc_steps_per_round):
                disc_losses = self._train_discriminators(x, batch)

        # === Phase 2: Train VAE ===
        self.model_optimizer.zero_grad()

        with self.amp_manager.autocast_context():
            # Shared forward pass and loss computation
            losses, encodings = self._forward_and_compute_loss(batch, flow_loss_mode="training")

            # Capture tensor stats for latent monitoring
            x_embedding = encodings["x_embedding"]
            x_recon = encodings["x_recon"]
            self.latent_monitor.capture_tensor_stats(x_embedding, x_recon, encodings)

            # Pre-compute true mu (dropout disabled) for anti-collapse losses
            # CRITICAL: Must use true mu without dropout to detect actual collapse
            true_mu_dict = None
            needs_true_mu = (
                self.range_regularization_loss.enabled
                or self.contrastive_memory_loss.enabled
                or self.entropy_uniformity_loss.enabled
            )
            if needs_true_mu:
                true_mu_dict = self._get_mu_without_dropout(x)

            # Compute cluster separation loss - training only
            cluster_sep_loss = self._compute_cluster_separation_loss()
            if cluster_sep_loss > 0:
                losses["cluster_separation"] = cluster_sep_loss
                losses["total"] = losses["total"] + self.cluster_separation_weight * cluster_sep_loss

            # Compute range regularization loss - training only
            # CRITICAL: Uses true_mu_dict (dropout disabled) to see actual encoder state
            if self.range_regularization_loss.enabled:
                range_reg_total = torch.tensor(0.0, device=self.device)
                range_thresh = self.range_regularization_loss.min_range

                for level in self.level_names:
                    for enc_key in self.encoder_names:
                        # Use true mu (no dropout) for anti-collapse loss
                        mu = true_mu_dict[enc_key][level]
                        level_loss = self.range_regularization_loss.compute(mu)
                        range_reg_total = range_reg_total + level_loss

                        # Track range stats for epoch logging
                        latent_dim = mu.shape[1]
                        dim_range = mu.max(dim=0).values - mu.min(dim=0).values
                        range_violations = (dim_range < range_thresh).sum().item()
                        losses[f"range_violations_{enc_key}_{level}"] = range_violations
                        losses[f"range_dims_{enc_key}_{level}"] = latent_dim
                        losses[f"range_min_{enc_key}_{level}"] = dim_range.min().item()
                        losses[f"range_max_{enc_key}_{level}"] = dim_range.max().item()
                        losses[f"range_mean_{enc_key}_{level}"] = dim_range.mean().item()

                # Unified level - use true mu (no dropout)
                unified_mu = true_mu_dict["unified"]
                unified_loss = self.range_regularization_loss.compute(unified_mu)
                range_reg_total = range_reg_total + unified_loss

                # Track unified range stats
                latent_dim = unified_mu.shape[1]
                dim_range = unified_mu.max(dim=0).values - unified_mu.min(dim=0).values
                range_violations = (dim_range < range_thresh).sum().item()
                losses["range_violations_unified"] = range_violations
                losses["range_dims_unified"] = latent_dim
                losses["range_min_unified"] = dim_range.min().item()
                losses["range_max_unified"] = dim_range.max().item()
                losses["range_mean_unified"] = dim_range.mean().item()

                losses["range_regularization"] = range_reg_total
                losses["total"] = losses["total"] + self.range_regularization_weight * range_reg_total

            # Compute contrastive memory loss - training only
            # Uses true_mu_dict (dropout disabled)
            if self.contrastive_memory_loss.enabled:
                contrastive_total = torch.tensor(0.0, device=self.device)

                for level in self.level_names:
                    for enc_key in self.encoder_names:
                        mu = true_mu_dict[enc_key][level]
                        level_loss = self.contrastive_memory_loss.compute(mu, enc_key, level)
                        contrastive_total = contrastive_total + level_loss
                        # Update memory AFTER computing loss
                        self.contrastive_memory_loss.update_memory(mu, enc_key, level)

                # Unified level
                unified_mu = true_mu_dict["unified"]
                unified_loss = self.contrastive_memory_loss.compute(unified_mu, "unified", "unified")
                contrastive_total = contrastive_total + unified_loss
                self.contrastive_memory_loss.update_memory(unified_mu, "unified", "unified")

                losses["contrastive_memory"] = contrastive_total
                losses["total"] = losses["total"] + self.contrastive_memory_weight * contrastive_total

            # Compute entropy uniformity loss - training only
            # Uses true_mu_dict (dropout disabled)
            if self.entropy_uniformity_loss.enabled:
                entropy_total = torch.tensor(0.0, device=self.device)

                for level in self.level_names:
                    for enc_key in self.encoder_names:
                        mu = true_mu_dict[enc_key][level]
                        level_loss = self.entropy_uniformity_loss.compute(mu)
                        entropy_total = entropy_total + level_loss

                # Unified level
                unified_mu = true_mu_dict["unified"]
                unified_loss = self.entropy_uniformity_loss.compute(unified_mu)
                entropy_total = entropy_total + unified_loss

                losses["entropy_uniformity"] = entropy_total
                losses["total"] = losses["total"] + self.entropy_uniformity_weight * entropy_total

            # Compute diversity discriminator VAE loss - training only
            # Encourages VAE to produce representations that ARE distinguishable by engineer
            # (cooperative: higher discriminator accuracy = better representations)
            if self.diversity_enabled and self.diversity_disc_loss is not None:
                if "engineer_id" in batch:
                    engineer_id = batch["engineer_id"]
                    # Convert engineer_id to index (handle string IDs)
                    if isinstance(engineer_id, str):
                        if engineer_id not in self.engineer_id_to_idx:
                            self.engineer_id_to_idx[engineer_id] = len(self.engineer_id_to_idx)
                        engineer_idx = self.engineer_id_to_idx[engineer_id]
                    else:
                        engineer_idx = int(engineer_id)

                    z_unified = encodings["unified"]["z"]
                    diversity_vae_loss = self.diversity_disc_loss.vae_loss(z_unified, engineer_idx)
                    losses["diversity"] = diversity_vae_loss
                    losses["total"] = losses["total"] + self.diversity_weight * diversity_vae_loss

        # Backward pass
        self.amp_manager.backward(losses["total"])

        # Gradient clipping and optimizer step
        if self.clip_enabled:
            self.amp_manager.unscale_and_clip(
                self.model_optimizer,
                self.model.parameters(),
                self.clip_max_norm,
            )
        self.amp_manager.step(self.model_optimizer)

        # Update decoder EMA
        self.decoder_ema.update()

        # Record KL values during warmup (for capacity scheduler)
        in_warmup = self.capacity_scheduler.in_warmup(self.current_epoch)
        if in_warmup:
            self._record_warmup_kl(losses)

        # Update beta controllers
        self._update_beta_controllers(losses)

        self.global_step += 1

        # Eval-mode monitoring: check true mu variance periodically
        if self.global_step % self.log_every_n_batches == 0 and self.global_step > 0:
            self._log_eval_mode_variance(x)

        # Convert losses to floats for logging
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        metrics.update(disc_losses)
        metrics.update(self.beta_controller.get_all_betas())

        # Add decoder-specific metrics (prefixed with "decoder_")
        if "decoder_metrics" in encodings:
            for k, v in encodings["decoder_metrics"].items():
                val = v.item() if torch.is_tensor(v) else v
                metrics[f"decoder_{k}"] = val

        return metrics

    def _train_discriminators(
        self,
        x: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """
        Train all enabled discriminators.

        Args:
            x: Input embeddings
            batch: Full batch dict (needed for diversity discriminator engineer_id)

        Returns:
            Dictionary of discriminator losses
        """
        # Skip if no discriminators enabled
        if self.disc_optimizer is None:
            return {}

        self.disc_optimizer.zero_grad()

        with torch.no_grad():
            encodings = self.model.encode(x)

        disc_loss_total = torch.tensor(0.0, device=self.device)
        disc_losses = {}

        # TC Intra (per level, aggregated across encoders)
        if self.tc_intra_enabled:
            for level in self.level_names:
                disc = self.tc_intra_discriminators[level]
                level_loss = torch.tensor(0.0, device=self.device)

                for enc_key in self.encoder_names:
                    z = encodings[enc_key][level]["z"]
                    loss, _ = disc.compute_tc_loss(z, train_discriminator=True)
                    level_loss = level_loss + loss

                disc_losses[f"disc_tc_{level}"] = level_loss.item()
                disc_loss_total = disc_loss_total + level_loss

        # PC Inter (per level)
        if self.pc_inter_enabled:
            for level in self.level_names:
                disc = self.pc_inter_discriminators[level]
                encoder_outputs = [
                    encodings[enc_key][level]["z"]
                    for enc_key in self.encoder_names
                ]
                loss, _ = disc.compute_pc_loss(encoder_outputs, train_discriminator=True)
                disc_losses[f"disc_pc_{level}"] = loss.item()
                disc_loss_total = disc_loss_total + loss

        # TC Unified
        if self.tc_unified_enabled and self.tc_unified_discriminator is not None:
            z_unified = encodings["unified"]["z"]
            loss, _ = self.tc_unified_discriminator.compute_tc_loss(
                z_unified, train_discriminator=True
            )
            disc_losses["disc_tc_unified"] = loss.item()
            disc_loss_total = disc_loss_total + loss

        # Diversity discriminator (engineer classification from unified latent)
        # Only works with single-engineer batch mode
        if self.diversity_enabled and self.diversity_discriminator is not None:
            z_unified = encodings["unified"]["z"]

            # Get engineer_id from batch - required for diversity discriminator
            if batch is not None and "engineer_id" in batch:
                engineer_id = batch["engineer_id"]
                # Convert engineer_id to index (handle string IDs)
                if isinstance(engineer_id, str):
                    if engineer_id not in self.engineer_id_to_idx:
                        self.engineer_id_to_idx[engineer_id] = len(self.engineer_id_to_idx)
                    engineer_idx = self.engineer_id_to_idx[engineer_id]
                else:
                    # Already an integer index
                    engineer_idx = int(engineer_id)

                loss = self.diversity_disc_loss.discriminator_loss(z_unified, engineer_idx)
                disc_losses["disc_diversity"] = loss.item()
                disc_loss_total = disc_loss_total + loss

        # Only backward if we computed any losses
        if disc_loss_total.item() > 0:
            disc_loss_total.backward()

            # Collect all discriminator parameters
            disc_params = []
            if self.tc_intra_enabled:
                for disc in self.tc_intra_discriminators.values():
                    disc_params.extend(disc.parameters())
            if self.pc_inter_enabled:
                for disc in self.pc_inter_discriminators.values():
                    disc_params.extend(disc.parameters())
            if self.tc_unified_enabled and self.tc_unified_discriminator is not None:
                disc_params.extend(self.tc_unified_discriminator.parameters())
            if self.diversity_enabled and self.diversity_discriminator is not None:
                disc_params.extend(self.diversity_discriminator.parameters())

            # Apply gradient clipping to discriminators
            if self.clip_enabled:
                torch.nn.utils.clip_grad_norm_(disc_params, self.clip_max_norm)
            self.disc_optimizer.step()

        disc_losses["disc_total"] = disc_loss_total.item()
        return disc_losses

    def _compute_adversarial_losses(
        self,
        encodings: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Compute adversarial losses for VAE training.

        Args:
            encodings: Model encodings

        Returns:
            Tuple of (tc_losses, pc_losses) dictionaries
        """
        # TC Intra losses
        if self.tc_intra_enabled:
            tc_intra_total = torch.tensor(0.0, device=self.device)
            for level in self.level_names:
                disc = self.tc_intra_discriminators[level]
                for enc_key in self.encoder_names:
                    z = encodings[enc_key][level]["z"]
                    _, vae_loss = disc.compute_tc_loss(z, train_discriminator=False)
                    tc_intra_total = tc_intra_total + vae_loss
        else:
            tc_intra_total = torch.tensor(0.0, device=self.device)

        # TC Unified loss
        if self.tc_unified_enabled and self.tc_unified_discriminator is not None:
            z_unified = encodings["unified"]["z"]
            _, tc_unified = self.tc_unified_discriminator.compute_tc_loss(
                z_unified, train_discriminator=False
            )
        else:
            tc_unified = torch.tensor(0.0, device=self.device)

        tc_losses = {
            "tc_intra": tc_intra_total,
            "tc_unified": tc_unified,
        }

        # PC Inter losses
        if self.pc_inter_enabled:
            pc_inter_total = torch.tensor(0.0, device=self.device)
            for level in self.level_names:
                disc = self.pc_inter_discriminators[level]
                encoder_outputs = [
                    encodings[enc_key][level]["z"]
                    for enc_key in self.encoder_names
                ]
                _, vae_loss = disc.compute_pc_loss(encoder_outputs, train_discriminator=False)
                pc_inter_total = pc_inter_total + vae_loss
        else:
            pc_inter_total = torch.tensor(0.0, device=self.device)

        pc_losses = {
            "pc_inter": pc_inter_total,
        }

        return tc_losses, pc_losses

    def _record_warmup_kl(self, losses: dict[str, torch.Tensor]):
        """Record KL values during warmup for capacity scheduler (per level, aggregated across encoders)."""
        for enc_name in self.encoder_names:
            for level in self.level_names:
                raw_kl_key = f"{enc_name}_raw_kl_{level}"
                if raw_kl_key in losses:
                    kl_value = losses[raw_kl_key].item() if torch.is_tensor(losses[raw_kl_key]) else losses[raw_kl_key]
                    self.capacity_scheduler.record_kl(level, kl_value, self.current_epoch)

        # Record unified KL
        if "raw_kl_unified" in losses:
            kl_value = losses["raw_kl_unified"].item() if torch.is_tensor(losses["raw_kl_unified"]) else losses["raw_kl_unified"]
            self.capacity_scheduler.record_kl("unified", kl_value, self.current_epoch)

    def _update_beta_controllers(self, losses: dict[str, torch.Tensor]):
        """Update beta controllers based on current per-encoder RAW KL values (not loss)."""
        if not self.beta_controller.enabled:
            return

        # Skip beta updates during warmup (betas stay at initial values)
        if self.capacity_scheduler.in_warmup(self.current_epoch):
            return

        # Get current capacities from scheduler
        capacities = self.capacity_scheduler.get_all_capacities(self.current_epoch, self.model.num_encoders)

        # Update per-encoder, per-level controllers using RAW KL (not |KL-C| loss)
        for enc_name in self.encoder_names:
            for level in self.level_names:
                raw_kl_key = f"{enc_name}_raw_kl_{level}"
                if raw_kl_key in losses:
                    current_kl = losses[raw_kl_key].item() if torch.is_tensor(losses[raw_kl_key]) else losses[raw_kl_key]
                    capacity_key = f"{enc_name}_{level}"
                    target = capacities[capacity_key]
                    self.beta_controller.update_encoder_level(enc_name, level, current_kl, target)

        # Update unified controller using RAW KL
        if "raw_kl_unified" in losses:
            current_kl = losses["raw_kl_unified"].item() if torch.is_tensor(losses["raw_kl_unified"]) else losses["raw_kl_unified"]
            target = capacities["unified"]
            self.beta_controller.update_unified(current_kl, target)

    def train_epoch(
        self,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        total_epochs: int | None = None,
    ) -> tuple[dict[str, float], dict[str, float] | None, bool]:
        """
        Train for one epoch with optional validation.

        Args:
            dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            total_epochs: Total epochs for display (optional)

        Returns:
            Tuple of (train_metrics, val_metrics, should_stop)
            - train_metrics: Dictionary of average training metrics
            - val_metrics: Dictionary of validation metrics (None if no val loader)
            - should_stop: True if early stopping triggered
        """
        self.model.train()
        epoch_metrics = {}
        num_batches = 0
        epoch_start = time.time()

        for batch in dataloader:
            metrics = self.train_step(batch)

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            num_batches += 1

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        epoch_time = time.time() - epoch_start
        self.current_epoch += 1

        # Validation
        val_metrics = None
        should_stop = False

        if val_dataloader is not None:
            val_metrics = self.validate(val_dataloader)
            # Use configurable early stopping metric
            val_loss = val_metrics.get(
                "val_early_stopping",
                val_metrics.get("val_total", val_metrics.get("val_recon", float("inf")))
            )

            # Early stopping check (only after capacity warmup ends)
            in_warmup = self.capacity_scheduler.in_warmup(self.current_epoch)
            if self.early_stopping_enabled and not in_warmup:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if self.save_best:
                        # Save best model state (use _original_model for torch.compile compat)
                        self.best_model_state = {
                            k: v.cpu().clone() for k, v in self._original_model.state_dict().items()
                        }
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        should_stop = True
                        logger.info(f"Early stopping triggered at epoch {self.current_epoch}")

        # Log epoch metrics (includes latent monitoring)
        self._log_epoch_metrics(epoch_metrics, val_metrics, total_epochs, epoch_time)

        # Step schedulers
        self.model_scheduler.step()
        if self.disc_scheduler is not None:
            self.disc_scheduler.step()

        return epoch_metrics, val_metrics, should_stop

    def restore_best_model(self):
        """Restore the best model state if save_best was enabled."""
        if self.best_model_state is not None:
            # Use _original_model for state_dict operations (works with torch.compile)
            self._original_model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model (val_loss={self.best_val_loss:.4f})")

    def train(
        self,
        train_data,
        message_db: dict,
        checkpoint_path: str,
    ):
        """
        Full training loop for Modal cloud deployment.

        Args:
            train_data: Training data as numpy array (n_samples, features)
            message_db: Message database dict with metadata
            checkpoint_path: Path to save checkpoints
        """
        import numpy as np
        from src.data.datasets import create_dataset

        # Store message_db reference for checkpoint metadata
        self.message_db = message_db

        # Update metadata from message_db
        if "metadata" in message_db:
            self.metadata.update(message_db["metadata"])

        training_config = self.config["training"]
        total_epochs = training_config["epochs"]
        val_split = training_config.get("validation_split", 0.1)
        batch_size = training_config.get("batch_size", 32)

        # Split data
        n_samples = len(train_data)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_subset = train_data[train_indices]
        val_subset = train_data[val_indices] if n_val > 0 else None

        logger.info(f"Training with {len(train_indices)} samples, validating with {n_val} samples")

        # Create data loaders
        batching_config = self.config.get("batching", {})
        batching_mode = batching_config.get("mode", "random")

        # For cloud training, use simple random batching
        train_loader = self._create_simple_dataloader(train_subset, batch_size, shuffle=True)
        val_loader = self._create_simple_dataloader(val_subset, batch_size, shuffle=False) if val_subset is not None else None

        logger.info(f"Starting training for {total_epochs} epochs")

        for epoch in range(total_epochs):
            # Check stop signal
            if self.stop_signal is not None and self.stop_signal.is_set():
                logger.info("Stop signal received, ending training")
                break

            # Train one epoch
            train_metrics, val_metrics, should_stop = self.train_epoch(
                train_loader,
                val_loader,
                total_epochs=total_epochs,
            )

            # Determine if this is the best epoch
            is_best = self.patience_counter == 0 and self.current_epoch > 1

            # Call epoch callback if provided
            if self.on_epoch_end_callback is not None:
                all_metrics = {**train_metrics}
                if val_metrics:
                    all_metrics.update(val_metrics)
                self.on_epoch_end_callback(self.current_epoch, all_metrics, is_best)

            # Save checkpoint if best
            if is_best or self.current_epoch == 1:
                self.save_checkpoint(checkpoint_path, include_optimizer_state=False)

            if should_stop:
                logger.info("Early stopping triggered")
                break

        # Restore best model and save final checkpoint
        self.restore_best_model()
        self.save_checkpoint(checkpoint_path, include_optimizer_state=False)

        logger.info(f"Training complete. Best val_loss: {self.best_val_loss:.4f}")

    def _create_simple_dataloader(self, data, batch_size: int, shuffle: bool = True):
        """Create a simple DataLoader from numpy array.

        Returns batches as dicts with "embeddings" key to match train_step expectations.
        """
        import torch
        from torch.utils.data import DataLoader, Dataset

        class SimpleDataset(Dataset):
            def __init__(self, data_array):
                self.data = torch.from_numpy(data_array).float()

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {"embeddings": self.data[idx]}

        def collate_fn(batch):
            return {"embeddings": torch.stack([b["embeddings"] for b in batch])}

        dataset = SimpleDataset(data)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def _log_epoch_metrics(
        self,
        metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
        total_epochs: int | None = None,
        epoch_time: float | None = None,
    ):
        """Log structured per-epoch metrics with latent monitoring."""
        def fmt(v: float) -> str:
            """Format value for display. Uses k/m suffixes only when > 4 digits."""
            abs_v = abs(v)
            if abs_v >= 1_000_000:  # 1m+, use m suffix
                if abs_v >= 100_000_000:
                    return f"{v / 1_000_000:.0f}m"
                elif abs_v >= 10_000_000:
                    return f"{v / 1_000_000:.1f}m"
                else:
                    return f"{v / 1_000_000:.2f}m"
            elif abs_v >= 10_000:  # 10k-999k, use k suffix
                if abs_v >= 100_000:
                    return f"{v / 1_000:.1f}k"
                else:
                    return f"{v / 1_000:.2f}k"
            elif abs_v >= 100:  # 100-9999
                return f"{v:.1f}"
            elif abs_v >= 10:  # 10-99
                return f"{v:.2f}"
            elif abs_v >= 1:  # 1-9
                return f"{v:.3f}"
            elif abs_v == 0:
                return "0"
            else:  # < 1
                return f"{v:.4f}"

        epoch_str = f"Epoch {self.current_epoch}"
        if total_epochs:
            epoch_str += f"/{total_epochs}"
        if epoch_time is not None:
            epoch_str += f" ({epoch_time:.1f}s)"

        # Get loss weights for computing weighted contributions
        w = self.config["loss_weights"]
        decoder_type = self.config["model"]["decoder"]["type"]

        # Build log parts with new format: Component[train:weighted(base) val:weighted(base)]
        log_parts = [epoch_str]

        # Helper to format train/val pair: show "weighted(raw)" only if weight != 1
        def fmt_train_val(label: str, train_raw: float, val_raw: float | None, weight: float) -> str:
            train_weighted = weight * train_raw
            if val_raw is not None:
                val_weighted = weight * val_raw
                if weight == 1.0:
                    return f"{label}[train:{fmt(train_raw)} val:{fmt(val_raw)}]"
                else:
                    return f"{label}[train:{fmt(train_weighted)}({fmt(train_raw)}) val:{fmt(val_weighted)}({fmt(val_raw)})]"
            else:
                if weight == 1.0:
                    return f"{label}[train:{fmt(train_raw)}]"
                else:
                    return f"{label}[train:{fmt(train_weighted)}({fmt(train_raw)})]"

        # Get recon sub-weights
        recon_w = w["recon"]

        # Flow loss (velocity MSE for flow decoders)
        if decoder_type in ("flow_matching", "spherical_flow_matching"):
            train_flow = metrics.get("flow", 0)
            val_flow = val_metrics.get("val_flow", None) if val_metrics else None
            log_parts.append(fmt_train_val("Flow", train_flow, val_flow, recon_w["flow"]))

        # Base reconstruction loss (MLP MSE or diff ODE recon)
        train_base_recon = metrics.get("base_recon", 0)
        val_base_recon = val_metrics.get("val_base_recon", None) if val_metrics else None
        if train_base_recon > 0 or (val_base_recon is not None and val_base_recon > 0):
            log_parts.append(fmt_train_val("Recon", train_base_recon, val_base_recon, recon_w["base"]))

        # Focal loss (additional penalty for hard samples)
        if self.config["focal_loss"]["enabled"]:
            train_focal = metrics.get("focal", 0)
            val_focal = val_metrics.get("val_focal", None) if val_metrics else None
            if train_focal != 0:  # Only show if non-zero
                log_parts.append(fmt_train_val("Focal", train_focal, val_focal, recon_w["focal"]))

        # TC - combined intra + unified (only show if any TC discriminator enabled)
        if self.tc_intra_enabled or self.tc_unified_enabled:
            tc_intra_raw = metrics.get("tc_intra", 0)
            tc_unified_raw = metrics.get("tc_unified", 0)
            tc_total_raw = tc_intra_raw + tc_unified_raw
            # Combined weight (average of the two weights for display purposes)
            tc_avg_weight = (w["tc_intra"] + w["tc_unified"]) / 2
            tc_total_weighted = w["tc_intra"] * tc_intra_raw + w["tc_unified"] * tc_unified_raw
            # Get validation values
            val_tc_intra = val_metrics.get("val_tc_intra", 0) if val_metrics else 0
            val_tc_unified = val_metrics.get("val_tc_unified", 0) if val_metrics else 0
            val_tc_total_raw = val_tc_intra + val_tc_unified if val_metrics else None
            val_tc_total_weighted = (w["tc_intra"] * val_tc_intra + w["tc_unified"] * val_tc_unified) if val_metrics else None
            if val_tc_total_raw is not None:
                if tc_avg_weight == 1.0:
                    log_parts.append(f"TC[train:{fmt(tc_total_raw)} val:{fmt(val_tc_total_raw)}]")
                else:
                    log_parts.append(f"TC[train:{fmt(tc_total_weighted)}({fmt(tc_total_raw)}) val:{fmt(val_tc_total_weighted)}({fmt(val_tc_total_raw)})]")
            else:
                if tc_avg_weight == 1.0:
                    log_parts.append(f"TC[train:{fmt(tc_total_raw)}]")
                else:
                    log_parts.append(f"TC[train:{fmt(tc_total_weighted)}({fmt(tc_total_raw)})]")

        # PC (only show if PC discriminator enabled)
        if self.pc_inter_enabled:
            pc_raw = metrics.get("pc_inter", 0)
            val_pc_raw = val_metrics.get("val_pc_inter", None) if val_metrics else None
            log_parts.append(fmt_train_val("PC", pc_raw, val_pc_raw, w["pc_inter"]))

        # IWO - combined intra + unified (only show if IWO enabled)
        if self.loss_fn.iwo_enabled:
            iwo_intra_raw = metrics.get("iwo_intra", 0)
            iwo_unified_raw = metrics.get("iwo_unified", 0)
            iwo_total_raw = iwo_intra_raw + iwo_unified_raw
            iwo_total_weighted = w["iwo_intra"] * iwo_intra_raw + w["iwo_unified"] * iwo_unified_raw
            iwo_avg_weight = (w["iwo_intra"] + w["iwo_unified"]) / 2
            # Get validation values
            val_iwo_intra = val_metrics.get("val_iwo_intra", 0) if val_metrics else 0
            val_iwo_unified = val_metrics.get("val_iwo_unified", 0) if val_metrics else 0
            val_iwo_total_raw = val_iwo_intra + val_iwo_unified if val_metrics else None
            val_iwo_total_weighted = (w["iwo_intra"] * val_iwo_intra + w["iwo_unified"] * val_iwo_unified) if val_metrics else None
            if val_iwo_total_raw is not None:
                if iwo_avg_weight == 1.0:
                    log_parts.append(f"IWO[train:{fmt(iwo_total_raw)} val:{fmt(val_iwo_total_raw)}]")
                else:
                    log_parts.append(f"IWO[train:{fmt(iwo_total_weighted)}({fmt(iwo_total_raw)}) val:{fmt(val_iwo_total_weighted)}({fmt(val_iwo_total_raw)})]")
            else:
                if iwo_avg_weight == 1.0:
                    log_parts.append(f"IWO[train:{fmt(iwo_total_raw)}]")
                else:
                    log_parts.append(f"IWO[train:{fmt(iwo_total_weighted)}({fmt(iwo_total_raw)})]")

        # Hoyer - show if enabled in config
        if self.config["hoyer"]["enabled"]:
            hoyer_raw = metrics.get("hoyer", 0)
            val_hoyer_raw = val_metrics.get("val_hoyer", None) if val_metrics else None
            log_parts.append(fmt_train_val("Hoyer", hoyer_raw, val_hoyer_raw, w["hoyer"]))

        # Cluster Separation - show if enabled in config
        if self.cluster_separation_loss.enabled:
            cluster_sep_raw = metrics.get("cluster_separation", 0)
            val_cluster_sep = val_metrics.get("val_cluster_separation", None) if val_metrics else None
            log_parts.append(fmt_train_val("ClustSep", cluster_sep_raw, val_cluster_sep, w["cluster_separation"]))

        # Range Regularization - show if enabled in config
        if self.range_regularization_loss.enabled:
            range_reg_raw = metrics.get("range_regularization", 0)
            val_range_reg = val_metrics.get("val_range_regularization", None) if val_metrics else None
            log_parts.append(fmt_train_val("RangeReg", range_reg_raw, val_range_reg, w["range_regularization"]))

        # Contrastive Memory - show if enabled in config (training-only, uses memory bank)
        if self.contrastive_memory_loss.enabled:
            contrastive_raw = metrics.get("contrastive_memory", 0)
            log_parts.append(fmt_train_val("Contrastive", contrastive_raw, None, w["contrastive_memory"]))

        # Entropy Uniformity - show if enabled in config
        if self.entropy_uniformity_loss.enabled:
            entropy_raw = metrics.get("entropy_uniformity", 0)
            val_entropy = val_metrics.get("val_entropy_uniformity", None) if val_metrics else None
            log_parts.append(fmt_train_val("Entropy", entropy_raw, val_entropy, w["entropy_uniformity"]))

        # Diversity Discriminator - show if enabled in config
        if self.diversity_enabled:
            diversity_raw = metrics.get("diversity", 0)
            log_parts.append(fmt_train_val("Diversity", diversity_raw, None, w["diversity_discriminator"]))

        # Training and validation totals at the end
        train_total = metrics.get("total", 0)
        if val_metrics:
            val_total = val_metrics.get("val_total", 0)
            log_parts.append(f"Train:{fmt(train_total)} | Val:{fmt(val_total)}")
        else:
            log_parts.append(f"Train:{fmt(train_total)}")

        # Build KL detail line with raw KL, capacity, and loss (if kl_stats enabled)
        kl_line = None
        if self.kl_stats_enabled:
            # Format: KL/C=Loss (e.g., 263/14=249 means raw KL is 263, capacity is 14, loss is 249)
            kl_detail_parts = []
            for enc_name in self.encoder_names:
                enc_kls = []
                for level in self.level_names:
                    raw_kl = metrics.get(f"{enc_name}_raw_kl_{level}", 0)
                    cap = metrics.get(f"{enc_name}_cap_{level}", 0)
                    loss = metrics.get(f"{enc_name}_kl_{level}", 0)
                    # Format: raw/cap=loss (compact)
                    enc_kls.append(f"{level[0].upper()}:{fmt(raw_kl)}-{fmt(cap)}={fmt(loss)}")
                kl_detail_parts.append(f"{enc_name}[{' '.join(enc_kls)}]")

            # Unified KL
            raw_kl_uni = metrics.get("raw_kl_unified", 0)
            cap_uni = metrics.get("cap_unified", 0)
            loss_uni = metrics.get("kl_unified", 0)
            kl_detail_parts.append(f"uni:{fmt(raw_kl_uni)}-{fmt(cap_uni)}={fmt(loss_uni)}")
            kl_line = f"KL(raw-cap=loss): {' | '.join(kl_detail_parts)}"

            # Add warmup countdown OR beta values at end of KL line
            # Warmup is controlled by capacity scheduler (no KL penalty during warmup)
            in_warmup = self.capacity_scheduler.in_warmup(self.current_epoch)
            if in_warmup:
                epochs_left = self.capacity_scheduler.warmup_epochs - self.current_epoch
                kl_line += f" | warmup: {epochs_left} epochs left (no KL penalty)"
            elif self.beta_controller.enabled:
                # Show beta values after warmup (dynamic levels)
                avg_betas = self.beta_controller.get_level_avg_betas()
                beta_parts = []
                for level in self.level_names:
                    beta_key = f"beta_{level}"
                    beta_parts.append(f"{level[0].upper()}:{fmt(avg_betas.get(beta_key, 1.0))}")
                beta_parts.append(f"U:{fmt(avg_betas.get('beta_unified', 1.0))}")
                kl_line += f" | β[{' '.join(beta_parts)}]"

            # Build KL dimension stats (shows distribution uniformity across dimensions)
            # High std indicates some dimensions doing more "work" than others
            kl_dim_parts = []
            for level in self.level_names:
                dim_mean = metrics.get(f"kl_dim_mean_{level}", None)
                dim_std = metrics.get(f"kl_dim_std_{level}", None)
                if dim_mean is not None and dim_std is not None:
                    kl_dim_parts.append(f"{level[0].upper()}[μ:{fmt(dim_mean)} σ:{fmt(dim_std)}]")
            # Add unified
            dim_mean = metrics.get("kl_dim_mean_unified", None)
            dim_std = metrics.get("kl_dim_std_unified", None)
            if dim_mean is not None and dim_std is not None:
                kl_dim_parts.append(f"U[μ:{fmt(dim_mean)} σ:{fmt(dim_std)}]")

            # Append KL dim stats to the same line as β values
            if kl_dim_parts:
                kl_line += f" | KL/dim: {' '.join(kl_dim_parts)}"

        # Build range stats line (if range_stats enabled and range regularization is active)
        range_line = None
        if self.range_stats_enabled and self.range_regularization_loss.enabled:
            # Format: enc1[B:v/d(min-avg-max)] where v=violations, d=dims, min/avg/max=range stats
            range_parts = []
            for enc_key in self.encoder_names:
                enc_levels = []
                for level in self.level_names:
                    v = int(metrics.get(f"range_violations_{enc_key}_{level}", 0))
                    d = int(metrics.get(f"range_dims_{enc_key}_{level}", 0))
                    r_min = metrics.get(f"range_min_{enc_key}_{level}", 0)
                    r_max = metrics.get(f"range_max_{enc_key}_{level}", 0)
                    r_mean = metrics.get(f"range_mean_{enc_key}_{level}", 0)
                    level_abbrev = level[0].upper()
                    enc_levels.append(f"{level_abbrev}:{v}/{d}({fmt(r_min)}-{fmt(r_mean)}-{fmt(r_max)})")
                range_parts.append(f"{enc_key}[{' '.join(enc_levels)}]")
            # Add unified
            v = int(metrics.get("range_violations_unified", 0))
            d = int(metrics.get("range_dims_unified", 0))
            r_min = metrics.get("range_min_unified", 0)
            r_max = metrics.get("range_max_unified", 0)
            r_mean = metrics.get("range_mean_unified", 0)
            range_parts.append(f"uni:{v}/{d}({fmt(r_min)}-{fmt(r_mean)}-{fmt(r_max)})")
            range_line = f"Range(v/d min-avg-max): {' '.join(range_parts)}"

        # Build latent monitoring line (if enabled)
        latent_line = self.latent_monitor.format_monitoring_line_compact() if self.latent_monitor.enabled else None

        # Build decoder metrics line (train/val side by side like other losses)
        decoder_line = None
        dec_parts = []

        def fmt_dec(label: str, train_val: float | None, val_val: float | None, is_percent: bool = False) -> str | None:
            """Format decoder metric with train/val side by side."""
            if train_val is None and val_val is None:
                return None
            if is_percent:
                t_str = f"{fmt(train_val * 100)}%" if train_val is not None else "-"
                v_str = f"{fmt(val_val * 100)}%" if val_val is not None else "-"
            else:
                t_str = fmt(train_val) if train_val is not None else "-"
                v_str = fmt(val_val) if val_val is not None else "-"
            if val_val is not None:
                return f"{label}[t:{t_str} v:{v_str}]"
            else:
                return f"{label}[t:{t_str}]"

        if decoder_type in ("flow_matching", "spherical_flow_matching"):
            # Velocity metrics (flow matching training signal)
            train_vel_cos = metrics.get("decoder_velocity_cosine_sim", None)
            val_vel_cos = val_metrics.get("val_decoder_velocity_cosine_sim", None) if val_metrics else None
            if part := fmt_dec("vel_cos", train_vel_cos, val_vel_cos):
                dec_parts.append(part)

            train_dir = metrics.get("decoder_direction_loss", None)
            val_dir = val_metrics.get("val_decoder_direction_loss", None) if val_metrics else None
            if part := fmt_dec("dir", train_dir, val_dir):
                dec_parts.append(part)

            train_improv = metrics.get("decoder_improvement_vs_zero", None)
            val_improv = val_metrics.get("val_decoder_improvement_vs_zero", None) if val_metrics else None
            if part := fmt_dec("↑zero", train_improv, val_improv, is_percent=True):
                dec_parts.append(part)

            # Reconstruction metrics (actual output quality)
            train_recon_cos = metrics.get("decoder_reconstruction_cosine_sim", None)
            val_recon_cos = val_metrics.get("val_decoder_reconstruction_cosine_sim", None) if val_metrics else None
            if part := fmt_dec("recon_cos", train_recon_cos, val_recon_cos):
                dec_parts.append(part)

            # Centroid baseline (train and val)
            train_centroid = metrics.get("decoder_improvement_vs_centroid", None)
            val_centroid = val_metrics.get("val_decoder_improvement_vs_centroid", None) if val_metrics else None
            if part := fmt_dec("↑centroid", train_centroid, val_centroid, is_percent=True):
                dec_parts.append(part)

        elif decoder_type == "mlp":
            # MSE metrics
            train_mse = metrics.get("decoder_mse_normalized", None) or metrics.get("decoder_mse_raw", None)
            val_mse = val_metrics.get("val_decoder_mse_normalized", None) if val_metrics else None
            if val_mse is None and val_metrics:
                val_mse = val_metrics.get("val_decoder_reconstruction_mse_total", None)
            if part := fmt_dec("mse", train_mse, val_mse):
                dec_parts.append(part)

            # Cosine similarity
            train_cos = metrics.get("decoder_cosine_sim", None)
            val_cos = val_metrics.get("val_decoder_reconstruction_cosine_sim", None) if val_metrics else None
            if part := fmt_dec("cos_sim", train_cos, val_cos):
                dec_parts.append(part)

        # Add early stopping info (ES components + patience)
        if val_metrics and self.early_stopping_enabled:
            es_components = val_metrics.get("val_es_components", {})
            if es_components:
                es_parts = []
                for name, value in es_components.items():
                    short_name = name.replace("reconstruction_", "").replace("velocity_", "vel_")
                    es_parts.append(f"{short_name}:{fmt(value)}")
                dec_parts.append(f"|| ES[{' '.join(es_parts)}]")
            dec_parts.append(f"Patience:{self.patience_counter}/{self.patience}")

        if dec_parts:
            decoder_line = f"Decoder: {' | '.join(dec_parts)}"

        # Build combined multi-line log
        log_lines = [
            "",  # Empty line after logger prefix
            f"Training: {' | '.join(log_parts)}",
        ]

        if decoder_line:
            log_lines.append(f"| {decoder_line}")

        if kl_line:
            log_lines.append(f"| {kl_line}")

        if range_line:
            log_lines.append(f"| {range_line}")

        if latent_line:
            log_lines.append(f"Latent: {latent_line}")

        logger.info("\n".join(log_lines))

        # Health warnings (separate)
        warnings = self.latent_monitor.check_health()
        for warning in warnings:
            logger.warning(f"LATENT HEALTH: {warning}")

        # Log to wandb if enabled
        if self.wandb_enabled and self.wandb_run is not None:
            self._log_wandb_metrics(metrics, val_metrics)

    def _log_wandb_metrics(
        self,
        metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Log comprehensive metrics to Weights & Biases.

        Logs all training metrics including:
        - Main training losses (total, reconstruction, KL, TC, PC, IWO, etc.)
        - Decoder-specific metrics (flow matching or MLP)
        - KL statistics (per-encoder, per-level, beta values, dimension stats)
        - Range regularization statistics
        - Latent space statistics from LatentMonitor
        - All validation metrics

        Args:
            metrics: Training metrics dictionary
            val_metrics: Optional validation metrics dictionary
        """
        import wandb

        decoder_type = self.config["model"]["decoder"]["type"]
        w = self.config["loss_weights"]
        recon_w = w["recon"]

        wandb_metrics: dict[str, float] = {
            "epoch": self.current_epoch,
        }

        # ============================================================
        # MAIN TRAINING LOSSES
        # ============================================================
        wandb_metrics["train/total"] = metrics.get("total", 0)
        wandb_metrics["train/kl_total"] = metrics.get("kl_total", 0)

        # Flow loss (velocity MSE for flow decoders)
        if decoder_type in ("flow_matching", "spherical_flow_matching"):
            flow_raw = metrics.get("flow", 0)
            wandb_metrics["train/flow_raw"] = flow_raw
            wandb_metrics["train/flow_weighted"] = flow_raw * recon_w["flow"]

        # Base reconstruction loss
        base_recon = metrics.get("base_recon", 0)
        if base_recon > 0:
            wandb_metrics["train/base_recon_raw"] = base_recon
            wandb_metrics["train/base_recon_weighted"] = base_recon * recon_w["base"]

        # Focal loss
        if self.config["focal_loss"]["enabled"]:
            focal = metrics.get("focal", 0)
            wandb_metrics["train/focal_raw"] = focal
            wandb_metrics["train/focal_weighted"] = focal * recon_w["focal"]

        # TC discriminator losses
        if self.tc_intra_enabled:
            tc_intra = metrics.get("tc_intra", 0)
            wandb_metrics["train/tc_intra_raw"] = tc_intra
            wandb_metrics["train/tc_intra_weighted"] = tc_intra * w["tc_intra"]
        if self.tc_unified_enabled:
            tc_unified = metrics.get("tc_unified", 0)
            wandb_metrics["train/tc_unified_raw"] = tc_unified
            wandb_metrics["train/tc_unified_weighted"] = tc_unified * w["tc_unified"]

        # PC discriminator loss
        if self.pc_inter_enabled:
            pc_inter = metrics.get("pc_inter", 0)
            wandb_metrics["train/pc_inter_raw"] = pc_inter
            wandb_metrics["train/pc_inter_weighted"] = pc_inter * w["pc_inter"]

        # IWO losses
        if self.loss_fn.iwo_enabled:
            iwo_intra = metrics.get("iwo_intra", 0)
            iwo_unified = metrics.get("iwo_unified", 0)
            wandb_metrics["train/iwo_intra_raw"] = iwo_intra
            wandb_metrics["train/iwo_intra_weighted"] = iwo_intra * w["iwo_intra"]
            wandb_metrics["train/iwo_unified_raw"] = iwo_unified
            wandb_metrics["train/iwo_unified_weighted"] = iwo_unified * w["iwo_unified"]

        # Hoyer sparsity loss
        if self.config["hoyer"]["enabled"]:
            hoyer = metrics.get("hoyer", 0)
            wandb_metrics["train/hoyer_raw"] = hoyer
            wandb_metrics["train/hoyer_weighted"] = hoyer * w["hoyer"]

        # Cluster separation loss
        if self.cluster_separation_loss.enabled:
            cluster_sep = metrics.get("cluster_separation", 0)
            wandb_metrics["train/cluster_separation_raw"] = cluster_sep
            wandb_metrics["train/cluster_separation_weighted"] = cluster_sep * w["cluster_separation"]

        # Range regularization loss
        if self.range_regularization_loss.enabled:
            range_reg = metrics.get("range_regularization", 0)
            wandb_metrics["train/range_regularization_raw"] = range_reg
            wandb_metrics["train/range_regularization_weighted"] = range_reg * w["range_regularization"]

        # Contrastive memory loss
        if self.contrastive_memory_loss.enabled:
            contrastive = metrics.get("contrastive_memory", 0)
            wandb_metrics["train/contrastive_memory_raw"] = contrastive
            wandb_metrics["train/contrastive_memory_weighted"] = contrastive * w["contrastive_memory"]

        # Entropy uniformity loss
        if self.entropy_uniformity_loss.enabled:
            entropy = metrics.get("entropy_uniformity", 0)
            wandb_metrics["train/entropy_uniformity_raw"] = entropy
            wandb_metrics["train/entropy_uniformity_weighted"] = entropy * w["entropy_uniformity"]

        # Diversity discriminator loss
        if self.diversity_enabled:
            diversity = metrics.get("diversity", 0)
            wandb_metrics["train/diversity_raw"] = diversity
            wandb_metrics["train/diversity_weighted"] = diversity * w["diversity_discriminator"]

        # ============================================================
        # KL STATISTICS (per-encoder, per-level)
        # ============================================================
        # Per-encoder per-level KL
        for enc_name in self.encoder_names:
            for level in self.level_names:
                raw_kl = metrics.get(f"{enc_name}_raw_kl_{level}", 0)
                cap = metrics.get(f"{enc_name}_cap_{level}", 0)
                kl_loss = metrics.get(f"{enc_name}_kl_{level}", 0)
                wandb_metrics[f"kl/{enc_name}_{level}_raw"] = raw_kl
                wandb_metrics[f"kl/{enc_name}_{level}_capacity"] = cap
                wandb_metrics[f"kl/{enc_name}_{level}_loss"] = kl_loss

        # Unified KL
        wandb_metrics["kl/unified_raw"] = metrics.get("raw_kl_unified", 0)
        wandb_metrics["kl/unified_capacity"] = metrics.get("cap_unified", 0)
        wandb_metrics["kl/unified_loss"] = metrics.get("kl_unified", 0)

        # Beta values (only after warmup)
        in_warmup = self.capacity_scheduler.in_warmup(self.current_epoch)
        wandb_metrics["kl/in_warmup"] = 1 if in_warmup else 0

        if not in_warmup and self.beta_controller.enabled:
            avg_betas = self.beta_controller.get_level_avg_betas()
            for level in self.level_names:
                beta_key = f"beta_{level}"
                wandb_metrics[f"beta/{level}"] = avg_betas.get(beta_key, 1.0)
            wandb_metrics["beta/unified"] = avg_betas.get("beta_unified", 1.0)

        # KL dimension statistics (distribution uniformity across dimensions)
        for level in self.level_names:
            dim_mean = metrics.get(f"kl_dim_mean_{level}")
            dim_std = metrics.get(f"kl_dim_std_{level}")
            if dim_mean is not None:
                wandb_metrics[f"kl_dim/{level}_mean"] = dim_mean
            if dim_std is not None:
                wandb_metrics[f"kl_dim/{level}_std"] = dim_std

        dim_mean_uni = metrics.get("kl_dim_mean_unified")
        dim_std_uni = metrics.get("kl_dim_std_unified")
        if dim_mean_uni is not None:
            wandb_metrics["kl_dim/unified_mean"] = dim_mean_uni
        if dim_std_uni is not None:
            wandb_metrics["kl_dim/unified_std"] = dim_std_uni

        # ============================================================
        # RANGE REGULARIZATION STATISTICS
        # ============================================================
        if self.range_regularization_loss.enabled and self.range_stats_enabled:
            # Per-encoder per-level range stats
            for enc_key in self.encoder_names:
                for level in self.level_names:
                    violations = metrics.get(f"range_violations_{enc_key}_{level}", 0)
                    dims = metrics.get(f"range_dims_{enc_key}_{level}", 0)
                    r_min = metrics.get(f"range_min_{enc_key}_{level}", 0)
                    r_max = metrics.get(f"range_max_{enc_key}_{level}", 0)
                    r_mean = metrics.get(f"range_mean_{enc_key}_{level}", 0)

                    wandb_metrics[f"range/{enc_key}_{level}_violations"] = violations
                    wandb_metrics[f"range/{enc_key}_{level}_dims"] = dims
                    wandb_metrics[f"range/{enc_key}_{level}_min"] = r_min
                    wandb_metrics[f"range/{enc_key}_{level}_max"] = r_max
                    wandb_metrics[f"range/{enc_key}_{level}_mean"] = r_mean
                    if dims > 0:
                        wandb_metrics[f"range/{enc_key}_{level}_violation_ratio"] = violations / dims

            # Unified range stats
            wandb_metrics["range/unified_violations"] = metrics.get("range_violations_unified", 0)
            wandb_metrics["range/unified_dims"] = metrics.get("range_dims_unified", 0)
            wandb_metrics["range/unified_min"] = metrics.get("range_min_unified", 0)
            wandb_metrics["range/unified_max"] = metrics.get("range_max_unified", 0)
            wandb_metrics["range/unified_mean"] = metrics.get("range_mean_unified", 0)
            unified_dims = metrics.get("range_dims_unified", 0)
            if unified_dims > 0:
                wandb_metrics["range/unified_violation_ratio"] = (
                    metrics.get("range_violations_unified", 0) / unified_dims
                )

        # ============================================================
        # DECODER METRICS
        # ============================================================
        if decoder_type in ("flow_matching", "spherical_flow_matching"):
            # Velocity metrics (flow matching training signal)
            if "decoder_velocity_mse_normalized" in metrics:
                wandb_metrics["decoder/velocity_mse_normalized"] = metrics["decoder_velocity_mse_normalized"]
            if "decoder_velocity_cosine_sim" in metrics:
                wandb_metrics["decoder/velocity_cosine_sim"] = metrics["decoder_velocity_cosine_sim"]
            if "decoder_direction_loss" in metrics:
                wandb_metrics["decoder/direction_loss"] = metrics["decoder_direction_loss"]
            if "decoder_improvement_vs_zero" in metrics:
                wandb_metrics["decoder/improvement_vs_zero"] = metrics["decoder_improvement_vs_zero"]

            # Reconstruction metrics
            if "decoder_reconstruction_cosine_sim" in metrics:
                wandb_metrics["decoder/reconstruction_cosine_sim"] = metrics["decoder_reconstruction_cosine_sim"]
            if "decoder_improvement_vs_centroid" in metrics:
                wandb_metrics["decoder/improvement_vs_centroid"] = metrics["decoder_improvement_vs_centroid"]

        elif decoder_type == "mlp":
            if "decoder_mse_normalized" in metrics:
                wandb_metrics["decoder/mse_normalized"] = metrics["decoder_mse_normalized"]
            elif "decoder_mse_raw" in metrics:
                wandb_metrics["decoder/mse_raw"] = metrics["decoder_mse_raw"]
            if "decoder_cosine_sim" in metrics:
                wandb_metrics["decoder/cosine_sim"] = metrics["decoder_cosine_sim"]

        # ============================================================
        # LATENT SPACE STATISTICS (from LatentMonitor)
        # ============================================================
        if self.latent_monitor.enabled and self.latent_monitor.last_tensor_stats:
            stats = self.latent_monitor.last_tensor_stats

            # Input tensor stats
            if "input" in stats:
                wandb_metrics["latent/input_min"] = stats["input"]["min"]
                wandb_metrics["latent/input_max"] = stats["input"]["max"]
                wandb_metrics["latent/input_mean"] = stats["input"]["mean"]
                wandb_metrics["latent/input_std"] = stats["input"]["std"]

            # Output tensor stats
            if "output" in stats:
                wandb_metrics["latent/output_min"] = stats["output"]["min"]
                wandb_metrics["latent/output_max"] = stats["output"]["max"]
                wandb_metrics["latent/output_mean"] = stats["output"]["mean"]
                wandb_metrics["latent/output_std"] = stats["output"]["std"]

            # Per-encoder per-level latent stats
            for enc_name in self.encoder_names:
                for level in self.level_names:
                    key = f"{enc_name}_{level}"
                    if key in stats:
                        s = stats[key]
                        wandb_metrics[f"latent/{key}_min"] = s["min"]
                        wandb_metrics[f"latent/{key}_max"] = s["max"]
                        wandb_metrics[f"latent/{key}_mean"] = s["mean"]
                        wandb_metrics[f"latent/{key}_std"] = s["std"]
                        wandb_metrics[f"latent/{key}_active"] = s.get("active", 0)
                        wandb_metrics[f"latent/{key}_total"] = s.get("total", 0)
                        total = s.get("total", 0)
                        if total > 0:
                            wandb_metrics[f"latent/{key}_active_ratio"] = s.get("active", 0) / total

            # Unified latent stats
            if "unified" in stats:
                s = stats["unified"]
                wandb_metrics["latent/unified_min"] = s["min"]
                wandb_metrics["latent/unified_max"] = s["max"]
                wandb_metrics["latent/unified_mean"] = s["mean"]
                wandb_metrics["latent/unified_std"] = s["std"]
                wandb_metrics["latent/unified_active"] = s.get("active", 0)
                wandb_metrics["latent/unified_total"] = s.get("total", 0)
                total = s.get("total", 0)
                if total > 0:
                    wandb_metrics["latent/unified_active_ratio"] = s.get("active", 0) / total

            # Aggregated per-level stats (averaged across encoders)
            for level in self.level_names:
                means, stds = [], []
                for enc_name in self.encoder_names:
                    key = f"{enc_name}_{level}"
                    if key in stats:
                        means.append(stats[key]["mean"])
                        stds.append(stats[key]["std"])
                if means:
                    wandb_metrics[f"latent/level_{level}_avg_mean"] = sum(means) / len(means)
                    wandb_metrics[f"latent/level_{level}_avg_std"] = sum(stds) / len(stds)

        # ============================================================
        # VALIDATION METRICS
        # ============================================================
        if val_metrics:
            wandb_metrics["val/total"] = val_metrics.get("val_total", 0)
            wandb_metrics["val/early_stopping"] = val_metrics.get("val_early_stopping", 0)
            wandb_metrics["patience"] = self.patience_counter

            # Validation flow loss
            if decoder_type in ("flow_matching", "spherical_flow_matching"):
                val_flow = val_metrics.get("val_flow")
                if val_flow is not None:
                    wandb_metrics["val/flow_raw"] = val_flow

            # Validation base reconstruction
            val_base_recon = val_metrics.get("val_base_recon")
            if val_base_recon is not None:
                wandb_metrics["val/base_recon_raw"] = val_base_recon

            # Validation focal loss
            if self.config["focal_loss"]["enabled"]:
                val_focal = val_metrics.get("val_focal")
                if val_focal is not None:
                    wandb_metrics["val/focal_raw"] = val_focal

            # Validation TC/PC/IWO
            if self.tc_intra_enabled:
                val_tc_intra = val_metrics.get("val_tc_intra")
                if val_tc_intra is not None:
                    wandb_metrics["val/tc_intra_raw"] = val_tc_intra
            if self.tc_unified_enabled:
                val_tc_unified = val_metrics.get("val_tc_unified")
                if val_tc_unified is not None:
                    wandb_metrics["val/tc_unified_raw"] = val_tc_unified
            if self.pc_inter_enabled:
                val_pc_inter = val_metrics.get("val_pc_inter")
                if val_pc_inter is not None:
                    wandb_metrics["val/pc_inter_raw"] = val_pc_inter
            if self.loss_fn.iwo_enabled:
                val_iwo_intra = val_metrics.get("val_iwo_intra")
                val_iwo_unified = val_metrics.get("val_iwo_unified")
                if val_iwo_intra is not None:
                    wandb_metrics["val/iwo_intra_raw"] = val_iwo_intra
                if val_iwo_unified is not None:
                    wandb_metrics["val/iwo_unified_raw"] = val_iwo_unified

            # Validation Hoyer
            if self.config["hoyer"]["enabled"]:
                val_hoyer = val_metrics.get("val_hoyer")
                if val_hoyer is not None:
                    wandb_metrics["val/hoyer_raw"] = val_hoyer

            # Validation cluster separation
            if self.cluster_separation_loss.enabled:
                val_cluster_sep = val_metrics.get("val_cluster_separation")
                if val_cluster_sep is not None:
                    wandb_metrics["val/cluster_separation_raw"] = val_cluster_sep

            # Validation range regularization
            if self.range_regularization_loss.enabled:
                val_range_reg = val_metrics.get("val_range_regularization")
                if val_range_reg is not None:
                    wandb_metrics["val/range_regularization_raw"] = val_range_reg

            # Validation entropy uniformity
            if self.entropy_uniformity_loss.enabled:
                val_entropy = val_metrics.get("val_entropy_uniformity")
                if val_entropy is not None:
                    wandb_metrics["val/entropy_uniformity_raw"] = val_entropy

            # Validation decoder metrics
            if decoder_type in ("flow_matching", "spherical_flow_matching"):
                decoder_val_keys = [
                    "val_decoder_velocity_cosine_sim",
                    "val_decoder_direction_loss",
                    "val_decoder_improvement_vs_zero",
                    "val_decoder_reconstruction_cosine_sim",
                    "val_decoder_improvement_vs_centroid",
                ]
                for key in decoder_val_keys:
                    val = val_metrics.get(key)
                    if val is not None:
                        # Convert key: val_decoder_X -> val/decoder_X
                        wandb_key = key.replace("val_", "val/")
                        wandb_metrics[wandb_key] = val

            elif decoder_type == "mlp":
                val_mse = val_metrics.get("val_decoder_mse_normalized") or val_metrics.get(
                    "val_decoder_reconstruction_mse_total"
                )
                if val_mse is not None:
                    wandb_metrics["val/decoder_mse"] = val_mse
                val_cos = val_metrics.get("val_decoder_reconstruction_cosine_sim")
                if val_cos is not None:
                    wandb_metrics["val/decoder_cosine_sim"] = val_cos

            # Early stopping components (detailed breakdown)
            es_components = val_metrics.get("val_es_components", {})
            for name, value in es_components.items():
                wandb_metrics[f"val/es_{name}"] = value

        wandb.log(wandb_metrics)

    def validate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """
        Validate on a dataset.

        Computes two types of metrics:
        1. val_flow_loss: Same velocity MSE as training (for direct comparison)
        2. Decoder validation metrics: Actual reconstruction quality via ODE solve

        Early stopping uses the decoder validation metric configured in
        decoder_validation.<decoder_type>.early_stopping_metric (e.g., reconstruction_cosine_sim).

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of average validation metrics including:
            - val_flow_loss: Flow matching loss (same as training)
            - val_decoder_*: Decoder-specific validation metrics
            - val_early_stopping: Value used for early stopping
        """
        self.model.eval()
        val_metrics = {}
        num_batches = 0

        # Get validation config from decoder_validation based on decoder type
        decoder_type = self.config["model"]["decoder"]["type"]
        decoder_val_config = self.config["decoder_validation"][decoder_type]

        # Early stopping config: dict of metric_name -> {weight, mode}
        es_config = decoder_val_config["early_stopping"]

        # Use EMA weights for validation (more stable)
        self.decoder_ema.apply_shadow()

        try:
            with torch.no_grad():
                for batch in dataloader:
                    # Forward pass to get unified_z and x_embedding
                    raw_losses, encodings = self._forward_and_compute_loss(
                        batch,
                        flow_loss_mode="training",  # Always use training mode for comparable loss
                    )

                    # Compute decoder validation metrics (includes actual reconstruction)
                    unified_z = encodings["unified_z"]
                    x_embedding = encodings["x_embedding"]
                    decoder_val_output = self.model.decoder.compute_validation_metrics(
                        unified_z, x_embedding
                    )

                    # Add decoder validation metrics (prefixed with val_decoder_)
                    for key, value in decoder_val_output.metrics.items():
                        if torch.is_tensor(value):
                            value = value.item()
                        val_key = f"val_decoder_{key}"
                        val_metrics[val_key] = val_metrics.get(val_key, 0.0) + value

                    # Store the total validation loss (same components as training)
                    total_loss = raw_losses.get("total", 0)
                    if torch.is_tensor(total_loss):
                        total_loss = total_loss.item()
                    val_metrics["val_total"] = val_metrics.get("val_total", 0.0) + total_loss

                    # Also store the flow loss for comparison with training
                    flow_loss = raw_losses.get("flow_loss", raw_losses.get("reconstruction", 0))
                    if torch.is_tensor(flow_loss):
                        flow_loss = flow_loss.item()
                    val_metrics["val_flow_loss"] = val_metrics.get("val_flow_loss", 0.0) + flow_loss

                    # Extract all loss components for logging (same as training)
                    loss_keys_to_extract = [
                        "flow", "base_recon", "focal",
                        "tc_intra", "tc_unified", "pc_inter",
                        "iwo_intra", "iwo_unified",
                        "hoyer", "kl_balance", "kl_total", "kl_unified",
                    ]
                    for key in loss_keys_to_extract:
                        if key in raw_losses:
                            value = raw_losses[key]
                            if torch.is_tensor(value):
                                value = value.item()
                            val_key = f"val_{key}"
                            val_metrics[val_key] = val_metrics.get(val_key, 0.0) + value

                    # Compute regularization losses (range_reg, entropy, cluster_sep)
                    # Now included in validation for parity with training loss
                    x = batch["embeddings"].to(self.device)
                    reg_losses = self._compute_regularization_losses(encodings, x)
                    for key, value in reg_losses.items():
                        if torch.is_tensor(value):
                            value = value.item()
                        val_key = f"val_{key}"
                        val_metrics[val_key] = val_metrics.get(val_key, 0.0) + value

                        # Add weighted contribution to total
                        weight_attr = f"{key}_weight"
                        weight = getattr(self, weight_attr, 1.0)
                        val_metrics["val_total"] = val_metrics.get("val_total", 0.0) + weight * value

                    num_batches += 1

            # Average all metrics
            for key in val_metrics:
                val_metrics[key] /= num_batches

            # Add alias for DiffODE loss (decoder reconstruction MSE)
            # The log code expects "val_diff_ode_loss" but decoder metrics use "val_decoder_reconstruction_mse_total"
            if "val_decoder_reconstruction_mse_total" in val_metrics:
                val_metrics["val_diff_ode_loss"] = val_metrics["val_decoder_reconstruction_mse_total"]

            # Compute weighted early stopping value from configured metrics
            # Each metric has: weight (relative importance), mode ("max" or "min")
            # Final ES value is normalized so lower = better for patience comparison
            es_value = 0.0
            total_weight = 0.0
            es_components = {}

            for metric_name, metric_cfg in es_config.items():
                weight = metric_cfg["weight"]
                if weight <= 0:
                    continue

                mode = metric_cfg["mode"]
                val_key = f"val_decoder_{metric_name}"

                if val_key in val_metrics:
                    raw_value = val_metrics[val_key]
                    # Normalize: for "max" mode, negate so lower = better
                    normalized = -raw_value if mode == "max" else raw_value
                    es_value += weight * normalized
                    total_weight += weight
                    es_components[metric_name] = raw_value

            if total_weight > 0:
                es_value /= total_weight
            else:
                # Fallback to flow loss if no valid metrics
                es_value = val_metrics.get("val_flow_loss", float("inf"))

            val_metrics["val_early_stopping"] = es_value
            val_metrics["val_es_components"] = es_components

        finally:
            # Restore original weights for continued training
            self.decoder_ema.restore()

        return val_metrics

    def save_checkpoint(self, path: str | Path, include_optimizer_state: bool = False):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
            include_optimizer_state: If True, includes optimizer state dicts for training
                resumption. If False (default), saves model-only checkpoint (smaller file,
                suitable for inference). Matches EPP checkpoint format when False.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Core model state (always saved)
        # Use _original_model for state_dict to ensure consistent keys
        # (torch.compile changes the model's state_dict structure)
        checkpoint = {
            "model_state_dict": self._original_model.state_dict(),
            "tc_intra_discriminators": {
                k: v.state_dict() for k, v in self.tc_intra_discriminators.items()
            } if self.tc_intra_discriminators else {},
            "pc_inter_discriminators": {
                k: v.state_dict() for k, v in self.pc_inter_discriminators.items()
            } if self.pc_inter_discriminators else {},
            "tc_unified_discriminator": self.tc_unified_discriminator.state_dict()
                if self.tc_unified_discriminator is not None else None,
            "decoder_ema": self.decoder_ema.state_dict()
                if self.decoder_ema is not None else None,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "config": self.config,
            "preprocessing": {
                "normalization": self.normalization_params,
            },
            # Metadata from message_db (includes embedder info)
            "metadata": self.metadata,
        }

        # Add embedder version if embedder info is present
        if "embedder" in self.metadata:
            embedder_info = self.metadata["embedder"]
            model_name = embedder_info.get("model_name", "")
            checkpoint["embedder"] = {
                **embedder_info,
                "model_version": self._get_embedder_version(model_name),
            }

        # Optimizer state (optional, significantly increases file size)
        if include_optimizer_state:
            checkpoint["model_optimizer_state_dict"] = self.model_optimizer.state_dict()
            checkpoint["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
            checkpoint["beta_controller"] = self.beta_controller.state_dict()
            checkpoint["capacity_scheduler"] = self.capacity_scheduler.state_dict()
            checkpoint["amp_manager"] = self.amp_manager.state_dict()
            checkpoint["model_scheduler_state_dict"] = self.model_scheduler.state_dict()
            checkpoint["disc_scheduler_state_dict"] = self.disc_scheduler.state_dict()

        torch.save(checkpoint, path)
        mode = "full (with optimizer)" if include_optimizer_state else "model-only"
        logger.info(f"Saved {mode} checkpoint to {path}")

    def _get_embedder_version(self, model_name: str) -> str:
        """
        Get version identifier for the embedding model.

        For HuggingFace models, attempts to get the commit hash.
        Falls back to a timestamp-based version if unavailable.
        """
        if not model_name:
            return "unknown"

        try:
            from huggingface_hub import HfApi
            api = HfApi()
            model_info = api.model_info(model_name)
            return model_info.sha[:12] if model_info.sha else "unknown"
        except Exception:
            from datetime import datetime
            return f"snapshot-{datetime.now().strftime('%Y%m%d')}"

    def finish_logging(self):
        """Finish any active logging sessions (e.g., wandb)."""
        if self.wandb_enabled and self.wandb_run is not None:
            import wandb
            wandb.finish()
            logger.info("W&B logging finished")

    def load_checkpoint(self, path: str | Path):
        """
        Load training checkpoint.

        Handles both full checkpoints (with optimizer state) and model-only checkpoints.
        If optimizer state is missing, training will restart from scratch but with
        pre-trained model weights.

        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        # Core model state (always present)
        # Use _original_model for state_dict operations (works with torch.compile)
        self._original_model.load_state_dict(checkpoint["model_state_dict"])

        # Load discriminator states (only if they exist in both checkpoint and trainer)
        if checkpoint.get("tc_intra_discriminators") and self.tc_intra_discriminators:
            for k, state_dict in checkpoint["tc_intra_discriminators"].items():
                if k in self.tc_intra_discriminators:
                    self.tc_intra_discriminators[k].load_state_dict(state_dict)

        if checkpoint.get("pc_inter_discriminators") and self.pc_inter_discriminators:
            for k, state_dict in checkpoint["pc_inter_discriminators"].items():
                if k in self.pc_inter_discriminators:
                    self.pc_inter_discriminators[k].load_state_dict(state_dict)

        if checkpoint.get("tc_unified_discriminator") and self.tc_unified_discriminator is not None:
            self.tc_unified_discriminator.load_state_dict(checkpoint["tc_unified_discriminator"])

        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]

        if checkpoint.get("decoder_ema") and self.decoder_ema is not None:
            self.decoder_ema.load_state_dict(checkpoint["decoder_ema"])

        # Optimizer state (optional - only present in full checkpoints)
        has_optimizer_state = "model_optimizer_state_dict" in checkpoint
        if has_optimizer_state:
            self.model_optimizer.load_state_dict(checkpoint["model_optimizer_state_dict"])
            self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])

            if "beta_controller" in checkpoint:
                self.beta_controller.load_state_dict(checkpoint["beta_controller"])
            if "capacity_scheduler" in checkpoint:
                self.capacity_scheduler.load_state_dict(checkpoint["capacity_scheduler"])
            if "amp_manager" in checkpoint:
                self.amp_manager.load_state_dict(checkpoint["amp_manager"])
            if "model_scheduler_state_dict" in checkpoint:
                self.model_scheduler.load_state_dict(checkpoint["model_scheduler_state_dict"])
            if "disc_scheduler_state_dict" in checkpoint:
                self.disc_scheduler.load_state_dict(checkpoint["disc_scheduler_state_dict"])

        mode = "full" if has_optimizer_state else "model-only"
        logger.info(f"Loaded {mode} checkpoint from {path} (epoch {self.current_epoch})")
