# Training components for BehavioralPatternDiscovery

from .losses import (
    loss_registry,
    ReconstructionLoss,
    FocalReconstructionLoss,
    KLDivergenceLoss,
    IWOLoss,
    HoyerLoss,
    CompositeLoss,
)

from .optimizers import (
    optimizer_registry,
    Lion,
    AdEMAMix,
    create_optimizer,
)

from .beta_controller import (
    BaseBetaController,
    DualGradientController,
    PIController,
    BetaControllerManager,
)

from .performance import (
    AMPManager,
    enable_gradient_checkpointing,
    disable_gradient_checkpointing,
    checkpointed_forward,
    CheckpointedSequential,
)

from .scheduler import (
    scheduler_registry,
    CosineScheduler,
    LinearScheduler,
    ConstantScheduler,
    create_scheduler,
)

from .monitoring import LatentMonitor

from .ema import EMAWrapper

from .trainer import Trainer

__all__ = [
    # Losses
    "loss_registry",
    "ReconstructionLoss",
    "FocalReconstructionLoss",
    "KLDivergenceLoss",
    "IWOLoss",
    "HoyerLoss",
    "CompositeLoss",
    # Optimizers
    "optimizer_registry",
    "Lion",
    "AdEMAMix",
    "create_optimizer",
    # Beta Controller
    "BaseBetaController",
    "DualGradientController",
    "PIController",
    "BetaControllerManager",
    # Performance
    "AMPManager",
    "enable_gradient_checkpointing",
    "disable_gradient_checkpointing",
    "checkpointed_forward",
    "CheckpointedSequential",
    # Scheduler
    "scheduler_registry",
    "CosineScheduler",
    "LinearScheduler",
    "ConstantScheduler",
    "create_scheduler",
    # Monitoring
    "LatentMonitor",
    # EMA
    "EMAWrapper",
    # Trainer
    "Trainer",
]
