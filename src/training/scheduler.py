"""
Learning Rate Scheduler

Provides scheduler factory and implementations for learning rate scheduling.
Supports warmup and various annealing strategies.
"""

import logging
import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from src.core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Registry for scheduler types
scheduler_registry = ComponentRegistry("scheduler")


def create_scheduler(
    optimizer: Optimizer,
    config: dict[str, Any],
    total_epochs: int,
) -> LRScheduler:
    """
    Create a learning rate scheduler from config.

    Args:
        optimizer: The optimizer to schedule
        config: Scheduler configuration containing:
            - type: Scheduler type (e.g., "cosine", "linear", "constant")
            - warmup_epochs: Number of warmup epochs
            - min_lr_ratio: Minimum LR as ratio of initial LR
        total_epochs: Total number of training epochs

    Returns:
        Configured LRScheduler instance
    """
    scheduler_type = config["type"]
    warmup_epochs = config["warmup_epochs"]
    min_lr_ratio = config["min_lr_ratio"]

    scheduler_cls = scheduler_registry.get(scheduler_type)
    return scheduler_cls(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
        total_epochs=total_epochs,
    )


@scheduler_registry.register("cosine")
class CosineScheduler(LambdaLR):
    """
    Cosine annealing scheduler with linear warmup.

    During warmup: LR linearly increases from 0 to initial LR
    After warmup: LR follows cosine decay to min_lr_ratio * initial_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        min_lr_ratio: float,
        total_epochs: int,
    ):
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs

        super().__init__(optimizer, self._lr_lambda)

        logger.info(
            f"CosineScheduler: warmup={warmup_epochs} epochs, "
            f"min_lr_ratio={min_lr_ratio}, total={total_epochs} epochs"
        )

    def _lr_lambda(self, epoch: int) -> float:
        """Compute learning rate multiplier for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup from 0 to 1
            if self.warmup_epochs == 0:
                return 1.0
            return (epoch + 1) / self.warmup_epochs

        # Cosine annealing after warmup
        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay


@scheduler_registry.register("linear")
class LinearScheduler(LambdaLR):
    """
    Linear decay scheduler with linear warmup.

    During warmup: LR linearly increases from 0 to initial LR
    After warmup: LR linearly decays to min_lr_ratio * initial_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        min_lr_ratio: float,
        total_epochs: int,
    ):
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs

        super().__init__(optimizer, self._lr_lambda)

        logger.info(
            f"LinearScheduler: warmup={warmup_epochs} epochs, "
            f"min_lr_ratio={min_lr_ratio}, total={total_epochs} epochs"
        )

    def _lr_lambda(self, epoch: int) -> float:
        """Compute learning rate multiplier for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup from 0 to 1
            if self.warmup_epochs == 0:
                return 1.0
            return (epoch + 1) / self.warmup_epochs

        # Linear decay after warmup
        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - progress)


@scheduler_registry.register("constant")
class ConstantScheduler(LambdaLR):
    """
    Constant learning rate with optional warmup.

    During warmup: LR linearly increases from 0 to initial LR
    After warmup: LR stays constant at initial LR
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        min_lr_ratio: float,
        total_epochs: int,
    ):
        self.warmup_epochs = warmup_epochs
        # min_lr_ratio is ignored for constant scheduler but accepted for interface consistency

        super().__init__(optimizer, self._lr_lambda)

        logger.info(f"ConstantScheduler: warmup={warmup_epochs} epochs")

    def _lr_lambda(self, epoch: int) -> float:
        """Compute learning rate multiplier for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup from 0 to 1
            if self.warmup_epochs == 0:
                return 1.0
            return (epoch + 1) / self.warmup_epochs

        # Constant after warmup
        return 1.0
