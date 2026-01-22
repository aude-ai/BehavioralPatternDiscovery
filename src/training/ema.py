"""
Exponential Moving Average (EMA)

Maintains an exponential moving average of model parameters for stable inference.
Used primarily for flow decoder to reduce variance during generation.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMAWrapper:
    """
    Maintains EMA of model parameters for stable inference.

    The EMA weights can be temporarily swapped in for inference/validation,
    then restored to continue training with the original weights.

    Usage:
        ema = EMAWrapper(model, config)

        # During training:
        loss.backward()
        optimizer.step()
        ema.update()

        # During validation:
        ema.apply_shadow()
        validate(model)
        ema.restore()
    """

    def __init__(self, model: nn.Module, config: dict[str, Any]):
        """
        Args:
            model: The model to track EMA for
            config: EMA configuration containing:
                - enabled: Whether EMA is enabled
                - decay: EMA decay rate (e.g., 0.999)
        """
        self.model = model
        self.enabled = config["enabled"]
        self.decay = config["decay"]

        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        if self.enabled:
            self._init_shadow()
            logger.info(f"EMA enabled with decay={self.decay}")
        else:
            logger.info("EMA disabled")

    def _init_shadow(self) -> None:
        """Initialize shadow parameters as copy of model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """
        Update shadow parameters with current model parameters.

        shadow = decay * shadow + (1 - decay) * param
        """
        if not self.enabled:
            return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    def apply_shadow(self) -> None:
        """
        Temporarily replace model params with EMA shadow params.

        Must call restore() after inference to resume training.
        """
        if not self.enabled:
            return

        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original params after using shadow for inference."""
        if not self.enabled:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])

        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "enabled": self.enabled,
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        if not state.get("enabled", False):
            return

        device = next(self.model.parameters()).device
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}
        self.decay = state["decay"]
