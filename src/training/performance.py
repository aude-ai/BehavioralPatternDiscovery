"""
Performance Utilities for Training

Provides:
- Automatic Mixed Precision (AMP): Dynamic precision for faster training
- Gradient Checkpointing: Trade compute for memory
- Performance configuration extraction
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class AMPManager:
    """
    Automatic Mixed Precision training manager.

    Supports both bfloat16 (no scaling needed) and float16 (requires scaling).
    bfloat16 is preferred when available as it has the same dynamic range as float32.
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: str = "bfloat16",
    ):
        """
        Args:
            enabled: Whether to enable AMP
            dtype: Precision type ("bfloat16" or "float16")
        """
        self.enabled = enabled and torch.cuda.is_available()

        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
            self.use_scaler = False
        elif dtype == "float16":
            self.dtype = torch.float16
            self.use_scaler = True
        else:
            raise ValueError(f"Unknown dtype: {dtype}. Use 'bfloat16' or 'float16'")

        # GradScaler only needed for float16
        if self.enabled and self.use_scaler:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        if self.enabled:
            logger.info(f"AMP enabled with {dtype}")
        else:
            logger.info("AMP disabled, using FP32")

    def autocast_context(self):
        """Get autocast context manager for forward pass."""
        return torch.amp.autocast(
            "cuda",
            enabled=self.enabled,
            dtype=self.dtype if self.enabled else None,
        )

    def backward(self, loss: torch.Tensor):
        """
        Perform backward pass with optional scaling.

        Args:
            loss: Loss tensor to backpropagate
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_and_clip(
        self,
        optimizer: torch.optim.Optimizer,
        parameters,
        max_norm: float,
    ):
        """
        Unscale gradients and clip.

        Args:
            optimizer: Optimizer to unscale for
            parameters: Parameters to clip
            max_norm: Maximum gradient norm
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)

    def step(self, optimizer: torch.optim.Optimizer):
        """
        Step optimizer with optional scaler.

        Args:
            optimizer: Optimizer to step
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        state = {"enabled": self.enabled, "dtype": str(self.dtype)}
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]):
        """Load state from checkpoint."""
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


def enable_gradient_checkpointing(model: nn.Module) -> int:
    """
    Enable gradient checkpointing on model.

    Calls enable_gradient_checkpointing() on components that support it.

    Args:
        model: Model to enable checkpointing on

    Returns:
        Number of components with checkpointing enabled
    """
    count = 0

    # Check if model has the method directly
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        count += 1

    # Also check encoders
    if hasattr(model, "encoders"):
        for encoder in model.encoders:
            if hasattr(encoder, "enable_gradient_checkpointing"):
                encoder.enable_gradient_checkpointing()
                count += 1

    # Check decoder
    if hasattr(model, "decoder"):
        if hasattr(model.decoder, "enable_gradient_checkpointing"):
            model.decoder.enable_gradient_checkpointing()
            count += 1

    # Check unification
    if hasattr(model, "unification"):
        if hasattr(model.unification, "enable_gradient_checkpointing"):
            model.unification.enable_gradient_checkpointing()
            count += 1

    if count > 0:
        logger.info(f"Gradient checkpointing enabled on {count} components")
    else:
        logger.warning("No components found for gradient checkpointing")

    return count


def disable_gradient_checkpointing(model: nn.Module) -> int:
    """
    Disable gradient checkpointing on model.

    Args:
        model: Model to disable checkpointing on

    Returns:
        Number of components with checkpointing disabled
    """
    count = 0

    if hasattr(model, "disable_gradient_checkpointing"):
        model.disable_gradient_checkpointing()
        count += 1

    if hasattr(model, "encoders"):
        for encoder in model.encoders:
            if hasattr(encoder, "disable_gradient_checkpointing"):
                encoder.disable_gradient_checkpointing()
                count += 1

    if hasattr(model, "decoder"):
        if hasattr(model.decoder, "disable_gradient_checkpointing"):
            model.decoder.disable_gradient_checkpointing()
            count += 1

    if hasattr(model, "unification"):
        if hasattr(model.unification, "disable_gradient_checkpointing"):
            model.unification.disable_gradient_checkpointing()
            count += 1

    if count > 0:
        logger.info(f"Gradient checkpointing disabled on {count} components")

    return count


def checkpointed_forward(
    module: nn.Module,
    *args,
    enabled: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Forward pass with optional gradient checkpointing.

    Args:
        module: Module to forward through
        *args: Positional arguments to module
        enabled: Whether checkpointing is enabled
        **kwargs: Keyword arguments to module

    Returns:
        Output tensor
    """
    if enabled and module.training:
        # Wrap forward to handle kwargs
        def forward_fn(*inputs):
            return module(*inputs, **kwargs)

        return checkpoint(
            forward_fn,
            *args,
            use_reentrant=False,
            preserve_rng_state=True,
        )
    else:
        return module(*args, **kwargs)


class CheckpointedSequential(nn.Sequential):
    """
    Sequential module with built-in gradient checkpointing support.

    Drop-in replacement for nn.Sequential with checkpointing capability.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gradient_checkpointing and self.training:
            return checkpoint(
                super().forward,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return super().forward(x)
