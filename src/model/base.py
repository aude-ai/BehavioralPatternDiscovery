"""
Base VAE Interface

Defines the contract that all VAE implementations must follow.
This enables the Trainer to work with any VAE architecture.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from src.core.config import ModelDimensions


class BaseVAE(ABC, nn.Module):
    """
    Abstract base class for VAE models.

    All VAE implementations must provide these methods to work with Trainer.
    This interface decouples the Trainer from specific VAE implementations,
    allowing different architectures to be swapped without trainer changes.

    Required attributes (set in subclass __init__):
        dims: ModelDimensions - Model dimension configuration
        num_encoders: int - Number of encoders
        input_dim: int - Input dimension
        output_dim: int - Output dimension
    """

    # These attributes must be set by subclasses in __init__
    dims: ModelDimensions
    num_encoders: int
    input_dim: int
    output_dim: int

    @abstractmethod
    def encode(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Encode input to latent representations.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Dictionary with encoder outputs including z and params per level.
            Structure depends on implementation but should include:
                - Per-encoder outputs (e.g., enc1, enc2, ...) with z and params
                - unified: Unified latent representation with z and params
        """
        pass

    @abstractmethod
    def decode(
        self,
        z: torch.Tensor,
        x_target: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Decode from latent to reconstruction.

        Args:
            z: Latent tensor (batch, latent_dim)
            x_target: Target tensor for training (optional)

        Returns:
            If x_target is None or not training: (batch, output_dim) reconstruction
            If x_target provided during training: (reconstruction, loss) tuple
        """
        pass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Full forward pass: encode → decode.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Dictionary with encodings and reconstruction
        """
        pass

    @abstractmethod
    def get_level_names(self) -> list[str]:
        """Return ordered list of encoder level names."""
        pass

    @abstractmethod
    def get_encoder_names(self) -> list[str]:
        """Return list of encoder names (e.g., ['enc1', 'enc2', ...])."""
        pass

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        pass

    @abstractmethod
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        pass
