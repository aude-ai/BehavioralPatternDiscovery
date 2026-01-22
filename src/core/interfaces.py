"""
Abstract Base Classes for Swappable Components

Defines the interfaces that all swappable components must implement.
These interfaces enable:
- Configuration-driven component selection
- SHAP compatibility through deterministic forward passes
- Clear contracts for component behavior
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class BaseEncoder(ABC, nn.Module):
    """
    Interface for all encoder implementations.

    Encoders transform input features into latent representations.
    Multiple encoders with different random seeds can be used in parallel
    to discover different aspects of the data.

    SHAP Compatibility:
        - forward() must be deterministic when training=False
        - Use distribution.get_mean() instead of sampling
        - No in-place operations
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any], distribution: "LatentDistribution"):
        """
        Initialize the encoder.

        Args:
            config: Encoder configuration (no defaults allowed)
            distribution: Latent distribution for reparameterization
        """
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        aux: Tensor,
        training: bool = True
    ) -> dict[str, Any]:
        """
        Encode input to latent representation.

        Args:
            x: Input embedding tensor (batch, embedding_dim)
            aux: Auxiliary features tensor (batch, aux_dim)
            training: If True, use reparameterization sampling.
                     If False, use deterministic mean (for SHAP).

        Returns:
            Dictionary containing:
            - 'latent': Final latent representation (batch, latent_dim)
            - 'distribution_params': Parameters for the latent distribution
            - Additional keys depend on implementation (e.g., hierarchical levels)
        """
        pass

    @abstractmethod
    def get_latent_dim(self) -> int:
        """Return the dimensionality of the output latent space."""
        pass


class BaseUnification(ABC, nn.Module):
    """
    Interface for unification layer implementations.

    The unification layer combines outputs from multiple base encoders
    into a single unified latent representation.

    SHAP Compatibility:
        - forward() must be deterministic when training=False
        - Use distribution.get_mean() instead of sampling
        - No in-place operations
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any], distribution: "LatentDistribution"):
        """
        Initialize the unification layer.

        Args:
            config: Unification configuration (no defaults allowed)
            distribution: Latent distribution for reparameterization
        """
        super().__init__()

    @abstractmethod
    def forward(
        self,
        encoder_outputs: list[Tensor],
        training: bool = True
    ) -> dict[str, Any]:
        """
        Combine encoder outputs into unified representation.

        Args:
            encoder_outputs: List of top-level latent tensors from each encoder
                           [(batch, encoder_latent_dim), ...]
            training: If True, use reparameterization sampling.
                     If False, use deterministic mean (for SHAP).

        Returns:
            Dictionary containing:
            - 'unified_z': Unified latent representation (batch, unified_dim)
            - 'unified_params': Parameters for the unified distribution
        """
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        """Return expected total input dimension (sum of encoder outputs)."""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the unified latent dimensionality."""
        pass


class BaseDecoder(ABC, nn.Module):
    """
    Interface for all decoder implementations.

    Decoders reconstruct the original embedding from the unified
    latent representation.

    SHAP Compatibility:
        - forward() should produce consistent outputs for same inputs
        - For flow-based decoders, use fixed number of ODE steps
        - No in-place operations
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the decoder.

        Args:
            config: Decoder configuration (no defaults allowed)
        """
        super().__init__()

    @abstractmethod
    def forward(self, z: Tensor, **kwargs: Any) -> Tensor:
        """
        Decode latent representation to output.

        Args:
            z: Unified latent tensor (batch, latent_dim)
            **kwargs: Implementation-specific arguments
                     (e.g., target for flow matching training)

        Returns:
            Reconstructed embedding (batch, output_dim)
        """
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        """Return expected input latent dimensionality."""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return output embedding dimensionality."""
        pass


class LatentDistribution(ABC):
    """
    Interface for latent distribution implementations.

    Handles the probabilistic structure of latent spaces:
    - Creating parameter layers (e.g., mu/logvar for Gaussian)
    - Computing distribution parameters from hidden states
    - Reparameterization for training
    - KL divergence computation

    SHAP Compatibility:
        - get_mean() provides deterministic outputs for SHAP analysis
        - All operations must maintain gradient flow
    """

    @abstractmethod
    def create_param_layers(
        self,
        in_dim: int,
        out_dim: int
    ) -> nn.ModuleDict:
        """
        Create layers that output distribution parameters.

        Args:
            in_dim: Input dimension from encoder hidden state
            out_dim: Latent dimension (number of latent variables)

        Returns:
            ModuleDict containing parameter projection layers
            (e.g., {"mu": Linear, "logvar": Linear} for Gaussian)
        """
        pass

    @abstractmethod
    def forward_params(
        self,
        h: Tensor,
        layers: nn.ModuleDict
    ) -> dict[str, Tensor]:
        """
        Compute distribution parameters from hidden state.

        Args:
            h: Hidden state tensor (batch, hidden_dim)
            layers: Parameter layers from create_param_layers()

        Returns:
            Dictionary of distribution parameters
            (e.g., {"mu": Tensor, "logvar": Tensor} for Gaussian)
        """
        pass

    @abstractmethod
    def reparameterize(
        self,
        params: dict[str, Tensor],
        training: bool
    ) -> Tensor:
        """
        Sample from the distribution using reparameterization trick.

        Args:
            params: Distribution parameters from forward_params()
            training: If True, sample with noise (reparameterization trick).
                     If False, return deterministic mean.

        Returns:
            Sampled latent tensor (batch, latent_dim)
        """
        pass

    @abstractmethod
    def get_mean(self, params: dict[str, Tensor]) -> Tensor:
        """
        Return deterministic mean of the distribution.

        CRITICAL for SHAP compatibility: This method is used during
        SHAP analysis to ensure deterministic gradient computation.

        Args:
            params: Distribution parameters from forward_params()

        Returns:
            Mean tensor (batch, latent_dim)
        """
        pass

    @abstractmethod
    def kl_divergence(self, params: dict[str, Tensor]) -> Tensor:
        """
        Compute KL divergence from the prior distribution.

        Args:
            params: Distribution parameters from forward_params()

        Returns:
            KL divergence tensor (batch,) or (batch, latent_dim)
        """
        pass


class BaseDiscriminator(ABC, nn.Module):
    """
    Interface for discriminator implementations.

    Discriminators are used for adversarial training to encourage
    statistical independence between latent dimensions (Total Correlation)
    or between encoder outputs (Partial Correlation).
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the discriminator.

        Args:
            config: Discriminator configuration (no defaults allowed)
        """
        super().__init__()

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        """
        Discriminate real vs shuffled samples.

        Args:
            z: Latent samples (batch, latent_dim)

        Returns:
            Logits (batch, 2) for real (index 0) vs fake/shuffled (index 1)
        """
        pass

    @abstractmethod
    def shuffle(self, z: Tensor) -> Tensor:
        """
        Create shuffled (fake) samples from real samples.

        The shuffle method varies by discriminator type:
        - TC: Dimension-wise independent shuffling
        - PC: Group-wise shuffling (encoder groups stay together)

        Args:
            z: Real latent samples (batch, latent_dim)

        Returns:
            Shuffled samples (batch, latent_dim)
        """
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        """Return expected input dimension."""
        pass


class BaseLoss(ABC, nn.Module):
    """
    Interface for loss function implementations.

    Loss functions are composable components that compute individual
    loss terms. They are combined by the training loop to form the
    total training objective.
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the loss function.

        Args:
            config: Loss configuration (no defaults allowed)
        """
        super().__init__()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return a unique name for this loss.

        Used for logging and identification in composite losses.
        """
        pass

    @abstractmethod
    def forward(self, **kwargs: Any) -> Tensor:
        """
        Compute the loss value.

        Args:
            **kwargs: Loss-specific inputs

        Returns:
            Scalar loss tensor
        """
        pass
