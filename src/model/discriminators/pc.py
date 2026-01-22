"""
Partial Correlation (PC) Discriminator

Used for inter-encoder independence. Encourages different encoders to
capture complementary (non-overlapping) information.

The discriminator operates on concatenated outputs from multiple encoders
at the same hierarchical level. The shuffle method permutes encoder outputs
as groups, preserving within-encoder correlations but breaking between-encoder
correlations.

Example for 3 encoders at bottom level (90 dims each):
- Real: [enc1_z, enc2_z, enc3_z] concatenated (270 dims)
- Fake: [enc1_z[perm1], enc2_z[perm2], enc3_z[perm3]] different permutations

This encourages the encoders to discover independent aspects of the data.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from src.model.layers import make_activation
from .tc import discriminator_registry

logger = logging.getLogger(__name__)


def permute_groups(z: torch.Tensor, group_sizes: list[int]) -> torch.Tensor:
    """
    Permute latent dimensions in groups (encoder-wise permutation).

    Each group (encoder's output) is shuffled independently across the batch,
    but dimensions within a group stay together.

    Uses vectorized approach for efficiency.

    Args:
        z: Concatenated latent codes (batch_size, sum(group_sizes))
        group_sizes: List of dimension sizes for each group (encoder)

    Returns:
        Group-permuted latent codes
    """
    batch_size = z.shape[0]
    device = z.device
    num_groups = len(group_sizes)

    # Generate all permutations at once: (num_groups, batch_size)
    rand_vals = torch.rand(num_groups, batch_size, device=device)
    perm_indices = torch.argsort(rand_vals, dim=1)  # (num_groups, batch_size)

    # Build output by gathering each group with its permutation
    z_permuted_parts = []
    start_idx = 0
    for i, group_size in enumerate(group_sizes):
        end_idx = start_idx + group_size
        # Get the permutation for this group
        perm = perm_indices[i]  # (batch_size,)
        # Gather the group using the permutation
        z_permuted_parts.append(z[perm, start_idx:end_idx])
        start_idx = end_idx

    return torch.cat(z_permuted_parts, dim=1)


@discriminator_registry.register("pc")
class PCDiscriminator(nn.Module):
    """
    Partial Correlation Discriminator.

    Distinguishes between joint encoder outputs and group-shuffled outputs
    to encourage inter-encoder independence.

    Used for:
    - PC Inter: Independence between encoders at each level (3 discriminators)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Discriminator configuration containing:
                - latent_dim_per_encoder: Dimension of each encoder's output
                - num_encoders: Number of encoders
                - hidden_dims: List of hidden layer dimensions
                - dropout: Dropout rate
                - activation: Activation config
                - pc_loss_formulation: "averaged" (EPP) or "real_only" (original BPD)
        """
        super().__init__()

        latent_dim_per_encoder = config["latent_dim_per_encoder"]
        num_encoders = config["num_encoders"]
        hidden_dims = config["hidden_dims"]
        dropout = config["dropout"]

        # PC loss formulation: "averaged" (EPP default) or "real_only"
        self.pc_loss_formulation = config["pc_loss_formulation"]

        self.input_dim = latent_dim_per_encoder * num_encoders
        self.group_sizes = [latent_dim_per_encoder] * num_encoders
        self.num_encoders = num_encoders

        activation_type = config["activation"]["type"]
        if activation_type in config["activation"]:
            activation_config = config["activation"][activation_type]
        else:
            activation_config = {}

        # Build MLP
        layers = []
        current_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                make_activation(activation_type, activation_config),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        # Output: 2 logits (real vs fake)
        layers.append(nn.Linear(current_dim, 2))

        self.model = nn.Sequential(*layers)
        self._init_weights(activation_type)

    def _init_weights(self, activation_type: str = "relu"):
        """
        Initialize weights with He initialization appropriate for the activation.

        Args:
            activation_type: The activation function type being used.
        """
        if activation_type in ("xielu", "elu", "leaky_relu"):
            nonlinearity = "leaky_relu"
            a = 1.0
        elif activation_type in ("swish", "gelu"):
            nonlinearity = "leaky_relu"
            a = 0.5
        else:
            nonlinearity = "relu"
            a = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if nonlinearity == "leaky_relu":
                    nn.init.kaiming_normal_(m.weight, a=a, mode="fan_in", nonlinearity=nonlinearity)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Concatenated latent codes (batch_size, input_dim)

        Returns:
            Logits (batch_size, 2) for real (index 0) vs fake (index 1)
        """
        return self.model(z)

    def shuffle(self, z: torch.Tensor) -> torch.Tensor:
        """
        Create shuffled (fake) samples using group-wise permutation.

        Each encoder's dimensions are permuted together as a group.

        Args:
            z: Real concatenated latent samples (batch_size, input_dim)

        Returns:
            Group-shuffled samples (batch_size, input_dim)
        """
        return permute_groups(z, self.group_sizes)

    def get_input_dim(self) -> int:
        """Return expected input dimension."""
        return self.input_dim

    def compute_pc_loss(
        self,
        encoder_outputs: list[torch.Tensor],
        train_discriminator: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PC loss for both discriminator and VAE.

        Args:
            encoder_outputs: List of encoder outputs at same level
                           [(batch, dim), (batch, dim), ...]
            train_discriminator: If True, compute discriminator loss.
                                If False, compute VAE PC penalty.

        Returns:
            Tuple of (discriminator_loss, vae_pc_loss)
            - discriminator_loss: Cross-entropy loss for discriminator training
            - vae_pc_loss: PC penalty for VAE training
        """
        # Concatenate encoder outputs
        z_real = torch.cat(encoder_outputs, dim=1)
        batch_size = z_real.shape[0]
        device = z_real.device

        # Create fake samples
        z_fake = self.shuffle(z_real)

        # Get discriminator predictions
        logits_real = self.forward(z_real)
        logits_fake = self.forward(z_fake)

        # Clamp logits to prevent extreme values causing NaN in cross_entropy
        logits_real = torch.clamp(logits_real, min=-50.0, max=50.0)
        logits_fake = torch.clamp(logits_fake, min=-50.0, max=50.0)

        # Labels: real=0, fake=1
        labels_real = torch.zeros(batch_size, dtype=torch.long, device=device)
        labels_fake = torch.ones(batch_size, dtype=torch.long, device=device)

        # Discriminator loss (cross-entropy) - same for both formulations
        disc_loss = (
            nn.functional.cross_entropy(logits_real, labels_real) +
            nn.functional.cross_entropy(logits_fake, labels_fake)
        ) / 2

        # VAE PC penalty: depends on formulation
        if self.pc_loss_formulation == "averaged":
            # EPP formulation: average of both classification losses
            vae_pc_loss = 0.5 * (
                nn.functional.cross_entropy(logits_real, labels_real) +
                nn.functional.cross_entropy(logits_fake, labels_fake)
            )
        else:
            # "real_only" formulation: only penalize real samples being detected
            vae_pc_loss = nn.functional.cross_entropy(logits_real, labels_fake)

        return disc_loss, vae_pc_loss
