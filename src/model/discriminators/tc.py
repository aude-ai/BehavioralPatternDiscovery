"""
Total Correlation (TC) Discriminator

Used for FactorVAE-style adversarial training to encourage independence
between latent dimensions within a single encoder or the unified space.

The discriminator learns to distinguish between:
- Real samples: z ~ q(z|x) (joint distribution)
- Fake samples: z_permuted (product of marginals)

TC is estimated using the density ratio trick:
    TC(z) ≈ E[log D(z) - log D(z_permuted)]

VAE TC Loss Formulations (configurable via tc_loss_formulation):
- "averaged" (EPP default): 0.5 * (CE(real, real_label) + CE(fake, fake_label))
    Standard FactorVAE formulation, averages both classification losses
- "real_only": CE(real, fake_label)
    Only penalizes when discriminator correctly identifies real samples

References:
- FactorVAE: Disentangling by Factorising (Kim & Mnih, 2018)
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from src.core.registry import ComponentRegistry
from src.model.layers import make_activation

logger = logging.getLogger(__name__)

# Registry for discriminators
discriminator_registry = ComponentRegistry[nn.Module]("discriminator")


def permute_dims(z: torch.Tensor) -> torch.Tensor:
    """
    Permute latent dimensions independently to create product of marginals.

    Each dimension is shuffled across the batch independently, breaking
    any correlation between dimensions while preserving marginal distributions.

    Uses vectorized argsort(rand()) instead of Python loop for 10-50x speedup.

    Args:
        z: Latent codes (batch_size, latent_dim)

    Returns:
        Permuted latent codes (product of marginals)
    """
    batch_size, latent_dim = z.shape

    # Vectorized approach: generate random values and argsort to get permutations
    # Shape: (latent_dim, batch_size) - one permutation per dimension
    rand_vals = torch.rand(latent_dim, batch_size, device=z.device)
    # Get permutation indices for each dimension
    perm_indices = torch.argsort(rand_vals, dim=1)  # (latent_dim, batch_size)

    # Use advanced indexing to gather permuted values
    # z.T is (latent_dim, batch_size), we gather along batch dimension
    z_t = z.T  # (latent_dim, batch_size)
    z_permuted_t = torch.gather(z_t, dim=1, index=perm_indices)

    return z_permuted_t.T  # Back to (batch_size, latent_dim)


@discriminator_registry.register("tc")
class TCDiscriminator(nn.Module):
    """
    Total Correlation Discriminator.

    Distinguishes between joint distribution and product of marginals
    to estimate Total Correlation penalty.

    Used for:
    - TC Intra: Independence within each encoder level (9 discriminators)
    - TC Unified: Independence in unified latent space (1 discriminator)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Discriminator configuration containing:
                - input_dim: Latent dimension to discriminate
                - hidden_dims: List of hidden layer dimensions
                - dropout: Dropout rate
                - activation: Activation config
                - tc_loss_formulation: "averaged" (EPP) or "real_only" (original BPD)
        """
        super().__init__()

        self.input_dim = config["input_dim"]
        hidden_dims = config["hidden_dims"]
        dropout = config["dropout"]

        # TC loss formulation: "averaged" (EPP default) or "real_only"
        self.tc_loss_formulation = config["tc_loss_formulation"]

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
            z: Latent codes (batch_size, input_dim)

        Returns:
            Logits (batch_size, 2) for real (index 0) vs fake (index 1)
        """
        return self.model(z)

    def shuffle(self, z: torch.Tensor) -> torch.Tensor:
        """
        Create shuffled (fake) samples using dimension-wise permutation.

        Args:
            z: Real latent samples (batch_size, input_dim)

        Returns:
            Shuffled samples (batch_size, input_dim)
        """
        return permute_dims(z)

    def get_input_dim(self) -> int:
        """Return expected input dimension."""
        return self.input_dim

    def compute_tc_loss(
        self,
        z_real: torch.Tensor,
        train_discriminator: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute TC loss for both discriminator and VAE.

        Args:
            z_real: Real latent samples (batch_size, input_dim)
            train_discriminator: If True, compute discriminator loss.
                                If False, compute VAE TC penalty.

        Returns:
            Tuple of (discriminator_loss, vae_tc_loss)
            - discriminator_loss: Cross-entropy loss for discriminator training
            - vae_tc_loss: TC penalty for VAE training
        """
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

        # VAE TC penalty: depends on formulation
        if self.tc_loss_formulation == "averaged":
            # EPP formulation: average of both classification losses
            # Standard FactorVAE approach
            vae_tc_loss = 0.5 * (
                nn.functional.cross_entropy(logits_real, labels_real) +
                nn.functional.cross_entropy(logits_fake, labels_fake)
            )
        else:
            # "real_only" formulation: only penalize real samples being detected
            # Train VAE to make discriminator think real samples are from marginals (fake)
            vae_tc_loss = nn.functional.cross_entropy(logits_real, labels_fake)

        return disc_loss, vae_tc_loss
