"""
Diversity Discriminator for Engineer-Distinguishing Representations

Predicts engineer ID from aggregated unified latent representation.
VAE trained to HELP the discriminator succeed (cooperative, not adversarial).
This ensures latent space captures meaningful behavioral differences between engineers.

SCOPE: Currently unified-layer only. Future expansion may extend to per-encoder
per-level discrimination.

BATCH MODE: Only used with single-engineer batch mode. Uses Empirical Bayes
aggregation across messages in batch before classification.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class DiversityDiscriminator(nn.Module):
    """
    Discriminator that predicts engineer ID from aggregated unified latent.

    SCOPE: Unified layer only. May be extended to per-encoder per-level in future.

    Architecture: Empirical Bayes aggregation + MLP classifier.
    Uses batch-level aggregation because single messages are insufficient
    to reliably identify an engineer.
    """

    def __init__(self, config: dict, input_dim: int, num_engineers: int):
        """
        Initialize DiversityDiscriminator.

        Args:
            config: discriminators.diversity config section with keys:
                - enabled: bool
                - hidden_dims: list[int] (hidden layer sizes)
                - prior_variance: float (Empirical Bayes prior variance)
            input_dim: Unified latent dimension (from ModelDimensions)
            num_engineers: Number of unique engineers to classify

        Raises:
            KeyError: If required config keys are missing
        """
        super().__init__()

        self.enabled = config["enabled"]
        if not self.enabled:
            logger.info("DiversityDiscriminator disabled")
            return

        self.input_dim = input_dim
        self.num_engineers = num_engineers
        self.prior_variance = config["prior_variance"]
        hidden_dims = config["hidden_dims"]

        # Build classifier network
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_engineers))
        self.network = nn.Sequential(*layers)

        logger.info(
            f"Initialized DiversityDiscriminator "
            f"(input_dim={self.input_dim}, hidden_dims={hidden_dims}, "
            f"num_engineers={num_engineers}, prior_variance={self.prior_variance})"
        )

    def _empirical_bayes_aggregate(self, z: Tensor) -> Tensor:
        """
        Aggregate message-level latents using Empirical Bayes shrinkage.

        Shrinks batch mean toward prior (zero) based on within-batch variance.
        More messages = less shrinkage. High variance = more shrinkage.

        Args:
            z: (batch_size, latent_dim) message-level latents from single engineer

        Returns:
            (1, latent_dim) aggregated engineer representation
        """
        batch_size = z.shape[0]

        if batch_size == 1:
            # Single message: heavy shrinkage toward prior
            shrinkage = self.prior_variance / (self.prior_variance + 1.0)
            return z * (1 - shrinkage)

        # Compute within-batch statistics
        batch_mean = z.mean(dim=0, keepdim=True)  # (1, latent_dim)
        batch_var = z.var(dim=0, keepdim=True)  # (1, latent_dim)

        # Empirical Bayes shrinkage factor per dimension
        # shrinkage = prior_var / (prior_var + observed_var / n)
        # Higher n or lower observed_var = less shrinkage (trust the data more)
        observed_var_of_mean = batch_var / batch_size
        shrinkage = self.prior_variance / (
            self.prior_variance + observed_var_of_mean.clamp(min=1e-6)
        )

        # Shrink toward zero prior
        aggregated = batch_mean * (1 - shrinkage)

        return aggregated

    def forward(self, z: Tensor) -> Tensor:
        """
        Predict engineer logits from batch of unified latents.

        Aggregates all messages in batch (assumed from single engineer)
        using Empirical Bayes, then classifies.

        Args:
            z: (batch_size, input_dim) unified latents from single engineer's messages

        Returns:
            (1, num_engineers) classification logits for the engineer
        """
        if not self.enabled:
            raise RuntimeError("DiversityDiscriminator is disabled")

        # Aggregate batch to single engineer representation
        aggregated = self._empirical_bayes_aggregate(z)  # (1, input_dim)

        # Classify
        return self.network(aggregated)


class DiversityDiscriminatorLoss:
    """
    Manages discriminator and VAE losses for cooperative engineer classification.

    COOPERATIVE (not adversarial): Both discriminator and VAE want accurate
    classification. This ensures the VAE learns engineer-distinguishing patterns.

    SCOPE: Unified layer only. May be extended to per-encoder per-level in future.

    Discriminator: Minimize cross-entropy (maximize accuracy)
    VAE: Also minimize cross-entropy (help discriminator succeed)
    """

    def __init__(self, config: dict, discriminator: DiversityDiscriminator):
        """
        Initialize DiversityDiscriminatorLoss.

        Args:
            config: discriminators.diversity config section
            discriminator: The DiversityDiscriminator instance
        """
        self.discriminator = discriminator
        self.enabled = config["enabled"]

    def discriminator_loss(self, z: Tensor, engineer_idx: int) -> Tensor:
        """
        Compute discriminator training loss.

        Args:
            z: (batch_size, dim) unified latents from single engineer (detached internally)
            engineer_idx: Integer index of the engineer (0 to num_engineers-1)

        Returns:
            Cross-entropy loss (minimize to improve classification)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=z.device)

        logits = self.discriminator(z.detach())  # (1, num_engineers)
        target = torch.tensor([engineer_idx], device=z.device)
        return F.cross_entropy(logits, target)

    def vae_loss(self, z: Tensor, engineer_idx: int) -> Tensor:
        """
        Compute VAE training loss (HELP the discriminator).

        COOPERATIVE: VAE wants discriminator to correctly classify the engineer.
        This encourages the VAE to learn engineer-distinguishing representations.

        Args:
            z: (batch_size, dim) unified latents from single engineer (with gradients)
            engineer_idx: Integer index of the engineer (0 to num_engineers-1)

        Returns:
            Cross-entropy loss (minimize to help classification accuracy)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=z.device)

        logits = self.discriminator(z)  # (1, num_engineers)
        target = torch.tensor([engineer_idx], device=z.device)

        # Cooperative: VAE also minimizes classification error
        return F.cross_entropy(logits, target)
