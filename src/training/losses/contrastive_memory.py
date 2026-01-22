"""
Contrastive Memory Loss for Inter-Batch Diversity

Maintains separate memory banks per encoder per level.
Encourages current batch to differ from historical batches.
One-sided gradients (current batch only) but effective for diversity.

No cross-encoder or cross-level comparisons - each memory bank is independent.
"""

import logging
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ContrastiveMemoryLoss:
    """
    Contrastive loss using memory bank for inter-batch diversity.

    Maintains separate FIFO queues per encoder per level.
    Trainer calls compute() and update_memory() per encoder/level.
    """

    def __init__(self, config: dict):
        """
        Initialize ContrastiveMemoryLoss.

        Args:
            config: contrastive_memory_loss config section with keys:
                - enabled: bool
                - memory_size: int (number of batches to store per memory bank)
                - temperature: float (contrastive sharpness, lower = sharper)

        Raises:
            KeyError: If required config keys are missing
        """
        self.enabled = config["enabled"]
        if not self.enabled:
            logger.info("ContrastiveMemoryLoss disabled")
            return

        self.memory_size = config["memory_size"]
        self.temperature = config["temperature"]

        # Separate memory bank per encoder per level
        # Structure: {key: deque of tensors} where key = "encoder_level" or "unified"
        self.memory_banks: dict[str, deque] = {}

        logger.info(
            f"Initialized ContrastiveMemoryLoss "
            f"(memory_size={self.memory_size}, temperature={self.temperature})"
        )

    def _get_memory_key(self, encoder_name: str, level_name: str) -> str:
        """Get memory bank key for encoder/level."""
        return f"{encoder_name}_{level_name}"

    def _ensure_memory_bank(self, key: str) -> deque:
        """Get or create memory bank for key."""
        if key not in self.memory_banks:
            self.memory_banks[key] = deque(maxlen=self.memory_size)
        return self.memory_banks[key]

    def compute(self, mu: Tensor, encoder_name: str, level_name: str) -> Tensor:
        """
        Compute contrastive loss for single encoder/level against its memory bank.

        Args:
            mu: (batch, latent_dim) current batch mu values
            encoder_name: Name of encoder (e.g., "enc_0") or "unified"
            level_name: Name of level (e.g., "bottom") or "unified"

        Returns:
            Scalar loss tensor
        """
        if not self.enabled:
            return torch.tensor(0.0, device=mu.device)

        key = self._get_memory_key(encoder_name, level_name)
        memory_bank = self._ensure_memory_bank(key)

        if len(memory_bank) == 0:
            return torch.tensor(0.0, device=mu.device)

        # Stack memory bank: (total_memory_samples, latent_dim)
        memory = torch.cat(list(memory_bank), dim=0)

        # Normalize for cosine similarity
        current_norm = F.normalize(mu, p=2, dim=1)  # (batch, latent_dim)
        memory_norm = F.normalize(memory, p=2, dim=1)  # (memory_samples, latent_dim)

        # Compute similarity matrix: (batch, memory_samples)
        similarity = torch.mm(current_norm, memory_norm.t()) / self.temperature

        # Loss: penalize high similarity to memory bank
        # Softmax focuses on the most similar (worst) cases
        soft_max_sim = F.softmax(similarity, dim=1)
        weighted_sim = (soft_max_sim * similarity).sum(dim=1)

        return weighted_sim.mean()

    def update_memory(self, mu: Tensor, encoder_name: str, level_name: str) -> None:
        """
        Add current batch mu to memory bank (detached).

        Call this AFTER computing loss for the batch.

        Args:
            mu: (batch, latent_dim) current batch mu values
            encoder_name: Name of encoder (e.g., "enc_0") or "unified"
            level_name: Name of level (e.g., "bottom") or "unified"
        """
        if not self.enabled:
            return

        key = self._get_memory_key(encoder_name, level_name)
        memory_bank = self._ensure_memory_bank(key)
        memory_bank.append(mu.detach().clone())
