"""
Jina Embeddings v4 Encoder

Multimodal embedding model based on Qwen2.5-VL-3B.
Supports text, images, and visual documents.
Outputs 2048-dim embeddings (or Matryoshka truncated).
"""

import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel

from .base import BaseTextEncoder

logger = logging.getLogger(__name__)


class JinaV4Encoder(BaseTextEncoder):
    """Jina v4 embeddings encoder using Transformers."""

    def __init__(self, config: dict):
        """
        Initialize Jina v4 encoder.

        Args:
            config: Full config dict with processing.text_encoder.jina_v4 section
        """
        jina_config = config["processing"]["text_encoder"]["jina_v4"]

        self._model_name = jina_config["model_name"]
        self.task = jina_config["task"]
        self.prompt_name = jina_config.get("prompt_name", "passage")
        self.max_tokens = jina_config["max_tokens"]
        self.batch_size = jina_config["batch_size"]
        self.output_dim = jina_config.get("output_dim", 2048)
        self.matryoshka_dim = jina_config.get("matryoshka_dim")

        logger.info(f"Loading Jina v4 model: {self._model_name}")

        # Load model with trust_remote_code (required for Jina v4)
        self.model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.model.eval()

        # Determine actual embedding dimension
        if self.matryoshka_dim:
            self._embedding_dim = self.matryoshka_dim
        else:
            self._embedding_dim = self.output_dim

        logger.info(
            f"Jina v4 encoder initialized: dim={self._embedding_dim}, "
            f"task={self.task}, prompt={self.prompt_name}"
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts."""
        with torch.no_grad():
            # Jina v4 uses encode_text method with task and prompt_name
            embeddings = self.model.encode_text(
                texts=texts,
                task=self.task,
                prompt_name=self.prompt_name,
                max_length=self.max_tokens,
            )

            # Apply Matryoshka truncation if specified
            if self.matryoshka_dim and self.matryoshka_dim < embeddings.shape[1]:
                embeddings = embeddings[:, :self.matryoshka_dim]

            # Convert to numpy
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().float().numpy()

        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        return self.encode([text])[0]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return self._model_name
