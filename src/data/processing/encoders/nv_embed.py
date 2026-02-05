"""
NV-Embed-v2 Encoder

NVIDIA's embedding model with 4-bit quantization support.
Based on Mistral-7B with latent attention layer.
"""

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .base import BaseTextEncoder

logger = logging.getLogger(__name__)


class NVEmbedEncoder(BaseTextEncoder):
    """NV-Embed-v2 encoder with quantization support."""

    def __init__(self, config: dict):
        """
        Initialize NV-Embed-v2 encoder.

        Args:
            config: Full config dict with processing.text_encoder.nv_embed section
        """
        nv_config = config["processing"]["text_encoder"]["nv_embed"]

        self._model_name = nv_config["model_name"]
        self.max_tokens = nv_config["max_tokens"]
        self.batch_size = nv_config["batch_size"]
        self.instruction_prefix = nv_config["instruction_prefix"]

        quant_config = nv_config["quantization"]
        self.quantization_enabled = quant_config["enabled"]

        logger.info(f"Loading NV-Embed-v2: {self._model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )

        # Load model with quantization if enabled
        self.model = self._load_model(quant_config)
        self.model.eval()

        # NV-Embed-v2 has 4096-dimensional embeddings
        self._embedding_dim = 4096

        logger.info(
            f"NV-Embed-v2 initialized: dim={self._embedding_dim}, "
            f"quantized={self.quantization_enabled}"
        )

    def _load_model(self, quant_config: dict) -> Any:
        """Load model with optional quantization."""
        if self.quantization_enabled:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, quant_config["compute_dtype"]),
                bnb_4bit_quant_type=quant_config["quant_type"],
                bnb_4bit_use_double_quant=quant_config["use_double_quant"],
            )
            model = AutoModel.from_pretrained(
                self._model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto",
            )
        else:
            model = AutoModel.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
            )
        return model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts using NV-Embed-v2's built-in encode method."""
        with torch.no_grad():
            # NV-Embed-v2 has a built-in encode() method that handles everything
            embeddings = self.model.encode(
                texts,
                instruction=self.instruction_prefix,
                max_length=self.max_tokens,
            )
            # Return raw embeddings without normalization

        return embeddings.cpu().float().numpy()

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        return self.encode([text])[0]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return self._model_name
