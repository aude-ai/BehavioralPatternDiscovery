"""
Qwen Raw Embeddings Encoder

Extracts raw hidden states from Qwen text models.
Supports:
- Qwen2.5-3B-Instruct (smaller, can run unquantized)
- Qwen2.5-7B-Instruct (larger, may require quantization)
- Other Qwen2/Qwen2.5 text models
"""

import logging
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .base import BaseTextEncoder

logger = logging.getLogger(__name__)


class QwenRawEncoder(BaseTextEncoder):
    """Raw Qwen hidden state encoder."""

    def __init__(self, config: dict):
        """
        Initialize Qwen raw encoder.

        Args:
            config: Full config dict with processing.text_encoder.qwen_raw section
        """
        qwen_config = config["processing"]["text_encoder"]["qwen_raw"]

        self._model_name = qwen_config["model_name"]
        self.max_tokens = qwen_config["max_tokens"]
        self.batch_size = qwen_config["batch_size"]

        quant_config = qwen_config["quantization"]
        self.quantization_type = quant_config["type"]

        logger.info(f"Loading Qwen model: {self._model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
        self.model = self._load_model(quant_config)
        self.model.eval()

        # Get embedding dimension from model config
        self._embedding_dim = self.model.config.hidden_size

        logger.info(
            f"Qwen encoder initialized: dim={self._embedding_dim}, "
            f"quantization={self.quantization_type}"
        )

    def _load_model(self, quant_config: dict) -> Any:
        """Load model with optional quantization."""
        if self.quantization_type == "none":
            model = AutoModel.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif self.quantization_type == "int8":
            model = AutoModel.from_pretrained(
                self._model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
        elif self.quantization_type == "int4":
            int4_config = quant_config["int4"]
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, int4_config["compute_dtype"]),
                bnb_4bit_quant_type=int4_config["quant_type"],
                bnb_4bit_use_double_quant=int4_config["use_double_quant"],
            )
            model = AutoModel.from_pretrained(
                self._model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unknown quantization type: {self.quantization_type}")

        return model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to raw hidden state embeddings."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

        embeddings = np.concatenate(all_embeddings, axis=0)

        return embeddings

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts to raw hidden states."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(self.model.device)

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract last hidden state of last token for each sequence
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # Get the last non-padding token position for each sequence
        sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
        batch_size = last_hidden.shape[0]

        embeddings = []
        for i in range(batch_size):
            last_token_idx = sequence_lengths[i].item()
            embedding = last_hidden[i, last_token_idx, :]
            embeddings.append(embedding)

        return torch.stack(embeddings).cpu().float().numpy()

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        return self.encode([text])[0]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return self._model_name
