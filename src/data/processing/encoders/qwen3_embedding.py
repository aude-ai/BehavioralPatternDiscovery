"""
Qwen3-Embedding-8B Encoder

8B parameter embedding model with 4096-dim output.
Supports up to 32K context length.
Uses last-token pooling for embeddings.
"""

import logging
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .base import BaseTextEncoder

# Ensure logs go to stdout for Modal
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class Qwen3EmbeddingEncoder(BaseTextEncoder):
    """Qwen3-Embedding-8B encoder using Transformers."""

    def __init__(self, config: dict):
        """
        Initialize Qwen3-Embedding-8B encoder.

        Args:
            config: Full config dict with processing.text_encoder.qwen3_embedding section
        """
        qwen_config = config["processing"]["text_encoder"]["qwen3_embedding"]

        self._model_name = qwen_config["model_name"]
        self.max_tokens = qwen_config["max_tokens"]
        self.batch_size = qwen_config["batch_size"]
        self.output_dim = qwen_config.get("output_dim", 4096)
        self.use_instruction = qwen_config.get("use_instruction", True)
        self.instruction = qwen_config.get(
            "instruction",
            "Given a text, retrieve semantically similar passages"
        )
        self.use_flash_attention = qwen_config.get("flash_attention", True)

        quant_config = qwen_config.get("quantization", {})
        self.quantization_type = quant_config.get("type", "none")

        logger.info(f"Loading Qwen3-Embedding: {self._model_name}")

        # Log GPU memory before loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU memory BEFORE model load: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

        # Load tokenizer with left padding (required for last-token pooling)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = self._load_model(quant_config)
        self.model.eval()

        # Get embedding dimension from model config
        self._embedding_dim = self.model.config.hidden_size

        logger.info(
            f"Qwen3-Embedding encoder initialized: dim={self._embedding_dim}, "
            f"quantization={self.quantization_type}, "
            f"flash_attention={self.use_flash_attention}"
        )

    def _load_model(self, quant_config: dict) -> Any:
        """Load model with optional quantization and flash attention."""
        # Following Hugging Face docs exactly:
        # model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B',
        #     attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()

        if self.quantization_type == "none":
            model_kwargs = {"torch_dtype": torch.float16}

            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            model = AutoModel.from_pretrained(self._model_name, **model_kwargs).cuda()

            # Log actual dtype and memory to verify
            param = next(model.parameters())
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"Model loaded: dtype={param.dtype}, device={param.device}")
            logger.info(f"GPU memory AFTER model load: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

        elif self.quantization_type == "int8":
            model = AutoModel.from_pretrained(
                self._model_name,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16,
            )

        elif self.quantization_type == "int4":
            int4_config = quant_config.get("int4", {})
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, int4_config.get("compute_dtype", "float16")
                ),
                bnb_4bit_quant_type=int4_config.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=int4_config.get("use_double_quant", True),
            )
            model = AutoModel.from_pretrained(
                self._model_name,
                device_map="auto",
                quantization_config=bnb_config,
            )

        else:
            raise ValueError(f"Unknown quantization type: {self.quantization_type}")

        return model

    def _last_token_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool embeddings using last token position.

        With left padding, the last token is always at position -1.
        """
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            # Fallback for right padding
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def _format_text(self, text: str) -> str:
        """Format text with instruction if enabled."""
        if self.use_instruction:
            return f"Instruct: {self.instruction}\nQuery: {text}"
        return text

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
        # Format texts with instruction
        formatted_texts = [self._format_text(t) for t in texts]

        # Tokenize
        inputs = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(self.model.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )
            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

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
