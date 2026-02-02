"""
Jina Embeddings v3 Encoder

Jina v3 model via SentenceTransformer.
Outputs raw embeddings (normalization removed from pipeline).
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize

from .base import BaseTextEncoder

logger = logging.getLogger(__name__)


class JinaV3Encoder(BaseTextEncoder):
    """Jina v3 embeddings encoder using SentenceTransformer."""

    def __init__(self, config: dict):
        """
        Initialize Jina v3 encoder.

        Args:
            config: Full config dict with processing.text_encoder.jina_v3 section
                   (also accepts legacy 'jina' key for backward compatibility)
        """
        text_encoder_config = config["processing"]["text_encoder"]
        # Support both jina_v3 and legacy jina key
        if "jina_v3" in text_encoder_config:
            jina_config = text_encoder_config["jina_v3"]
        else:
            jina_config = text_encoder_config["jina"]

        self._model_name = jina_config["model_name"]
        self.task = jina_config["task"]
        self.max_tokens = jina_config["max_tokens"]
        self.batch_size = jina_config["batch_size"]

        logger.info(f"Loading Jina model: {self._model_name}")

        # SentenceTransformer handles device placement automatically
        model_kwargs = {"trust_remote_code": True}
        self.model = SentenceTransformer(self._model_name, **model_kwargs)

        # Remove the built-in Normalize layer from the pipeline to get raw embeddings
        # Jina v3 includes: transformer -> pooler -> normalizer
        # We want: transformer -> pooler (no normalizer)
        self._remove_normalize_layer()

        # Get embedding dimension from model
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Jina encoder initialized: dim={self._embedding_dim}")

    def _remove_normalize_layer(self):
        """Remove the Normalize layer from the SentenceTransformer pipeline."""
        modules_to_keep = []
        removed = False

        for idx, module in enumerate(self.model._modules.values()):
            if isinstance(module, Normalize):
                logger.info(f"Removing built-in Normalize layer from Jina pipeline (index {idx})")
                removed = True
            else:
                modules_to_keep.append(module)

        if removed:
            # Rebuild the model with the filtered modules
            from collections import OrderedDict
            new_modules = OrderedDict()
            for i, module in enumerate(modules_to_keep):
                new_modules[str(i)] = module
            self.model._modules = new_modules
            logger.info(f"Pipeline now has {len(modules_to_keep)} modules (normalizer removed)")
        else:
            logger.info("No Normalize layer found in pipeline (already raw)")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to raw embeddings (no normalization)."""
        embeddings = self.model.encode(
            texts,
            task=self.task,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=False,
        )
        return np.array(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        return self.encode([text])[0]

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return self._model_name
