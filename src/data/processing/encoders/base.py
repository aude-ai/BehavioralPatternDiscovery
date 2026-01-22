"""
Base Text Encoder Interface

Defines the contract all text encoders must implement.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseTextEncoder(ABC):
    """Abstract base class for text encoders."""

    @abstractmethod
    def __init__(self, config: dict):
        """
        Initialize encoder from config.

        Args:
            config: Full config dict with processing.text_encoder section
        """
        pass

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding.

        Args:
            text: Text string to encode

        Returns:
            numpy array of shape (embedding_dim,)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for logging."""
        pass
