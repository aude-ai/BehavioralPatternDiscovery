"""
Text Encoder Module

Registry-based text encoding with multiple backend support:
- Jina (v3, v4): Contrastive-trained embeddings via SentenceTransformer
- NV-Embed-v2: NVIDIA's embedding model with 4-bit quantization
- Qwen Raw: Raw LLM hidden states for maximum variance preservation
"""

from src.core.registry import ComponentRegistry

from .base import BaseTextEncoder
from .jina import JinaEncoder
from .nv_embed import NVEmbedEncoder
from .qwen_raw import QwenRawEncoder

# Registry for text encoders
text_encoder_registry = ComponentRegistry[BaseTextEncoder]("text_encoder")

# Register encoders
text_encoder_registry.register_class("jina", JinaEncoder)
text_encoder_registry.register_class("nv_embed", NVEmbedEncoder)
text_encoder_registry.register_class("qwen_raw", QwenRawEncoder)


def create_text_encoder(config: dict) -> BaseTextEncoder:
    """
    Create a text encoder from config.

    Args:
        config: Full config dict with processing.text_encoder section

    Returns:
        Configured text encoder instance
    """
    encoder_config = config["processing"]["text_encoder"]
    encoder_type = encoder_config["type"]
    return text_encoder_registry.create(encoder_type, config=config)


__all__ = [
    "BaseTextEncoder",
    "JinaEncoder",
    "NVEmbedEncoder",
    "QwenRawEncoder",
    "text_encoder_registry",
    "create_text_encoder",
]
