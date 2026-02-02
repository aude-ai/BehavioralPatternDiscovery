"""
Text Encoder Module

Registry-based text encoding with multiple backend support:
- Jina v3: Contrastive-trained embeddings via SentenceTransformer (1024 dim)
- Jina v4: Multimodal embeddings based on Qwen2.5-VL (2048 dim)
- NV-Embed-v2: NVIDIA's embedding model with 4-bit quantization (4096 dim)
- Qwen Raw: Raw LLM hidden states from Qwen2.5 (variable dim)
- Qwen3-Embedding-8B: Dedicated embedding model with 4096 dim output
"""

from src.core.registry import ComponentRegistry

from .base import BaseTextEncoder
from .jina import JinaV3Encoder
from .jina_v4 import JinaV4Encoder
from .nv_embed import NVEmbedEncoder
from .qwen_raw import QwenRawEncoder
from .qwen3_embedding import Qwen3EmbeddingEncoder

# Registry for text encoders
text_encoder_registry = ComponentRegistry[BaseTextEncoder]("text_encoder")

# Register encoders
text_encoder_registry.register_class("jina", JinaV3Encoder)  # Backward compat alias
text_encoder_registry.register_class("jina_v3", JinaV3Encoder)
text_encoder_registry.register_class("jina_v4", JinaV4Encoder)
text_encoder_registry.register_class("nv_embed", NVEmbedEncoder)
text_encoder_registry.register_class("qwen_raw", QwenRawEncoder)
text_encoder_registry.register_class("qwen3_embedding", Qwen3EmbeddingEncoder)


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
    "JinaV3Encoder",
    "JinaV4Encoder",
    "NVEmbedEncoder",
    "QwenRawEncoder",
    "Qwen3EmbeddingEncoder",
    "text_encoder_registry",
    "create_text_encoder",
]
