"""
Data Module

Contains data collection, synthetic generation, and processing.

Structure:
- collection/: NDJSON loading and activity parsing (no torch)
- synthetic/: LLM-based synthetic profile generation (no torch)
- processing/: Text encoding, feature extraction, normalization (requires torch)

This module only exports non-torch components.
For torch-dependent components, import directly from submodules:
    from src.data.processing import JinaV3Encoder, NormalizationPipeline
"""

from .collection import EngineerManager

__all__ = [
    "EngineerManager",
]
