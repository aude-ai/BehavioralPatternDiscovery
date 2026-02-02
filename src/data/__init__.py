"""
Data Module

Contains data collection, synthetic generation, processing, and dataset utilities.

Structure:
- collection/: NDJSON loading and activity parsing (no torch)
- synthetic/: LLM-based synthetic profile generation (no torch)
- processing/: Text encoding, feature extraction, normalization (requires torch)
- datasets.py: PyTorch datasets for training (requires torch)

This module only exports non-torch components.
For torch-dependent components, import directly from submodules:
    from src.data.datasets import EngineerSequenceDataset
    from src.data.processing import JinaEncoder, NormalizationPipeline
"""

from .collection import EngineerManager

__all__ = [
    "EngineerManager",
]
