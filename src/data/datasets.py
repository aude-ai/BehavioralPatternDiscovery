"""
Flexible Batching Datasets

Supports multiple batching strategies:
- Engineer-specific: Each batch = all messages from one engineer
- Multi-engineer: Each batch = messages from N engineers (stratified)
- Random: Completely random message sampling across all engineers

Data Loading:
- Datasets load pre-combined training data from numpy arrays
- Message metadata (engineer_id, text, etc.) from message_database.pkl
- TrainingDataPreparer creates the combined arrays based on aux_features config
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.core.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Registry for dataset implementations
dataset_registry = ComponentRegistry[Dataset]("dataset")


def _load_message_database(path: str | Path) -> tuple[list[dict], dict]:
    """
    Load message database and metadata.

    Handles both old format (list) and new format (dict with messages/metadata).

    Args:
        path: Path to message_database.pkl

    Returns:
        Tuple of (messages list, metadata dict)
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # New format: dict with 'messages' and 'metadata'
    if isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
        metadata = data["metadata"]
    else:
        # Old format: just a list of messages (backward compatibility)
        messages = data
        # Infer metadata from first message
        if messages:
            first_msg = messages[0]
            metadata = {
                "embedding_dim": first_msg["embedding"].shape[0],
                "aux_features_dim": first_msg["aux_features"].shape[0],
                "num_messages": len(messages),
            }
        else:
            metadata = {}

    return messages, metadata


def _load_training_data(
    message_database_path: str | Path,
    training_input_path: str | Path | None,
) -> tuple[list[dict], np.ndarray, dict]:
    """
    Load message metadata and training input data.

    Args:
        message_database_path: Path to message_database.pkl (for metadata)
        training_input_path: Path to prepared training input numpy array
                            If None, falls back to loading from message_database

    Returns:
        Tuple of (messages list, training data array, metadata dict)
    """
    messages, metadata = _load_message_database(message_database_path)

    if training_input_path is not None:
        # Load pre-combined training data
        training_data = np.load(training_input_path)
        logger.info(f"Loaded training data from {training_input_path}: shape={training_data.shape}")

        # Validate row counts match
        if len(messages) != training_data.shape[0]:
            raise ValueError(
                f"Row count mismatch: message_database has {len(messages)} messages, "
                f"training data has {training_data.shape[0]} rows"
            )
    else:
        # Fallback: extract from message_database (backward compatibility)
        logger.warning("No training_input_path provided, falling back to message_database extraction")
        training_data = np.array([
            np.concatenate([msg["embedding"], msg["aux_features"]])
            for msg in messages
        ], dtype=np.float32)

    return messages, training_data, metadata


# =============================================================================
# Multi-Engineer Dataset (Uses ALL Messages with Diversity)
# =============================================================================


@dataset_registry.register("multi_engineer")
class MultiEngineerDataset(Dataset):
    """
    Dataset that uses ALL messages, distributed into batches with engineer diversity.

    Unlike random batching, this ensures:
    - All messages are used each epoch (no discarding)
    - Each batch contains messages from multiple engineers
    - No single engineer dominates a batch (max messages_per_engineer per batch)

    Returns pre-built batches. Use with batch_size=1 and engineer_collate_fn.
    """

    def __init__(
        self,
        message_database_path: str | Path,
        engineer_ids: list[str],
        batch_size: int = 512,
        messages_per_engineer: int = 32,
        training_input_path: str | Path | None = None,
    ):
        """
        Args:
            message_database_path: Path to message_database.pkl
            engineer_ids: List of engineer IDs to include
            batch_size: Target size for each batch
            messages_per_engineer: Max messages from one engineer per batch
            training_input_path: Path to prepared training input numpy array
        """
        self.batch_size = batch_size
        self.messages_per_engineer = messages_per_engineer
        self.engineer_ids_set = set(engineer_ids)

        # Load message metadata and training data
        logger.info(f"Loading message database from {message_database_path}")
        messages, self.training_data, metadata = _load_training_data(
            message_database_path, training_input_path
        )

        logger.info(f"  Input dim: {self.training_data.shape[1]}")

        # Group message indices by engineer
        self.engineer_indices: dict[str, list[int]] = {}
        for idx, msg in enumerate(messages):
            eng_id = msg["engineer_id"]
            if eng_id in self.engineer_ids_set:
                if eng_id not in self.engineer_indices:
                    self.engineer_indices[eng_id] = []
                self.engineer_indices[eng_id].append(idx)

        # Count total messages
        self.total_messages = sum(len(idxs) for idxs in self.engineer_indices.values())
        self.num_engineers = len(self.engineer_indices)

        logger.info(
            f"MultiEngineerDataset: {self.num_engineers} engineers, "
            f"{self.total_messages} total messages, "
            f"batch_size={batch_size}, max {messages_per_engineer} per engineer per batch"
        )

        # Build initial batches
        self._build_batches()

    def _build_batches(self):
        """Build batches distributing all messages with engineer diversity."""
        # Shuffle indices within each engineer
        engineer_queues = {}
        for eng_id, indices in self.engineer_indices.items():
            shuffled = list(indices)
            np.random.shuffle(shuffled)
            engineer_queues[eng_id] = shuffled

        # Build batches by round-robin from engineers (store indices, not data)
        self.batches: list[list[int]] = []
        current_batch: list[int] = []
        current_batch_counts: dict[str, int] = {}

        # Keep cycling until all messages are assigned
        while any(engineer_queues.values()):
            added_this_round = False

            # Try to add one message from each engineer (round-robin)
            for eng_id in list(engineer_queues.keys()):
                if not engineer_queues[eng_id]:
                    continue

                # Check if this engineer can add to current batch
                count = current_batch_counts.get(eng_id, 0)
                if count < self.messages_per_engineer:
                    idx = engineer_queues[eng_id].pop(0)
                    current_batch.append(idx)
                    current_batch_counts[eng_id] = count + 1
                    added_this_round = True

                    # Batch full - save and start new one
                    if len(current_batch) >= self.batch_size:
                        self.batches.append(current_batch)
                        current_batch = []
                        current_batch_counts = {}

            # If no messages added but some remain, all engineers hit their per-batch limit
            # Start a new batch to continue
            if not added_this_round and any(engineer_queues.values()):
                if current_batch:
                    self.batches.append(current_batch)
                current_batch = []
                current_batch_counts = {}

        # Save final partial batch
        if current_batch:
            self.batches.append(current_batch)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a pre-built batch.

        Args:
            idx: Batch index

        Returns:
            Dictionary with embeddings tensor
        """
        batch_indices = self.batches[idx]

        # Get data for batch indices
        batch_tensor = self.training_data[batch_indices].astype(np.float32)

        # Shuffle within batch
        np.random.shuffle(batch_tensor)

        return {
            "embeddings": torch.from_numpy(batch_tensor),
        }

    def on_epoch_end(self):
        """Rebuild batches with fresh shuffling for next epoch."""
        self._build_batches()


# =============================================================================
# Random Message Dataset (Uses ALL Messages)
# =============================================================================


@dataset_registry.register("random")
class RandomMessageDataset(Dataset):
    """
    Dataset that uses ALL messages with random shuffling each epoch.

    Maximum diversity - each batch contains random messages from any engineer.
    All messages are used exactly once per epoch.

    Returns pre-built batches. Use with batch_size=1 and engineer_collate_fn.
    """

    def __init__(
        self,
        message_database_path: str | Path,
        engineer_ids: list[str],
        batch_size: int = 512,
        training_input_path: str | Path | None = None,
    ):
        """
        Args:
            message_database_path: Path to message_database.pkl
            engineer_ids: List of engineer IDs to include
            batch_size: Number of messages per batch
            training_input_path: Path to prepared training input numpy array
        """
        self.batch_size = batch_size
        self.engineer_ids_set = set(engineer_ids)

        # Load message metadata and training data
        logger.info(f"Loading message database from {message_database_path}")
        messages, self.training_data, metadata = _load_training_data(
            message_database_path, training_input_path
        )

        logger.info(f"  Input dim: {self.training_data.shape[1]}")

        # Filter to get indices for selected engineers
        self.all_indices = [
            idx for idx, msg in enumerate(messages)
            if msg["engineer_id"] in self.engineer_ids_set
        ]

        self.num_messages = len(self.all_indices)

        logger.info(
            f"RandomMessageDataset: {self.num_messages} messages, "
            f"batch_size={batch_size}"
        )

        # Build initial batches
        self._build_batches()

    def _build_batches(self):
        """Shuffle all indices and build batches."""
        shuffled = list(self.all_indices)
        np.random.shuffle(shuffled)

        self.batches: list[list[int]] = []
        for i in range(0, len(shuffled), self.batch_size):
            batch = shuffled[i:i + self.batch_size]
            if batch:  # Include partial final batch
                self.batches.append(batch)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a pre-built batch.

        Args:
            idx: Batch index

        Returns:
            Dictionary with embeddings tensor
        """
        batch_indices = self.batches[idx]

        # Get data for batch indices
        batch_tensor = self.training_data[batch_indices].astype(np.float32)

        return {
            "embeddings": torch.from_numpy(batch_tensor),
        }

    def on_epoch_end(self):
        """Rebuild batches with fresh shuffling for next epoch."""
        self._build_batches()


# =============================================================================
# Original Engineer Sequence Dataset
# =============================================================================


@dataset_registry.register("engineer")
class EngineerSequenceDataset(Dataset):
    """
    Dataset that returns all messages for a single engineer as one sample.

    Each sample = all messages from one engineer (variable count).
    DataLoader batch_size MUST be 1.

    Outputs per sample:
        - embeddings: (N_messages, input_dim)
        - engineer_id: str
        - num_messages: int
    """

    def __init__(
        self,
        message_database_path: str | Path,
        engineer_ids: list[str],
        max_messages: int,
        training_input_path: str | Path | None = None,
    ):
        """
        Args:
            message_database_path: Path to message_database.pkl
            engineer_ids: List of engineer IDs to include
            max_messages: Maximum messages per engineer (truncate if exceeded)
            training_input_path: Path to prepared training input numpy array
        """
        self.max_messages = max_messages
        self.engineer_ids = engineer_ids

        # Load message metadata and training data
        logger.info(f"Loading message database from {message_database_path}")
        messages, self.training_data, metadata = _load_training_data(
            message_database_path, training_input_path
        )

        logger.info(f"  Input dim: {self.training_data.shape[1]}")

        # Group message indices by engineer
        self.engineer_indices: dict[str, list[int]] = {}
        for idx, msg in enumerate(messages):
            eng_id = msg["engineer_id"]
            if eng_id in engineer_ids:
                if eng_id not in self.engineer_indices:
                    self.engineer_indices[eng_id] = []
                self.engineer_indices[eng_id].append(idx)

        # Log statistics
        message_counts = [len(idxs) for idxs in self.engineer_indices.values()]
        if message_counts:
            logger.info(f"Loaded {len(self.engineer_indices)} engineers")
            logger.info(
                f"Message counts - min: {min(message_counts)}, "
                f"max: {max(message_counts)}, "
                f"mean: {np.mean(message_counts):.1f}, "
                f"median: {np.median(message_counts):.1f}"
            )
        else:
            raise ValueError("No messages found for any engineer in provided list")

    def __len__(self) -> int:
        """Number of engineers in dataset."""
        return len(self.engineer_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get all messages for one engineer.

        Args:
            idx: Engineer index

        Returns:
            Dictionary with embeddings, engineer_id, num_messages
        """
        eng_id = self.engineer_ids[idx]
        indices = self.engineer_indices[eng_id]

        # Truncate if too many messages
        if len(indices) > self.max_messages:
            indices = self._truncate_indices(indices, eng_id)

        # Get data for indices
        combined = self.training_data[indices].astype(np.float32)

        return {
            "embeddings": torch.from_numpy(combined),
            "engineer_id": eng_id,
            "num_messages": len(indices),
        }

    def _truncate_indices(
        self, indices: list[int], eng_id: str
    ) -> list[int]:
        """
        Truncate indices keeping recent + oldest + random middle.

        Args:
            indices: List of message indices
            eng_id: Engineer ID for logging

        Returns:
            Truncated list of indices
        """
        original_count = len(indices)

        # Strategy: Keep recent + oldest + random middle
        recent = indices[-self.max_messages // 4 :]
        oldest = indices[: self.max_messages // 4]

        middle_count = self.max_messages - len(recent) - len(oldest)
        middle_start = len(oldest)
        middle_end = len(indices) - len(recent)

        if middle_end > middle_start and middle_count > 0:
            middle_range = range(middle_start, middle_end)
            selected_middle = np.random.choice(
                middle_range,
                size=min(middle_count, len(middle_range)),
                replace=False,
            )
            middle = [indices[i] for i in sorted(selected_middle)]
        else:
            middle = []

        truncated = oldest + middle + recent

        logger.debug(
            f"Truncated engineer {eng_id} from {original_count} to {len(truncated)} messages"
        )

        return truncated


def engineer_collate_fn(batch: list[dict]) -> dict[str, Any]:
    """
    Collate function for pre-batched datasets.

    Receives a list of length 1 (batch_size=1).
    Extracts the single item's data.

    Works with:
    - EngineerSequenceDataset
    - MultiEngineerDataset
    - RandomMessageDataset

    Args:
        batch: List with 1 element

    Returns:
        Dictionary with embeddings (and optionally engineer_id, num_messages)
    """
    if len(batch) != 1:
        raise ValueError(
            f"Batch size must be 1 for pre-batched datasets. "
            f"Got batch size: {len(batch)}"
        )

    sample = batch[0]
    result = {"embeddings": sample["embeddings"]}

    # Include optional fields if present (for EngineerSequenceDataset)
    if "engineer_id" in sample:
        result["engineer_id"] = sample["engineer_id"]
    if "num_messages" in sample:
        result["num_messages"] = sample["num_messages"]

    return result


def create_dataset(
    mode: str,
    message_database_path: str | Path,
    engineer_ids: list[str],
    batch_size: int,
    max_messages: int,
    messages_per_engineer: int,
    training_input_path: str | Path | None = None,
) -> Dataset:
    """
    Create a dataset using the registry.

    Args:
        mode: Dataset type ("engineer", "multi_engineer", "random")
        message_database_path: Path to message_database.pkl
        engineer_ids: List of engineer IDs to include
        batch_size: Batch size for multi_engineer/random modes
        max_messages: Max messages per engineer (engineer mode only)
        messages_per_engineer: Max messages per engineer per batch (multi_engineer only)
        training_input_path: Path to prepared training input numpy array

    Returns:
        Dataset instance

    Raises:
        KeyError: If mode is not registered
    """
    if mode == "engineer":
        return dataset_registry.create(
            mode,
            message_database_path=message_database_path,
            engineer_ids=engineer_ids,
            max_messages=max_messages,
            training_input_path=training_input_path,
        )
    elif mode == "multi_engineer":
        return dataset_registry.create(
            mode,
            message_database_path=message_database_path,
            engineer_ids=engineer_ids,
            batch_size=batch_size,
            messages_per_engineer=messages_per_engineer,
            training_input_path=training_input_path,
        )
    elif mode == "random":
        return dataset_registry.create(
            mode,
            message_database_path=message_database_path,
            engineer_ids=engineer_ids,
            batch_size=batch_size,
            training_input_path=training_input_path,
        )
    else:
        raise KeyError(
            f"Unknown dataset mode: '{mode}'. "
            f"Available: {dataset_registry.list_registered()}"
        )
