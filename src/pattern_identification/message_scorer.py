"""
Message Scorer

Computes and stores per-message activation scores for flexible querying.
Runs during batch scoring (B.6) after activations are computed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class MessageScorer:
    """Store per-message pattern scores for flexible querying."""

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Config with paths and scoring settings
        """
        self.config = config
        self.percentile_levels = [25, 50, 75, 90, 95]

    def save_message_scores(
        self,
        activations: dict[str, np.ndarray],
        message_database: list[dict],
        output_path: Path,
    ) -> dict[str, Any]:
        """
        Save per-message activation scores to HDF5.

        Args:
            activations: Dict mapping level keys to (n_messages, n_dims) arrays
            message_database: List of message dicts with 'engineer_id'
            output_path: Path for HDF5 output

        Returns:
            Index metadata for quick lookups
        """
        n_messages = len(message_database)
        logger.info(f"Saving message scores for {n_messages} messages")

        engineer_ids = [msg.get("engineer_id", "unknown") for msg in message_database]
        unique_engineers = sorted(set(engineer_ids))

        engineer_index = {eng: {"message_indices": [], "n_messages": 0} for eng in unique_engineers}
        for idx, eng_id in enumerate(engineer_ids):
            engineer_index[eng_id]["message_indices"].append(idx)
            engineer_index[eng_id]["n_messages"] += 1

        percentile_thresholds = {}
        for level_key, scores in activations.items():
            percentile_thresholds[level_key] = {}
            for p in self.percentile_levels:
                percentile_thresholds[level_key][f"p{p}"] = np.percentile(
                    scores, p, axis=0
                ).tolist()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("engineer_ids", data=engineer_ids, dtype=dt)

            f.create_dataset("message_indices", data=np.arange(n_messages))

            for level_key, scores in activations.items():
                f.create_dataset(
                    level_key,
                    data=scores.astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )

            f.attrs["n_messages"] = n_messages
            f.attrs["created_at"] = np.bytes_(str(np.datetime64("now")))

        logger.info(f"Saved message scores to {output_path}")

        index = {
            "n_messages": n_messages,
            "engineers": engineer_index,
            "levels": {
                level_key: {"n_dims": scores.shape[1]}
                for level_key, scores in activations.items()
            },
            "percentile_thresholds": percentile_thresholds,
        }

        return index

    @staticmethod
    def load_message_scores(
        h5_path: Path,
        level_key: str,
        message_indices: list[int] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Load scores for specific messages and level.

        Args:
            h5_path: Path to HDF5 file
            level_key: Level to load (e.g., "enc1_bottom")
            message_indices: Optional subset of indices to load

        Returns:
            (scores array, engineer_ids list)
        """
        with h5py.File(h5_path, "r") as f:
            if message_indices is not None:
                indices = np.array(message_indices)
                scores = f[level_key][indices]
                engineer_ids = []
                for i in indices:
                    eid = f["engineer_ids"][i]
                    if isinstance(eid, bytes):
                        engineer_ids.append(eid.decode())
                    else:
                        engineer_ids.append(str(eid))
            else:
                scores = f[level_key][:]
                engineer_ids = []
                for eid in f["engineer_ids"][:]:
                    if isinstance(eid, bytes):
                        engineer_ids.append(eid.decode())
                    else:
                        engineer_ids.append(str(eid))

        return scores, engineer_ids

    @staticmethod
    def get_top_messages_for_pattern(
        h5_path: Path,
        level_key: str,
        pattern_idx: int,
        message_database: list[dict],
        limit: int = 20,
        message_indices: list[int] | None = None,
    ) -> list[dict]:
        """
        Get top messages for a specific pattern dimension.

        Args:
            h5_path: Path to HDF5 file
            level_key: Level to query (e.g., "enc1_bottom")
            pattern_idx: Dimension index within the level
            message_database: List of message dicts
            limit: Max messages to return
            message_indices: Optional subset to query (e.g., for single engineer)

        Returns:
            List of message dicts with score info
        """
        scores, eng_ids = MessageScorer.load_message_scores(
            h5_path, level_key, message_indices
        )

        dim_scores = scores[:, pattern_idx]
        top_k = min(limit, len(dim_scores))
        top_indices = np.argsort(-np.abs(dim_scores))[:top_k]

        if message_indices is not None:
            actual_indices = [message_indices[i] for i in top_indices]
        else:
            actual_indices = top_indices.tolist()

        messages = []
        for i, idx in enumerate(top_indices):
            actual_idx = actual_indices[i]
            msg = message_database[actual_idx]
            messages.append({
                "text": msg.get("text", "")[:500],
                "score": float(dim_scores[idx]),
                "engineer_id": eng_ids[idx],
                "message_idx": actual_idx,
                "source": msg.get("source"),
                "activity_type": msg.get("activity_type"),
            })

        return messages
