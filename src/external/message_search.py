"""
Message Search Module

Provides pattern-based message search for evidence snippets.
Uses message_scores.h5 for efficient per-message score queries.
"""

import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

from src.external.schemas import MessageSearchResult, MessageSearchResponse
from src.pattern_identification.message_scorer import MessageScorer

logger = logging.getLogger(__name__)


class MessageSearcher:
    """Search messages by pattern activation using message_scores.h5."""

    def __init__(self, data_dir: Path):
        """
        Initialize searcher.

        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.pattern_id_dir = self.data_dir / "pattern_identification"

        # Lazy-loaded data
        self._message_db: Optional[List[Dict]] = None
        self._h5_path: Optional[Path] = None

    @property
    def h5_path(self) -> Path:
        """Path to message_scores.h5 file."""
        if self._h5_path is None:
            self._h5_path = self.pattern_id_dir / "scoring" / "message_scores.h5"
        return self._h5_path

    @property
    def message_db(self) -> List[Dict]:
        """Lazy load message database."""
        if self._message_db is None:
            db_path = self.pattern_id_dir / "scoring" / "message_database.pkl"
            if db_path.exists():
                with open(db_path, 'rb') as f:
                    data = pickle.load(f)
                    self._message_db = data.get("messages", data)
                logger.info(f"Loaded message database with {len(self._message_db)} messages")
            else:
                logger.warning(f"Message database not found at {db_path}")
                self._message_db = []
        return self._message_db

    def _parse_pattern_id(self, pattern_id: str) -> tuple:
        """
        Parse pattern ID into components.

        Args:
            pattern_id: e.g., "unified_3", "enc1_bottom_5"

        Returns:
            (level_key, pattern_idx) e.g., ("unified", 3) or ("enc1_bottom", 5)
        """
        parts = pattern_id.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid pattern_id format: {pattern_id}")

        level_key, dim_str = parts

        try:
            pattern_idx = int(dim_str)
        except ValueError:
            raise ValueError(f"Invalid pattern_id format: {pattern_id}")

        return (level_key, pattern_idx)

    def _parse_timestamp(self, ts_value: Any) -> datetime:
        """Parse timestamp from various formats."""
        if isinstance(ts_value, datetime):
            return ts_value

        if isinstance(ts_value, str):
            try:
                return datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return datetime.now()

    def search_by_pattern(
        self,
        pattern_id: str,
        top_k: int = 20,
        engineer_id: Optional[str] = None
    ) -> MessageSearchResponse:
        """
        Search for messages that activate a pattern.

        Args:
            pattern_id: Pattern identifier (e.g., "unified_3", "enc1_bottom_5")
            top_k: Maximum number of results
            engineer_id: Optional filter to specific engineer

        Returns:
            MessageSearchResponse with matching messages
        """
        if not self.h5_path.exists():
            logger.warning(f"Message scores file not found at {self.h5_path}")
            return MessageSearchResponse(
                pattern_id=pattern_id,
                total_matches=0,
                returned_count=0,
                messages=[],
            )

        level_key, pattern_idx = self._parse_pattern_id(pattern_id)

        # Query messages from message_scores.h5
        pattern_examples = MessageScorer.get_top_messages_for_pattern(
            h5_path=self.h5_path,
            level_key=level_key,
            pattern_idx=pattern_idx,
            message_database=self.message_db,
            limit=top_k * 5 if engineer_id else top_k,  # Get more if filtering
        )

        results: List[MessageSearchResult] = []

        for rank, example in enumerate(pattern_examples):
            msg_idx = example.get("message_idx")
            if msg_idx is None:
                continue

            msg_data = self.message_db[msg_idx] if msg_idx < len(self.message_db) else {}
            if not msg_data:
                continue

            msg_engineer = example.get("engineer_id", msg_data.get("engineer_id", "unknown"))

            # Filter by engineer if specified
            if engineer_id and msg_engineer != engineer_id:
                continue

            timestamp = self._parse_timestamp(msg_data.get("timestamp"))

            result = MessageSearchResult(
                message_id=str(msg_idx),
                raw_ref_id=msg_data.get("raw_ref_id"),
                raw_ref_collection=msg_data.get("raw_ref_collection"),
                engineer_id=msg_engineer,
                timestamp=timestamp,
                source=msg_data.get("source", "unknown"),
                text=msg_data.get("text", ""),
                pattern_id=pattern_id,
                activation_score=example.get("score", 0.0),
                activation_rank=len(results) + 1,
            )
            results.append(result)

            if len(results) >= top_k:
                break

        return MessageSearchResponse(
            pattern_id=pattern_id,
            total_matches=len(pattern_examples),
            returned_count=len(results),
            messages=results,
        )

    def get_user_pattern_messages(
        self,
        engineer_id: str,
        pattern_id: str,
        top_k: int = 5
    ) -> MessageSearchResponse:
        """
        Get a specific user's top messages for a pattern.

        This is useful for generating evidence snippets in evaluations.
        """
        return self.search_by_pattern(
            pattern_id=pattern_id,
            top_k=top_k,
            engineer_id=engineer_id
        )

    def get_message_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get full message data by ID."""
        try:
            idx = int(message_id)
            if 0 <= idx < len(self.message_db):
                return self.message_db[idx]
        except (ValueError, TypeError):
            pass
        return None

    def get_all_patterns(self) -> List[str]:
        """Get list of all available pattern IDs from message_scores.h5."""
        import h5py

        if not self.h5_path.exists():
            return []

        pattern_ids = []

        with h5py.File(self.h5_path, 'r') as f:
            for level_key in f.keys():
                if level_key in ('engineer_ids', 'message_indices'):
                    continue
                n_dims = f[level_key].shape[1]
                for dim_idx in range(n_dims):
                    pattern_ids.append(f"{level_key}_{dim_idx}")

        return sorted(pattern_ids)
