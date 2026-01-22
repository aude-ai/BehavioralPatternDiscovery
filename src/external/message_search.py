"""
Message Search Module

Provides pattern-based message search for evidence snippets.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

from src.external.schemas import MessageSearchResult, MessageSearchResponse

logger = logging.getLogger(__name__)


class MessageSearcher:
    """Search messages by pattern activation."""

    def __init__(self, data_dir: Path):
        """
        Initialize searcher.

        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.pattern_id_dir = self.data_dir / "pattern_identification"

        # Lazy-loaded data
        self._message_db: Optional[Dict] = None
        self._message_examples: Optional[Dict] = None

    @property
    def message_db(self) -> Dict:
        """Lazy load message database."""
        if self._message_db is None:
            db_path = self.pattern_id_dir / "assignment" / "message_database.json"
            if db_path.exists():
                with open(db_path, 'r') as f:
                    self._message_db = json.load(f)
                logger.info(f"Loaded message database with {len(self._message_db)} messages")
            else:
                logger.warning(f"Message database not found at {db_path}")
                self._message_db = {}
        return self._message_db

    @property
    def message_examples(self) -> Dict:
        """Lazy load message examples."""
        if self._message_examples is None:
            ex_path = self.pattern_id_dir / "assignment" / "message_examples.json"
            if ex_path.exists():
                with open(ex_path, 'r') as f:
                    self._message_examples = json.load(f)
                logger.info(f"Loaded message examples for {len(self._message_examples)} pattern groups")
            else:
                logger.warning(f"Message examples not found at {ex_path}")
                self._message_examples = {}
        return self._message_examples

    def _parse_pattern_id(self, pattern_id: str) -> tuple:
        """
        Parse pattern ID into components.

        Args:
            pattern_id: e.g., "unified_3", "enc1_bottom_5"

        Returns:
            (key, dim_key) e.g., ("unified", "unified_3") or ("enc1_bottom", "bottom_5")
        """
        parts = pattern_id.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid pattern_id format: {pattern_id}")

        key_part, dim_str = parts

        try:
            dim_idx = int(dim_str)
        except ValueError:
            raise ValueError(f"Invalid pattern_id format: {pattern_id}")

        if key_part == "unified":
            return ("unified", f"unified_{dim_idx}")
        else:
            # e.g., "enc1_bottom" -> key="enc1_bottom", need to extract level
            level_parts = key_part.split("_", 1)
            if len(level_parts) == 2:
                level = level_parts[1]
                return (key_part, f"{level}_{dim_idx}")
            return (key_part, f"{key_part}_{dim_idx}")

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
            pattern_id: Pattern identifier
            top_k: Maximum number of results
            engineer_id: Optional filter to specific engineer

        Returns:
            MessageSearchResponse with matching messages
        """
        key, dim_key = self._parse_pattern_id(pattern_id)

        # Get examples from message_examples.json
        key_examples = self.message_examples.get(key, {})
        pattern_examples = key_examples.get(dim_key, [])

        results: List[MessageSearchResult] = []
        total_matches = len(pattern_examples)

        for rank, example in enumerate(pattern_examples):
            msg_idx = example.get("message_idx")
            if msg_idx is None:
                continue

            msg_data = self.message_db.get(str(msg_idx), {})
            if not msg_data:
                continue

            msg_engineer = msg_data.get("engineer_id", "unknown")

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
                activation_score=example.get("activation", 0.0),
                activation_rank=rank + 1,
            )
            results.append(result)

            if len(results) >= top_k:
                break

        return MessageSearchResponse(
            pattern_id=pattern_id,
            total_matches=total_matches,
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
        return self.message_db.get(str(message_id))

    def get_all_patterns(self) -> List[str]:
        """Get list of all available pattern IDs."""
        pattern_ids = []

        for key, dim_dict in self.message_examples.items():
            for dim_key in dim_dict.keys():
                # Convert dim_key back to pattern_id format
                # dim_key is like "bottom_0", we need "enc1_bottom_0" or "unified_0"
                if key == "unified":
                    parts = dim_key.split("_")
                    if len(parts) >= 2:
                        pattern_ids.append(f"unified_{parts[-1]}")
                else:
                    parts = dim_key.split("_")
                    if len(parts) >= 2:
                        pattern_ids.append(f"{key}_{parts[-1]}")

        return sorted(pattern_ids)
