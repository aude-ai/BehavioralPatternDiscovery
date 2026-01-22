"""
Text Cleanup and Spam Detection

Provides configurable text cleanup and spam filtering for data loaders.
Includes quality filtering for low-information messages and per-engineer deduplication.
"""

import logging
import re
import unicodedata
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and filter text content based on configuration."""

    def __init__(self, config: dict):
        """
        Initialize text cleaner.

        Args:
            config: text_cleanup section from data config
        """
        self.enabled = config["enabled"]
        self.normalization = config["normalization"]
        self.filtering = config["filtering"]
        self.spam_config = config["spam_patterns"]
        self.url_config = config["urls"]
        self.code_config = config["code_blocks"]

        # Quality filter config
        self.quality_config = config["quality_filter"]

        # Deduplication config
        self.dedup_config = config["deduplication"]

        # Compile spam patterns
        self._spam_patterns: list[re.Pattern] = []
        if self.spam_config["enabled"]:
            for pattern in self.spam_config["patterns"]:
                try:
                    self._spam_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid spam pattern '{pattern}': {e}")

        # Compile quality filter patterns
        self._quality_patterns: list[re.Pattern] = []
        if self.quality_config["enabled"]:
            for pattern in self.quality_config["patterns"]:
                try:
                    self._quality_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid quality filter pattern '{pattern}': {e}")

        # URL pattern
        self._url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
        )

        # Code block patterns (markdown and common formats)
        self._code_block_pattern = re.compile(
            r'```[\s\S]*?```|`[^`\n]+`|<code>[\s\S]*?</code>',
            re.MULTILINE
        )

        # Deduplication state tracking (per-engineer text counts)
        self._dedup_counts: dict[tuple[str, str], int] = defaultdict(int)

        # Statistics tracking
        self._stats = {
            "excluded_by_quality": 0,
            "excluded_by_dedup": 0,
        }

        logger.info(f"TextCleaner initialized (enabled={self.enabled})")
        if self.enabled and self._spam_patterns:
            logger.info(f"  Spam patterns: {len(self._spam_patterns)}")
        if self.enabled and self._quality_patterns:
            logger.info(f"  Quality filter patterns: {len(self._quality_patterns)}")
        if self.enabled and self.dedup_config["enabled"]:
            logger.info(f"  Deduplication enabled: max {self.dedup_config['max_duplicates_per_engineer']} per engineer")

    def clean(self, text: str, engineer_id: Optional[str] = None) -> Optional[str]:
        """
        Clean text and check for spam/quality/duplicates.

        Args:
            text: Raw text content
            engineer_id: Optional engineer ID for per-engineer deduplication

        Returns:
            Cleaned text, or None if filtered out (spam/quality/duplicate/etc.)
        """
        if not self.enabled:
            return text

        if not text:
            return None

        # Apply cleanup steps in order
        text = self._handle_urls(text)
        text = self._handle_code_blocks(text)
        text = self._normalize(text)

        # Check filtering rules
        if not self._passes_filters(text):
            return None

        # Check spam patterns
        if self._is_spam(text):
            return None

        # Check quality filter patterns
        if self._is_low_quality(text):
            self._stats["excluded_by_quality"] += 1
            return None

        # Check per-engineer deduplication (requires engineer_id)
        if engineer_id and not self._passes_deduplication(text, engineer_id):
            self._stats["excluded_by_dedup"] += 1
            return None

        return text

    def _normalize(self, text: str) -> str:
        """Apply normalization rules."""
        if self.normalization["remove_null_bytes"]:
            text = text.replace("\x00", "")

        if self.normalization["normalize_unicode"]:
            text = unicodedata.normalize("NFKC", text)

        if self.normalization["collapse_whitespace"]:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

        if self.normalization["strip_leading_trailing"]:
            text = text.strip()

        return text

    def _handle_urls(self, text: str) -> str:
        """Handle URLs based on config."""
        mode = self.url_config["mode"]

        if mode == "keep":
            return text
        elif mode == "remove":
            return self._url_pattern.sub("", text)
        elif mode == "replace":
            return self._url_pattern.sub(self.url_config["replacement"], text)

        return text

    def _handle_code_blocks(self, text: str) -> str:
        """Handle code blocks based on config."""
        mode = self.code_config["mode"]

        if mode == "keep":
            max_lines = self.code_config["max_lines"]
            if max_lines:
                # Truncate code blocks to max lines
                def truncate_code(match):
                    code = match.group(0)
                    lines = code.split("\n")
                    if len(lines) > max_lines:
                        return "\n".join(lines[:max_lines]) + "\n..."
                    return code
                return self._code_block_pattern.sub(truncate_code, text)
            return text
        elif mode == "remove":
            return self._code_block_pattern.sub("", text)
        elif mode == "summarize":
            return self._code_block_pattern.sub(self.code_config["summary_text"], text)

        return text

    def _passes_filters(self, text: str) -> bool:
        """Check if text passes filtering rules."""
        # Min length
        min_len = self.filtering["min_text_length"]
        if min_len and len(text) < min_len:
            return False

        # Max length - truncate instead of filter
        max_len = self.filtering["max_text_length"]
        if max_len and len(text) > max_len:
            # Truncation is handled by truncate() method, not filtering
            pass

        # Min word count
        min_words = self.filtering["min_word_count"]
        if min_words:
            word_count = len(text.split())
            if word_count < min_words:
                return False

        # Repetition ratio (spam indicator)
        max_rep = self.filtering["max_repetition_ratio"]
        if max_rep and len(text) > 0:
            # Count most repeated character
            char_counts = {}
            for c in text.lower():
                if c.isalnum():
                    char_counts[c] = char_counts.get(c, 0) + 1
            if char_counts:
                max_count = max(char_counts.values())
                total_alnum = sum(char_counts.values())
                if total_alnum > 0 and max_count / total_alnum > max_rep:
                    return False

        return True

    def _is_spam(self, text: str) -> bool:
        """Check if text matches spam patterns."""
        if not self.spam_config["enabled"]:
            return False

        for pattern in self._spam_patterns:
            if pattern.search(text):
                return True

        return False

    def _is_low_quality(self, text: str) -> bool:
        """Check if text matches low-quality patterns."""
        if not self.quality_config["enabled"]:
            return False

        for pattern in self._quality_patterns:
            if pattern.match(text):
                return True

        return False

    def _normalize_text_for_dedup(self, text: str) -> str:
        """Normalize text for deduplication comparison."""
        normalized = text
        if self.dedup_config["case_insensitive"]:
            normalized = normalized.lower()
        if self.dedup_config["normalize_whitespace"]:
            normalized = " ".join(normalized.split())
        return normalized

    def _passes_deduplication(self, text: str, engineer_id: str) -> bool:
        """
        Check if text passes per-engineer deduplication.

        Returns True if the text should be kept, False if it's a duplicate.
        """
        if not self.dedup_config["enabled"]:
            return True

        normalized = self._normalize_text_for_dedup(text)
        key = (engineer_id, normalized)
        current_count = self._dedup_counts[key]
        max_duplicates = self.dedup_config["max_duplicates_per_engineer"]

        if current_count >= max_duplicates:
            return False

        self._dedup_counts[key] += 1
        return True

    def reset_deduplication_state(self) -> None:
        """Reset deduplication tracking state for a new load."""
        self._dedup_counts.clear()
        self._stats["excluded_by_quality"] = 0
        self._stats["excluded_by_dedup"] = 0

    def get_stats(self) -> dict:
        """Get filtering statistics."""
        return self._stats.copy()

    def log_stats(self) -> None:
        """Log filtering statistics."""
        if self._stats["excluded_by_quality"] > 0:
            logger.info(f"  Excluded by quality filter: {self._stats['excluded_by_quality']}")
        if self._stats["excluded_by_dedup"] > 0:
            logger.info(f"  Excluded by deduplication: {self._stats['excluded_by_dedup']}")

    def truncate(self, text: str) -> str:
        """
        Truncate text to max length if configured.

        Called separately from clean() to allow explicit truncation.
        """
        max_len = self.filtering["max_text_length"]
        if max_len and len(text) > max_len:
            return text[:max_len]
        return text
