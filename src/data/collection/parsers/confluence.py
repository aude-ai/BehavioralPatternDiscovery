"""
Confluence Page Parser

Parses Confluence page activities from MongoDB.
Handles Airbyte data format where data is wrapped in '_airbyte_data' field.
"""

import logging
import re
from typing import Any

from .base import BaseParser, parser_registry

logger = logging.getLogger(__name__)


@parser_registry.register("confluence")
class ConfluenceParser(BaseParser):
    """Parser for Confluence page activities."""

    def __init__(self, config: dict):
        """
        Initialize Confluence parser.

        Args:
            config: Parser config with keys:
                - skip_minor_edits: bool
                - skip_empty_messages: bool
        """
        self.skip_minor_edits = config["skip_minor_edits"]
        self.skip_empty_messages = config["skip_empty_messages"]

    def parse(self, data: dict, user_mapping: dict[str, str], collection_type: str) -> list[dict[str, Any]]:
        """Parse Confluence page activity into standardized format."""
        activities = []

        # Skip minor edits if configured
        if self.skip_minor_edits and data.get("minorEdit", False):
            return activities

        # Get author info (check various field names used in different Confluence formats)
        author = (
            data.get("versionAuthor", {}) or
            data.get("author", {}) or
            data.get("version", {}).get("by", {})
        )

        # Get engineer ID
        engineer_id = self._get_engineer_id(author, user_mapping)
        if not engineer_id:
            return activities

        # Determine activity type
        activity_type = self._determine_activity_type(data)

        # Extract text
        text = self._extract_text(data)

        if self.skip_empty_messages and (not text or not text.strip()):
            return activities

        # Get timestamp
        timestamp = (
            data.get("when") or
            data.get("version", {}).get("when") or
            data.get("created") or
            data.get("timestamp")
        )

        activities.append({
            "engineer_id": engineer_id,
            "text": text.strip() if text else "",
            "source": "confluence",
            "activity_type": activity_type,
            "timestamp": timestamp,
            "metadata": {
                "space": data.get("space", {}).get("key") or data.get("space_key"),
                "page_id": data.get("id") or data.get("page_id"),
                "page_title": data.get("title") or data.get("page_title"),
                "version": data.get("version", {}).get("number"),
                "minor_edit": data.get("minorEdit", False),
            }
        })

        return activities

    def _get_engineer_id(self, author: dict, user_mapping: dict[str, str]) -> str:
        """Extract engineer ID from author object. Prioritizes name over email."""
        # Try to find in user_mapping first (check all possible identifiers)
        for val in [
            author.get("accountId"),
            author.get("email"),
            author.get("displayName"),
            author.get("publicName"),
            author.get("username"),
        ]:
            if val and str(val) in user_mapping:
                return user_mapping[str(val)]

        # Fall back to using name directly (prioritize name over email for display)
        return (
            author.get("displayName") or
            author.get("publicName") or
            author.get("email") or
            author.get("username") or
            author.get("accountId") or
            ""
        )

    def _determine_activity_type(self, data: dict) -> str:
        """Determine activity type from Confluence data."""
        version_num = data.get("version", {}).get("number", 1)
        if version_num == 1:
            return "page_create"
        return "page_edit"

    def _extract_text(self, data: dict) -> str:
        """Extract text content from Confluence page."""
        # Try version message first
        version_message = data.get("version", {}).get("message", "")
        if version_message:
            return version_message

        # Fall back to title + body excerpt
        title = data.get("title", "")
        body = data.get("body", {})

        # Body can be in different formats
        body_text = ""
        if isinstance(body, dict):
            # Try storage format
            storage = body.get("storage", {})
            body_text = storage.get("value", "")
            # Try view format
            if not body_text:
                view = body.get("view", {})
                body_text = view.get("value", "")
            # Try excerpt
            if not body_text:
                excerpt = body.get("excerpt", {})
                body_text = excerpt.get("value", "")
        elif isinstance(body, str):
            body_text = body

        # Clean HTML if present (basic cleanup)
        if body_text and "<" in body_text:
            body_text = re.sub(r"<[^>]+>", " ", body_text)
            body_text = re.sub(r"\s+", " ", body_text).strip()

        # Limit body text length
        if len(body_text) > 1000:
            body_text = body_text[:1000] + "..."

        return f"{title}\n{body_text}".strip()

    def get_source_name(self) -> str:
        """Return source name."""
        return "confluence"

    def get_collection_patterns(self) -> dict[str, str]:
        """Return collection patterns this parser handles."""
        return {
            "pages": r"confluence_pages",
            "activities": r"_Confluence_activities$",
        }
