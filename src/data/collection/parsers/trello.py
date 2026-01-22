"""
Trello Activity Parser

Parses Trello cards and actions from MongoDB.
Handles Airbyte data format where data is wrapped in '_airbyte_data' field.
"""

import logging
from typing import Any

from .base import BaseParser, parser_registry

logger = logging.getLogger(__name__)


@parser_registry.register("trello")
class TrelloParser(BaseParser):
    """Parser for Trello activities."""

    def __init__(self, config: dict):
        """
        Initialize Trello parser.

        Args:
            config: Parser config with keys:
                - activity_types: list[str]
        """
        self.activity_types = config["activity_types"]

    def parse(self, data: dict, user_mapping: dict[str, str], collection_type: str) -> list[dict[str, Any]]:
        """Parse Trello activity into standardized format."""
        activities = []

        # Get member/user info
        member = data.get("member", {}) or data.get("memberCreator", {})

        # Get engineer ID
        engineer_id = self._get_engineer_id(member, user_mapping)
        if not engineer_id:
            return activities

        # Determine activity type
        activity_type = self._determine_activity_type(data)
        if activity_type not in self.activity_types:
            return activities

        # Extract text
        text = self._extract_text(data, activity_type)
        if not text or not text.strip():
            return activities

        # Get timestamp
        timestamp = data.get("date") or data.get("created_at") or data.get("timestamp")

        activities.append({
            "engineer_id": engineer_id,
            "text": text.strip(),
            "source": "trello",
            "activity_type": activity_type,
            "timestamp": timestamp,
            "metadata": {
                "board": data.get("board", {}).get("name") or data.get("board_name"),
                "list": data.get("list", {}).get("name") or data.get("list_name"),
                "card_id": data.get("card", {}).get("id") or data.get("card_id"),
                "action_type": data.get("type"),
            }
        })

        return activities

    def _get_engineer_id(self, member: dict, user_mapping: dict[str, str]) -> str:
        """Extract engineer ID from member object. Prioritizes name over email."""
        # Try to find in user_mapping first (check ID first for canonical mapping)
        for val in [
            member.get("id"),
            member.get("email"),
            member.get("username"),
            member.get("fullName"),
        ]:
            if val and str(val) in user_mapping:
                return user_mapping[str(val)]

        # Fall back to using name directly (prioritize name over email for display)
        return (
            member.get("fullName") or
            member.get("username") or
            member.get("email") or
            member.get("id") or
            ""
        )

    def _determine_activity_type(self, data: dict) -> str:
        """Determine activity type from Trello action."""
        action_type = data.get("type", "")

        if action_type in ["createCard", "copyCard"]:
            return "card_create"
        if action_type in ["updateCard", "moveCardToBoard", "moveCardFromBoard"]:
            return "card_update"
        if action_type in ["commentCard", "addComment"]:
            return "comment"

        # Fallback based on data structure
        if "card" in data and not data.get("type"):
            return "card_create"

        return "card_update"

    def _extract_text(self, data: dict, activity_type: str) -> str:
        """Extract text content from Trello activity."""
        if activity_type == "comment":
            text_data = data.get("data", {})
            return text_data.get("text", "") or data.get("text", "")

        if activity_type in ["card_create", "card_update"]:
            card = data.get("card", {}) or data.get("data", {}).get("card", {})
            name = card.get("name", "")
            desc = card.get("desc", "") or ""
            return f"{name}\n{desc}".strip()

        return data.get("text", "") or data.get("data", {}).get("text", "")

    def get_source_name(self) -> str:
        """Return source name."""
        return "trello"

    def get_collection_patterns(self) -> dict[str, str]:
        """Return collection patterns this parser handles."""
        return {
            "actions": r"Trello_actions",
            "cards": r"Trello_cards",
            "activities": r"_Trello_activities$",
        }
