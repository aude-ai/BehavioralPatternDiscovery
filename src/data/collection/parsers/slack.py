"""
Slack Message Parser

Parses Slack messages from MongoDB with bot filtering.
Handles Airbyte data format where data is wrapped in '_airbyte_data' field.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .base import BaseParser, parser_registry

logger = logging.getLogger(__name__)


@parser_registry.register("slack")
class SlackParser(BaseParser):
    """Parser for Slack messages."""

    def __init__(self, config: dict):
        """
        Initialize Slack parser.

        Args:
            config: Parser config with keys:
                - filter_bots: bool
                - filter_app_users: bool
                - filter_bot_usernames: bool
        """
        self.filter_bots = config["filter_bots"]
        self.filter_app_users = config["filter_app_users"]
        self.filter_bot_usernames = config["filter_bot_usernames"]

    def parse(self, data: dict, user_mapping: dict[str, str], collection_type: str) -> list[dict[str, Any]]:
        """Parse Slack message into standardized format."""
        activities = []

        user = data.get("user")
        if not user or not isinstance(user, dict):
            return activities

        # Bot filtering
        if self.filter_bots and user.get("is_bot", False):
            return activities

        if self.filter_app_users and user.get("is_app_user", False):
            return activities

        if self.filter_bot_usernames:
            user_name = user.get("name", "") or user.get("username", "")
            if "bot" in user_name.lower():
                return activities

        # Get engineer ID - try various fields
        engineer_id = self._get_engineer_id(data, user, user_mapping)
        if not engineer_id:
            return activities

        # Extract text
        text = data.get("text", "").strip()
        if not text:
            return activities

        # Parse timestamp (Slack uses Unix timestamp)
        timestamp = self._parse_timestamp(data.get("ts"))

        activities.append({
            "engineer_id": engineer_id,
            "text": text,
            "timestamp": timestamp,
            "source": "slack",
            "activity_type": "message",
            "metadata": {
                "channel": data.get("channel") or data.get("channel_id"),
                "channel_name": data.get("channel_name"),
                "team": data.get("team"),
                "user_id": user.get("id"),
                "real_name": user.get("real_name"),
                "thread_ts": data.get("thread_ts"),
                "reply_count": data.get("reply_count"),
            }
        })

        return activities

    def _get_engineer_id(self, data: dict, user: dict, user_mapping: dict[str, str]) -> str:
        """Extract engineer ID from user object. Prioritizes name over email."""
        profile = user.get("profile", {})

        # Try to find in user_mapping first (check all possible identifiers)
        for val in [
            user.get("id"),
            profile.get("email"),
            data.get("email"),
            user.get("real_name"),
            profile.get("real_name"),
            user.get("name"),
        ]:
            if val and str(val) in user_mapping:
                return user_mapping[str(val)]

        # Fall back to using name directly (prioritize name over email for display)
        return (
            user.get("real_name") or
            profile.get("real_name") or
            profile.get("display_name") or
            data.get("email") or
            profile.get("email") or
            user.get("name") or
            user.get("id") or
            ""
        )

    def _parse_timestamp(self, ts_value) -> str | None:
        """Convert Slack timestamp to ISO format."""
        if not ts_value:
            return None

        try:
            timestamp_float = float(ts_value)
            dt = datetime.fromtimestamp(timestamp_float)
            return dt.isoformat()
        except (ValueError, TypeError):
            return None

    def get_source_name(self) -> str:
        """Return source name."""
        return "slack"

    def get_collection_patterns(self) -> dict[str, str]:
        """Return collection patterns this parser handles."""
        return {
            "activities": r"_Slack_activities$",
            "messages": r"slack_messages",
        }
