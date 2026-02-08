"""
Base Parser Interface

Defines the interface all activity parsers must implement.
Uses the registry pattern for swappable parser implementations.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from src.core.registry import ComponentRegistry

parser_registry: ComponentRegistry["BaseParser"] = ComponentRegistry("parser")


class BaseParser(ABC):
    """
    Interface for all activity parsers.

    Each parser handles a specific data source (GitHub, Slack, etc.) and
    transforms raw MongoDB documents into a standardized activity format.
    """

    @abstractmethod
    def __init__(self, config: dict):
        """
        Initialize parser from config.

        Args:
            config: Parser-specific configuration. No defaults allowed.
        """
        pass

    def extract_airbyte_data(self, document: dict) -> dict:
        """
        Extract data from Airbyte document format.

        Airbyte wraps the actual data in an '_airbyte_data' field.
        """
        return document.get("_airbyte_data", document)

    @abstractmethod
    def parse(self, data: dict, user_mapping: dict[str, str], collection_type: str) -> list[dict[str, Any]]:
        """
        Parse activity data into standardized format.

        Args:
            data: Activity data (already extracted from _airbyte_data wrapper)
            user_mapping: Mapping of user IDs/logins to canonical engineer IDs
            collection_type: Type of collection (e.g., 'pull_requests', 'commits')

        Returns:
            List of standardized activity dicts with keys:
            - engineer_id: str
            - text: str
            - source: str
            - activity_type: str
            - timestamp: str (ISO 8601)
            - metadata: dict (source-specific additional data)

            Returns empty list if activity should be filtered out.
        """
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """
        Return the source name.

        Returns:
            Source identifier (e.g., 'github', 'slack')
        """
        pass

    @abstractmethod
    def get_collection_patterns(self) -> dict[str, str]:
        """
        Return MongoDB collection name patterns this parser handles.

        Returns:
            Dict mapping collection_type to regex pattern.
            E.g., {"pull_requests": "GitHub_pull_requests(?!_commits|_stats)"}
        """
        pass

    def match_collection(self, collection_name: str) -> str | None:
        """
        Check if this parser handles the given collection.

        Args:
            collection_name: Name of the MongoDB collection

        Returns:
            Collection type string if matched, None otherwise
        """
        for coll_type, pattern in self.get_collection_patterns().items():
            if re.search(pattern, collection_name):
                return coll_type
        return None
