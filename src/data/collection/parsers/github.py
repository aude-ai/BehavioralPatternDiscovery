"""
GitHub Activity Parser

Parses GitHub activities (commits, PRs, reviews, issues, comments) from MongoDB.
Handles Airbyte data format where data is wrapped in '_airbyte_data' field.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .base import BaseParser, parser_registry

logger = logging.getLogger(__name__)


@parser_registry.register("github")
class GitHubParser(BaseParser):
    """Parser for GitHub activities from Airbyte collections."""

    def __init__(self, config: dict):
        """
        Initialize GitHub parser.

        Args:
            config: Parser config with keys:
                - skip_bot_users: bool
                - activity_types: list[str]
        """
        self.skip_bot_users = config["skip_bot_users"]
        self.activity_types = config["activity_types"]

    def parse(self, data: dict, user_mapping: dict[str, str], collection_type: str) -> list[dict[str, Any]]:
        """Parse GitHub activity based on collection type."""
        if collection_type == "pull_requests":
            return self._parse_pull_request(data, user_mapping)
        elif collection_type == "pull_request_commits":
            return self._parse_commit(data, user_mapping)
        elif collection_type == "reviews":
            return self._parse_review(data, user_mapping)
        elif collection_type == "comments":
            return self._parse_comment(data, user_mapping)
        elif collection_type == "review_comments":
            return self._parse_review_comment(data, user_mapping)
        else:
            logger.debug(f"Unknown GitHub collection type: {collection_type}")
            return []

    def _get_engineer_id(self, data: dict, user_mapping: dict[str, str], fallback_name: str = None) -> str:
        """Extract engineer ID from data. Checks user_mapping for canonical IDs."""
        user = data.get("user", {}) if isinstance(data.get("user"), dict) else {}

        # Try to find in user_mapping first (check user ID, login, and fallback name)
        for val in [
            str(user.get("id", "")) if user.get("id") else None,
            user.get("login"),
            fallback_name,
        ]:
            if val and val in user_mapping:
                return user_mapping[val]

        # Fall back: use login if available, otherwise fallback_name
        if user.get("login"):
            return user["login"]
        if fallback_name:
            return fallback_name

        return ""

    def _parse_timestamp(self, timestamp_str: str) -> str | None:
        """Parse timestamp string to ISO format."""
        if not timestamp_str:
            return None
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.isoformat()
        except (ValueError, AttributeError):
            return None

    def _extract_repository(self, data: dict) -> str:
        """Extract repository name from data or URL."""
        if "repository" in data:
            return data["repository"]

        url = data.get("url", "") or data.get("html_url", "")
        if "github.com" in url or "api.github.com" in url:
            parts = url.split("/")
            for i, part in enumerate(parts):
                if part == "repos" and i + 2 < len(parts):
                    return f"{parts[i+1]}/{parts[i+2]}"

        return ""

    def _parse_pull_request(self, data: dict, user_mapping: dict[str, str]) -> list[dict]:
        """Parse a pull request document."""
        activities = []

        engineer_id = self._get_engineer_id(data, user_mapping)
        if not engineer_id:
            return activities

        # Check for bot user
        if self.skip_bot_users:
            user = data.get("user", {})
            if user.get("type") == "Bot":
                return activities

        timestamp = self._parse_timestamp(data.get("created_at"))
        repository = self._extract_repository(data)

        # PR title
        title = data.get("title", "").strip()
        if title and "pull_request" in self.activity_types:
            activities.append({
                "engineer_id": engineer_id,
                "text": title,
                "timestamp": timestamp,
                "source": "github",
                "activity_type": "pull_request",
                "metadata": {
                    "number": data.get("number"),
                    "state": data.get("state"),
                    "repository": repository,
                    "merged_at": data.get("merged_at"),
                    "additions": data.get("additions"),
                    "deletions": data.get("deletions"),
                    "changed_files": data.get("changed_files"),
                }
            })

        # PR body
        body = data.get("body", "")
        if body and body.strip() and "pull_request" in self.activity_types:
            activities.append({
                "engineer_id": engineer_id,
                "text": body.strip(),
                "timestamp": timestamp,
                "source": "github",
                "activity_type": "pull_request_body",
                "metadata": {
                    "number": data.get("number"),
                    "repository": repository,
                }
            })

        return activities

    def _parse_commit(self, data: dict, user_mapping: dict[str, str]) -> list[dict]:
        """Parse a commit document."""
        activities = []

        commit_data = data.get("commit", {})
        if not commit_data:
            return activities

        author_data = commit_data.get("author", {})
        author_name = author_data.get("name", "")
        engineer_id = self._get_engineer_id(data, user_mapping, author_name)

        if not engineer_id:
            return activities

        if "commit" not in self.activity_types:
            return activities

        message = commit_data.get("message", "").strip()
        if not message:
            return activities

        timestamp = self._parse_timestamp(author_data.get("date"))
        repository = self._extract_repository(data)

        activities.append({
            "engineer_id": engineer_id,
            "text": message,
            "timestamp": timestamp,
            "source": "github",
            "activity_type": "commit",
            "metadata": {
                "sha": data.get("sha"),
                "repository": repository,
                "author_email": author_data.get("email"),
            }
        })

        return activities

    def _parse_review(self, data: dict, user_mapping: dict[str, str]) -> list[dict]:
        """Parse a code review document."""
        activities = []

        engineer_id = self._get_engineer_id(data, user_mapping)
        if not engineer_id:
            return activities

        if "review" not in self.activity_types:
            return activities

        body = data.get("body", "").strip()
        state = data.get("state", "")

        # If no body, create text from state
        if not body:
            body = f"Review: {state}"

        timestamp = self._parse_timestamp(data.get("submitted_at") or data.get("created_at"))
        repository = self._extract_repository(data)

        activities.append({
            "engineer_id": engineer_id,
            "text": body,
            "timestamp": timestamp,
            "source": "github",
            "activity_type": "review",
            "metadata": {
                "state": state,
                "repository": repository,
                "pull_request_url": data.get("pull_request_url"),
            }
        })

        return activities

    def _parse_comment(self, data: dict, user_mapping: dict[str, str]) -> list[dict]:
        """Parse an issue/PR comment document."""
        activities = []

        engineer_id = self._get_engineer_id(data, user_mapping)
        if not engineer_id:
            return activities

        # Check for bot user
        if self.skip_bot_users:
            user = data.get("user", {})
            if user.get("type") == "Bot":
                return activities

        if "comment" not in self.activity_types:
            return activities

        body = data.get("body", "").strip()
        if not body:
            return activities

        timestamp = self._parse_timestamp(data.get("created_at"))
        repository = self._extract_repository(data)

        activities.append({
            "engineer_id": engineer_id,
            "text": body,
            "timestamp": timestamp,
            "source": "github",
            "activity_type": "comment",
            "metadata": {
                "id": data.get("id"),
                "repository": repository,
                "issue_url": data.get("issue_url"),
            }
        })

        return activities

    def _parse_review_comment(self, data: dict, user_mapping: dict[str, str]) -> list[dict]:
        """Parse a review comment document."""
        activities = []

        engineer_id = self._get_engineer_id(data, user_mapping)
        if not engineer_id:
            return activities

        if "comment" not in self.activity_types:
            return activities

        body = data.get("body", "").strip()
        if not body:
            return activities

        timestamp = self._parse_timestamp(data.get("created_at"))
        repository = self._extract_repository(data)

        activities.append({
            "engineer_id": engineer_id,
            "text": body,
            "timestamp": timestamp,
            "source": "github",
            "activity_type": "review_comment",
            "metadata": {
                "id": data.get("id"),
                "repository": repository,
                "path": data.get("path"),
                "commit_id": data.get("commit_id"),
            }
        })

        return activities

    def get_source_name(self) -> str:
        """Return source name."""
        return "github"

    def get_collection_patterns(self) -> dict[str, str]:
        """Return collection patterns this parser handles."""
        return {
            "pull_requests": r"GitHub_pull_requests(?!_commits|_stats)",
            "pull_request_commits": r"GitHub_pull_request_commits",
            "reviews": r"GitHub_reviews(?!_comments)",
            "comments": r"GitHub_comments(?!.*review)",
            "review_comments": r"GitHub_review_comments",
        }


def build_user_mapping_from_prs(pr_documents: list[dict]) -> dict[str, str]:
    """
    Build a mapping from commit author names to GitHub logins using PR data.

    Args:
        pr_documents: List of PR documents from MongoDB

    Returns:
        Dict mapping author names/logins to canonical login
    """
    mapping = {}

    for doc in pr_documents:
        data = doc.get("_airbyte_data", doc)
        login = data.get("user", {}).get("login")

        if not login:
            continue

        # Map login to itself
        mapping[login] = login

        # Map commit author names to login
        commits = data.get("commits", [])
        for commit in commits:
            commit_data = commit.get("commit", commit)
            author_name = commit_data.get("author", {}).get("name")
            if author_name:
                mapping[author_name] = login

    return mapping
