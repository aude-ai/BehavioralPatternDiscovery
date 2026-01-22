"""
Azure DevOps Activity Parser

Parses Azure DevOps activities from NDJSON records (7im format).

Handles sections:
- boards_work_item_discussions: Comments on work items
- boards_work_item_revisions: Work item state changes
- repos_pull_requests: Pull requests
- repos_commits: Git commits
- repos_pull_request_threads: PR review comments
- repos_pull_request_comments: PR review comments
"""

import logging
import re
from typing import Any

from .base import BaseParser, parser_registry

logger = logging.getLogger(__name__)


@parser_registry.register("azure_devops")
class AzureDevOpsParser(BaseParser):
    """Parser for Azure DevOps NDJSON records."""

    # Sections that contain meaningful behavioral text
    TEXT_SECTIONS = {
        "boards_work_item_discussions",
        "boards_work_item_revisions",
        "repos_pull_requests",
        "repos_commits",
        "repos_pull_request_threads",
        "repos_pull_request_comments",
    }

    def __init__(self, config: dict):
        """
        Initialize Azure DevOps parser.

        Args:
            config: Parser config with keys:
                - skip_system_users: bool - Filter out system/bot users
                - activity_types: list[str] - Activity types to include
                - min_text_length: int - Minimum text length to include
        """
        self.skip_system_users = config["skip_system_users"]
        self.activity_types = config["activity_types"]
        self.min_text_length = config["min_text_length"]

    def parse(self, data: dict, user_mapping: dict[str, str], collection_type: str, record_summary: str) -> list[dict[str, Any]]:
        """
        Parse Azure DevOps NDJSON record into standardized activities.

        Args:
            data: The 'record_details' object from the NDJSON record
            user_mapping: Mapping of user IDs to canonical engineer IDs
            collection_type: The 'section' from the NDJSON record
            record_summary: Human-readable summary string with key fields

        Returns:
            List of standardized activity dicts
        """
        # Skip sections without text content
        if collection_type not in self.TEXT_SECTIONS:
            return []

        # Route to appropriate parser method
        if collection_type == "boards_work_item_discussions":
            return self._parse_discussion(data, user_mapping, record_summary)
        elif collection_type == "boards_work_item_revisions":
            return self._parse_revision(data, user_mapping, record_summary)
        elif collection_type == "repos_pull_requests":
            return self._parse_pull_request(data, user_mapping, record_summary)
        elif collection_type == "repos_commits":
            return self._parse_commit(data, user_mapping, record_summary)
        elif collection_type in ("repos_pull_request_threads", "repos_pull_request_comments"):
            return self._parse_pr_comment(data, user_mapping, record_summary)
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")

    def _is_system_user(self, creator: dict) -> bool:
        """Check if creator is a system/bot user."""
        if not creator:
            return True

        display_name = creator.get("displayName", "")
        unique_name = creator.get("uniqueName", "")
        name = creator.get("name", "")

        system_patterns = [
            r"Microsoft\.VisualStudio\.Services",
            r"^Azure DevOps",
            r"^TFS$",
            r"^\[",
            r"Build Service",
            r"^SVC-",
        ]

        for pattern in system_patterns:
            if re.search(pattern, display_name, re.IGNORECASE):
                return True
            if re.search(pattern, unique_name, re.IGNORECASE):
                return True
            if re.search(pattern, name, re.IGNORECASE):
                return True

        user_id = creator.get("id", "")
        if user_id and user_id.startswith("00000002-0000-8888"):
            return True

        return False

    def _get_engineer_id(self, data: dict, user_mapping: dict[str, str], section: str) -> str:
        """Extract engineer ID from data, handling section-specific author locations."""
        if section == "repos_commits":
            creator = data["author"]
        elif section == "repos_pull_requests":
            creator = data["createdBy"]
        elif section == "boards_work_item_revisions":
            identities = data["identities"]
            creator = identities.get("changedBy") or identities["createdBy"]
        elif section in ("boards_work_item_discussions", "repos_pull_request_comments", "repos_pull_request_threads"):
            creator = data["author"]
        else:
            raise ValueError(f"Unknown section for author extraction: {section}")

        if self.skip_system_users and self._is_system_user(creator):
            return ""

        raw_id = (
            creator.get("email") or
            creator.get("uniqueName") or
            creator.get("displayName") or
            creator.get("name") or
            creator.get("id") or
            ""
        )

        if raw_id and str(raw_id) in user_mapping:
            return user_mapping[str(raw_id)]

        if raw_id and str(raw_id).lower() in user_mapping:
            return user_mapping[str(raw_id).lower()]

        return str(raw_id) if raw_id else ""

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.replace("&nbsp;", " ")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&amp;", "&")
        text = text.replace("&quot;", '"')
        return text.strip()

    def _extract_from_summary(self, record_summary: str, field: str) -> str:
        """
        Extract a field value from record_summary string.

        Format: "[Type] id | field1=value1 | field2=value2 | ..."
        Delimiter is ' | ' (space-pipe-space), not just '|'.
        Field values may contain '|' characters.
        """
        # Match field= followed by value up to next ' | ' delimiter or end of string
        pattern = rf'{field}=(.*?)(?= \| |$)'
        match = re.search(pattern, record_summary, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        raise ValueError(f"Field '{field}' not found in record_summary: {record_summary[:100]}")

    def _extract_last_field_from_summary(self, record_summary: str, field: str) -> str:
        """
        Extract the last field from record_summary (captures to end of string).

        Use this for fields that are always last and may contain | or newlines.
        Examples: body (discussions), content (PR comments)
        """
        # Use re.DOTALL so . matches newlines in field content
        # Use (.*) instead of (.+) to allow empty fields
        pattern = rf'{field}=(.*)$'
        match = re.search(pattern, record_summary, re.IGNORECASE | re.DOTALL)
        if match:
            return self._clean_html(match.group(1))
        raise ValueError(f"Field '{field}' not found in record_summary: {record_summary[:100]}")

    def _parse_discussion(self, data: dict, user_mapping: dict[str, str], record_summary: str) -> list[dict]:
        """Parse a work item discussion/comment."""
        if "discussion" not in self.activity_types:
            return []

        engineer_id = self._get_engineer_id(data, user_mapping, "boards_work_item_discussions")
        if not engineer_id:
            return []

        text = self._extract_last_field_from_summary(record_summary, "body")

        if len(text) < self.min_text_length:
            return []

        timestamp = data["changedDate"]

        return [{
            "engineer_id": engineer_id,
            "text": text,
            "timestamp": timestamp,
            "source": "azure_devops",
            "activity_type": "discussion",
            "metadata": {
                "work_item_id": data["workItemId"],
            }
        }]

    def _parse_revision(self, data: dict, user_mapping: dict[str, str], record_summary: str) -> list[dict]:
        """Parse a work item revision (state change)."""
        if "revision" not in self.activity_types:
            return []

        engineer_id = self._get_engineer_id(data, user_mapping, "boards_work_item_revisions")
        if not engineer_id:
            return []

        text = self._extract_from_summary(record_summary, "title")

        if len(text) < self.min_text_length:
            return []

        timestamp = data["changedDate"]
        state = self._extract_from_summary(record_summary, "state")

        return [{
            "engineer_id": engineer_id,
            "text": text,
            "timestamp": timestamp,
            "source": "azure_devops",
            "activity_type": "revision",
            "metadata": {
                "work_item_id": data["workItemId"],
                "state": state,
            }
        }]

    def _parse_pull_request(self, data: dict, user_mapping: dict[str, str], record_summary: str) -> list[dict]:
        """Parse a pull request."""
        if "pull_request" not in self.activity_types:
            return []

        engineer_id = self._get_engineer_id(data, user_mapping, "repos_pull_requests")
        if not engineer_id:
            return []

        title = self._extract_from_summary(record_summary, "title")
        description = data.get("description", "")

        text = f"{title}\n{description}".strip()
        if len(text) < self.min_text_length:
            return []

        timestamp = data["creationDate"]

        return [{
            "engineer_id": engineer_id,
            "text": text,
            "timestamp": timestamp,
            "source": "azure_devops",
            "activity_type": "pull_request",
            "metadata": {
                "pull_request_id": data["pullRequestId"],
                "repo_name": data.get("repo", {}).get("name"),
                "state": data.get("status"),
            }
        }]

    def _parse_commit(self, data: dict, user_mapping: dict[str, str], record_summary: str) -> list[dict]:
        """Parse a git commit."""
        if "commit" not in self.activity_types:
            return []

        engineer_id = self._get_engineer_id(data, user_mapping, "repos_commits")
        if not engineer_id:
            return []

        text = self._extract_from_summary(record_summary, "comment")

        if len(text) < self.min_text_length:
            return []

        timestamp = data["author"]["date"]
        change_counts = data.get("changeCounts", {})

        return [{
            "engineer_id": engineer_id,
            "text": text,
            "timestamp": timestamp,
            "source": "azure_devops",
            "activity_type": "commit",
            "metadata": {
                "commit_id": data["commitId"],
                "change_counts_add": change_counts.get("Add"),
                "change_counts_edit": change_counts.get("Edit"),
                "change_counts_delete": change_counts.get("Delete"),
            }
        }]

    def _parse_pr_comment(self, data: dict, user_mapping: dict[str, str], record_summary: str) -> list[dict]:
        """Parse a PR review comment."""
        if "pr_comment" not in self.activity_types:
            return []

        engineer_id = self._get_engineer_id(data, user_mapping, "repos_pull_request_comments")
        if not engineer_id:
            return []

        # content is the last field and may contain | or newlines
        text = self._extract_last_field_from_summary(record_summary, "content")

        if len(text) < self.min_text_length:
            return []

        timestamp = data["publishedDate"]

        return [{
            "engineer_id": engineer_id,
            "text": text,
            "timestamp": timestamp,
            "source": "azure_devops",
            "activity_type": "pr_comment",
            "metadata": {
                "pull_request_id": data["pullRequestId"],
                "thread_id": data.get("threadId"),
                "comment_id": data.get("commentId"),
            }
        }]

    def get_source_name(self) -> str:
        """Return source name."""
        return "azure_devops"

    def get_collection_patterns(self) -> dict[str, str]:
        """Return collection patterns this parser handles."""
        return {
            "boards_work_item_discussions": r"boards_work_item_discussions",
            "boards_work_item_revisions": r"boards_work_item_revisions",
            "repos_pull_requests": r"repos_pull_requests",
            "repos_commits": r"repos_commits",
            "repos_pull_request_threads": r"repos_pull_request_threads",
            "repos_pull_request_comments": r"repos_pull_request_comments",
        }
