"""
Activity Parsers Module

Provides parsers for different data sources (GitHub, Slack, Trello, Confluence, Azure DevOps).
Each parser transforms raw documents into standardized activity format.
"""

from .base import BaseParser, parser_registry

# Import parsers to trigger registration
from . import github
from . import slack
from . import trello
from . import confluence
from . import azure_devops

__all__ = [
    "BaseParser",
    "parser_registry",
]
