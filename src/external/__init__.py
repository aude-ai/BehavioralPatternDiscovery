"""
External Integration Module

Provides interfaces for the larger evaluation system to:
1. Export discovered patterns for category mapping
2. Search messages by pattern activation
3. Query model metadata
"""

from src.external.pattern_export import PatternExporter
from src.external.message_search import MessageSearcher
from src.external.schemas import (
    PatternExport,
    PatternListResponse,
    MessageSearchResult,
    MessageSearchResponse,
    ModelMetadata,
    EncoderMetadata,
    NDJSONIngestRequest,
    IngestResponse,
    UserPatternMessagesRequest,
)

__all__ = [
    "PatternExporter",
    "MessageSearcher",
    "PatternExport",
    "PatternListResponse",
    "MessageSearchResult",
    "MessageSearchResponse",
    "ModelMetadata",
    "EncoderMetadata",
    "NDJSONIngestRequest",
    "IngestResponse",
    "UserPatternMessagesRequest",
]
