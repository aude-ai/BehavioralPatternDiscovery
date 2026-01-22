"""
Pydantic schemas for external API requests and responses.

These schemas define the contract between BehavioralPatternDiscovery
and the larger evaluation system.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# Pattern Export Schemas
# =============================================================================

class PatternExport(BaseModel):
    """
    A discovered pattern exported for category mapping.

    This format allows the larger system to:
    1. Display pattern info for human review
    2. Compute semantic similarity to CTO-defined categories
    3. Create pattern -> category mappings with signs and weights
    """
    pattern_id: str = Field(
        ...,
        description="Unique pattern identifier (e.g., 'unified_0', 'enc1_bottom_5')"
    )
    model_version_id: str = Field(
        ...,
        description="Version of the VAE model that discovered this pattern"
    )
    level: str = Field(
        ...,
        description="Hierarchy level: 'bottom', 'mid', 'top', or 'unified'"
    )
    encoder: Optional[str] = Field(
        None,
        description="Encoder name (null for unified patterns)"
    )

    # LLM-generated naming
    name: str = Field(..., description="Human-readable pattern name from LLM")
    description: str = Field(..., description="Pattern description from LLM")

    # Statistics for mapping weight calculation
    activation_count: int = Field(
        ...,
        description="Total messages with significant activation for this pattern"
    )
    unique_engineers: int = Field(
        ...,
        description="Number of unique engineers who exhibit this pattern"
    )
    mean_activation: float = Field(
        ...,
        description="Mean activation score across all messages"
    )
    std_activation: float = Field(
        ...,
        description="Standard deviation of activation scores"
    )

    # Representative examples for semantic similarity
    example_message_ids: List[str] = Field(
        ...,
        description="IDs of top representative messages"
    )
    example_texts: List[str] = Field(
        ...,
        description="Text of top representative messages (for embedding)"
    )

    # Composition info (for non-bottom patterns)
    composed_from: Optional[List[str]] = Field(
        None,
        description="Pattern IDs this pattern is composed from (via SHAP)"
    )
    composition_weights: Optional[Dict[str, float]] = Field(
        None,
        description="SHAP weights for composition"
    )


class PatternListResponse(BaseModel):
    """Response containing all exported patterns."""
    model_version_id: str
    export_timestamp: datetime
    pattern_count: int
    patterns: List[PatternExport]


# =============================================================================
# Message Search Schemas
# =============================================================================

class MessageSearchResult(BaseModel):
    """A single message result from pattern-based search."""
    message_id: str = Field(..., description="Internal message identifier")
    raw_ref_id: Optional[str] = Field(
        None,
        description="Original source document ID"
    )
    raw_ref_collection: Optional[str] = Field(
        None,
        description="Original source collection"
    )

    engineer_id: str
    timestamp: datetime
    source: str = Field(..., description="Source type: github_pr, slack, jira, etc.")
    text: str

    # Pattern activation info
    pattern_id: str
    activation_score: float = Field(
        ...,
        description="How strongly this message activates the pattern"
    )
    activation_rank: int = Field(
        ...,
        description="Rank among all messages for this pattern (1 = strongest)"
    )


class MessageSearchResponse(BaseModel):
    """Response for message search queries."""
    pattern_id: str
    total_matches: int
    returned_count: int
    messages: List[MessageSearchResult]


class UserPatternMessagesRequest(BaseModel):
    """Request to get a specific user's messages for a pattern."""
    engineer_id: str
    pattern_id: str
    top_k: int = Field(5, ge=1, le=50)


# =============================================================================
# Model Metadata Schemas
# =============================================================================

class EncoderMetadata(BaseModel):
    """Metadata for a single encoder."""
    name: str
    levels: List[str]
    dimensions_per_level: Dict[str, int]


class ModelMetadata(BaseModel):
    """
    Metadata about the trained VAE model.

    Useful for the mapping system to understand:
    - How many patterns exist at each level
    - The hierarchical structure
    - Model version for tracking
    """
    model_version_id: str
    trained_at: Optional[datetime] = None

    # Architecture info
    num_encoders: int
    encoders: List[EncoderMetadata]
    unified_dimension: int

    # Training info
    num_training_messages: int = 0
    num_engineers: int = 0

    # Pattern counts
    total_patterns: int
    patterns_per_level: Dict[str, int]


# =============================================================================
# Ingest Schemas
# =============================================================================

class NDJSONIngestRequest(BaseModel):
    """Request to ingest NDJSON data."""
    input_path: str = Field(
        ...,
        description="Path to NDJSON file or directory"
    )
    sections: Optional[List[str]] = Field(
        None,
        description="Sections to include (null = all)"
    )
    engineer_id_mapping_path: Optional[str] = Field(
        None,
        description="Path to engineer ID mapping JSON"
    )

    # Optional window filtering
    window_start: Optional[datetime] = Field(
        None,
        description="Start of evaluation window"
    )
    window_end: Optional[datetime] = Field(
        None,
        description="End of evaluation window"
    )


class IngestResponse(BaseModel):
    """Response from data ingestion."""
    status: str
    activities_loaded: int
    unique_engineers: int
    sources: Dict[str, int]
    output_path: str
