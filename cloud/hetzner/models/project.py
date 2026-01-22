"""Project models."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    """Project status enum."""

    CREATED = "created"
    DATA_LOADED = "data_loaded"
    PREPROCESSED = "preprocessed"
    EMBEDDED = "embedded"
    TRAINED = "trained"
    SCORED = "scored"
    PATTERNS_IDENTIFIED = "patterns_identified"
    READY = "ready"


class ProjectCreate(BaseModel):
    """Schema for creating a project."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    config_overrides: Optional[dict] = None


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    config_overrides: Optional[dict] = None


class Project(BaseModel):
    """Project response schema."""

    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    status: ProjectStatus
    config_overrides: Optional[dict]
    metadata: Optional[dict] = Field(None, alias="metadata_")

    class Config:
        from_attributes = True
        populate_by_name = True


class ProjectWithStats(Project):
    """Project with additional statistics."""

    engineer_count: int = 0
    message_count: int = 0
    has_checkpoint: bool = False
    has_patterns: bool = False
