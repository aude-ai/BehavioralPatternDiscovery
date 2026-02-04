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
    metadata: Optional[dict] = Field(default=None, validation_alias="metadata_")

    # Variant support (Phase 5)
    parent_id: Optional[str] = None
    owned_files: Optional[dict] = None

    class Config:
        from_attributes = True

    @property
    def is_variant(self) -> bool:
        """Check if this project is a variant of another project."""
        return self.parent_id is not None

    @property
    def is_root(self) -> bool:
        """Check if this project is a root project (not a variant)."""
        return self.parent_id is None


class ProjectWithStats(Project):
    """Project with additional statistics."""

    engineer_count: int = 0
    message_count: int = 0
    has_checkpoint: bool = False
    has_patterns: bool = False
    variant_count: int = 0


# Variant support (Phase 5)

class VariantCreate(BaseModel):
    """Schema for creating a project variant."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    starting_step: str = Field(..., description="Pipeline step to start from (e.g., 'B.5', 'B.6', 'B.8')")


class BorrowedFile(BaseModel):
    """A file borrowed by a variant from its parent."""

    variant_id: str
    variant_name: str
    file_key: str


class DeletionBlockedResponse(BaseModel):
    """Response when project deletion is blocked by variants borrowing files."""

    error: str = "deletion_blocked"
    message: str
    borrowed_files: list[BorrowedFile]


class DeletionSuccessResponse(BaseModel):
    """Response when project deletion succeeds."""

    deleted: bool = True
    detached_variants: list[str] = Field(default_factory=list)
