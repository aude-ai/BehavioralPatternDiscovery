"""Job models."""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enum."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Job type enum."""

    FETCH_DATA = "fetch_data"
    PREPROCESS = "preprocess"
    EMBED = "embed"
    NORMALIZE = "normalize"
    TRAIN = "train"
    BATCH_SCORE = "batch_score"
    SHAP_ANALYZE = "shap_analyze"
    NAME_PATTERNS = "name_patterns"
    SCORE_INDIVIDUAL = "score_individual"
    GENERATE_REPORT = "generate_report"


class JobCreate(BaseModel):
    """Schema for creating a job."""

    project_id: str
    job_type: JobType


class Job(BaseModel):
    """Job response schema."""

    id: str
    project_id: str
    job_type: JobType
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: float
    progress_message: Optional[str]
    result: Optional[dict]
    error: Optional[str]
    modal_call_id: Optional[str]
    metadata: Optional[dict] = Field(default=None, validation_alias="metadata_")

    class Config:
        from_attributes = True
