"""Pipeline orchestration routes."""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Job, JobType
from ..services import ProjectService, StorageService
from ..services.r2_service import get_all_r2_file_info, validate_prerequisites
from ..tasks import cpu_tasks, gpu_tasks

router = APIRouter()

# Cache for loaded configs
_config_cache = {}


def load_all_configs() -> dict:
    """Load all config files from config/ directory."""
    if _config_cache:
        return _config_cache

    import logging
    from src.core.config import load_config

    logger = logging.getLogger(__name__)
    config_dir = Path(__file__).parent.parent.parent.parent / "config"

    # Required config files
    required_files = [
        "data.yaml",
        "model.yaml",
        "training.yaml",
        "pattern_identification.yaml",
        "scoring.yaml",
        "cloud.yaml",
    ]

    for filename in required_files:
        config_path = config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Required config file not found: {config_path}")
        key = filename.replace(".yaml", "")
        _config_cache[key] = load_config(config_path)
        logger.info(f"Loaded config: {filename}")

    return _config_cache


def get_pipeline_config(user_overrides: dict = None) -> dict:
    """
    Build merged pipeline config from config files and user overrides.

    Returns a flattened config structure for tasks.
    """
    configs = load_all_configs()

    # Validate required config sections exist
    if "data" not in configs:
        raise ValueError("Missing required config: data.yaml")
    if "model" not in configs:
        raise ValueError("Missing required config: model.yaml")
    if "training" not in configs:
        raise ValueError("Missing required config: training.yaml")
    if "pattern_identification" not in configs:
        raise ValueError("Missing required config: pattern_identification.yaml")
    if "scoring" not in configs:
        raise ValueError("Missing required config: scoring.yaml")
    if "cloud" not in configs:
        raise ValueError("Missing required config: cloud.yaml")

    data_config = configs["data"]

    # Validate required sections in data config
    if "processing" not in data_config:
        raise ValueError("Missing required config section: data.processing")

    merged = {
        "processing": data_config["processing"],
        "collection": data_config.get("collection", {}),  # Optional
        "paths": data_config.get("paths", {}),  # Optional
        "model": configs["model"],
        "training": configs["training"],
        "cloud": configs["cloud"],
    }

    # Flatten pattern_identification.yaml keys to top level
    for key, value in configs["pattern_identification"].items():
        merged[key] = value

    # Flatten scoring.yaml keys to top level
    for key, value in configs["scoring"].items():
        merged[key] = value

    # Apply user overrides with deep merge
    if user_overrides:
        merged = _deep_merge(merged, user_overrides)

    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge override into base dict.

    For nested dicts, recursively merges instead of replacing.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_services(project_id: str, db: Session = Depends(get_db)):
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    storage = StorageService(project_id)
    return service, storage, project


# =============================================================================
# SEGMENT B: Processing Pipeline (triggers Modal)
# =============================================================================


class ProcessRequest(BaseModel):
    """Request body for processing pipeline."""
    starting_step: str = "B.1"
    config: Optional[dict] = None


@router.post("/process", response_model=Job)
def start_processing_pipeline(
    project_id: str,
    request: ProcessRequest,
    db: Session = Depends(get_db),
):
    """
    Start the unified processing pipeline on Modal (Segment B).

    Starting points:
    - B.1: Full pipeline (requires activities.csv on Hetzner)
    - B.5: From training (requires embeddings, aux_features in R2)
    - B.6: From batch scoring (requires checkpoint, train_input, message_database in R2)
    - B.8: SHAP only (requires checkpoint, activations in R2)

    Pipeline runs B.1 -> B.2 -> B.3 -> B.4 -> B.5 -> B.6 -> B.7 -> B.8 sequentially,
    starting from the specified step.
    """
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    # Validate Hetzner prerequisites
    if request.starting_step == "B.1":
        if not storage.activities_path.exists():
            raise HTTPException(
                status_code=400,
                detail="activities.csv not found. Run data collection first (Segment A).",
            )

    # Validate R2 prerequisites for later starting points
    if request.starting_step != "B.1":
        valid, error = validate_prerequisites(project_id, request.starting_step)
        if not valid:
            raise HTTPException(status_code=400, detail=error)

    # Create job
    job = service.create_job(project_id, JobType.TRAIN)
    merged_config = get_pipeline_config(request.config)

    # Trigger Modal processing pipeline
    gpu_tasks.trigger_processing_pipeline.delay(
        project_id=project_id,
        job_id=job.id,
        starting_step=request.starting_step,
        config=merged_config,
    )

    return job


@router.get("/r2-status")
def get_r2_status(project_id: str, db: Session = Depends(get_db)):
    """
    Get R2 and Hetzner file status for a project.

    Returns existence and metadata for all files used by Segment B.
    """
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    # R2 files (large processing outputs)
    r2_status = get_all_r2_file_info(project_id)

    # Hetzner files (local storage)
    hetzner_status = {
        "activities": {
            "exists": storage.activities_path.exists(),
            "size_bytes": storage.activities_path.stat().st_size if storage.activities_path.exists() else 0,
        },
        "population_stats": {
            "exists": storage.population_stats_path.exists(),
            "size_bytes": storage.population_stats_path.stat().st_size if storage.population_stats_path.exists() else 0,
        },
        "message_examples": {
            "exists": storage.message_examples_path.exists(),
            "size_bytes": storage.message_examples_path.stat().st_size if storage.message_examples_path.exists() else 0,
        },
        "hierarchical_weights": {
            "exists": storage.hierarchical_weights_path.exists(),
            "size_bytes": storage.hierarchical_weights_path.stat().st_size if storage.hierarchical_weights_path.exists() else 0,
        },
        "pattern_names": {
            "exists": storage.pattern_names_path.exists(),
            "size_bytes": storage.pattern_names_path.stat().st_size if storage.pattern_names_path.exists() else 0,
        },
    }

    return {
        "r2": r2_status,
        "hetzner": hetzner_status,
    }


# =============================================================================
# SEGMENT C: Pattern Naming (runs on Hetzner)
# =============================================================================


@router.post("/name-patterns", response_model=Job)
def start_pattern_naming(
    project_id: str,
    config: dict = None,
    db: Session = Depends(get_db),
):
    """
    Name patterns using LLM (Segment C).

    Runs on Hetzner. Requires:
    - message_examples.json (from B.7, sent to Hetzner)
    - hierarchical_weights.json (from B.8, sent to Hetzner)
    """
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    if not storage.message_examples_path.exists():
        raise HTTPException(
            status_code=400,
            detail="message_examples.json not found. Run processing pipeline first (Segment B).",
        )

    if not storage.hierarchical_weights_path.exists():
        raise HTTPException(
            status_code=400,
            detail="hierarchical_weights.json not found. Run processing pipeline first (Segment B).",
        )

    job = service.create_job(project_id, JobType.NAME_PATTERNS)
    merged_config = get_pipeline_config(config)

    cpu_tasks.name_patterns.delay(project_id, job.id, merged_config)

    return job


# =============================================================================
# JOB MANAGEMENT
# =============================================================================


@router.get("/jobs", response_model=list[Job])
def list_jobs(
    project_id: str,
    job_type: str = None,
    status: str = None,
    db: Session = Depends(get_db),
):
    """List jobs for a project."""
    service = ProjectService(db)
    return service.get_project_jobs(
        project_id,
        job_type=JobType(job_type) if job_type else None,
        status=status,
    )


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(
    project_id: str,
    job_id: str,
    db: Session = Depends(get_db),
):
    """Get job details."""
    service = ProjectService(db)
    job = service.get_job(job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
