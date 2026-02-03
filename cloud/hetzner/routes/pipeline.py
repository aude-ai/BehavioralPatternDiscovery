"""Pipeline orchestration routes."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Job, JobType
from ..services import ProjectService, StorageService
from ..tasks import cpu_tasks, gpu_tasks

router = APIRouter()

# Cache for loaded configs
_config_cache = {}


def load_all_configs() -> dict:
    """Load all config files from config/ directory."""
    if _config_cache:
        return _config_cache

    from src.core.config import load_config

    config_dir = Path(__file__).parent.parent.parent.parent / "config"

    config_files = ["data.yaml", "model.yaml", "training.yaml", "pattern_identification.yaml", "scoring.yaml"]

    for filename in config_files:
        config_path = config_dir / filename
        if config_path.exists():
            key = filename.replace(".yaml", "")
            _config_cache[key] = load_config(config_path)

    return _config_cache


def get_pipeline_config(user_overrides: dict = None) -> dict:
    """
    Build merged pipeline config from config files and user overrides.

    Returns a flattened config structure for tasks. Keys from pattern_identification.yaml
    and scoring.yaml are merged at top level for direct access by tasks.
    """
    configs = load_all_configs()

    # Start with processing config from data.yaml
    data_config = configs.get("data", {})
    merged = {
        "processing": data_config.get("processing", {}),
        "collection": data_config.get("collection", {}),
        "paths": data_config.get("paths", {}),
        "model": configs.get("model", {}),
        "training": configs.get("training", {}),
    }

    # Flatten pattern_identification.yaml keys to top level
    # (message_assignment, pattern_naming, shap, batch_scoring, etc.)
    pattern_id_config = configs.get("pattern_identification", {})
    for key, value in pattern_id_config.items():
        merged[key] = value

    # Flatten scoring.yaml keys to top level
    scoring_config = configs.get("scoring", {})
    for key, value in scoring_config.items():
        merged[key] = value

    # Apply user overrides
    if user_overrides:
        for key, value in user_overrides.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

    return merged


def get_services(project_id: str, db: Session = Depends(get_db)):
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    storage = StorageService(project_id)
    return service, storage, project


@router.post("/preprocess", response_model=Job)
def start_preprocessing(
    project_id: str,
    config: dict = None,
    db: Session = Depends(get_db),
):
    """Start preprocessing pipeline."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    job = service.create_job(project_id, JobType.PREPROCESS)
    merged_config = get_pipeline_config(config)

    cpu_tasks.preprocess_data.delay(project_id, job.id, merged_config)

    return job


@router.post("/embed", response_model=Job)
def start_embedding(
    project_id: str,
    config: dict = None,
    db: Session = Depends(get_db),
):
    """Start embedding on Modal."""
    service = ProjectService(db)
    storage = StorageService(project_id)

    if not storage.train_aux_vars_path.exists():
        raise HTTPException(status_code=400, detail="Run preprocessing first")

    # Load texts from activities
    import pandas as pd

    activities = pd.read_csv(storage.activities_path)
    texts = activities["text"].tolist()

    job = service.create_job(project_id, JobType.EMBED)
    merged_config = get_pipeline_config(config)

    gpu_tasks.trigger_embedding.delay(project_id, job.id, texts, merged_config)

    return job


@router.post("/train", response_model=Job)
def start_training(
    project_id: str,
    config: dict = None,
    db: Session = Depends(get_db),
):
    """
    Start VAE training on Modal.

    Automatically prepares training data (combines embeddings + aux features,
    creates message_database) before spawning the Modal training function.
    """
    service = ProjectService(db)
    storage = StorageService(project_id)

    if not storage.train_features_path.exists():
        raise HTTPException(status_code=400, detail="Run embedding first")

    job = service.create_job(project_id, JobType.TRAIN)
    merged_config = get_pipeline_config(config)

    gpu_tasks.trigger_training.delay(project_id, job.id, merged_config)

    return job


@router.post("/batch-score", response_model=Job)
def start_batch_scoring(
    project_id: str,
    config: dict = None,
    db: Session = Depends(get_db),
):
    """Start batch scoring on Modal."""
    service = ProjectService(db)
    storage = StorageService(project_id)

    if not storage.checkpoint_path.exists():
        raise HTTPException(status_code=400, detail="Train model first")

    job = service.create_job(project_id, JobType.BATCH_SCORE)
    merged_config = get_pipeline_config(config)

    gpu_tasks.trigger_batch_score.delay(project_id, job.id, merged_config)

    return job


@router.post("/shap", response_model=Job)
def start_shap_analysis(
    project_id: str,
    config: dict = None,
    db: Session = Depends(get_db),
):
    """Start SHAP analysis on Modal."""
    service = ProjectService(db)
    storage = StorageService(project_id)

    if not storage.activations_path.exists():
        raise HTTPException(status_code=400, detail="Run batch scoring first")

    job = service.create_job(project_id, JobType.SHAP_ANALYZE)
    merged_config = get_pipeline_config(config)

    gpu_tasks.trigger_shap_analysis.delay(project_id, job.id, merged_config)

    return job


@router.post("/assign-messages", response_model=Job)
def start_message_assignment(
    project_id: str,
    config: dict = None,
    db: Session = Depends(get_db),
):
    """Assign messages to patterns."""
    service = ProjectService(db)
    storage = StorageService(project_id)

    if not storage.activations_path.exists():
        raise HTTPException(status_code=400, detail="Run batch scoring first")

    job = service.create_job(project_id, JobType.NAME_PATTERNS)  # Reusing type
    merged_config = get_pipeline_config(config)

    cpu_tasks.assign_messages.delay(project_id, job.id, merged_config)

    return job


@router.post("/name-patterns", response_model=Job)
def start_pattern_naming(
    project_id: str,
    config: dict = None,
    db: Session = Depends(get_db),
):
    """Name patterns using LLM."""
    service = ProjectService(db)
    storage = StorageService(project_id)

    if not storage.message_examples_path.exists():
        raise HTTPException(status_code=400, detail="Run message assignment first")

    job = service.create_job(project_id, JobType.NAME_PATTERNS)
    merged_config = get_pipeline_config(config)

    cpu_tasks.name_patterns.delay(project_id, job.id, merged_config)

    return job


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
