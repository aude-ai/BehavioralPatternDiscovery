"""Project management routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Project, ProjectCreate, ProjectUpdate
from ..services import ProjectService, StorageService

router = APIRouter()


def get_project_service(db: Session = Depends(get_db)) -> ProjectService:
    return ProjectService(db)


@router.post("", response_model=Project)
def create_project(
    data: ProjectCreate,
    service: ProjectService = Depends(get_project_service),
):
    """Create a new project."""
    return service.create_project(data)


@router.get("", response_model=list[Project])
def list_projects(
    skip: int = 0,
    limit: int = 100,
    service: ProjectService = Depends(get_project_service),
):
    """List all projects."""
    return service.list_projects(skip=skip, limit=limit)


@router.get("/{project_id}", response_model=Project)
def get_project(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
):
    """Get project by ID."""
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/{project_id}", response_model=Project)
def update_project(
    project_id: str,
    data: ProjectUpdate,
    service: ProjectService = Depends(get_project_service),
):
    """Update project."""
    project = service.update_project(project_id, data)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
):
    """Delete project."""
    if not service.delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted"}


@router.get("/{project_id}/status")
def get_project_status(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
):
    """Get detailed project status including pipeline readiness."""
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    jobs = service.get_project_jobs(project_id)
    storage = StorageService(project_id)

    # Check what files exist to determine pipeline readiness
    has_activities = storage.file_exists(storage.activities_path)
    has_embeddings = storage.file_exists(storage.train_features_path)
    has_checkpoint = storage.file_exists(storage.checkpoint_path)
    has_activations = storage.file_exists(storage.activations_path)
    has_population_stats = storage.file_exists(storage.population_stats_path)
    has_hierarchical_weights = storage.file_exists(storage.hierarchical_weights_path)
    has_message_examples = storage.file_exists(storage.message_examples_path)
    has_pattern_names = storage.file_exists(storage.pattern_names_path)

    # Check for running jobs
    running_jobs = [j for j in jobs if j.status == "running"]
    current_task = running_jobs[0].job_type if running_jobs else None

    return {
        "status": {
            "project_status": project.status,
            "data_collected": has_activities,
            "preprocessed": has_embeddings,
            "models_trained": has_checkpoint,
            "batch_scored": has_activations and has_population_stats,
            "patterns_interpreted": has_hierarchical_weights,
            "patterns_identified": has_pattern_names,
            "current_task": current_task,
            "last_error": None,
        },
        "can_run": {
            "preprocess": has_activities,
            "train_vae": has_embeddings,
            "batch_score": has_checkpoint,
            "shap_analysis": has_activations and has_message_examples,
            "identify_patterns": has_hierarchical_weights and has_message_examples,
            "population_viewer": has_activations,
            "score_engineer": has_checkpoint and has_population_stats,
        },
        "project": project,
        "jobs": jobs,
    }
