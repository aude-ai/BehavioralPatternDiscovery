"""Project management routes."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (
    Project, ProjectCreate, ProjectUpdate,
    VariantCreate, DeletionBlockedResponse, DeletionSuccessResponse,
)
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


@router.get("/{project_id}/config")
def get_project_config(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
):
    """Get project configuration including thresholds for viewers."""
    from .pipeline import get_pipeline_config

    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Load base pipeline config
    config = get_pipeline_config()

    # Extract viewer-relevant settings
    report_config = config.get("report", {})

    return {
        "strength_threshold": report_config.get("strength_threshold", 70),
        "weakness_threshold": report_config.get("weakness_threshold", 40),
        "report_viewer": config.get("report_viewer", {}),
    }


@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
):
    """Delete project.

    Returns 200 with deletion info on success.
    Returns 409 if variants still borrow files from this project.
    """
    result = service.delete_project(project_id)

    if result.get("error") == "not_found":
        raise HTTPException(status_code=404, detail="Project not found")

    if result.get("error") == "deletion_blocked":
        return JSONResponse(
            status_code=409,
            content=result,
        )

    return result


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
    # Local Hetzner files
    has_activities = storage.file_exists(storage.activities_path)
    has_population_stats = storage.file_exists(storage.population_stats_path)
    has_hierarchical_weights = storage.file_exists(storage.hierarchical_weights_path)
    has_message_scores_index = storage.file_exists(storage.message_scores_index_path)
    has_pattern_names = storage.file_exists(storage.pattern_names_path)

    # R2 files (large processing outputs stored in cloud)
    from ..services.r2_service import r2_file_exists
    has_embeddings = r2_file_exists(project_id, "embeddings")
    has_checkpoint = r2_file_exists(project_id, "checkpoint")
    has_activations = r2_file_exists(project_id, "activations")

    # Check for running jobs
    running_jobs = [j for j in jobs if j.status == "running"]
    current_task = running_jobs[0].job_type if running_jobs else None

    # Get variant count if this is a root project
    variant_count = service.get_variant_count(project_id) if project.is_root else 0

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
            "shap_analysis": has_activations and has_message_scores_index,
            "identify_patterns": has_hierarchical_weights and has_message_scores_index,
            "population_viewer": has_activations,
            "score_engineer": has_checkpoint and has_population_stats,
        },
        "project": project,
        "jobs": jobs,
        "variant_count": variant_count,
    }


# =============================================================================
# VARIANT ENDPOINTS (Phase 5)
# =============================================================================


@router.post("/{project_id}/variants", response_model=Project)
def create_variant(
    project_id: str,
    data: VariantCreate,
    service: ProjectService = Depends(get_project_service),
):
    """Create a variant of an existing project.

    Variants inherit data from their parent and only store data
    for pipeline steps they have run.
    """
    result = service.create_variant(project_id, data)

    if result.get("error") == "not_found":
        raise HTTPException(status_code=404, detail="Parent project not found")

    if result.get("error") == "invalid_parent":
        raise HTTPException(status_code=400, detail=result.get("message"))

    return result["variant"]


@router.get("/{project_id}/variants", response_model=list[Project])
def list_variants(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
):
    """List all variants of a project."""
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return service.get_variants(project_id)


@router.delete("/{project_id}/variants/{variant_id}")
def delete_variant(
    project_id: str,
    variant_id: str,
    service: ProjectService = Depends(get_project_service),
):
    """Delete a variant project.

    Only deletes files owned by the variant, not inherited files.
    """
    # Verify the variant belongs to this parent
    variant = service.get_project(variant_id)
    if not variant:
        raise HTTPException(status_code=404, detail="Variant not found")

    if variant.parent_id != project_id:
        raise HTTPException(status_code=400, detail="Variant does not belong to this project")

    if not service.delete_variant(variant_id):
        raise HTTPException(status_code=500, detail="Failed to delete variant")

    return {"status": "deleted", "variant_id": variant_id}
