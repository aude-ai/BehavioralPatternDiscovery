"""Project management routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Project, ProjectCreate, ProjectUpdate
from ..services import ProjectService

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
    """Get detailed project status including jobs."""
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    jobs = service.get_project_jobs(project_id)

    return {
        "project": project,
        "jobs": jobs,
    }
