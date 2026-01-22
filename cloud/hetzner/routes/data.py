"""Data management routes."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Job, JobType
from ..services import ProjectService, StorageService
from ..tasks import cpu_tasks

router = APIRouter()


@router.post("/upload/activities")
async def upload_activities(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload activities CSV file."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)
    storage.ensure_directories()

    content = await file.read()
    with open(storage.activities_path, "wb") as f:
        f.write(content)

    return {"status": "uploaded", "filename": file.filename, "size": len(content)}


@router.post("/upload/engineer-metadata")
async def upload_engineer_metadata(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload engineer metadata CSV file."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)
    storage.ensure_directories()

    content = await file.read()
    with open(storage.engineer_metadata_path, "wb") as f:
        f.write(content)

    return {"status": "uploaded", "filename": file.filename, "size": len(content)}


@router.post("/fetch/mongodb", response_model=Job)
def fetch_from_mongodb(
    project_id: str,
    config: dict,
    db: Session = Depends(get_db),
):
    """Fetch data from MongoDB."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    job = service.create_job(project_id, JobType.FETCH_DATA)
    cpu_tasks.fetch_mongodb_data.delay(project_id, job.id, config)

    return job


@router.post("/fetch/ndjson", response_model=Job)
def fetch_from_ndjson(
    project_id: str,
    config: dict,
    db: Session = Depends(get_db),
):
    """Fetch data from NDJSON files."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    job = service.create_job(project_id, JobType.FETCH_DATA)
    cpu_tasks.fetch_ndjson_data.delay(project_id, job.id, config)

    return job


@router.get("/status")
def get_data_status(
    project_id: str,
    db: Session = Depends(get_db),
):
    """Get status of data files for a project."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    return {
        "activities": {
            "exists": storage.file_exists(storage.activities_path),
            "size": storage.get_file_size(storage.activities_path),
        },
        "engineer_metadata": {
            "exists": storage.file_exists(storage.engineer_metadata_path),
            "size": storage.get_file_size(storage.engineer_metadata_path),
        },
        "message_database": {
            "exists": storage.file_exists(storage.message_database_path),
            "size": storage.get_file_size(storage.message_database_path),
        },
        "embeddings": {
            "exists": storage.file_exists(storage.train_features_path),
            "size": storage.get_file_size(storage.train_features_path),
        },
        "checkpoint": {
            "exists": storage.file_exists(storage.checkpoint_path),
            "size": storage.get_file_size(storage.checkpoint_path),
        },
    }
