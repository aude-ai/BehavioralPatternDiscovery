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


@router.post("/upload/ndjson")
async def upload_ndjson(
    project_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload NDJSON zip file for processing."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)
    storage.ensure_directories()

    # Save uploaded file to project's data collection directory
    upload_path = storage.base_path / "data/collection/uploaded_data.zip"
    content = await file.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "size": len(content),
        "path": str(upload_path),
    }


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


@router.get("/engineers")
def list_engineers(
    project_id: str,
    db: Session = Depends(get_db),
):
    """List engineers with activity counts and metadata."""
    import pandas as pd

    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    if not storage.file_exists(storage.activities_path):
        return {
            "engineers": [],
            "train_count": 0,
            "validation_count": 0,
            "bot_count": 0,
            "total_engineers": 0,
        }

    df = pd.read_csv(storage.activities_path)

    # Group by engineer
    engineers = []
    for engineer_id, group in df.groupby("engineer_id"):
        eng_data = {
            "engineer_id": engineer_id,
            "activity_count": len(group),
            "split": group["split"].iloc[0] if "split" in group.columns else "train",
            "sources": group["source"].unique().tolist() if "source" in group.columns else [],
            "is_bot": bool(group["is_bot"].iloc[0]) if "is_bot" in group.columns else False,
            "is_internal": bool(group["is_internal"].iloc[0]) if "is_internal" in group.columns else None,
            "projects": ",".join(group["project"].unique().tolist()) if "project" in group.columns else "",
        }
        engineers.append(eng_data)

    # Count summaries
    train_count = sum(1 for e in engineers if e["split"] == "train" and not e["is_bot"])
    validation_count = sum(1 for e in engineers if e["split"] == "validation" and not e["is_bot"])
    bot_count = sum(1 for e in engineers if e["is_bot"])

    return {
        "engineers": engineers,
        "train_count": train_count,
        "validation_count": validation_count,
        "bot_count": bot_count,
        "total_engineers": len(engineers),
    }


@router.post("/engineers/set-split")
def set_engineer_split(
    project_id: str,
    request: dict,
    db: Session = Depends(get_db),
):
    """Set train/validation split for selected engineers."""
    import pandas as pd

    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    if not storage.file_exists(storage.activities_path):
        raise HTTPException(status_code=404, detail="No activities file found")

    engineer_ids = request.get("engineer_ids", [])
    split = request.get("split", "train")

    if split not in ("train", "validation"):
        raise HTTPException(status_code=400, detail="Split must be 'train' or 'validation'")

    df = pd.read_csv(storage.activities_path)

    if "split" not in df.columns:
        df["split"] = "train"

    mask = df["engineer_id"].isin(engineer_ids)
    rows_updated = mask.sum()
    df.loc[mask, "split"] = split

    df.to_csv(storage.activities_path, index=False)

    return {
        "message": f"Set {len(engineer_ids)} engineers to {split}",
        "rows_updated": int(rows_updated),
    }


@router.post("/engineers/remove")
def remove_engineers(
    project_id: str,
    request: dict,
    db: Session = Depends(get_db),
):
    """Remove selected engineers from the dataset."""
    import pandas as pd

    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    if not storage.file_exists(storage.activities_path):
        raise HTTPException(status_code=404, detail="No activities file found")

    engineer_ids = request.get("engineer_ids", [])

    df = pd.read_csv(storage.activities_path)
    original_count = len(df)

    df = df[~df["engineer_id"].isin(engineer_ids)]
    rows_deleted = original_count - len(df)

    df.to_csv(storage.activities_path, index=False)

    return {
        "message": f"Removed {len(engineer_ids)} engineers",
        "rows_deleted": int(rows_deleted),
    }


@router.post("/engineers/merge")
def merge_engineers(
    project_id: str,
    request: dict,
    db: Session = Depends(get_db),
):
    """Merge multiple engineer identities into one."""
    import pandas as pd

    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    if not storage.file_exists(storage.activities_path):
        raise HTTPException(status_code=404, detail="No activities file found")

    source_ids = request.get("source_ids", [])
    target_id = request.get("target_id")

    if not target_id:
        raise HTTPException(status_code=400, detail="target_id is required")

    df = pd.read_csv(storage.activities_path)

    mask = df["engineer_id"].isin(source_ids)
    rows_updated = mask.sum()
    df.loc[mask, "engineer_id"] = target_id

    df.to_csv(storage.activities_path, index=False)

    return {
        "message": f"Merged {len(source_ids)} engineers into {target_id}",
        "rows_updated": int(rows_updated),
    }
