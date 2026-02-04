"""Data management routes."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Job, JobType
from ..services import ProjectService, StorageService
from ..tasks import cpu_tasks

router = APIRouter()


def load_data_config() -> dict:
    """Load the main data config from config/data.yaml."""
    from src.core.config import load_config

    config_path = Path(__file__).parent.parent.parent.parent / "config" / "data.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Data config not found at {config_path}")
    return load_config(config_path)


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
    config: dict = None,
    db: Session = Depends(get_db),
):
    """Fetch data from NDJSON files."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Load base config from config/data.yaml
    data_config = load_data_config()

    if "collection" not in data_config:
        raise HTTPException(status_code=500, detail="Missing required config section: collection in data.yaml")

    collection_config = data_config["collection"]

    # Build config for NDJSONLoader from the data.yaml structure
    ndjson_config = {
        "parser_configs": collection_config.get("parsers", {}),  # Optional parser overrides
        "excluded_sections": collection_config.get("ndjson", {}).get("excluded_sections", []),  # Optional exclusions
        "text_cleanup": collection_config.get("text_cleanup", {}),  # Optional cleanup config
    }

    # Merge with user-provided overrides
    if config:
        for key, value in config.items():
            if key in ndjson_config and isinstance(ndjson_config[key], dict) and isinstance(value, dict):
                ndjson_config[key] = {**ndjson_config[key], **value}
            else:
                ndjson_config[key] = value

    # If no input_path provided, use uploaded file
    storage = StorageService(project_id)
    if "input_path" not in ndjson_config or not ndjson_config["input_path"]:
        upload_path = storage.base_path / "data/collection/uploaded_data.zip"
        if upload_path.exists():
            ndjson_config["input_path"] = str(upload_path)
        else:
            raise HTTPException(status_code=400, detail="No input_path provided and no uploaded file found")

    job = service.create_job(project_id, JobType.FETCH_DATA)
    cpu_tasks.fetch_ndjson_data.delay(project_id, job.id, ndjson_config)

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

    # R2 file status (large processing outputs)
    from ..services.r2_service import get_r2_file_info
    embeddings_info = get_r2_file_info(project_id, "embeddings")
    checkpoint_info = get_r2_file_info(project_id, "checkpoint")
    message_db_info = get_r2_file_info(project_id, "message_database")

    return {
        # Local Hetzner files
        "activities": {
            "exists": storage.file_exists(storage.activities_path),
            "size": storage.get_file_size(storage.activities_path),
        },
        "engineer_metadata": {
            "exists": storage.file_exists(storage.engineer_metadata_path),
            "size": storage.get_file_size(storage.engineer_metadata_path),
        },
        # R2 files
        "message_database": {
            "exists": message_db_info.get("exists", False),
            "size": message_db_info.get("size_bytes", 0),
            "location": "r2",
        },
        "embeddings": {
            "exists": embeddings_info.get("exists", False),
            "size": embeddings_info.get("size_bytes", 0),
            "location": "r2",
        },
        "checkpoint": {
            "exists": checkpoint_info.get("exists", False),
            "size": checkpoint_info.get("size_bytes", 0),
            "location": "r2",
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
        # Use .any() for boolean flags to preserve True if any row has it
        is_bot = bool(group["is_bot"].any()) if "is_bot" in group.columns else False
        is_internal = bool(group["is_internal"].any()) if "is_internal" in group.columns else None

        # Use 'validation' if any row has it, else 'train'
        split = "train"
        if "split" in group.columns:
            split = "validation" if (group["split"] == "validation").any() else "train"

        eng_data = {
            "engineer_id": engineer_id,
            "activity_count": len(group),
            "split": split,
            "sources": group["source"].unique().tolist() if "source" in group.columns else [],
            "is_bot": is_bot,
            "is_internal": is_internal,
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

    if "engineer_ids" not in request:
        raise HTTPException(status_code=400, detail="engineer_ids is required")

    engineer_ids = request["engineer_ids"]
    split = request.get("split", "train")  # Default to train if not specified

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

    if "engineer_ids" not in request:
        raise HTTPException(status_code=400, detail="engineer_ids is required")

    engineer_ids = request["engineer_ids"]

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
    """Merge multiple engineer identities into one.

    Preserves non-default features:
    - is_internal: True if ANY source engineer was internal
    - is_bot: True if ANY source engineer was a bot
    - split: Uses 'validation' if ANY source was validation, else 'train'
    """
    import pandas as pd

    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    if not storage.file_exists(storage.activities_path):
        raise HTTPException(status_code=404, detail="No activities file found")

    if "source_ids" not in request:
        raise HTTPException(status_code=400, detail="source_ids is required")
    if "target_id" not in request:
        raise HTTPException(status_code=400, detail="target_id is required")

    source_ids = request["source_ids"]
    target_id = request["target_id"]

    if not target_id:
        raise HTTPException(status_code=400, detail="target_id is required")

    df = pd.read_csv(storage.activities_path)

    mask = df["engineer_id"].isin(source_ids)
    rows_updated = mask.sum()

    if rows_updated == 0:
        raise HTTPException(status_code=404, detail="No matching engineers found")

    # Determine merged feature values (preserve non-default values)
    source_rows = df[mask]

    # is_internal: True if any source had is_internal=True
    merged_is_internal = False
    if "is_internal" in df.columns:
        merged_is_internal = source_rows["is_internal"].any()

    # is_bot: True if any source had is_bot=True
    merged_is_bot = False
    if "is_bot" in df.columns:
        merged_is_bot = source_rows["is_bot"].any()

    # split: 'validation' if any was validation, else 'train'
    merged_split = "train"
    if "split" in df.columns:
        if (source_rows["split"] == "validation").any():
            merged_split = "validation"

    # Update all matching rows
    df.loc[mask, "engineer_id"] = target_id

    if "is_internal" in df.columns:
        df.loc[mask, "is_internal"] = merged_is_internal

    if "is_bot" in df.columns:
        df.loc[mask, "is_bot"] = merged_is_bot

    if "split" in df.columns:
        df.loc[mask, "split"] = merged_split

    df.to_csv(storage.activities_path, index=False)

    return {
        "message": f"Merged {len(source_ids)} engineers into {target_id}",
        "rows_updated": int(rows_updated),
        "merged_features": {
            "is_internal": merged_is_internal,
            "is_bot": merged_is_bot,
            "split": merged_split,
        }
    }
