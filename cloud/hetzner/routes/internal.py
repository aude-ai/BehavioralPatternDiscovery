"""Internal routes for Modal callbacks."""
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db, JobModel
from ..models import JobStatus
from ..services import StorageService
from ..websocket import broadcast_to_project

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# Chunk size for streaming (64KB)
CHUNK_SIZE = 65536


async def verify_internal_key(x_internal_key: str = Header(...)):
    """Verify internal API key."""
    if x_internal_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid internal key")


def iter_file(path: Path, chunk_size: int = CHUNK_SIZE):
    """Generator that yields file chunks."""
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk


# =============================================================================
# JOB EVENTS (Modal -> Hetzner)
# =============================================================================


@router.post("/jobs/{job_id}/event", dependencies=[Depends(verify_internal_key)])
async def job_event(job_id: str, data: dict, db: Session = Depends(get_db)):
    """
    Receive event from Modal.

    Event types:
    - status: Progress message update
    - progress: Progress percentage + optional message
    - epoch: Training epoch metrics
    - completed: Job finished successfully
    - failed: Job failed with error
    """
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    event_type = data.get("type")
    project_id = job.project_id

    update_data = {}

    if event_type == "status":
        update_data["progress_message"] = data.get("message")

    elif event_type == "progress":
        update_data["progress"] = data.get("progress", 0)
        if "message" in data:
            update_data["progress_message"] = data["message"]
        # Support step-level progress from unified pipeline
        if "step" in data:
            metadata = job.metadata_ or {}
            metadata["current_step"] = data.get("step")
            metadata["step_name"] = data.get("step_name")
            update_data["metadata_"] = metadata

    elif event_type == "epoch":
        metadata = job.metadata_ or {}
        metadata["last_epoch"] = data.get("epoch")
        metadata["last_metrics"] = data.get("metrics")
        update_data["metadata_"] = metadata

    elif event_type == "completed":
        update_data["status"] = JobStatus.COMPLETED
        update_data["completed_at"] = datetime.utcnow()
        update_data["progress"] = 1.0
        update_data["progress_message"] = data.get("message", "Completed")

    elif event_type == "failed":
        update_data["status"] = JobStatus.FAILED
        update_data["completed_at"] = datetime.utcnow()
        update_data["error"] = data.get("error")

    if update_data:
        db.query(JobModel).filter(JobModel.id == job_id).update(update_data)
        db.commit()

    # Broadcast to WebSocket clients
    section = data.get("section", "processing")
    await broadcast_to_project(project_id, {
        "type": f"job_{event_type}",
        "job_id": job_id,
        "job_type": job.job_type,
        "section": section,
        **data,
    })

    return {"status": "ok"}


# =============================================================================
# SMALL JSON ENDPOINTS (Modal -> Hetzner)
# These are the only files that come back from Modal to Hetzner.
# Large files stay in R2.
# =============================================================================


@router.post("/projects/{project_id}/population-stats", dependencies=[Depends(verify_internal_key)])
async def upload_population_stats(project_id: str, stats: dict):
    """Receive population stats from Modal (small JSON, from B.6)."""
    storage = StorageService(project_id)
    storage.save_json(storage.population_stats_path, stats)
    return {"status": "ok"}


@router.get("/projects/{project_id}/population-stats", dependencies=[Depends(verify_internal_key)])
async def get_population_stats(project_id: str):
    """Serve population stats to Modal (for D.1 individual scoring)."""
    storage = StorageService(project_id)

    if not storage.population_stats_path.exists():
        raise HTTPException(status_code=404, detail="Population stats not found")

    stats = storage.load_json(storage.population_stats_path)
    return stats


@router.post("/projects/{project_id}/message-examples", dependencies=[Depends(verify_internal_key)])
async def upload_message_examples(project_id: str, examples: dict):
    """Receive message examples from Modal (small JSON, from B.7)."""
    storage = StorageService(project_id)
    storage.save_json(storage.message_examples_path, examples)
    return {"status": "ok"}


@router.post("/projects/{project_id}/shap-weights", dependencies=[Depends(verify_internal_key)])
async def upload_shap_weights(project_id: str, weights: dict):
    """Receive SHAP weights from Modal (small JSON, from B.8)."""
    storage = StorageService(project_id)
    storage.save_json(storage.hierarchical_weights_path, weights)
    return {"status": "ok"}


# =============================================================================
# ACTIVITIES CSV (Hetzner -> Modal)
# This is the only file Modal needs from Hetzner to start processing.
# =============================================================================


@router.get("/projects/{project_id}/activities", dependencies=[Depends(verify_internal_key)])
async def get_activities(project_id: str):
    """
    Serve activities CSV for Modal processing (Segment B input).

    This is a small file (typically < 50MB) that Modal needs to start B.1.
    """
    storage = StorageService(project_id)

    if not storage.activities_path.exists():
        raise HTTPException(status_code=404, detail="Activities not found")

    file_size = storage.activities_path.stat().st_size

    return StreamingResponse(
        iter_file(storage.activities_path),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=activities.csv",
            "Content-Length": str(file_size),
        },
    )
