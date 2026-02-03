"""Internal routes for Modal callbacks."""
import logging
import tarfile
import tempfile
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


async def stream_upload_to_file(request: Request, output_path: Path) -> int:
    """
    Stream upload body directly to file.

    Returns total bytes written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    with open(output_path, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            total_bytes += len(chunk)

    return total_bytes


# =============================================================================
# JOB EVENTS (unchanged - small JSON payloads)
# =============================================================================


@router.post("/jobs/{job_id}/event", dependencies=[Depends(verify_internal_key)])
async def job_event(job_id: str, data: dict, db: Session = Depends(get_db)):
    """Receive event from Modal."""
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

    # Broadcast to WebSocket clients with section info for log routing
    section = data.get("section", "general")
    await broadcast_to_project(project_id, {
        "type": f"job_{event_type}",
        "job_id": job_id,
        "job_type": job.job_type,
        "section": section,
        **data,
    })

    return {"status": "ok"}


# =============================================================================
# TRAINING DATA - DOWNLOAD (streaming)
# =============================================================================


@router.get("/projects/{project_id}/training-data", dependencies=[Depends(verify_internal_key)])
async def get_training_data(project_id: str):
    """
    Serve training data bundle (tarball of compressed files).

    Creates tarball on-the-fly and streams it.
    Files are already compressed (.zst), so tarball uses no compression.
    """
    storage = StorageService(project_id)

    if not storage.train_input_path.exists():
        raise HTTPException(status_code=404, detail="Training data not found")

    # Create tarball in temp file (not memory)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with tarfile.open(tmp_path, mode="w") as tar:
            tar.add(storage.train_input_path, arcname="train_input.npy.zst")
            tar.add(storage.message_database_path, arcname="message_database.pkl.zst")

        file_size = tmp_path.stat().st_size

        def cleanup_iter():
            try:
                yield from iter_file(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

        return StreamingResponse(
            cleanup_iter(),
            media_type="application/x-tar",
            headers={
                "Content-Disposition": "attachment; filename=training_data.tar",
                "Content-Length": str(file_size),
            },
        )
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# =============================================================================
# CHECKPOINT - DOWNLOAD/UPLOAD (streaming)
# =============================================================================


@router.get("/projects/{project_id}/checkpoint", dependencies=[Depends(verify_internal_key)])
async def get_checkpoint(project_id: str):
    """
    Serve model checkpoint (compressed).

    Streams directly from disk - constant memory usage.
    """
    storage = StorageService(project_id)

    if not storage.checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    file_size = storage.checkpoint_path.stat().st_size

    return StreamingResponse(
        iter_file(storage.checkpoint_path),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=best_model.pt.zst",
            "Content-Length": str(file_size),
        },
    )


@router.post("/projects/{project_id}/checkpoint", dependencies=[Depends(verify_internal_key)])
async def upload_checkpoint(project_id: str, request: Request):
    """
    Receive trained checkpoint from Modal (compressed).

    Streams directly to disk - constant memory usage.
    """
    storage = StorageService(project_id)
    storage.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = await stream_upload_to_file(request, storage.checkpoint_path)

    return {"status": "ok", "size": total_bytes}


# =============================================================================
# TRAIN INPUT - DOWNLOAD (streaming)
# =============================================================================


@router.get("/projects/{project_id}/train-input", dependencies=[Depends(verify_internal_key)])
async def get_train_input(project_id: str):
    """
    Serve training input (embeddings + aux features, compressed).

    Streams directly from disk - constant memory usage.
    """
    storage = StorageService(project_id)

    if not storage.train_input_path.exists():
        raise HTTPException(status_code=404, detail="Training input not found")

    file_size = storage.train_input_path.stat().st_size

    return StreamingResponse(
        iter_file(storage.train_input_path),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=train_input.npy.zst",
            "Content-Length": str(file_size),
        },
    )


# =============================================================================
# MESSAGES - DOWNLOAD (streaming)
# =============================================================================


@router.get("/projects/{project_id}/messages", dependencies=[Depends(verify_internal_key)])
async def get_messages(project_id: str):
    """
    Serve message database (compressed).

    Streams directly from disk - constant memory usage.
    """
    storage = StorageService(project_id)

    if not storage.message_database_path.exists():
        raise HTTPException(status_code=404, detail="Message database not found")

    file_size = storage.message_database_path.stat().st_size

    return StreamingResponse(
        iter_file(storage.message_database_path),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=message_database.pkl.zst",
            "Content-Length": str(file_size),
        },
    )


# =============================================================================
# EMBEDDINGS - UPLOAD (streaming)
# =============================================================================


@router.post("/projects/{project_id}/embeddings", dependencies=[Depends(verify_internal_key)])
async def upload_embeddings(project_id: str, request: Request):
    """
    Receive embeddings from Modal (compressed).

    Streams directly to disk - constant memory usage.
    Modal sends pre-compressed .npy.zst file.
    """
    storage = StorageService(project_id)

    total_bytes = await stream_upload_to_file(request, storage.train_features_path)

    return {"status": "ok", "size": total_bytes}


# =============================================================================
# ACTIVATIONS - DOWNLOAD/UPLOAD (streaming)
# =============================================================================


@router.get("/projects/{project_id}/activations", dependencies=[Depends(verify_internal_key)])
async def get_activations(project_id: str):
    """
    Serve activations file (compressed).

    Streams directly from disk - constant memory usage.
    """
    storage = StorageService(project_id)

    if not storage.activations_path.exists():
        raise HTTPException(status_code=404, detail="Activations not found")

    file_size = storage.activations_path.stat().st_size

    return StreamingResponse(
        iter_file(storage.activations_path),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=activations.h5.zst",
            "Content-Length": str(file_size),
        },
    )


@router.post("/projects/{project_id}/activations", dependencies=[Depends(verify_internal_key)])
async def upload_activations(project_id: str, request: Request):
    """
    Receive activations from Modal (compressed).

    Streams directly to disk - constant memory usage.
    Population stats sent separately via JSON endpoint.
    """
    storage = StorageService(project_id)

    total_bytes = await stream_upload_to_file(request, storage.activations_path)

    return {"status": "ok", "size": total_bytes}


@router.post("/projects/{project_id}/population-stats", dependencies=[Depends(verify_internal_key)])
async def upload_population_stats(project_id: str, stats: dict):
    """Receive population stats (small JSON, no streaming needed)."""
    storage = StorageService(project_id)
    storage.save_json(storage.population_stats_path, stats)
    return {"status": "ok"}


# =============================================================================
# SHAP WEIGHTS - UPLOAD (small JSON, no streaming needed)
# =============================================================================


@router.post("/projects/{project_id}/shap-weights", dependencies=[Depends(verify_internal_key)])
async def upload_shap_weights(project_id: str, weights: dict):
    """Receive SHAP weights from Modal."""
    storage = StorageService(project_id)
    storage.save_json(storage.hierarchical_weights_path, weights)
    return {"status": "ok"}
