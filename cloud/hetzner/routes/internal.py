"""Internal routes for Modal callbacks."""
import gzip
import io
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db, JobModel
from ..models import JobStatus
from ..services import StorageService
from ..websocket import broadcast_to_project

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


async def verify_internal_key(x_internal_key: str = Header(...)):
    """Verify internal API key."""
    if x_internal_key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid internal key")


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
        # Training epoch - store in metadata
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

    # Broadcast to WebSocket
    await broadcast_to_project(project_id, {
        "type": f"job_{event_type}",
        "job_id": job_id,
        "job_type": job.job_type,
        **data,
    })

    return {"status": "ok"}


@router.get("/projects/{project_id}/training-data", dependencies=[Depends(verify_internal_key)])
async def get_training_data(project_id: str):
    """Package and serve training data for Modal."""
    import tarfile
    from fastapi.responses import StreamingResponse

    storage = StorageService(project_id)

    if not storage.train_input_path.exists():
        raise HTTPException(status_code=404, detail="Training data not found")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(storage.train_input_path, arcname="train_input.npy")
        tar.add(storage.message_database_path, arcname="message_database.pkl")

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/gzip",
        headers={"Content-Disposition": "attachment; filename=training_data.tar.gz"},
    )


@router.get("/projects/{project_id}/checkpoint", dependencies=[Depends(verify_internal_key)])
async def get_checkpoint(project_id: str):
    """Serve model checkpoint for Modal."""
    from fastapi.responses import FileResponse

    storage = StorageService(project_id)

    if not storage.checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    return FileResponse(
        storage.checkpoint_path,
        media_type="application/octet-stream",
        filename="best_model.pt",
    )


@router.post("/projects/{project_id}/checkpoint", dependencies=[Depends(verify_internal_key)])
async def upload_checkpoint(project_id: str, checkpoint: UploadFile = File(...)):
    """Receive trained checkpoint from Modal."""
    storage = StorageService(project_id)
    storage.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    content = await checkpoint.read()
    with open(storage.checkpoint_path, "wb") as f:
        f.write(content)

    return {"status": "ok", "size": len(content)}


@router.get("/projects/{project_id}/messages", dependencies=[Depends(verify_internal_key)])
async def get_messages(project_id: str):
    """Serve message database for Modal."""
    from fastapi.responses import Response

    storage = StorageService(project_id)

    if not storage.message_database_path.exists():
        raise HTTPException(status_code=404, detail="Message database not found")

    with open(storage.message_database_path, "rb") as f:
        content = f.read()

    return Response(content=content, media_type="application/octet-stream")


@router.post("/projects/{project_id}/embeddings", dependencies=[Depends(verify_internal_key)])
async def upload_embeddings(
    project_id: str,
    embeddings: UploadFile = File(...),
    embedding_dim: str = Form(...),
):
    """Receive embeddings from Modal."""
    import numpy as np

    storage = StorageService(project_id)

    content = await embeddings.read()
    decompressed = gzip.decompress(content)
    arr = np.load(io.BytesIO(decompressed))

    storage.save_numpy(storage.train_features_path, arr)

    return {"status": "ok", "shape": list(arr.shape), "embedding_dim": int(embedding_dim)}


@router.get("/projects/{project_id}/activations", dependencies=[Depends(verify_internal_key)])
async def get_activations(project_id: str):
    """Serve activations for Modal."""
    from fastapi.responses import FileResponse

    storage = StorageService(project_id)

    if not storage.activations_path.exists():
        raise HTTPException(status_code=404, detail="Activations not found")

    return FileResponse(
        storage.activations_path,
        media_type="application/octet-stream",
        filename="activations.h5",
    )


@router.post("/projects/{project_id}/activations", dependencies=[Depends(verify_internal_key)])
async def upload_activations(
    project_id: str,
    activations: UploadFile = File(...),
    population_stats: str = Form(None),
):
    """Receive activations from Modal."""
    storage = StorageService(project_id)

    content = await activations.read()
    with open(storage.activations_path, "wb") as f:
        f.write(content)

    if population_stats:
        storage.save_json(storage.population_stats_path, json.loads(population_stats))

    return {"status": "ok"}


@router.post("/projects/{project_id}/shap-weights", dependencies=[Depends(verify_internal_key)])
async def upload_shap_weights(project_id: str, weights: dict):
    """Receive SHAP weights from Modal."""
    storage = StorageService(project_id)
    storage.save_json(storage.hierarchical_weights_path, weights)
    return {"status": "ok"}
