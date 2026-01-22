"""External API routes for third-party integration."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..services import ProjectService, StorageService

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.get("/patterns")
def list_patterns(
    project_id: str = Query(...),
    level: Optional[str] = None,
    encoder: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    List all patterns with metadata.

    Used by external systems to get discovered patterns.
    """
    service = ProjectService(db)
    storage = StorageService(project_id)

    if not service.get_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

    pattern_names = storage.load_json(storage.pattern_names_path)
    if not pattern_names:
        raise HTTPException(status_code=404, detail="Patterns not found")

    message_examples = storage.load_json(storage.message_examples_path) or {}
    hierarchical_weights = storage.load_json(storage.hierarchical_weights_path) or {}
    population_stats = storage.load_json(storage.population_stats_path) or {}

    patterns = []
    for pattern_id, info in pattern_names.items():
        # Parse pattern ID (e.g., "enc1_bottom_0", "unified_3")
        parts = pattern_id.rsplit("_", 1)
        if len(parts) == 2:
            pattern_level = parts[0]
            pattern_idx = int(parts[1])
        else:
            continue

        # Filter by level/encoder if specified
        if level and level not in pattern_level:
            continue
        if encoder and not pattern_level.startswith(encoder):
            continue

        examples = message_examples.get(pattern_id, {}).get("examples", [])
        stats = population_stats.get(pattern_id, {})

        patterns.append({
            "id": pattern_id,
            "name": info.get("name"),
            "description": info.get("description"),
            "level": pattern_level,
            "index": pattern_idx,
            "examples": examples[:5],
            "statistics": stats,
        })

    return {"patterns": patterns, "count": len(patterns)}


@router.get("/patterns/{pattern_id}")
def get_pattern(
    pattern_id: str,
    project_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """Get detailed information about a specific pattern."""
    storage = StorageService(project_id)

    pattern_names = storage.load_json(storage.pattern_names_path)
    if not pattern_names or pattern_id not in pattern_names:
        raise HTTPException(status_code=404, detail="Pattern not found")

    info = pattern_names[pattern_id]
    message_examples = storage.load_json(storage.message_examples_path) or {}
    examples = message_examples.get(pattern_id, {}).get("examples", [])

    return {
        "id": pattern_id,
        "name": info.get("name"),
        "description": info.get("description"),
        "examples": examples,
    }


@router.get("/patterns/{pattern_id}/messages")
def get_pattern_messages(
    pattern_id: str,
    project_id: str = Query(...),
    top_k: int = Query(20, ge=1, le=100),
    engineer_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Get messages that activate a pattern.

    Used to provide evidence snippets for evaluations.
    """
    storage = StorageService(project_id)

    message_examples = storage.load_json(storage.message_examples_path)
    if not message_examples or pattern_id not in message_examples:
        raise HTTPException(status_code=404, detail="Pattern not found")

    examples = message_examples[pattern_id].get("examples", [])

    # Filter by engineer if specified
    if engineer_id:
        examples = [e for e in examples if e.get("engineer_id") == engineer_id]

    return {
        "pattern_id": pattern_id,
        "messages": examples[:top_k],
        "total": len(examples),
    }


@router.get("/model/metadata")
def get_model_metadata(
    project_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """Get model architecture and training metadata."""
    import torch

    storage = StorageService(project_id)

    if not storage.checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    checkpoint = torch.load(storage.checkpoint_path, map_location="cpu")

    return {
        "config": checkpoint.get("config"),
        "metadata": checkpoint.get("metadata"),
        "training_epochs": checkpoint.get("epoch"),
    }
