"""External API routes for third-party integration."""
import pickle
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..services import ProjectService, StorageService
from ..services.r2_service import download_h5_from_r2, download_pickle_from_r2
from src.pattern_identification.message_scorer import MessageScorer

router = APIRouter()

# Load external config
_external_config = None


def get_external_config() -> dict:
    """Load external.yaml config."""
    global _external_config
    if _external_config is None:
        from src.core.config import load_config
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "external.yaml"
        _external_config = load_config(config_path)
    return _external_config


def _get_h5_and_messages(storage: StorageService, project_id: str) -> tuple[Path, list[dict]]:
    """Get cached message_scores.h5 and message database for querying."""
    h5_path = storage.get_cached_h5("message_scores")
    if not h5_path.exists():
        download_h5_from_r2(project_id, "message_scores", h5_path)

    msg_cache_path = storage.base_path / "cache" / "message_database.pkl"
    if not msg_cache_path.exists():
        msg_data = download_pickle_from_r2(project_id, "message_database")
        msg_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(msg_cache_path, "wb") as f:
            pickle.dump(msg_data, f)
    else:
        with open(msg_cache_path, "rb") as f:
            msg_data = pickle.load(f)

    messages_list = msg_data.get("messages", msg_data) if isinstance(msg_data, dict) else msg_data
    return h5_path, messages_list


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

    hierarchical_weights = storage.load_json(storage.hierarchical_weights_path) or {}
    population_stats = storage.load_json(storage.population_stats_path) or {}

    config = get_external_config()
    top_k_examples = config.get("pattern_export", {}).get("top_k_examples", 5)

    h5_path, messages_list = _get_h5_and_messages(storage, project_id)

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

        examples = MessageScorer.get_top_messages_for_pattern(
            h5_path=h5_path,
            level_key=pattern_level,
            pattern_idx=pattern_idx,
            message_database=messages_list,
            limit=top_k_examples,
        )
        stats = population_stats.get(pattern_id, {})

        patterns.append({
            "id": pattern_id,
            "name": info.get("name"),
            "description": info.get("description"),
            "level": pattern_level,
            "index": pattern_idx,
            "examples": examples,
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

    # Parse pattern_id to query examples
    parts = pattern_id.rsplit("_", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail=f"Invalid pattern_id format: {pattern_id}")

    level_key, idx_str = parts
    pattern_idx = int(idx_str)

    h5_path, messages_list = _get_h5_and_messages(storage, project_id)
    examples = MessageScorer.get_top_messages_for_pattern(
        h5_path=h5_path,
        level_key=level_key,
        pattern_idx=pattern_idx,
        message_database=messages_list,
        limit=20,
    )

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
    top_k: Optional[int] = Query(None, ge=1),
    engineer_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Get messages that activate a pattern.

    Used to provide evidence snippets for evaluations.
    """
    config = get_external_config()
    search_config = config.get("message_search", {})
    default_top_k = search_config.get("default_top_k", 20)
    max_top_k = search_config.get("max_top_k", 100)

    # Apply defaults and limits from config
    if top_k is None:
        top_k = default_top_k
    top_k = min(top_k, max_top_k)

    storage = StorageService(project_id)

    # Parse pattern_id
    parts = pattern_id.rsplit("_", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail=f"Invalid pattern_id format: {pattern_id}")

    level_key, idx_str = parts
    pattern_idx = int(idx_str)

    h5_path, messages_list = _get_h5_and_messages(storage, project_id)

    # Get message indices filtered by engineer if specified
    message_indices = None
    if engineer_id:
        index = storage.load_json(storage.message_scores_index_path)
        if index and engineer_id in index.get("engineers", {}):
            message_indices = index["engineers"][engineer_id]["message_indices"]
        else:
            return {"pattern_id": pattern_id, "messages": [], "total": 0}

    examples = MessageScorer.get_top_messages_for_pattern(
        h5_path=h5_path,
        level_key=level_key,
        pattern_idx=pattern_idx,
        message_database=messages_list,
        limit=top_k,
        message_indices=message_indices,
    )

    return {
        "pattern_id": pattern_id,
        "messages": examples,
        "total": len(examples),
    }


@router.get("/model/metadata")
def get_model_metadata(
    project_id: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Get model architecture and training metadata.

    Reads from model_metadata.json stored in R2 (created during training on Modal).
    This avoids torch dependency on Hetzner.
    """
    from ..services.r2_service import get_r2_file_info, download_json_from_r2

    # Check if model metadata exists in R2
    file_info = get_r2_file_info(project_id, "model_metadata")
    if not file_info.get("exists"):
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found. Run training pipeline first.",
        )

    # Download and return the JSON metadata
    try:
        metadata = download_json_from_r2(project_id, "model_metadata")
        return metadata
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model metadata: {e}",
        )
