"""Pattern management routes."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..services import ProjectService, StorageService

router = APIRouter()


@router.get("")
def list_patterns(
    project_id: str,
    level: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all discovered patterns."""
    service = ProjectService(db)
    storage = StorageService(project_id)

    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    pattern_names = storage.load_json(storage.pattern_names_path)
    if not pattern_names:
        # Return empty list if no patterns yet
        return {"patterns": [], "count": 0}

    patterns = []
    # Pattern names structure: {"enc1_bottom": {"bottom_0": {...}, "bottom_1": {...}}, "unified": {"unified_0": {...}}}
    for level_key, level_patterns in pattern_names.items():
        if not isinstance(level_patterns, dict):
            continue

        # Filter by level if specified
        if level and level not in level_key:
            continue

        for dim_key, info in level_patterns.items():
            if not isinstance(info, dict):
                continue

            # Extract dimension index from dim_key (e.g., "bottom_0" -> 0, "unified_0" -> 0)
            dim_idx = 0
            parts = dim_key.split("_")
            if len(parts) >= 2:
                try:
                    dim_idx = int(parts[-1])
                except ValueError:
                    pass

            pattern_id = f"{level_key}_{dim_key}"
            patterns.append({
                "id": pattern_id,
                "name": info.get("name"),
                "description": info.get("description"),
                "level": level_key,
                "index": dim_idx,
            })

    return {"patterns": patterns, "count": len(patterns)}


@router.get("/names")
def get_pattern_names(
    project_id: str,
    db: Session = Depends(get_db),
):
    """Get pattern names dictionary."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)
    pattern_names = storage.load_json(storage.pattern_names_path)
    if not pattern_names:
        return {}

    return pattern_names


@router.get("/hierarchical-weights")
def get_hierarchical_weights(
    project_id: str,
    db: Session = Depends(get_db),
):
    """Get SHAP hierarchical weights."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)
    hierarchical_weights = storage.load_json(storage.hierarchical_weights_path)
    if not hierarchical_weights:
        raise HTTPException(status_code=404, detail="Hierarchical weights not found - run SHAP analysis first")

    return hierarchical_weights


@router.get("/population-data")
def get_population_data(
    project_id: str,
    engineer_ids: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get population viewer data (raw population stats)."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    # Load population stats
    population_stats = storage.load_json(storage.population_stats_path)
    if not population_stats:
        raise HTTPException(status_code=404, detail="Population stats not found - run batch scoring first")

    return population_stats


@router.get("/shap-interpretations/{model_name}")
def get_shap_interpretations(
    project_id: str,
    model_name: str,
    db: Session = Depends(get_db),
):
    """Get SHAP interpretations for a model."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    # Try to load SHAP interpretations file
    shap_file = storage.project_dir / "shap" / f"shap_interpretations_{model_name}.json"
    if not shap_file.exists():
        raise HTTPException(status_code=404, detail=f"SHAP interpretations for {model_name} not found")

    return storage.load_json(shap_file)


@router.get("/hierarchy")
def get_pattern_hierarchy(
    project_id: str,
    db: Session = Depends(get_db),
):
    """Get hierarchical structure of patterns."""
    storage = StorageService(project_id)

    hierarchical_weights = storage.load_json(storage.hierarchical_weights_path)
    if not hierarchical_weights:
        raise HTTPException(status_code=404, detail="Hierarchical weights not found - run SHAP analysis first")

    pattern_names = storage.load_json(storage.pattern_names_path) or {}

    # Build hierarchy tree
    hierarchy = {
        "unified": [],
        "encoders": {},
    }

    for pattern_id, weights in hierarchical_weights.items():
        name_info = pattern_names.get(pattern_id, {})

        if pattern_id.startswith("unified_"):
            hierarchy["unified"].append({
                "id": pattern_id,
                "name": name_info.get("name"),
                "children": weights.get("children", []),
            })
        else:
            # Extract encoder name
            parts = pattern_id.split("_")
            if len(parts) >= 2:
                encoder = parts[0]
                level = parts[1]

                if encoder not in hierarchy["encoders"]:
                    hierarchy["encoders"][encoder] = {"bottom": [], "top": []}

                if level in hierarchy["encoders"][encoder]:
                    hierarchy["encoders"][encoder][level].append({
                        "id": pattern_id,
                        "name": name_info.get("name"),
                        "children": weights.get("children", []),
                    })

    return hierarchy


# NOTE: Routes with path parameters must come AFTER specific routes
# to avoid catching specific paths as pattern IDs

@router.get("/{pattern_id}")
def get_pattern(
    project_id: str,
    pattern_id: str,
    db: Session = Depends(get_db),
):
    """Get detailed information about a pattern."""
    storage = StorageService(project_id)

    pattern_names = storage.load_json(storage.pattern_names_path)
    if not pattern_names or pattern_id not in pattern_names:
        raise HTTPException(status_code=404, detail="Pattern not found")

    info = pattern_names[pattern_id]
    hierarchical_weights = storage.load_json(storage.hierarchical_weights_path) or {}
    population_stats = storage.load_json(storage.population_stats_path) or {}

    weights = hierarchical_weights.get(pattern_id, {})
    stats = population_stats.get(pattern_id, {})

    return {
        "id": pattern_id,
        "name": info.get("name"),
        "description": info.get("description"),
        "hierarchical_weights": weights,
        "statistics": stats,
    }


@router.get("/{pattern_id}/messages")
def get_pattern_messages(
    project_id: str,
    pattern_id: str,
    top_k: int = 50,
    db: Session = Depends(get_db),
):
    """
    Get example messages for a pattern.

    Use the new /messages endpoint for flexible queries:
    GET /projects/{project_id}/messages?pattern_key=enc1_bottom&pattern_idx=0&limit=50
    """
    storage = StorageService(project_id)

    pattern_key, pattern_idx = _parse_pattern_id(pattern_id)
    if pattern_key is None:
        raise HTTPException(status_code=400, detail=f"Invalid pattern_id format: {pattern_id}")

    index = storage.load_json(storage.message_scores_index_path)
    if not index:
        raise HTTPException(status_code=404, detail="Message scores not found - run batch scoring first")

    if pattern_key not in index["levels"]:
        raise HTTPException(status_code=404, detail=f"Pattern key {pattern_key} not found")

    n_dims = index["levels"][pattern_key]["n_dims"]
    if pattern_idx >= n_dims:
        raise HTTPException(status_code=400, detail=f"Pattern index {pattern_idx} out of range")

    from ..services.r2_service import download_h5_from_r2, download_pickle_from_r2

    h5_path = storage.get_cached_h5("message_scores")
    if not h5_path.exists():
        download_h5_from_r2(project_id, "message_scores", h5_path)

    try:
        message_database = download_pickle_from_r2(project_id, "message_database")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Message database not found")

    messages_list = message_database.get("messages", message_database)
    if isinstance(messages_list, dict):
        messages_list = messages_list.get("messages", [])

    from src.pattern_identification.message_scorer import MessageScorer
    examples = MessageScorer.get_top_messages_for_pattern(
        h5_path,
        pattern_key,
        pattern_idx,
        messages_list,
        limit=top_k,
    )

    return {
        "pattern_id": pattern_id,
        "messages": examples,
        "total": len(examples),
    }


def _parse_pattern_id(pattern_id: str) -> tuple[str | None, int | None]:
    """
    Parse pattern_id into pattern_key and pattern_idx.

    Examples:
    - "enc1_bottom_bottom_0" -> ("enc1_bottom", 0)
    - "unified_unified_3" -> ("unified", 3)
    """
    parts = pattern_id.rsplit("_", 1)
    if len(parts) != 2:
        return None, None

    try:
        pattern_idx = int(parts[1])
    except ValueError:
        return None, None

    remaining = parts[0]
    remaining_parts = remaining.rsplit("_", 1)
    if len(remaining_parts) != 2:
        if remaining in ("unified",):
            return remaining, pattern_idx
        return None, None

    pattern_key = remaining_parts[0]
    return pattern_key, pattern_idx
