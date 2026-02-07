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
    # Pattern names structure: {"enc1_bottom": {"dim_0": {...}, "dim_1": {...}}, "unified": {...}}
    for level_key, level_patterns in pattern_names.items():
        if not isinstance(level_patterns, dict):
            continue

        # Filter by level if specified
        if level and level not in level_key:
            continue

        for dim_key, info in level_patterns.items():
            if not isinstance(info, dict):
                continue

            # Extract dimension index from dim_key (e.g., "dim_0" -> 0)
            dim_idx = 0
            if dim_key.startswith("dim_"):
                try:
                    dim_idx = int(dim_key.split("_")[1])
                except (IndexError, ValueError):
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


@router.get("/message-distribution")
def get_message_distribution(
    project_id: str,
    pattern_key: str,
    pattern_idx: int,
    engineer_ids: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get message distribution for a pattern."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    # Load message examples
    message_examples = storage.load_json(storage.message_examples_path)
    if not message_examples:
        raise HTTPException(status_code=404, detail="Message examples not found")

    # Construct pattern ID from key and index
    pattern_id = f"{pattern_key}_{pattern_idx}"
    if pattern_id not in message_examples:
        raise HTTPException(status_code=404, detail=f"Pattern {pattern_id} not found")

    examples = message_examples[pattern_id].get("examples", [])

    # Filter by engineer IDs if provided
    if engineer_ids:
        ids = set(engineer_ids.split(","))
        examples = [e for e in examples if e.get("engineer_id") in ids]

    # Calculate statistics
    if examples:
        scores = [e.get("score", 0) for e in examples]
        stats = {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "count": len(scores),
        }
    else:
        stats = {"min": 0, "max": 0, "mean": 0, "count": 0}

    return {
        "messages": examples,
        "stats": stats,
    }


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
    message_examples = storage.load_json(storage.message_examples_path) or {}
    hierarchical_weights = storage.load_json(storage.hierarchical_weights_path) or {}
    population_stats = storage.load_json(storage.population_stats_path) or {}

    examples = message_examples.get(pattern_id, {}).get("examples", [])
    weights = hierarchical_weights.get(pattern_id, {})
    stats = population_stats.get(pattern_id, {})

    return {
        "id": pattern_id,
        "name": info.get("name"),
        "description": info.get("description"),
        "examples": examples,
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
    """Get example messages for a pattern."""
    storage = StorageService(project_id)

    message_examples = storage.load_json(storage.message_examples_path)
    if not message_examples or pattern_id not in message_examples:
        raise HTTPException(status_code=404, detail="Pattern not found")

    examples = message_examples[pattern_id].get("examples", [])

    return {
        "pattern_id": pattern_id,
        "messages": examples[:top_k],
        "total": len(examples),
    }
