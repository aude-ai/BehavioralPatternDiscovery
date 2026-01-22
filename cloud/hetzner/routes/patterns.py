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
        raise HTTPException(status_code=404, detail="Patterns not found - run pattern naming first")

    patterns = []
    for pattern_id, info in pattern_names.items():
        # Parse pattern ID
        parts = pattern_id.rsplit("_", 1)
        if len(parts) == 2:
            pattern_level = parts[0]
            pattern_idx = int(parts[1])
        else:
            continue

        # Filter by level if specified
        if level and level not in pattern_level:
            continue

        patterns.append({
            "id": pattern_id,
            "name": info.get("name"),
            "description": info.get("description"),
            "level": pattern_level,
            "index": pattern_idx,
        })

    return {"patterns": patterns, "count": len(patterns)}


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
