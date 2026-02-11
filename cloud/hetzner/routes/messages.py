"""Message query routes."""
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..services import ProjectService, StorageService
from ..services.r2_service import download_h5_from_r2, download_pickle_from_r2

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("")
def query_messages(
    project_id: str,
    engineer_id: Optional[str] = None,
    pattern_key: Optional[str] = None,
    pattern_idx: Optional[int] = None,
    min_percentile: Optional[int] = None,
    max_percentile: Optional[int] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    limit: int = Query(default=50, le=500),
    offset: int = 0,
    sort_by: str = "score",
    sort_order: str = "desc",
    db: Session = Depends(get_db),
):
    """
    Query messages with flexible filtering.

    Examples:
    - All messages from engineer: ?engineer_id=eng_001
    - Top messages for pattern: ?pattern_key=enc1_bottom&pattern_idx=0&limit=20
    - Engineer's high-scoring messages: ?engineer_id=eng_001&pattern_key=unified&pattern_idx=0&min_percentile=75
    """
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)

    index = storage.load_json(storage.message_scores_index_path)
    if not index:
        raise HTTPException(
            status_code=404,
            detail="Message scores not found - run batch scoring first"
        )

    message_database = _load_message_database(project_id)
    if not message_database:
        raise HTTPException(status_code=404, detail="Message database not found")

    messages_list = message_database.get("messages", message_database)
    if isinstance(messages_list, dict):
        messages_list = messages_list.get("messages", [])

    if engineer_id:
        if engineer_id not in index["engineers"]:
            raise HTTPException(status_code=404, detail=f"Engineer {engineer_id} not found")
        candidate_indices = index["engineers"][engineer_id]["message_indices"]
    else:
        candidate_indices = list(range(index["n_messages"]))

    if pattern_key and pattern_idx is not None:
        if pattern_key not in index["levels"]:
            raise HTTPException(status_code=404, detail=f"Pattern key {pattern_key} not found")

        n_dims = index["levels"][pattern_key]["n_dims"]
        if pattern_idx >= n_dims:
            raise HTTPException(
                status_code=400,
                detail=f"Pattern index {pattern_idx} out of range (max: {n_dims - 1})"
            )

        h5_path = storage.get_cached_h5("message_scores")
        if not h5_path.exists():
            logger.info(f"Downloading message_scores.h5 from R2 for project {project_id}")
            download_h5_from_r2(project_id, "message_scores", h5_path)

        from src.pattern_identification.message_scorer import MessageScorer
        scores, eng_ids = MessageScorer.load_message_scores(
            h5_path, pattern_key, candidate_indices
        )

        dim_scores = scores[:, pattern_idx]

        thresholds = index["percentile_thresholds"].get(pattern_key, {})

        mask = np.ones(len(candidate_indices), dtype=bool)

        if min_score is not None:
            mask &= dim_scores >= min_score
        if max_score is not None:
            mask &= dim_scores <= max_score

        if min_percentile is not None:
            threshold_key = f"p{min_percentile}"
            if threshold_key in thresholds:
                threshold_val = thresholds[threshold_key][pattern_idx]
                mask &= dim_scores >= threshold_val
            else:
                threshold_val = np.percentile(dim_scores, min_percentile)
                mask &= dim_scores >= threshold_val

        if max_percentile is not None:
            threshold_key = f"p{max_percentile}"
            if threshold_key in thresholds:
                threshold_val = thresholds[threshold_key][pattern_idx]
                mask &= dim_scores <= threshold_val
            else:
                threshold_val = np.percentile(dim_scores, max_percentile)
                mask &= dim_scores <= threshold_val

        filtered_indices = np.array(candidate_indices)[mask]
        filtered_scores = dim_scores[mask]
        filtered_eng_ids = np.array(eng_ids)[mask]

        if sort_order == "desc":
            sort_idx = np.argsort(-filtered_scores)
        else:
            sort_idx = np.argsort(filtered_scores)

        filtered_indices = filtered_indices[sort_idx]
        filtered_scores = filtered_scores[sort_idx]
        filtered_eng_ids = filtered_eng_ids[sort_idx]

        total = len(filtered_indices)
        filtered_indices = filtered_indices[offset:offset + limit]
        filtered_scores = filtered_scores[offset:offset + limit]
        filtered_eng_ids = filtered_eng_ids[offset:offset + limit]

        messages = []
        for idx, score, eng_id in zip(filtered_indices, filtered_scores, filtered_eng_ids):
            msg = messages_list[int(idx)]
            messages.append({
                "message_idx": int(idx),
                "engineer_id": str(eng_id),
                "text": msg.get("text", ""),
                "score": float(score),
                "source": msg.get("source"),
                "activity_type": msg.get("activity_type"),
                "timestamp": msg.get("timestamp"),
            })

        return {
            "messages": messages,
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": {
                "engineer_id": engineer_id,
                "pattern_key": pattern_key,
                "pattern_idx": pattern_idx,
                "min_percentile": min_percentile,
                "max_percentile": max_percentile,
            },
        }

    else:
        if not engineer_id:
            raise HTTPException(
                status_code=400,
                detail="Must specify either engineer_id or pattern_key+pattern_idx"
            )

        total = len(candidate_indices)
        paginated_indices = candidate_indices[offset:offset + limit]

        messages = []
        for idx in paginated_indices:
            msg = messages_list[idx]
            messages.append({
                "message_idx": idx,
                "engineer_id": msg.get("engineer_id"),
                "text": msg.get("text", ""),
                "source": msg.get("source"),
                "activity_type": msg.get("activity_type"),
                "timestamp": msg.get("timestamp"),
            })

        return {
            "messages": messages,
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": {"engineer_id": engineer_id},
        }


@router.get("/stats")
def get_message_stats(
    project_id: str,
    engineer_id: Optional[str] = None,
    pattern_key: Optional[str] = None,
    pattern_idx: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    Get statistics for messages matching query.

    Returns count, score distribution, percentile breakdown.
    """
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    storage = StorageService(project_id)
    index = storage.load_json(storage.message_scores_index_path)
    if not index:
        raise HTTPException(status_code=404, detail="Message scores not found")

    stats = {
        "total_messages": index["n_messages"],
        "total_engineers": len(index["engineers"]),
    }

    if engineer_id:
        if engineer_id not in index["engineers"]:
            raise HTTPException(status_code=404, detail=f"Engineer {engineer_id} not found")
        stats["engineer_messages"] = index["engineers"][engineer_id]["n_messages"]

    if pattern_key:
        if pattern_key not in index["levels"]:
            raise HTTPException(status_code=404, detail=f"Pattern key {pattern_key} not found")
        stats["pattern_dims"] = index["levels"][pattern_key]["n_dims"]
        stats["percentile_thresholds"] = index["percentile_thresholds"].get(pattern_key, {})

    return stats


def _load_message_database(project_id: str):
    """Load message database from R2 with caching."""
    try:
        return download_pickle_from_r2(project_id, "message_database")
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Failed to load message database: {e}")
        return None
