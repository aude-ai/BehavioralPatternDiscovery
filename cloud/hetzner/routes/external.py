"""
External Integration API — Single entry point for third-party consumers.

This file is the ONLY interface external projects need to interact with
BehavioralPatternDiscovery. The frontend uses internal routes directly;
external consumers use these endpoints exclusively.

All endpoints are under:  /external/projects/{project_id}/...

=============================================================================
QUICK START — Typical usage flow for scoring an engineer:
=============================================================================

  1. TRIGGER SCORING (async)
     POST /external/projects/{project_id}/score-engineer
     Body: {"engineer_id": "eng_001"}
     Returns: {"id": "job_abc", "status": "PENDING", ...}

  2. POLL UNTIL COMPLETE
     GET /external/projects/{project_id}/jobs/{job_id}
     Returns: {"id": "job_abc", "status": "RUNNING", "progress": 0.5, ...}
     Keep polling every 2-3 seconds until status is "COMPLETED" or "FAILED".

  3. TRIGGER REPORT GENERATION (async, optional but recommended)
     POST /external/projects/{project_id}/generate-report/eng_001
     Returns: {"id": "job_def", "status": "PENDING", ...}
     Poll again until COMPLETED.

  4. GET EVERYTHING IN ONE CALL
     GET /external/projects/{project_id}/evaluation/eng_001
     Returns: scores, report, pattern context, and top messages — all at once.

=============================================================================
POLLING PATTERN — How async jobs work:
=============================================================================

  Scoring and report generation run on GPU/CPU workers asynchronously.
  The POST endpoints return a Job object immediately. Poll the job
  status endpoint until completion:

    job = POST /external/projects/{project_id}/score-engineer
    while True:
        status = GET /external/projects/{project_id}/jobs/{job.id}
        if status.status == "COMPLETED":
            break
        if status.status == "FAILED":
            raise Error(status.error)
        sleep(2)

  Job statuses: PENDING → RUNNING → COMPLETED | FAILED | CANCELLED

=============================================================================
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Job, JobType
from ..services import ProjectService, StorageService
from ..services.r2_service import download_h5_from_r2, download_pickle_from_r2
from ..tasks import gpu_tasks, cpu_tasks
from .pipeline import get_pipeline_config
from src.pattern_identification.message_scorer import MessageScorer

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_external_config = None


def _get_external_config() -> dict:
    """Load external.yaml config (top_k_examples, default_top_k, max_top_k)."""
    global _external_config
    if _external_config is None:
        from src.core.config import load_config
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "external.yaml"
        _external_config = load_config(config_path)
    return _external_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_project_or_404(project_id: str, db: Session) -> object:
    """Validate project exists, raise 404 if not."""
    service = ProjectService(db)
    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _get_h5_and_messages(storage: StorageService, project_id: str) -> tuple[Path, list[dict]]:
    """Load message_scores.h5 and message database, caching locally."""
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


# ===========================================================================
# 1. HEALTH CHECK
# ===========================================================================

@router.get("/health")
def health_check():
    """Simple health check. Returns {"status": "healthy"} if the API is up."""
    return {"status": "healthy"}


# ===========================================================================
# 2. PATTERNS — List all discovered behavioral patterns
# ===========================================================================

@router.get("/projects/{project_id}/patterns")
def list_patterns(
    project_id: str,
    level: Optional[str] = None,
    encoder: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    List all discovered behavioral patterns with names, descriptions,
    population statistics, and example messages.

    Optional filters:
      - level: filter by hierarchy level (e.g., "bottom", "mid", "top")
      - encoder: filter by encoder prefix (e.g., "enc1", "enc2")

    Returns:
      {
        "patterns": [
          {
            "id": "enc1_bottom_0",
            "name": "Code Review Engagement",
            "description": "...",
            "level": "enc1_bottom",
            "index": 0,
            "examples": [...],
            "statistics": {...}
          },
          ...
        ],
        "count": 42
      }
    """
    _get_project_or_404(project_id, db)
    storage = StorageService(project_id)

    pattern_names = storage.load_json(storage.pattern_names_path)
    if not pattern_names:
        raise HTTPException(status_code=404, detail="Patterns not found — run naming pipeline first")

    hierarchical_weights = storage.load_json(storage.hierarchical_weights_path) or {}
    population_stats = storage.load_json(storage.population_stats_path) or {}

    config = _get_external_config()
    top_k_examples = config["pattern_export"]["top_k_examples"]

    h5_path, messages_list = _get_h5_and_messages(storage, project_id)

    patterns = []
    for pattern_id, info in pattern_names.items():
        parts = pattern_id.rsplit("_", 1)
        if len(parts) != 2:
            continue

        pattern_level = parts[0]
        pattern_idx = int(parts[1])

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

        patterns.append({
            "id": pattern_id,
            "name": info.get("name"),
            "description": info.get("description"),
            "level": pattern_level,
            "index": pattern_idx,
            "examples": examples,
            "statistics": population_stats.get(pattern_id, {}),
        })

    return {"patterns": patterns, "count": len(patterns)}


# ===========================================================================
# 3. POPULATION STATS — Overall averages across all engineers
# ===========================================================================

@router.get("/projects/{project_id}/population-stats")
def get_population_stats(
    project_id: str,
    db: Session = Depends(get_db),
):
    """
    Get population-level statistics from batch scoring.

    These are the baseline averages/distributions that individual
    engineer scores are compared against (percentiles, quartiles, etc.).

    Returns the full population_stats.json structure.
    """
    _get_project_or_404(project_id, db)
    storage = StorageService(project_id)

    stats = storage.load_json(storage.population_stats_path)
    if not stats:
        raise HTTPException(status_code=404, detail="Population stats not found — run batch scoring first")

    return stats


# ===========================================================================
# 4. INDIVIDUAL SCORES — Get pre-computed scores for an engineer
# ===========================================================================

@router.get("/projects/{project_id}/scores/{engineer_id}")
def get_individual_scores(
    project_id: str,
    engineer_id: str,
    db: Session = Depends(get_db),
):
    """
    Get individual engineer scores (must have been computed already via
    POST /score-engineer).

    Returns:
      {
        "engineer_id": "eng_001",
        "n_messages": 150,
        "scores": {
          "enc1_bottom": {
            "raw_mean": [...],
            "posterior_mean": [...],
            "percentiles": [...],
            "shrinkage": 0.85
          },
          ...
        }
      }

    Percentiles are 0-100, showing where the engineer ranks relative to
    the population for each pattern dimension.
    """
    _get_project_or_404(project_id, db)
    storage = StorageService(project_id)

    scores_path = storage.base_path / f"scoring/individual/{engineer_id}.json"
    scores = storage.load_json(scores_path)

    if not scores:
        raise HTTPException(
            status_code=404,
            detail=f"Scores not found for '{engineer_id}' — trigger scoring first via POST /score-engineer",
        )

    return scores


# ===========================================================================
# 5. REPORT — Get pre-generated markdown report for an engineer
# ===========================================================================

@router.get("/projects/{project_id}/report/{engineer_id}")
def get_report(
    project_id: str,
    engineer_id: str,
    db: Session = Depends(get_db),
):
    """
    Get the LLM-generated evaluation report for an engineer (must have
    been generated already via POST /generate-report/{engineer_id}).

    Returns:
      {
        "engineer_id": "eng_001",
        "overall_summary": "## Performance Summary\\n\\n..."
      }

    The overall_summary is markdown-formatted text covering strengths,
    weaknesses, and actionable recommendations.
    """
    _get_project_or_404(project_id, db)
    storage = StorageService(project_id)

    report_path = storage.base_path / f"scoring/reports/{engineer_id}.json"
    report = storage.load_json(report_path)

    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Report not found for '{engineer_id}' — trigger generation first via POST /generate-report/{engineer_id}",
        )

    return report


# ===========================================================================
# 6. MESSAGES — Query messages by pattern and/or engineer
# ===========================================================================

@router.get("/projects/{project_id}/messages")
def query_messages(
    project_id: str,
    pattern_key: str = Query(..., description="Pattern level key, e.g. 'enc1_bottom', 'unified'"),
    pattern_idx: int = Query(..., description="Pattern dimension index, e.g. 0, 1, 2"),
    engineer_id: Optional[str] = Query(None, description="Filter to a specific engineer's messages"),
    top_k: Optional[int] = Query(None, ge=1, description="Number of results (default from config)"),
    db: Session = Depends(get_db),
):
    """
    Get messages that most strongly activate a specific pattern.

    Optionally filter to a single engineer's messages. Results are sorted
    by activation score (highest first).

    Returns:
      {
        "pattern_key": "enc1_bottom",
        "pattern_idx": 0,
        "messages": [
          {"text": "...", "score": 0.95, "source": "github", ...},
          ...
        ],
        "total": 20
      }
    """
    _get_project_or_404(project_id, db)
    storage = StorageService(project_id)

    config = _get_external_config()
    search_config = config["message_search"]
    if top_k is None:
        top_k = search_config["default_top_k"]
    top_k = min(top_k, search_config["max_top_k"])

    h5_path, messages_list = _get_h5_and_messages(storage, project_id)

    # Optionally filter to a specific engineer's message indices
    message_indices = None
    if engineer_id:
        index = storage.load_json(storage.message_scores_index_path)
        if index and engineer_id in index.get("engineers", {}):
            message_indices = index["engineers"][engineer_id]["message_indices"]
        else:
            return {"pattern_key": pattern_key, "pattern_idx": pattern_idx, "messages": [], "total": 0}

    examples = MessageScorer.get_top_messages_for_pattern(
        h5_path=h5_path,
        level_key=pattern_key,
        pattern_idx=pattern_idx,
        message_database=messages_list,
        limit=top_k,
        message_indices=message_indices,
    )

    return {
        "pattern_key": pattern_key,
        "pattern_idx": pattern_idx,
        "messages": examples,
        "total": len(examples),
    }


# ===========================================================================
# 7. EVALUATION — "Give me everything" for an engineer in one call
# ===========================================================================

@router.get("/projects/{project_id}/evaluation/{engineer_id}")
def get_full_evaluation(
    project_id: str,
    engineer_id: str,
    top_k_messages: int = Query(default=5, ge=1, le=50, description="Messages per notable pattern"),
    db: Session = Depends(get_db),
):
    """
    Get a complete evaluation of an engineer in a single response.

    Aggregates: scores, report, pattern names, population stats, and
    top messages for the engineer's most notable patterns (strengths
    and weaknesses).

    PREREQUISITE: Individual scoring must have been run first via
    POST /score-engineer. Report generation is optional — if a report
    exists it's included, otherwise report is null.

    Returns:
      {
        "engineer_id": "eng_001",
        "scores": { ... },
        "report": { "overall_summary": "..." } | null,
        "patterns": { "enc1_bottom_0": {"name": "...", "description": "..."}, ... },
        "population_stats": { ... },
        "top_messages": {
          "enc1_bottom:0": [{"text": "...", "score": 0.95}, ...],
          ...
        }
      }

    top_messages only includes patterns where the engineer scored at or
    above the 70th percentile (strengths) or at or below the 40th
    percentile (weaknesses).
    """
    _get_project_or_404(project_id, db)
    storage = StorageService(project_id)

    # --- Scores (required) ---
    scores_path = storage.base_path / f"scoring/individual/{engineer_id}.json"
    scores = storage.load_json(scores_path)
    if not scores:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Scores not found for '{engineer_id}'. "
                "Trigger scoring first: POST /external/projects/{project_id}/score-engineer "
                "with body {\"engineer_id\": \"" + engineer_id + "\"}"
            ),
        )

    # --- Report (optional — included if available) ---
    report_path = storage.base_path / f"scoring/reports/{engineer_id}.json"
    report = storage.load_json(report_path)

    # --- Pattern names ---
    pattern_names = storage.load_json(storage.pattern_names_path) or {}

    # --- Population stats ---
    population_stats = storage.load_json(storage.population_stats_path) or {}

    # --- Top messages for notable patterns ---
    # Identify patterns where this engineer is notably strong or weak
    notable_patterns = _find_notable_patterns(scores)

    top_messages = {}
    if notable_patterns:
        h5_path, messages_list = _get_h5_and_messages(storage, project_id)

        # Get this engineer's message indices for filtering
        message_indices = None
        index = storage.load_json(storage.message_scores_index_path)
        if index and engineer_id in index.get("engineers", {}):
            message_indices = index["engineers"][engineer_id]["message_indices"]

        for level_key, dim_idx in notable_patterns:
            examples = MessageScorer.get_top_messages_for_pattern(
                h5_path=h5_path,
                level_key=level_key,
                pattern_idx=dim_idx,
                message_database=messages_list,
                limit=top_k_messages,
                message_indices=message_indices,
            )
            top_messages[f"{level_key}:{dim_idx}"] = examples

    return {
        "engineer_id": engineer_id,
        "scores": scores,
        "report": report,
        "patterns": pattern_names,
        "population_stats": population_stats,
        "top_messages": top_messages,
    }


def _find_notable_patterns(
    scores: dict,
    strength_threshold: float = 70.0,
    weakness_threshold: float = 40.0,
) -> list[tuple[str, int]]:
    """
    Identify patterns where the engineer is notably strong or weak.

    Returns list of (level_key, dim_index) tuples for patterns at or
    above strength_threshold or at or below weakness_threshold percentile.
    """
    notable = []
    score_data = scores.get("scores", {})

    for level_key, level_scores in score_data.items():
        percentiles = level_scores.get("percentiles", [])
        for dim_idx, pct in enumerate(percentiles):
            if pct >= strength_threshold or pct <= weakness_threshold:
                notable.append((level_key, dim_idx))

    return notable


# ===========================================================================
# 8. SCORE ENGINEER — Trigger individual scoring (async)
# ===========================================================================

class ScoreEngineerRequest(BaseModel):
    """Request body for triggering individual scoring."""
    engineer_id: str


@router.post("/projects/{project_id}/score-engineer", response_model=Job)
def score_engineer(
    project_id: str,
    request: ScoreEngineerRequest,
    db: Session = Depends(get_db),
):
    """
    Trigger individual scoring for an engineer. This runs asynchronously
    on a GPU worker.

    Returns a Job object with an id field. Poll the job status at:
      GET /external/projects/{project_id}/jobs/{job.id}

    Once status is "COMPLETED", retrieve scores at:
      GET /external/projects/{project_id}/scores/{engineer_id}

    Prerequisites:
      - Processing pipeline must have been run (model trained, batch scored)
    """
    import pandas as pd

    _get_project_or_404(project_id, db)
    service = ProjectService(db)
    storage = StorageService(project_id)

    # Verify model exists
    from ..services.r2_service import r2_file_exists
    if not r2_file_exists(project_id, "checkpoint"):
        raise HTTPException(status_code=400, detail="Model not trained yet — run the processing pipeline first")

    # Verify population stats exist
    population_stats = storage.load_json(storage.population_stats_path)
    if not population_stats:
        raise HTTPException(status_code=400, detail="Population stats not found — run batch scoring first")

    # Load engineer's messages
    if not storage.file_exists(storage.activities_path):
        raise HTTPException(status_code=400, detail="No activities data found")

    df = pd.read_csv(storage.activities_path)
    engineer_df = df[df["engineer_id"] == request.engineer_id]

    if len(engineer_df) == 0:
        raise HTTPException(status_code=404, detail=f"Engineer '{request.engineer_id}' not found in activities data")

    messages = engineer_df.to_dict(orient="records")

    job = service.create_job(project_id, JobType.SCORE_INDIVIDUAL)
    config = get_pipeline_config()

    gpu_tasks.trigger_individual_score.delay(
        project_id=project_id,
        job_id=job.id,
        engineer_id=request.engineer_id,
        messages=messages,
        population_stats=population_stats,
        config=config,
    )

    return job


# ===========================================================================
# 9. GENERATE REPORT — Trigger LLM report generation (async)
# ===========================================================================

@router.post("/projects/{project_id}/generate-report/{engineer_id}", response_model=Job)
def generate_report(
    project_id: str,
    engineer_id: str,
    db: Session = Depends(get_db),
):
    """
    Trigger LLM-based report generation for an engineer. This runs
    asynchronously on a CPU worker.

    Returns a Job object with an id field. Poll the job status at:
      GET /external/projects/{project_id}/jobs/{job.id}

    Once status is "COMPLETED", retrieve the report at:
      GET /external/projects/{project_id}/report/{engineer_id}

    Prerequisites:
      - Individual scoring must have been completed first
    """
    _get_project_or_404(project_id, db)
    service = ProjectService(db)
    storage = StorageService(project_id)

    scores_path = storage.base_path / f"scoring/individual/{engineer_id}.json"
    if not scores_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Score engineer first via POST /external/projects/{project_id}/score-engineer",
        )

    job = service.create_job(project_id, JobType.GENERATE_REPORT)
    config = get_pipeline_config()

    cpu_tasks.generate_report.delay(project_id, job.id, engineer_id, config)

    return job


# ===========================================================================
# 10. JOB STATUS — Poll async job progress
# ===========================================================================

@router.get("/projects/{project_id}/jobs/{job_id}", response_model=Job)
def get_job_status(
    project_id: str,
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Check the status of an async job (scoring or report generation).

    Job statuses:
      - PENDING:   Queued, waiting for a worker
      - RUNNING:   In progress (check progress field for 0.0-1.0)
      - COMPLETED: Done — retrieve results from the appropriate GET endpoint
      - FAILED:    Error occurred — check the error field for details
      - CANCELLED: Job was cancelled

    Poll this endpoint every 2-3 seconds until status is COMPLETED or FAILED.
    """
    service = ProjectService(db)
    job = service.get_job(job_id)
    if not job or job.project_id != project_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
