"""Scoring routes."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Job, JobType
from ..services import ProjectService, StorageService
from ..tasks import gpu_tasks, cpu_tasks
from .pipeline import get_pipeline_config

router = APIRouter()


class ScoreRequest(BaseModel):
    """Request body for individual scoring."""
    engineer_id: str


@router.post("/individual", response_model=Job)
def score_individual(
    project_id: str,
    request: ScoreRequest,
    db: Session = Depends(get_db),
):
    """Score an individual engineer."""
    import pandas as pd

    service = ProjectService(db)
    storage = StorageService(project_id)

    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check R2 for checkpoint (not local storage)
    from ..services.r2_service import r2_file_exists
    if not r2_file_exists(project_id, "checkpoint"):
        raise HTTPException(status_code=400, detail="Model not trained yet. Run processing pipeline first.")

    # Load population stats
    population_stats = storage.load_json(storage.population_stats_path)
    if not population_stats:
        raise HTTPException(status_code=400, detail="Run batch scoring first")

    # Load messages for this engineer from activities.csv
    if not storage.file_exists(storage.activities_path):
        raise HTTPException(status_code=400, detail="No activities data found")

    df = pd.read_csv(storage.activities_path)
    engineer_df = df[df["engineer_id"] == request.engineer_id]

    if len(engineer_df) == 0:
        raise HTTPException(status_code=404, detail=f"Engineer '{request.engineer_id}' not found")

    messages = engineer_df.to_dict(orient="records")

    job = service.create_job(project_id, JobType.SCORE_INDIVIDUAL)

    # Get pipeline config for the scorer
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


@router.get("/individual/{engineer_id}")
def get_individual_scores(
    project_id: str,
    engineer_id: str,
    db: Session = Depends(get_db),
):
    """Get scores for an individual engineer."""
    storage = StorageService(project_id)

    scores_path = storage.base_path / f"scoring/individual/{engineer_id}.json"
    scores = storage.load_json(scores_path)

    if not scores:
        raise HTTPException(status_code=404, detail="Scores not found")

    return scores


@router.post("/report/{engineer_id}", response_model=Job)
def generate_report(
    project_id: str,
    engineer_id: str,
    config: dict = None,
    db: Session = Depends(get_db),
):
    """Generate a report for an engineer."""
    service = ProjectService(db)
    storage = StorageService(project_id)

    project = service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    scores_path = storage.base_path / f"scoring/individual/{engineer_id}.json"
    if not scores_path.exists():
        raise HTTPException(status_code=400, detail="Score individual first")

    job = service.create_job(project_id, JobType.GENERATE_REPORT)
    merged_config = config or {}

    cpu_tasks.generate_report.delay(project_id, job.id, engineer_id, merged_config)

    return job


@router.get("/report/{engineer_id}")
def get_report(
    project_id: str,
    engineer_id: str,
    db: Session = Depends(get_db),
):
    """Get generated report for an engineer."""
    storage = StorageService(project_id)

    report_path = storage.base_path / f"scoring/reports/{engineer_id}.json"
    report = storage.load_json(report_path)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return report


@router.get("/population-stats")
def get_population_stats(
    project_id: str,
    db: Session = Depends(get_db),
):
    """Get population statistics from batch scoring."""
    storage = StorageService(project_id)

    stats = storage.load_json(storage.population_stats_path)
    if not stats:
        raise HTTPException(status_code=404, detail="Population stats not found")

    return stats
