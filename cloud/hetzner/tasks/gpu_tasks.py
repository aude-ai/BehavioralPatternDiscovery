"""GPU tasks that trigger Modal functions."""
import logging

import modal

from ..celery_app import celery_app
from ..config import get_settings
from ..database import get_db_context, JobModel
from ..models import JobStatus
from ..services import StorageService

logger = logging.getLogger(__name__)
settings = get_settings()


def get_modal_function(app_name: str, function_name: str):
    """Get a Modal function handle."""
    return modal.Function.lookup(app_name, function_name)


# =============================================================================
# SEGMENT B: Processing Pipeline (Modal)
# =============================================================================


@celery_app.task(bind=True, max_retries=1)
def trigger_processing_pipeline(
    self,
    project_id: str,
    job_id: str,
    starting_step: str,
    config: dict,
    force: bool = False,
):
    """
    Trigger the unified processing pipeline on Modal (Segment B).

    Runs steps B.1 through B.8 sequentially, starting from the specified step.
    All large files are stored in R2, only small JSONs come back to Hetzner.

    Args:
        force: If True, re-run steps even if outputs exist. If False (default),
               skip steps whose outputs already exist in R2.
    """
    with get_db_context() as db:
        try:
            force_msg = " (force mode)" if force else ""
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.RUNNING,
                "progress_message": f"Starting processing pipeline from {starting_step}{force_msg}...",
            })
            db.commit()

            # Call the unified Modal function
            pipeline_fn = get_modal_function("bpd-processing", "run_processing_pipeline")
            call = pipeline_fn.spawn(
                project_id=project_id,
                job_id=job_id,
                starting_step=starting_step,
                config=config,
                force=force,
            )

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "modal_call_id": call.object_id,
            })
            db.commit()

            logger.info(
                f"Triggered processing pipeline for project {project_id}, "
                f"starting from {starting_step}, Modal call: {call.object_id}"
            )

            return {"modal_call_id": call.object_id}

        except Exception as e:
            logger.exception(f"Failed to trigger processing pipeline: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


# =============================================================================
# SEGMENT D.1: Individual Scoring (Modal)
# =============================================================================


@celery_app.task(bind=True, max_retries=3)
def trigger_individual_score(
    self,
    project_id: str,
    job_id: str,
    engineer_id: str,
    messages: list[dict],
    population_stats: dict,
    config: dict,
):
    """
    Trigger individual scoring on Modal (Segment D.1).

    Uses the warm ScoringService for low latency.
    """
    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.RUNNING,
            })
            db.commit()

            # Use the warm scoring service
            scoring_service = modal.Cls.lookup("bpd-processing", "ScoringService")
            result = scoring_service().score.remote(
                project_id=project_id,
                engineer_id=engineer_id,
                messages=messages,
                population_stats=population_stats,
                config=config,
            )

            # Save result to Hetzner
            storage = StorageService(project_id)
            scores_path = storage.base_path / f"scoring/individual/{engineer_id}.json"
            storage.save_json(scores_path, result)

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
                "result": {"engineer_id": engineer_id},
            })
            db.commit()

            return result

        except Exception as e:
            logger.exception(f"Failed to score individual: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise
