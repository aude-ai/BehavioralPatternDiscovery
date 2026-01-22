"""GPU tasks that trigger Modal functions."""
import logging

import modal

from ..celery_app import celery_app
from ..config import get_settings
from ..database import get_db, JobModel
from ..models import JobStatus

logger = logging.getLogger(__name__)
settings = get_settings()


def get_modal_function(app_name: str, function_name: str):
    """Get a Modal function handle."""
    return modal.Function.lookup(app_name, function_name)


@celery_app.task(bind=True, max_retries=2)
def trigger_embedding(self, project_id: str, job_id: str, texts: list[str], config: dict):
    """Trigger embedding on Modal."""
    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.RUNNING,
                "progress_message": "Starting embedding on Modal...",
            })
            db.commit()

            # Call Modal function
            embed_fn = get_modal_function("bpd-embedding", "embed_all_texts")
            call = embed_fn.spawn(
                project_id=project_id,
                job_id=job_id,
                texts=texts,
                task=config.get("embedding", {}).get("task", "retrieval.passage"),
            )

            # Store Modal call ID
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "modal_call_id": call.object_id,
            })
            db.commit()

            logger.info(f"Triggered embedding for project {project_id}, Modal call: {call.object_id}")

            return {"modal_call_id": call.object_id}

        except Exception as e:
            logger.exception(f"Failed to trigger embedding: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


@celery_app.task(bind=True, max_retries=1)
def trigger_training(self, project_id: str, job_id: str, config: dict):
    """Trigger VAE training on Modal."""
    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.RUNNING,
                "progress_message": "Starting training on Modal...",
            })
            db.commit()

            train_fn = get_modal_function("bpd-vae", "train_vae")
            call = train_fn.spawn(
                project_id=project_id,
                job_id=job_id,
                config=config,
            )

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "modal_call_id": call.object_id,
            })
            db.commit()

            logger.info(f"Triggered training for project {project_id}, Modal call: {call.object_id}")

            return {"modal_call_id": call.object_id}

        except Exception as e:
            logger.exception(f"Failed to trigger training: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


@celery_app.task(bind=True, max_retries=2)
def trigger_batch_score(self, project_id: str, job_id: str, config: dict):
    """Trigger batch scoring on Modal."""
    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.RUNNING,
                "progress_message": "Starting batch scoring on Modal...",
            })
            db.commit()

            score_fn = get_modal_function("bpd-vae", "batch_score")
            call = score_fn.spawn(
                project_id=project_id,
                job_id=job_id,
                config=config,
            )

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "modal_call_id": call.object_id,
            })
            db.commit()

            return {"modal_call_id": call.object_id}

        except Exception as e:
            logger.exception(f"Failed to trigger batch scoring: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


@celery_app.task(bind=True, max_retries=2)
def trigger_shap_analysis(self, project_id: str, job_id: str, config: dict):
    """Trigger SHAP analysis on Modal."""
    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.RUNNING,
                "progress_message": "Starting SHAP analysis on Modal...",
            })
            db.commit()

            shap_fn = get_modal_function("bpd-vae", "shap_analyze")
            call = shap_fn.spawn(
                project_id=project_id,
                job_id=job_id,
                config=config,
            )

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "modal_call_id": call.object_id,
            })
            db.commit()

            return {"modal_call_id": call.object_id}

        except Exception as e:
            logger.exception(f"Failed to trigger SHAP analysis: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


@celery_app.task(bind=True, max_retries=3)
def trigger_individual_score(
    self,
    project_id: str,
    job_id: str,
    engineer_id: str,
    messages: list[dict],
    population_stats: dict,
):
    """Trigger individual scoring on Modal."""
    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.RUNNING,
            })
            db.commit()

            # Use the warm scoring service for lower latency
            scoring_service = modal.Cls.lookup("bpd-vae", "ScoringService")
            result = scoring_service().score.remote(
                project_id=project_id,
                engineer_id=engineer_id,
                messages=messages,
                population_stats=population_stats,
            )

            # Save result
            from ..services import StorageService

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
