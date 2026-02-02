"""GPU tasks that trigger Modal functions."""
import logging

import modal
import numpy as np
import pandas as pd

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


def prepare_training_data(project_id: str, config: dict) -> dict:
    """
    Prepare training data by combining embeddings with aux features
    and creating the message database.

    Called automatically before training - not a separate pipeline step.
    """
    storage = StorageService(project_id)

    # Load activities
    activities_df = pd.read_csv(storage.activities_path)

    # Load embeddings (compressed)
    embeddings = storage.load_numpy_compressed(storage.train_features_path)

    # Load aux features (uncompressed - small file)
    aux_features = storage.load_numpy(storage.train_aux_vars_path)

    # Combine embeddings + aux features if aux features exist
    if aux_features is not None and config.get("include_aux_features", True):
        train_input = np.concatenate([embeddings, aux_features], axis=1)
    else:
        train_input = embeddings

    # Create message database
    messages = []
    for idx, row in activities_df.iterrows():
        messages.append({
            "index": idx,
            "text": row.get("text", ""),
            "engineer_id": row.get("engineer_id", ""),
            "source": row.get("source", ""),
            "timestamp": row.get("timestamp", ""),
        })

    # Extract encoder metadata from config for reproducibility
    encoder_config = config.get("processing", {}).get("text_encoder", {})
    encoder_type = encoder_config.get("type", "unknown")
    encoder_settings = encoder_config.get(encoder_type, {})
    model_name = encoder_settings.get("model_name", "unknown")

    message_db = {
        "messages": messages,
        "metadata": {
            "num_messages": len(messages),
            "embedding_dim": embeddings.shape[1],
            "aux_dim": aux_features.shape[1] if aux_features is not None else 0,
            "total_dim": train_input.shape[1],
            # Embedder metadata for reproducibility
            "embedder": {
                "type": encoder_type,
                "model_name": model_name,
                "config": encoder_settings,
            },
        },
    }

    # Save compressed files
    storage.save_numpy_compressed(storage.train_input_path, train_input)
    storage.save_pickle_compressed(storage.message_database_path, message_db)

    return {
        "train_input_shape": list(train_input.shape),
        "num_messages": len(messages),
    }


@celery_app.task(bind=True, max_retries=2)
def trigger_embedding(self, project_id: str, job_id: str, texts: list[str], config: dict):
    """Trigger embedding on Modal with full config for encoder selection."""
    with get_db_context() as db:
        try:
            # Validate config has required sections
            if "processing" not in config or "text_encoder" not in config.get("processing", {}):
                raise ValueError("Config must contain processing.text_encoder section")

            encoder_type = config["processing"]["text_encoder"]["type"]

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.RUNNING,
                "progress_message": f"Starting {encoder_type} embedding on Modal...",
            })
            db.commit()

            # Call Modal function with full config
            embed_fn = get_modal_function("bpd-embedding", "embed_all_texts")
            call = embed_fn.spawn(
                project_id=project_id,
                job_id=job_id,
                texts=texts,
                config=config,
            )

            # Store Modal call ID
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "modal_call_id": call.object_id,
            })
            db.commit()

            logger.info(
                f"Triggered {encoder_type} embedding for project {project_id}, "
                f"Modal call: {call.object_id}"
            )

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
    """
    Trigger VAE training on Modal.

    Automatically prepares training data (combines embeddings + aux features,
    creates message_database) before spawning the Modal training function.
    """
    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.RUNNING,
                "progress_message": "Preparing training data...",
            })
            db.commit()

            # Prepare training data (creates train_input.npy.zst and message_database.pkl.zst)
            storage = StorageService(project_id)
            if not storage.train_input_path.exists():
                prep_result = prepare_training_data(project_id, config)
                logger.info(f"Prepared training data: {prep_result}")

            db.query(JobModel).filter(JobModel.id == job_id).update({
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
    with get_db_context() as db:
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
    with get_db_context() as db:
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
    with get_db_context() as db:
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
