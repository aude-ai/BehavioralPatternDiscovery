"""CPU-bound Celery tasks that run on Hetzner."""
import logging
import sys
from datetime import datetime
from pathlib import Path

from ..celery_app import celery_app
from ..config import get_settings
from ..database import get_db_context, JobModel, ProjectModel
from ..models import JobStatus, ProjectStatus
from ..services import StorageService

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)
settings = get_settings()


# =============================================================================
# SEGMENT A: Data Collection (Hetzner)
# =============================================================================


@celery_app.task(bind=True, max_retries=3)
def fetch_ndjson_data(self, project_id: str, job_id: str, config: dict):
    """
    Fetch data from NDJSON source (Segment A).

    Supports multiple source types via config["input_path"]:
    - Local folder path (e.g., "/data/ndjson/")
    - Local zip file (e.g., "/data/export.zip")
    - URL to zip file (e.g., "https://api.example.com/export.zip")

    Output: activities.csv saved to Hetzner storage
    """
    # Direct imports - these modules have no torch dependencies
    from src.data.collection.ndjson_loader import NDJSONLoader
    from src.data.collection.data_source import resolve_data_source

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            if "input_path" not in config or not config["input_path"]:
                raise ValueError("input_path is required in config")

            source_path = config["input_path"]
            with resolve_data_source(source_path) as resolved_path:
                resolved_config = config.copy()
                resolved_config["input_path"] = str(resolved_path)

                if "identity_file" not in config or not config["identity_file"]:
                    resolved_config["identity_file"] = str(resolved_path / "adoIdentities.ndjson")

                loader = NDJSONLoader(resolved_config)
                activities_df = loader.load()

                activities_df.to_csv(storage.activities_path, index=False)

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
                "result": {"num_activities": len(activities_df)},
            })
            db.query(ProjectModel).filter(ProjectModel.id == project_id).update({
                "status": ProjectStatus.DATA_LOADED,
                "updated_at": datetime.utcnow(),
            })
            db.commit()

            return {"status": "completed", "num_activities": len(activities_df)}

        except Exception as e:
            logger.exception(f"Failed to fetch NDJSON data: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


# =============================================================================
# SEGMENT C: Pattern Naming (Hetzner)
# =============================================================================


@celery_app.task(bind=True)
def name_patterns(self, project_id: str, job_id: str, config: dict):
    """
    Generate pattern names using LLM (Segment C).

    Requires:
    - message_scores.h5 (from B.6, in R2)
    - message_scores_index.json (from B.6, on Hetzner)
    - hierarchical_weights.json (from B.7, on Hetzner)
    - message_database.pkl.zst (from B.4, in R2)

    Output: pattern_names.json saved to Hetzner storage
    """
    from src.pattern_identification.pattern_naming import PatternNamer
    from src.pattern_identification.message_scorer import MessageScorer
    from ..services.r2_service import download_pickle_from_r2, download_h5_from_r2

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            hierarchical_weights = storage.load_json(storage.hierarchical_weights_path)
            if hierarchical_weights is None:
                raise FileNotFoundError(f"hierarchical_weights.json not found at {storage.hierarchical_weights_path}")

            message_database = download_pickle_from_r2(project_id, "message_database")

            h5_path = storage.get_cached_h5("message_scores")
            if not h5_path.exists():
                logger.info(f"Downloading message_scores.h5 from R2...")
                download_h5_from_r2(project_id, "message_scores", h5_path)

            messages_list = message_database.get("messages", message_database)
            if isinstance(messages_list, dict):
                messages_list = messages_list.get("messages", [])

            # Load word attributions if available
            word_attributions = storage.load_json(storage.word_attributions_path) or {}
            if word_attributions:
                logger.info(f"Loaded word attributions for {len(word_attributions)} levels")

            def query_examples(pattern_key: str, pattern_idx: int, limit: int) -> list[dict]:
                """Query top examples for a pattern from message_scores.h5."""
                examples = MessageScorer.get_top_messages_for_pattern(
                    h5_path=h5_path,
                    level_key=pattern_key,
                    pattern_idx=pattern_idx,
                    message_database=messages_list,
                    limit=limit,
                )

                # Merge word attributions if available
                level_attrs = word_attributions.get(pattern_key, {})
                pattern_attrs = level_attrs.get(str(pattern_idx), [])

                return {
                    "examples": examples,
                    "aggregated_word_attributions": pattern_attrs,
                }

            def update_progress(current: int, total: int, message: str):
                """Update job progress in database."""
                progress = current / total if total > 0 else 0
                db.query(JobModel).filter(JobModel.id == job_id).update({
                    "progress": progress,
                    "progress_message": message,
                })
                db.commit()
                logger.info(f"Progress: {message} ({progress:.0%})")

            namer = PatternNamer(config, output_path=storage.pattern_names_path)
            pattern_names = namer.name_all_patterns(
                hierarchical_weights=hierarchical_weights,
                message_database=messages_list,
                query_examples_fn=query_examples,
                progress_callback=update_progress,
            )

            # Save pattern names to Hetzner storage
            storage.save_json(storage.pattern_names_path, pattern_names)

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
                "result": {"num_patterns": len(pattern_names)},
            })
            db.query(ProjectModel).filter(ProjectModel.id == project_id).update({
                "status": ProjectStatus.PATTERNS_IDENTIFIED,
                "updated_at": datetime.utcnow(),
            })
            db.commit()

            return {"status": "completed", "num_patterns": len(pattern_names)}

        except Exception as e:
            logger.exception(f"Failed to name patterns: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


# =============================================================================
# SEGMENT D.2: Report Generation (Hetzner)
# =============================================================================


@celery_app.task(bind=True)
def generate_report(self, project_id: str, job_id: str, engineer_id: str, config: dict):
    """
    Generate report for an engineer using LLM (Segment D.2).

    Requires:
    - Individual scores (from D.1, saved to Hetzner)
    - Pattern names (from Segment C)
    - message_scores.h5 (from B.6, cached on Hetzner or in R2)
    - message_database.pkl.zst (from B.4, in R2)

    Output: Report JSON saved to Hetzner storage
    """
    from src.scoring.report_generator import ReportGenerator
    from src.pattern_identification.message_scorer import MessageScorer
    from ..services.r2_service import download_pickle_from_r2, download_h5_from_r2

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            scores_path = storage.base_path / f"scoring/individual/{engineer_id}.json"
            scores = storage.load_json(scores_path)
            if scores is None:
                raise FileNotFoundError(f"Individual scores not found at {scores_path}")

            pattern_names = storage.load_json(storage.pattern_names_path)
            if pattern_names is None:
                raise FileNotFoundError(f"pattern_names.json not found at {storage.pattern_names_path}")

            index = storage.load_json(storage.message_scores_index_path)
            if index is None:
                raise FileNotFoundError(f"message_scores_index.json not found at {storage.message_scores_index_path}")

            h5_path = storage.get_cached_h5("message_scores")
            if not h5_path.exists():
                logger.info(f"Downloading message_scores.h5 from R2...")
                download_h5_from_r2(project_id, "message_scores", h5_path)

            message_database = download_pickle_from_r2(project_id, "message_database")
            messages_list = message_database.get("messages", message_database)
            if isinstance(messages_list, dict):
                messages_list = messages_list.get("messages", [])

            engineer_indices = None
            if engineer_id in index["engineers"]:
                engineer_indices = index["engineers"][engineer_id]["message_indices"]

            def query_examples(pattern_key: str, pattern_idx: int, limit: int) -> list[dict]:
                """Query top examples for engineer from message_scores.h5."""
                return MessageScorer.get_top_messages_for_pattern(
                    h5_path=h5_path,
                    level_key=pattern_key,
                    pattern_idx=pattern_idx,
                    message_database=messages_list,
                    limit=limit,
                    message_indices=engineer_indices,
                )

            debug_dir = storage.base_path / "scoring" / "debug"
            generator = ReportGenerator(config, debug_dir=debug_dir)

            # Transform scores into patterns list for report generator
            patterns = []
            raw_scores = scores.get("scores", {})
            for level_key, level_data in raw_scores.items():
                percentiles = level_data.get("percentiles", [])
                level_names = pattern_names.get(level_key, {})
                parts = level_key.split("_")
                level = parts[1] if len(parts) >= 2 else level_key
                for dim_idx, pct in enumerate(percentiles):
                    dim_key = f"{level}_{dim_idx}"
                    name_info = level_names.get(dim_key, {})
                    name = name_info.get("name", f"{level_key}_dim{dim_idx}")
                    patterns.append({
                        "level": level_key,
                        "dim": dim_idx,
                        "name": name,
                        "percentile": pct,
                    })

            transformed_scores = {
                "engineer_id": scores.get("engineer_id", engineer_id),
                "n_messages": scores.get("n_messages", 0),
                "patterns": patterns,
            }

            report = generator.generate_report(
                engineer_id=engineer_id,
                scores=transformed_scores,
                pattern_names=pattern_names,
                query_examples_fn=query_examples,
            )

            report_path = storage.base_path / f"scoring/reports/{engineer_id}.json"
            storage.save_json(report_path, report)

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
            })
            db.query(ProjectModel).filter(ProjectModel.id == project_id).update({
                "status": ProjectStatus.READY,
                "updated_at": datetime.utcnow(),
            })
            db.commit()

            return {"status": "completed", "engineer_id": engineer_id}

        except Exception as e:
            logger.exception(f"Failed to generate report: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise
