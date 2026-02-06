"""CPU-bound Celery tasks that run on Hetzner."""
import logging
import sys
from pathlib import Path

from ..celery_app import celery_app
from ..config import get_settings
from ..database import get_db_context, JobModel
from ..models import JobStatus
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
    - message_examples.json (from B.7, on Hetzner)
    - hierarchical_weights.json (from B.8, on Hetzner)
    - message_database.pkl.zst (from B.4, in R2)

    Output: pattern_names.json saved to Hetzner storage
    """
    from src.pattern_identification.pattern_naming import PatternNamer
    from ..services.r2_service import download_pickle_from_r2

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            message_examples = storage.load_json(storage.message_examples_path)
            if message_examples is None:
                raise FileNotFoundError(f"message_examples.json not found at {storage.message_examples_path}")

            hierarchical_weights = storage.load_json(storage.hierarchical_weights_path)
            if hierarchical_weights is None:
                raise FileNotFoundError(f"hierarchical_weights.json not found at {storage.hierarchical_weights_path}")

            message_database = download_pickle_from_r2(project_id, "message_database")

            namer = PatternNamer(config)
            pattern_names = namer.name_all_patterns(
                message_examples=message_examples,
                hierarchical_weights=hierarchical_weights,
                message_database=message_database,
            )

            # Save pattern names to Hetzner storage
            storage.save_json(storage.pattern_names_path, pattern_names)

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
                "result": {"num_patterns": len(pattern_names)},
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

    Output: Report JSON saved to Hetzner storage
    """
    # Direct import - this module only uses LLM APIs, no torch
    from src.scoring.report_generator import ReportGenerator

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

            generator = ReportGenerator(config)
            report = generator.generate(
                engineer_id=engineer_id,
                scores=scores,
                pattern_names=pattern_names,
            )

            report_path = storage.base_path / f"scoring/reports/{engineer_id}.json"
            storage.save_json(report_path, report)

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
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
