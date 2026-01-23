"""CPU-bound Celery tasks that run on Hetzner."""
import logging
import sys
from pathlib import Path

from ..celery_app import celery_app
from ..config import get_settings
from ..database import get_db, JobModel
from ..models import JobStatus, ProjectStatus
from ..services import StorageService

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(bind=True, max_retries=3)
def fetch_mongodb_data(self, project_id: str, job_id: str, config: dict):
    """Fetch data from MongoDB."""
    from src.data.collection.mongodb_loader import MongoDBLoader

    with get_db() as db:
        try:
            # Update job status
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            loader = MongoDBLoader(config)
            activities_df = loader.load()
            activities_df.to_csv(storage.activities_path, index=False)

            # Update job as completed
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
                "result": {"num_activities": len(activities_df)},
            })
            db.commit()

            return {"status": "completed", "num_activities": len(activities_df)}

        except Exception as e:
            logger.exception(f"Failed to fetch MongoDB data: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


@celery_app.task(bind=True, max_retries=3)
def fetch_ndjson_data(self, project_id: str, job_id: str, config: dict):
    """
    Fetch data from NDJSON source.

    Supports multiple source types via config["input_path"]:
    - Local folder path (e.g., "/data/ndjson/")
    - Local zip file (e.g., "/data/export.zip")
    - URL to zip file (e.g., "https://api.example.com/export.zip")
    """
    from src.data.collection.ndjson_loader import NDJSONLoader
    from src.data.collection.data_source import resolve_data_source

    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            # Resolve data source (handles folder, zip, or URL)
            source_path = config.get("input_path", "")
            with resolve_data_source(source_path) as resolved_path:
                # Update config with resolved path
                resolved_config = config.copy()
                resolved_config["input_path"] = str(resolved_path)

                # Also update identity_file path if it's relative to input_path
                if "identity_file" not in config or not config["identity_file"]:
                    # Default to adoIdentities.ndjson in the resolved folder
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


@celery_app.task(bind=True)
def preprocess_data(self, project_id: str, job_id: str, config: dict):
    """
    Run preprocessing (CPU portion only).

    This extracts statistical features and prepares text for embedding.
    The actual embedding is done via Modal.
    """
    import pandas as pd

    from src.data.processing.statistical_features import StatisticalFeatureExtractor

    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            # Load activities
            activities_df = pd.read_csv(storage.activities_path)

            # Extract statistical features
            feature_extractor = StatisticalFeatureExtractor(config.get("processing", {}))
            aux_features = feature_extractor.extract(activities_df)

            # Save auxiliary features
            storage.save_numpy(storage.train_aux_vars_path, aux_features)

            # Prepare texts for embedding (will be sent to Modal)
            texts = activities_df["text"].tolist()

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
                "result": {
                    "num_texts": len(texts),
                    "aux_features_shape": list(aux_features.shape),
                },
            })
            db.commit()

            return {
                "status": "completed",
                "texts": texts,  # Pass to embedding task
                "num_texts": len(texts),
            }

        except Exception as e:
            logger.exception(f"Failed to preprocess: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


@celery_app.task(bind=True)
def normalize_embeddings(self, project_id: str, job_id: str, config: dict):
    """Apply normalization pipeline to embeddings."""
    from src.data.processing.normalizer import NormalizationPipeline

    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            # Load embeddings
            embeddings = storage.load_numpy(storage.train_features_path)

            # Apply normalization
            pipeline_config = config.get("processing", {}).get("normalization", "")
            if pipeline_config:
                pipeline = NormalizationPipeline.from_config(pipeline_config)
                pipeline.fit(embeddings)
                normalized = pipeline.transform(embeddings)
                storage.save_numpy(storage.train_features_path, normalized)

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
            })
            db.commit()

            return {"status": "completed"}

        except Exception as e:
            logger.exception(f"Failed to normalize: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


@celery_app.task(bind=True)
def assign_messages(self, project_id: str, job_id: str, config: dict):
    """Assign messages to patterns based on activations."""
    from src.pattern_identification.message_assigner import MessageAssigner

    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            # Load data
            message_db = storage.load_pickle(storage.message_database_path)
            population_stats = storage.load_json(storage.population_stats_path)

            # Load activations
            import h5py

            activations = {}
            with h5py.File(storage.activations_path, "r") as f:
                for key in f.keys():
                    activations[key] = f[key][:]

            # Assign messages
            assigner = MessageAssigner(config.get("message_assignment", {}))
            message_examples = assigner.assign(
                activations=activations,
                message_db=message_db,
                population_stats=population_stats,
            )

            storage.save_json(storage.message_examples_path, message_examples)

            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.COMPLETED,
                "progress": 1.0,
            })
            db.commit()

            return {"status": "completed"}

        except Exception as e:
            logger.exception(f"Failed to assign messages: {e}")
            db.query(JobModel).filter(JobModel.id == job_id).update({
                "status": JobStatus.FAILED,
                "error": str(e),
            })
            db.commit()
            raise


@celery_app.task(bind=True)
def name_patterns(self, project_id: str, job_id: str, config: dict):
    """Generate pattern names using LLM."""
    from src.pattern_identification.pattern_naming import PatternNamer

    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            # Load data
            message_examples = storage.load_json(storage.message_examples_path)
            hierarchical_weights = storage.load_json(storage.hierarchical_weights_path)

            # Name patterns
            namer = PatternNamer(config.get("pattern_naming", {}))
            pattern_names = namer.name_patterns(
                message_examples=message_examples,
                hierarchical_weights=hierarchical_weights,
                output_path=storage.pattern_names_path,
            )

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


@celery_app.task(bind=True)
def generate_report(self, project_id: str, job_id: str, engineer_id: str, config: dict):
    """Generate report for an engineer."""
    from src.scoring.report_generator import ReportGenerator

    with get_db() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            # Load scores
            scores_path = storage.base_path / f"scoring/individual/{engineer_id}.json"
            scores = storage.load_json(scores_path)

            # Load pattern names
            pattern_names = storage.load_json(storage.pattern_names_path)

            # Generate report
            generator = ReportGenerator(config.get("report_generation", {}))
            report = generator.generate(
                engineer_id=engineer_id,
                scores=scores,
                pattern_names=pattern_names,
            )

            # Save report
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
