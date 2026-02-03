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

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            source_path = config.get("input_path", "")
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


@celery_app.task(bind=True)
def preprocess_data(self, project_id: str, job_id: str, config: dict):
    """
    Run preprocessing (CPU portion only).

    This extracts statistical features and prepares text for embedding.
    The actual embedding is done via Modal.
    """
    import pandas as pd

    from src.data.processing.statistical_features import StatisticalFeatureExtractor

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            activities_df = pd.read_csv(storage.activities_path)

            feature_extractor = StatisticalFeatureExtractor(config)
            aux_features = feature_extractor.extract(activities_df)

            storage.save_numpy(storage.train_aux_vars_path, aux_features)

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
                "texts": texts,
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

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            # Load compressed embeddings
            embeddings = storage.load_numpy_compressed(storage.train_features_path)

            norm_config = config.get("processing", {}).get("normalization", {})
            if norm_config.get("enabled", True) and norm_config.get("pipeline"):
                pipeline = NormalizationPipeline(config)
                pipeline.fit(embeddings)
                normalized = pipeline.transform(embeddings)
                # Save back compressed
                storage.save_numpy_compressed(storage.train_features_path, normalized)

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
    import h5py

    from src.pattern_identification.message_assigner import MessageAssigner

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            # Load compressed message database
            message_db = storage.load_pickle_compressed(storage.message_database_path)
            population_stats = storage.load_json(storage.population_stats_path)

            # Decompress activations to temp file for h5py access
            tmp_activations = storage.decompress_to_temp(storage.activations_path)
            try:
                activations = {}
                with h5py.File(tmp_activations, "r") as f:
                    for key in f.keys():
                        activations[key] = f[key][:]
            finally:
                tmp_activations.unlink()

            assigner = MessageAssigner(config)
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

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            message_examples = storage.load_json(storage.message_examples_path)
            hierarchical_weights = storage.load_json(storage.hierarchical_weights_path)

            namer = PatternNamer(config)
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

    with get_db_context() as db:
        try:
            db.query(JobModel).filter(JobModel.id == job_id).update(
                {"status": JobStatus.RUNNING}
            )
            db.commit()

            storage = StorageService(project_id)

            scores_path = storage.base_path / f"scoring/individual/{engineer_id}.json"
            scores = storage.load_json(scores_path)

            pattern_names = storage.load_json(storage.pattern_names_path)

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
