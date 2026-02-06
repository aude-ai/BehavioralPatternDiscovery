"""
Modal App: Processing Pipeline

Unified pipeline for Segment B (Processing):
- B.1: Statistical Features
- B.2: Text Embedding
- B.3: Normalization
- B.4: Training Preparation
- B.5: VAE Training
- B.6: Batch Scoring
- B.7: Message Assignment
- B.8: SHAP Analysis

All large data is stored in R2. Only small JSONs are sent to Hetzner.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import modal
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Version marker for deployment verification
PIPELINE_VERSION = "2026.02.05.8"

app = modal.App("bpd-processing")

# Shared Modal configurations
PYTHON_VERSION = "3.11"

# Single unified image with all dependencies
# A100 has 40GB VRAM - plenty for both embedding models and VAE
# Using PyTorch base image with CUDA and cuDNN pre-configured
# Versions aligned with working local Dockerfile/requirements.txt
app_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel",
    )
    .apt_install("git", "wget")
    .pip_install(
        # Core
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyyaml",
        "requests",
        "boto3",
        "zstandard",
        "fastapi",
        "uvicorn[standard]",
        # ML
        "transformers>=4.51.0",
        "sentence-transformers",
        "datasets",
        "bitsandbytes",
        "einops",
        "flash-attn",
        "h5py",
        "peft==0.13.2",
        "pillow",
        "accelerate",
        # Explainability
        "shap>=0.50.0",
        # Utilities
        "tqdm",
        "wandb",
    )
    .env({
        "HF_HOME": "/cache/models",
        "TRANSFORMERS_CACHE": "/cache/models",
    })
    .add_local_dir("src", "/app/src")
    .add_local_dir("config", "/app/config")
    .add_local_dir("cloud/modal_apps/common", "/app/cloud/modal_apps/common")
)

# Volumes
model_cache = modal.Volume.from_name("bpd-model-cache", create_if_missing=True)
training_volume = modal.Volume.from_name("bpd-training", create_if_missing=True)

# Step metadata loaded from cloud.yaml
def _load_pipeline_config() -> dict:
    """Load pipeline config from cloud.yaml."""
    import yaml

    config_path = Path("/app/config/cloud.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Cloud config not found at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "pipeline" not in config:
        raise ValueError("Missing required config section: pipeline in cloud.yaml")
    if "steps" not in config["pipeline"]:
        raise ValueError("Missing required config section: pipeline.steps in cloud.yaml")

    return config["pipeline"]


def get_step_metadata() -> dict[str, dict]:
    """Get step metadata from cloud.yaml."""
    pipeline_config = _load_pipeline_config()
    return pipeline_config["steps"]


def get_step_order() -> list[str]:
    """Get ordered list of steps from cloud.yaml."""
    steps = get_step_metadata()
    # Sort by step key (B.1, B.2, etc.)
    return sorted(steps.keys())


def get_steps_to_run(starting_step: str) -> list[str]:
    """Get list of steps to run from starting_points config."""
    config = _load_pipeline_config()
    starting_points = config.get("starting_points", {})

    if starting_step not in starting_points:
        raise ValueError(f"Invalid starting step: {starting_step}. Valid: {list(starting_points.keys())}")

    return starting_points[starting_step]["steps_to_run"]


def _get_step_outputs(step: str) -> dict:
    """Get the R2 and Hetzner outputs for a step from cloud.yaml."""
    config = _load_pipeline_config()
    step_config = config["steps"].get(step, {})
    return {
        "r2_outputs": step_config.get("r2_outputs", []),
        "hetzner_outputs": step_config.get("hetzner_outputs", []),
    }


def get_hetzner_url() -> str:
    """Get Hetzner base URL from environment."""
    url = os.environ.get("HETZNER_BASE_URL")
    if url is None:
        raise ValueError("HETZNER_BASE_URL environment variable is required")
    return url


def get_headers() -> dict:
    """Get headers for Hetzner API calls."""
    return {"X-Internal-Key": os.environ["HETZNER_INTERNAL_KEY"]}


# =============================================================================
# PROGRESS CALLBACK
# =============================================================================


class ProgressCallback:
    """Helper for sending progress updates to Hetzner."""

    def __init__(self, hetzner_url: str, job_id: str, headers: dict, section: str = "processing"):
        self.hetzner_url = hetzner_url
        self.job_id = job_id
        self.headers = headers
        self.section = section
        self._first_call = True
        logger.info(f"ProgressCallback initialized: url={hetzner_url}, job_id={job_id}")

    def __call__(self, event_type: str, data: dict = None):
        """Send event to Hetzner."""
        import requests

        payload = {
            "type": event_type,
            "section": self.section,
            **(data or {}),
        }

        url = f"{self.hetzner_url}/internal/jobs/{self.job_id}/event"

        # Log first call prominently
        if self._first_call:
            logger.info(f"First callback to Hetzner: {url}")
            self._first_call = False

        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            logger.info(f"Callback {event_type} sent successfully")
        except requests.exceptions.HTTPError as e:
            logger.error(
                f"Callback {event_type} FAILED: HTTP {e.response.status_code} - "
                f"URL: {url} - Response: {e.response.text[:500]}"
            )
        except Exception as e:
            logger.error(f"Callback {event_type} FAILED: {e} - URL: {url}")

    def status(self, message: str):
        """Send status message."""
        self("status", {"message": message})

    def progress(self, progress: float, **kwargs):
        """Send progress update."""
        self("progress", {"progress": progress, **kwargs})

    def completed(self, message: str = "Completed"):
        """Send completion event."""
        self("completed", {"message": message})

    def failed(self, error: str):
        """Send failure event."""
        self("failed", {"error": error})


def upload_json(url: str, headers: dict, data: dict, timeout: int = 30):
    """Upload JSON data to Hetzner."""
    import requests

    response = requests.post(
        url,
        headers={**headers, "Content-Type": "application/json"},
        json=data,
        timeout=timeout,
    )
    response.raise_for_status()
    logger.info(f"Uploaded JSON to: {url}")


def download_file_streaming(url: str, headers: dict, output_path: Path, timeout: int = 300):
    """Download file with streaming to avoid memory issues."""
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)

    logger.info(f"Downloaded: {url} -> {output_path}")


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================


@app.function(
    image=app_image,
    gpu="A100",
    volumes={"/cache": model_cache, "/training": training_volume},
    timeout=86400,
    secrets=[
        modal.Secret.from_name("hetzner-internal-key"),
        modal.Secret.from_name("r2-credentials"),
    ],
)
def run_processing_pipeline(
    project_id: str,
    job_id: str,
    starting_step: str,
    config: dict,
) -> dict:
    """
    Run the processing pipeline from a given starting step.

    All steps from starting_step onwards will be executed. Large files are
    stored in R2, small JSONs are sent to Hetzner.

    Args:
        project_id: Project identifier
        job_id: Job ID for progress tracking
        starting_step: Starting point key (full, from_training, from_scoring, or shap_only)
        config: Full merged pipeline configuration

    Returns:
        Dict with completion status and metadata
    """
    sys.path.insert(0, "/app")

    from cloud.modal_apps.common.r2_storage import validate_prerequisites

    # Log version for deployment verification
    logger.info(f"=== Processing Pipeline v{PIPELINE_VERSION} ===")
    logger.info(f"Project: {project_id}, Job: {job_id}, Starting: {starting_step}")

    # Set up environment
    os.environ["HF_HOME"] = "/cache/models"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/models"

    hetzner_url = get_hetzner_url()
    headers = get_headers()
    callback = ProgressCallback(hetzner_url, job_id, headers, section="processing")

    steps_to_run = get_steps_to_run(starting_step)
    total_steps = len(steps_to_run)

    # Extract inheritance info from config
    parent_id = config.get("parent_id")
    owned_files = config.get("owned_files") or {}

    try:
        # Validate prerequisites
        callback.status(f"Validating prerequisites for {starting_step}...")
        valid, error = validate_prerequisites(
            project_id, starting_step,
            parent_id=parent_id, owned_files=owned_files
        )
        if not valid:
            raise ValueError(f"Prerequisites not met: {error}")

        # Download activities.csv from Hetzner (needed for most steps)
        activities_df = None
        if starting_step in ["full", "from_training"]:
            callback.status("Downloading activities from Hetzner...")
            activities_df = _download_activities(project_id, hetzner_url, headers)

            # Validate required columns
            required_cols = ["engineer_id", "text"]
            missing_cols = [c for c in required_cols if c not in activities_df.columns]
            if missing_cols:
                raise ValueError(
                    f"activities.csv missing required columns: {missing_cols}. "
                    f"Found columns: {list(activities_df.columns)}"
                )

            logger.info(f"Loaded activities: {len(activities_df)} rows, columns: {list(activities_df.columns)}")

            # Apply per-engineer message limit if configured
            max_per_engineer = (
                config.get("processing", {})
                .get("sampling", {})
                .get("max_activities_per_engineer")
            )
            if max_per_engineer and max_per_engineer > 0:
                import pandas as pd_sampling

                original_count = len(activities_df)
                logger.info(f"Applying sampling: max {max_per_engineer} messages per engineer")

                # Sample using groupby-apply pattern
                # Note: groupby preserves all columns, reset_index only affects index
                sampled_groups = []
                for eng_id, group in activities_df.groupby("engineer_id", sort=False):
                    n_sample = min(len(group), max_per_engineer)
                    sampled_groups.append(group.sample(n=n_sample, random_state=42))

                if sampled_groups:
                    activities_df = pd_sampling.concat(sampled_groups, ignore_index=True)
                else:
                    logger.warning("No groups after sampling - keeping original DataFrame")

                # Defensive check - ensure column wasn't lost
                if "engineer_id" not in activities_df.columns:
                    raise ValueError(
                        f"BUG: 'engineer_id' column lost after sampling. "
                        f"Columns after sampling: {list(activities_df.columns)}"
                    )

                callback.status(f"Sampled {len(activities_df)}/{original_count} messages (max {max_per_engineer}/engineer)")
                logger.info(f"Sampled activities: {original_count} -> {len(activities_df)} (max {max_per_engineer}/engineer)")
                logger.info(f"Columns after sampling: {list(activities_df.columns)}")

        # Initialize state that persists between steps
        state = PipelineState(project_id, config, callback)
        state.activities_df = activities_df

        # Run each step
        step_metadata = get_step_metadata()
        steps_run = []

        for i, step in enumerate(steps_to_run):
            step_name = step_metadata[step]["name"]
            overall_progress = i / total_steps

            callback.status(f"Running {step}: {step_name}...")
            callback("progress", {
                "segment": "B",
                "step": step,
                "step_name": step_name,
                "step_progress": 0.0,
                "overall_progress": overall_progress,
            })

            # Run the step
            try:
                step_fn = STEP_FUNCTIONS[step]
                step_fn(state)
                steps_run.append(step)
            except Exception as e:
                logger.exception(f"Pipeline failed at step {step} ({step_name}): {e}")
                callback("failed", {
                    "error": str(e),
                    "failed_step": step,
                    "failed_step_name": step_name,
                    "steps_completed": steps_run,
                })
                raise RuntimeError(f"Step {step} ({step_name}) failed: {e}") from e

            # Report step completion
            callback("progress", {
                "segment": "B",
                "step": step,
                "step_name": step_name,
                "step_progress": 1.0,
                "overall_progress": (i + 1) / total_steps,
            })

        callback.completed("Processing pipeline completed successfully")

        return {
            "status": "completed",
            "steps_requested": steps_to_run,
            "steps_run": steps_run,
            "project_id": project_id,
        }

    except Exception as e:
        # This catches errors before steps start (e.g., prerequisite validation)
        logger.exception(f"Pipeline failed: {e}")
        callback.failed(str(e))
        raise


class PipelineState:
    """
    Holds state that persists between pipeline steps.

    This allows steps to pass data directly without R2 round-trips
    when running sequentially.
    """

    def __init__(self, project_id: str, config: dict, callback: ProgressCallback):
        self.project_id = project_id
        self.config = config
        self.callback = callback

        # Variant inheritance support (Phase 5)
        self.parent_id = config.get("parent_id")
        self.owned_files = config.get("owned_files") or {}

        # Data that can be passed between steps
        self.activities_df = None
        self.aux_features: np.ndarray | None = None
        self.embeddings: np.ndarray | None = None
        self.normalized_embeddings: np.ndarray | None = None
        self.train_input: np.ndarray | None = None
        self.message_database: dict | None = None
        self.checkpoint_path: Path | None = None
        self.model = None
        self.activations: dict[str, np.ndarray] | None = None
        self.population_stats: dict | None = None


def _download_activities(project_id: str, hetzner_url: str, headers: dict):
    """Download activities.csv from Hetzner."""
    import pandas as pd

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_file_streaming(
            f"{hetzner_url}/internal/projects/{project_id}/activities",
            headers,
            tmp_path,
        )
        df = pd.read_csv(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return df


# =============================================================================
# INDIVIDUAL STEP IMPLEMENTATIONS
# =============================================================================


def step_b1_statistical_features(state: PipelineState):
    """B.1: Extract statistical features from activities."""
    import numpy as np

    from src.data.processing.statistical_features import StatisticalFeatureExtractor

    from cloud.modal_apps.common.r2_storage import upload_numpy_to_r2

    # Validate activities_df exists
    if state.activities_df is None:
        raise ValueError("activities_df is None - was it downloaded? Check starting_step is 'full' or 'from_training'")

    # Log DataFrame info for debugging
    logger.info(f"step_b1: activities_df shape={state.activities_df.shape}")
    logger.info(f"step_b1: activities_df columns={list(state.activities_df.columns)}")
    logger.info(f"step_b1: activities_df dtypes:\n{state.activities_df.dtypes}")

    # Validate required columns exist
    required_cols = ["engineer_id", "text", "activity_type", "timestamp"]
    missing_cols = [c for c in required_cols if c not in state.activities_df.columns]
    if missing_cols:
        raise ValueError(
            f"activities_df missing required columns: {missing_cols}. "
            f"Available columns: {list(state.activities_df.columns)}"
        )

    state.callback.status("Extracting statistical features...")

    extractor = StatisticalFeatureExtractor(state.config)

    # Extract features per engineer (returns dict[str, np.ndarray])
    features_by_engineer = extractor.extract(state.activities_df)

    # Normalize features to [-1, 1]
    normalized_features, scale_factors = extractor.normalize_features(features_by_engineer)

    # Convert to per-message array (each message gets its engineer's features)
    aux_features = np.array([
        normalized_features[str(row["engineer_id"])]
        for _, row in state.activities_df.iterrows()
    ], dtype=np.float32)

    # Store in state for next step
    state.aux_features = aux_features

    # Upload to R2
    upload_numpy_to_r2(aux_features, state.project_id, "aux_features")

    state.callback.status(f"Extracted {aux_features.shape[1]} features for {aux_features.shape[0]} messages")


def step_b2_text_embedding(state: PipelineState):
    """B.2: Generate text embeddings."""
    import gc
    import torch

    from src.data.processing.encoders import create_text_encoder

    from cloud.modal_apps.common.r2_storage import upload_numpy_to_r2

    # Clear any leftover GPU memory from previous steps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    state.callback.status("Loading embedding model...")

    encoder = create_text_encoder(state.config)
    model_cache.commit()

    # Log memory after model load
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU memory after model load: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    texts = state.activities_df["text"].tolist()
    total = len(texts)

    if "processing" not in state.config:
        raise ValueError("Missing required config section: processing")
    if "text_encoder" not in state.config["processing"]:
        raise ValueError("Missing required config section: processing.text_encoder")

    encoder_config = state.config["processing"]["text_encoder"]
    if "type" not in encoder_config:
        raise ValueError("Missing required config key: processing.text_encoder.type")

    encoder_type = encoder_config["type"]
    encoder_settings = encoder_config.get(encoder_type, {})
    # batch_size has documented default of 32 if not specified
    batch_size = encoder_settings.get("batch_size", 32)

    state.callback.status(f"Embedding {total} texts with {encoder_type}...")

    all_embeddings = []
    num_batches = (total + batch_size - 1) // batch_size
    last_reported_pct = 0
    logged_peak_memory = False

    for batch_idx, i in enumerate(range(0, total, batch_size)):
        batch = texts[i : i + batch_size]
        batch_embeddings = encoder.encode(batch)
        all_embeddings.append(batch_embeddings)

        # Log peak memory after first batch
        if not logged_peak_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU memory at peak (after first batch): allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
            logged_peak_memory = True

        # Only report progress every 10% or on last batch
        progress = min((i + len(batch)) / total, 1.0)
        current_pct = int(progress * 10)
        if current_pct > last_reported_pct or batch_idx == num_batches - 1:
            last_reported_pct = current_pct
            state.callback("progress", {
                "segment": "B",
                "step": "B.2",
                "step_name": "Text Embedding",
                "step_progress": progress,
                "message": f"Embedded {i + len(batch)}/{total} texts",
            })

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Store in state
    state.embeddings = embeddings

    # Upload to R2
    upload_numpy_to_r2(embeddings, state.project_id, "embeddings")

    state.callback.status(f"Generated embeddings: {embeddings.shape}")


def step_b3_normalization(state: PipelineState):
    """B.3: Apply normalization pipeline to embeddings."""
    from src.data.processing.normalizer import NormalizationPipeline

    from cloud.modal_apps.common.r2_storage import (
        download_numpy_from_r2,
        upload_numpy_to_r2,
    )

    # Load embeddings if not in state (starting from B.5)
    if state.embeddings is None:
        state.callback.status("Loading embeddings from R2...")
        state.embeddings = download_numpy_from_r2(
            state.project_id, "embeddings",
            parent_id=state.parent_id, owned_files=state.owned_files
        )

    state.callback.status("Applying normalization pipeline...")

    if "processing" not in state.config:
        raise ValueError("Missing required config section: processing")
    if "normalization" not in state.config["processing"]:
        raise ValueError("Missing required config section: processing.normalization")

    norm_config = state.config["processing"]["normalization"]

    if norm_config.get("enabled", True) and "pipeline" in norm_config:
        pipeline = NormalizationPipeline(state.config)
        pipeline.fit(state.embeddings)
        normalized = pipeline.transform(state.embeddings)
    else:
        # No normalization configured, pass through
        normalized = state.embeddings

    # Store in state
    state.normalized_embeddings = normalized

    # Upload to R2
    upload_numpy_to_r2(normalized, state.project_id, "normalized_embeddings")

    state.callback.status(f"Normalized embeddings: {normalized.shape}")


def step_b4_training_prep(state: PipelineState):
    """B.4: Prepare training data (combine embeddings + aux features, create message database)."""
    from cloud.modal_apps.common.r2_storage import (
        download_numpy_from_r2,
        upload_numpy_to_r2,
        upload_pickle_to_r2,
    )

    # Load if not in state
    if state.normalized_embeddings is None:
        state.callback.status("Loading normalized embeddings from R2...")
        state.normalized_embeddings = download_numpy_from_r2(
            state.project_id, "normalized_embeddings",
            parent_id=state.parent_id, owned_files=state.owned_files
        )

    if state.aux_features is None:
        state.callback.status("Loading aux features from R2...")
        state.aux_features = download_numpy_from_r2(
            state.project_id, "aux_features",
            parent_id=state.parent_id, owned_files=state.owned_files
        )

    state.callback.status("Preparing training data...")

    # Check model config for aux_features setting
    include_aux = state.config["model"]["input"]["aux_features"]["enabled"]
    if include_aux:
        train_input = np.concatenate([state.normalized_embeddings, state.aux_features], axis=1)
    else:
        train_input = state.normalized_embeddings

    # Create message database - DataFrame columns are required
    required_columns = ["text", "engineer_id", "source", "timestamp"]
    missing_columns = [col for col in required_columns if col not in state.activities_df.columns]
    if missing_columns:
        raise ValueError(f"activities_df missing required columns: {missing_columns}")

    messages = []
    for idx, row in state.activities_df.iterrows():
        messages.append({
            "index": idx,
            "text": row["text"],
            "engineer_id": row["engineer_id"],
            "source": row["source"],
            "timestamp": str(row["timestamp"]),
        })

    # Extract encoder metadata - config is required
    if "processing" not in state.config:
        raise ValueError("Missing required config section: processing")
    if "text_encoder" not in state.config["processing"]:
        raise ValueError("Missing required config section: processing.text_encoder")

    encoder_config = state.config["processing"]["text_encoder"]
    if "type" not in encoder_config:
        raise ValueError("Missing required config key: processing.text_encoder.type")

    encoder_type = encoder_config["type"]
    encoder_settings = encoder_config.get(encoder_type, {})
    model_name = encoder_settings.get("model_name", encoder_type)

    message_database = {
        "messages": messages,
        "metadata": {
            "num_messages": len(messages),
            "embedding_dim": state.normalized_embeddings.shape[1],
            "aux_dim": state.aux_features.shape[1] if state.aux_features is not None else 0,
            "total_dim": train_input.shape[1],
            "embedder": {
                "type": encoder_type,
                "model_name": model_name,
                "config": encoder_settings,
            },
        },
    }

    # Store in state
    state.train_input = train_input
    state.message_database = message_database

    # Upload to R2
    upload_numpy_to_r2(train_input, state.project_id, "train_input")
    upload_pickle_to_r2(message_database, state.project_id, "message_database")

    state.callback.status(f"Training data prepared: {train_input.shape}")


def step_b5_vae_training(state: PipelineState):
    """B.5: Train VAE model."""
    import torch
    import wandb

    from src.core.config import ModelDimensions
    from src.model.vae import MultiEncoderVAE
    from src.training.trainer import Trainer

    from cloud.modal_apps.common.r2_storage import (
        download_numpy_from_r2,
        download_pickle_from_r2,
        upload_checkpoint_to_r2,
        upload_json_to_r2,
    )

    # Load if not in state
    if state.train_input is None:
        state.callback.status("Loading training data from R2...")
        state.train_input = download_numpy_from_r2(
            state.project_id, "train_input",
            parent_id=state.parent_id, owned_files=state.owned_files
        )
        state.message_database = download_pickle_from_r2(
            state.project_id, "message_database",
            parent_id=state.parent_id, owned_files=state.owned_files
        )

    state.callback.status("Initializing model...")

    metadata = state.message_database.get("metadata", {})
    dims = ModelDimensions.from_config(state.config["model"], metadata)

    model = MultiEncoderVAE(state.config["model"], dims, state.config.get("training"))
    model = model.to("cuda")

    # Initialize wandb if API key available
    wandb_enabled = os.environ.get("WANDB_API_KEY") is not None
    if wandb_enabled:
        wandb.init(
            project="behavioral-pattern-discovery",
            name=f"train-{state.project_id[:8]}",
            config=state.config,
            tags=["modal", "cloud"],
        )

    state.callback.status("Starting training...")

    if "training" not in state.config:
        raise ValueError("Missing required config section: training")

    # training.yaml has nested "training" section with epochs
    training_loop_config = state.config["training"].get("training", {})
    if "epochs" not in training_loop_config:
        raise ValueError("Missing required config key: training.training.epochs")

    max_epochs = training_loop_config["epochs"]

    def on_epoch_end(epoch: int, metrics: dict, is_best: bool):
        state.callback("progress", {
            "segment": "B",
            "step": "B.5",
            "step_name": "VAE Training",
            "step_progress": epoch / max_epochs,
            "message": f"Epoch {epoch}, loss: {metrics.get('total_loss', 0):.4f}",
        })
        if wandb_enabled:
            # Flatten nested dicts and filter to numeric values only
            flat_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            flat_metrics[f"{k}/{sub_k}"] = float(sub_v)
                elif isinstance(v, (int, float)):
                    flat_metrics[k] = float(v)
            wandb.log(flat_metrics, step=epoch)

    # Trainer expects full merged config with model, training, loss_weights, etc.
    # Build trainer config by merging training.yaml contents with model config
    trainer_config = {
        **state.config["training"],  # All of training.yaml
        "model": state.config["model"],  # Add model config for distribution settings
    }

    trainer = Trainer(
        model=model,
        config=trainer_config,
        dims=dims,
        on_epoch_end=on_epoch_end,
        metadata=metadata,
    )

    # Save checkpoint locally first
    checkpoint_dir = Path(f"/training/{state.project_id}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    trainer.train(
        train_data=state.train_input,
        message_db=state.message_database,
        checkpoint_path=str(checkpoint_path),
    )

    training_volume.commit()

    if wandb_enabled:
        wandb.finish()

    # Store in state
    state.checkpoint_path = checkpoint_path
    state.model = model

    # Upload to R2
    upload_checkpoint_to_r2(checkpoint_path, state.project_id)

    # Upload model metadata as JSON (for Hetzner access without torch)
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    model_metadata = {
        "config": checkpoint_data.get("config"),
        "metadata": checkpoint_data.get("metadata"),
        "training_epochs": checkpoint_data.get("epoch"),
    }
    upload_json_to_r2(model_metadata, state.project_id, "model_metadata")

    state.callback.status("Training completed, checkpoint and metadata uploaded to R2")


def step_b6_batch_scoring(state: PipelineState):
    """B.6: Score all messages through VAE."""
    import h5py
    import torch

    from src.core.config import ModelDimensions
    from src.model.vae import MultiEncoderVAE
    from src.pattern_identification.batch_scorer import BatchScorer

    from cloud.modal_apps.common.r2_storage import (
        download_checkpoint_from_r2,
        download_numpy_from_r2,
        download_pickle_from_r2,
        upload_h5_to_r2,
    )

    # Load model if not in state
    if state.model is None:
        state.callback.status("Loading model from R2...")
        checkpoint_path = Path("/tmp/checkpoint.pt")
        download_checkpoint_from_r2(
            state.project_id, checkpoint_path,
            parent_id=state.parent_id, owned_files=state.owned_files
        )

        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model_config = checkpoint["config"]
        metadata = checkpoint["metadata"]
        dims = ModelDimensions.from_config(model_config, metadata)

        model = MultiEncoderVAE(model_config, dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to("cuda")
        model.eval()

        state.model = model
        checkpoint_path.unlink()

    # Load training data if not in state
    if state.train_input is None:
        state.callback.status("Loading training data from R2...")
        state.train_input = download_numpy_from_r2(
            state.project_id, "train_input",
            parent_id=state.parent_id, owned_files=state.owned_files
        )

    if state.message_database is None:
        state.message_database = download_pickle_from_r2(
            state.project_id, "message_database",
            parent_id=state.parent_id, owned_files=state.owned_files
        )

    state.callback.status("Running batch scoring...")

    scorer = BatchScorer(state.config)

    def progress_fn(progress, processed, total):
        state.callback("progress", {
            "segment": "B",
            "step": "B.6",
            "step_name": "Batch Scoring",
            "step_progress": progress,
            "message": f"Scored {processed}/{total} messages",
        })

    activations, population_stats = scorer.score_all(
        vae=state.model,
        train_input=state.train_input,
        progress_callback=progress_fn,
    )

    # Store in state
    state.activations = activations
    state.population_stats = population_stats

    # Save activations to HDF5 and upload to R2
    state.callback.status("Uploading activations to R2...")
    h5_path = Path("/tmp/activations.h5")
    with h5py.File(h5_path, "w") as f:
        for key, value in activations.items():
            f.create_dataset(key, data=value, compression="gzip")

    upload_h5_to_r2(h5_path, state.project_id)
    h5_path.unlink()

    # Send population stats to Hetzner (small JSON)
    state.callback.status("Sending population stats to Hetzner...")
    hetzner_url = get_hetzner_url()
    headers = get_headers()
    upload_json(
        f"{hetzner_url}/internal/projects/{state.project_id}/population-stats",
        headers,
        population_stats,
    )

    state.callback.status(f"Batch scoring complete: {len(activations)} activation sets")


def step_b7_message_assignment(state: PipelineState):
    """B.7: Assign top messages to patterns."""
    import h5py
    import requests

    from src.pattern_identification import MessageAssigner

    from cloud.modal_apps.common.r2_storage import (
        download_h5_from_r2,
        download_pickle_from_r2,
    )

    # Load activations if not in state
    if state.activations is None:
        state.callback.status("Loading activations from R2...")
        h5_path = Path("/tmp/activations.h5")
        download_h5_from_r2(
            state.project_id, h5_path,
            parent_id=state.parent_id, owned_files=state.owned_files
        )

        state.activations = {}
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                state.activations[key] = f[key][:]
        h5_path.unlink()

    if state.message_database is None:
        state.callback.status("Loading message database from R2...")
        state.message_database = download_pickle_from_r2(
            state.project_id, "message_database",
            parent_id=state.parent_id, owned_files=state.owned_files
        )

    if state.population_stats is None:
        # Load from Hetzner
        hetzner_url = get_hetzner_url()
        headers = get_headers()
        response = requests.get(
            f"{hetzner_url}/internal/projects/{state.project_id}/population-stats",
            headers=headers,
        )
        response.raise_for_status()
        state.population_stats = response.json()

    state.callback.status("Assigning messages to patterns...")

    assigner = MessageAssigner(state.config)
    message_examples_raw = assigner.assign_all(
        activations=state.activations,
        message_database=state.message_database["messages"],
    )

    # Convert to JSON-serializable dict
    message_examples = MessageAssigner.to_dict(message_examples_raw)

    # Send to Hetzner (small JSON)
    state.callback.status("Sending message examples to Hetzner...")
    hetzner_url = get_hetzner_url()
    headers = get_headers()
    upload_json(
        f"{hetzner_url}/internal/projects/{state.project_id}/message-examples",
        headers,
        message_examples,
    )

    state.callback.status(f"Assigned messages to {len(message_examples)} patterns")


def step_b8_shap_analysis(state: PipelineState):
    """B.8: Run SHAP analysis to extract hierarchical weights."""
    import h5py
    import torch

    from src.core.config import ModelDimensions
    from src.model.vae import MultiEncoderVAE
    from src.pattern_identification.shap_analysis import SHAPAnalyzer

    from cloud.modal_apps.common.r2_storage import (
        download_checkpoint_from_r2,
        download_h5_from_r2,
    )

    # Load model if not in state
    if state.model is None:
        state.callback.status("Loading model from R2...")
        checkpoint_path = Path("/tmp/checkpoint.pt")
        download_checkpoint_from_r2(
            state.project_id, checkpoint_path,
            parent_id=state.parent_id, owned_files=state.owned_files
        )

        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model_config = checkpoint["config"]
        metadata = checkpoint["metadata"]
        dims = ModelDimensions.from_config(model_config, metadata)

        model = MultiEncoderVAE(model_config, dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to("cuda")
        model.eval()

        state.model = model
        checkpoint_path.unlink()

    # Load activations if not in state
    if state.activations is None:
        state.callback.status("Loading activations from R2...")
        h5_path = Path("/tmp/activations.h5")
        download_h5_from_r2(
            state.project_id, h5_path,
            parent_id=state.parent_id, owned_files=state.owned_files
        )

        state.activations = {}
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                state.activations[key] = f[key][:]
        h5_path.unlink()

    state.callback.status("Running SHAP analysis...")

    analyzer = SHAPAnalyzer(state.config)

    def progress_fn(progress):
        state.callback("progress", {
            "segment": "B",
            "step": "B.8",
            "step_name": "SHAP Analysis",
            "step_progress": progress,
        })

    hierarchical_weights = analyzer.analyze(
        vae=state.model,
        activations=state.activations,
        progress_callback=progress_fn,
    )

    # Send to Hetzner (small JSON)
    state.callback.status("Sending hierarchical weights to Hetzner...")
    hetzner_url = get_hetzner_url()
    headers = get_headers()
    upload_json(
        f"{hetzner_url}/internal/projects/{state.project_id}/shap-weights",
        headers,
        hierarchical_weights,
    )

    state.callback.status(f"SHAP analysis complete: {len(hierarchical_weights)} patterns analyzed")


# Step function mapping
STEP_FUNCTIONS = {
    "B.1": step_b1_statistical_features,
    "B.2": step_b2_text_embedding,
    "B.3": step_b3_normalization,
    "B.4": step_b4_training_prep,
    "B.5": step_b5_vae_training,
    "B.6": step_b6_batch_scoring,
    "B.7": step_b7_message_assignment,
    "B.8": step_b8_shap_analysis,
}


# =============================================================================
# INDIVIDUAL SCORING (SEGMENT D)
# =============================================================================


@app.function(
    image=app_image,
    gpu="A100",
    timeout=600,
    secrets=[
        modal.Secret.from_name("hetzner-internal-key"),
        modal.Secret.from_name("r2-credentials"),
    ],
)
def score_individual(
    project_id: str,
    engineer_id: str,
    messages: list[dict],
    population_stats: dict,
    config: dict,
) -> dict:
    """Score a single engineer against the population."""
    import torch

    sys.path.insert(0, "/app")

    from src.core.config import ModelDimensions
    from src.model.vae import MultiEncoderVAE
    from src.scoring.individual_scorer import IndividualScorer

    from cloud.modal_apps.common.r2_storage import download_checkpoint_from_r2

    # Extract inheritance info from config
    parent_id = config.get("parent_id")
    owned_files = config.get("owned_files") or {}

    # Download checkpoint from R2 (with inheritance support)
    checkpoint_path = Path("/tmp/checkpoint.pt")
    download_checkpoint_from_r2(
        project_id, checkpoint_path,
        parent_id=parent_id, owned_files=owned_files
    )

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model_config = checkpoint["config"]
    metadata = checkpoint["metadata"]
    dims = ModelDimensions.from_config(model_config, metadata)

    model = MultiEncoderVAE(model_config, dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to("cuda")
    model.eval()

    checkpoint_path.unlink()

    # Score
    scorer = IndividualScorer(config)
    result = scorer.score_engineer(
        engineer_id=engineer_id,
        vae=model,
        messages=messages,
        population_stats=population_stats,
    )

    return result


@app.cls(
    image=app_image,
    gpu="A100",
    scaledown_window=60,
    timeout=300,
    secrets=[
        modal.Secret.from_name("hetzner-internal-key"),
        modal.Secret.from_name("r2-credentials"),
    ],
)
class ScoringService:
    """Warm service for individual scoring requests."""

    @modal.enter()
    def setup(self):
        """Initialize model cache on container start."""
        self.models: dict = {}
        self.dims: dict = {}

    def _load_model(
        self,
        project_id: str,
        parent_id: str | None = None,
        owned_files: dict | None = None,
    ):
        """Load and cache model for project."""
        import torch

        sys.path.insert(0, "/app")

        if project_id in self.models:
            return

        from src.core.config import ModelDimensions
        from src.model.vae import MultiEncoderVAE

        from cloud.modal_apps.common.r2_storage import download_checkpoint_from_r2

        # Download from R2 (with inheritance support)
        checkpoint_path = Path(f"/tmp/checkpoint_{project_id}.pt")
        download_checkpoint_from_r2(
            project_id, checkpoint_path,
            parent_id=parent_id, owned_files=owned_files
        )

        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        checkpoint_path.unlink()

        model_config = checkpoint["config"]["model"]
        metadata = checkpoint["metadata"]
        dims = ModelDimensions.from_config(model_config, metadata)

        model = MultiEncoderVAE(model_config, dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to("cuda")
        model.eval()

        self.models[project_id] = model
        self.dims[project_id] = dims

    @modal.method()
    def score(
        self,
        project_id: str,
        engineer_id: str,
        messages: list[dict],
        population_stats: dict,
        config: dict,
    ) -> dict:
        """Score individual engineer with cached model."""
        import torch

        sys.path.insert(0, "/app")

        from src.scoring.individual_scorer import IndividualScorer
        from cloud.modal_apps.common.r2_storage import (
            download_pickle_from_r2,
            download_numpy_from_r2,
            download_checkpoint_from_r2,
        )

        parent_id = config.get("parent_id")
        owned_files = config.get("owned_files") or {}

        self._load_model(project_id, parent_id=parent_id, owned_files=owned_files)
        model = self.models[project_id]

        # Load message_database and train_input from R2
        message_database = download_pickle_from_r2(
            project_id, "message_database",
            parent_id=parent_id, owned_files=owned_files
        )
        train_input = download_numpy_from_r2(
            project_id, "train_input",
            parent_id=parent_id, owned_files=owned_files
        )

        # Build lookup for existing messages by text (to identify duplicates)
        existing_texts = {}
        for msg in message_database["messages"]:
            if msg["engineer_id"] == engineer_id:
                existing_texts[msg["text"]] = msg["index"]

        # Separate existing vs new messages
        engineer_messages = []
        new_messages = []

        for msg in messages:
            if msg["text"] in existing_texts:
                # Existing message - use stored embedding
                msg_copy = msg.copy()
                msg_copy["embedding"] = train_input[existing_texts[msg["text"]]]
                engineer_messages.append(msg_copy)
            else:
                # New message - needs embedding
                new_messages.append(msg)

        # Embed new messages if any
        if new_messages:
            checkpoint_path = Path(f"/tmp/checkpoint_{project_id}_score.pt")
            download_checkpoint_from_r2(
                project_id, checkpoint_path,
                parent_id=parent_id, owned_files=owned_files
            )
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            checkpoint_path.unlink()

            embedder_info = checkpoint.get("embedder", checkpoint.get("metadata", {}).get("embedder", {}))
            norm_params = checkpoint.get("preprocessing", {}).get("normalization")

            encoder_type = embedder_info.get("type", "jina_v3")
            encoder_config = {
                "processing": {
                    "text_encoder": {
                        "type": encoder_type,
                        encoder_type: embedder_info.get("config", {}),
                    },
                    "normalization": config.get("processing", {}).get("normalization", {}),
                }
            }

            from src.data.processing.encoders import create_text_encoder
            from src.data.processing.normalizer import NormalizationPipeline

            encoder = create_text_encoder(encoder_config)
            texts = [m["text"] for m in new_messages]
            embeddings = encoder.encode(texts)

            if norm_params:
                pipeline = NormalizationPipeline(encoder_config)
                pipeline.load_params(norm_params)
                embeddings = pipeline.transform(embeddings)

            # Append new embeddings to train_input and update message_database
            from cloud.modal_apps.common.r2_storage import (
                upload_numpy_to_r2,
                upload_pickle_to_r2,
            )

            start_idx = len(train_input)
            for i, m in enumerate(new_messages):
                m["embedding"] = embeddings[i]
                m["index"] = start_idx + i
                engineer_messages.append(m)
                # Add to message_database
                message_database["messages"].append({
                    "index": start_idx + i,
                    "text": m["text"],
                    "engineer_id": engineer_id,
                    "source": m.get("source", "unknown"),
                    "timestamp": str(m.get("timestamp", "")),
                })

            # Update R2 with new data
            updated_train_input = np.concatenate([train_input, embeddings], axis=0)
            message_database["metadata"]["num_messages"] = len(message_database["messages"])

            upload_numpy_to_r2(updated_train_input, project_id, "train_input")
            upload_pickle_to_r2(message_database, project_id, "message_database")

            logger.info(f"Embedded {len(new_messages)} new messages for {engineer_id}, updated R2")

        if not engineer_messages:
            raise ValueError(f"No messages found for engineer: {engineer_id}")

        logger.info(f"Scoring {engineer_id}: {len(engineer_messages)} total messages "
                    f"({len(engineer_messages) - len(new_messages)} existing, {len(new_messages)} new)")

        scorer = IndividualScorer(config)
        result = scorer.score_engineer(
            engineer_id=engineer_id,
            vae=model,
            messages=engineer_messages,
            population_stats=population_stats,
        )

        return result


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================


@app.function(
    image=app_image,
    secrets=[modal.Secret.from_name("r2-credentials")],
)
def get_r2_status(project_id: str) -> dict:
    """Get R2 file status for a project."""
    sys.path.insert(0, "/app")
    from cloud.modal_apps.common.r2_storage import get_all_r2_file_info

    return get_all_r2_file_info(project_id)


@app.function(
    image=app_image,
    secrets=[modal.Secret.from_name("r2-credentials")],
)
def delete_project_r2_files(project_id: str) -> dict:
    """Delete all R2 files for a project."""
    sys.path.insert(0, "/app")
    from cloud.modal_apps.common.r2_storage import delete_all_project_files

    return delete_all_project_files(project_id)


# =============================================================================
# LOCAL TESTING
# =============================================================================


@app.local_entrypoint()
def test(project_id: str = "test123"):
    """Test R2 connectivity by checking file status for a project."""
    import json

    print(f"Testing R2 connectivity for project: {project_id}")
    print("-" * 50)

    result = get_r2_status.remote(project_id)

    print(json.dumps(result, indent=2, default=str))
    print("-" * 50)
    print("R2 connection successful!")
