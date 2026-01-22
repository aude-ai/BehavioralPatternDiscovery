"""
FastAPI Backend for Behavioral Pattern Discovery

Orchestrates all training, pattern identification, and evaluation processes through REST API endpoints.

NOTE: This is a minimal version for testing. Full API will be implemented after
data, pattern_identification, and scoring components are complete.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import sys
import logging
import yaml
import threading
from pathlib import Path

import torch
import numpy as np

# Performance optimization: Enable cuDNN autotuning
# This benchmarks and selects the fastest cuDNN algorithms for your hardware
# Only beneficial when input sizes are constant (which they are in this application)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Setup logging
class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit for Docker compatibility"""
    def emit(self, record):
        try:
            super().emit(record)
            self.flush()
        except OSError:
            pass

os.makedirs("data/logs", exist_ok=True)

file_handler = logging.FileHandler("data/logs/application.log")
file_handler.setLevel(logging.INFO)

stream_handler = FlushingStreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[file_handler, stream_handler],
    force=True
)
logger = logging.getLogger(__name__)

# Filter out /api/status from uvicorn access logs
class StatusEndpointFilter(logging.Filter):
    def filter(self, record):
        return "/api/status" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(StatusEndpointFilter())

# FastAPI app
app = FastAPI(title="Behavioral Pattern Discovery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


def ensure_data_directories(config: dict) -> None:
    """Create all data directories from config paths on startup."""
    paths = config["paths"]

    # Collect all directory paths from config
    dirs_to_create = []

    # Data section paths
    data_paths = paths["data"]
    dirs_to_create.append(Path(data_paths["collection"]["activities_csv"]).parent)
    dirs_to_create.append(Path(data_paths["synthetic"]["templates_dir"]))
    dirs_to_create.append(Path(data_paths["synthetic"]["generated_dir"]))
    dirs_to_create.append(Path(data_paths["processing"]["message_database"]).parent)

    # Training section paths
    training_paths = paths["training"]
    dirs_to_create.append(Path(training_paths["checkpoints_dir"]))
    dirs_to_create.append(Path(training_paths["logs_dir"]))

    # Pattern identification paths
    pattern_paths = paths["pattern_identification"]
    dirs_to_create.append(Path(pattern_paths["scoring"]["activations"]).parent)
    dirs_to_create.append(Path(pattern_paths["messages"]["examples"]).parent)
    dirs_to_create.append(Path(pattern_paths["shap"]["hierarchical_weights"]).parent)
    dirs_to_create.append(Path(pattern_paths["naming"]["pattern_names"]).parent)

    # Scoring paths
    scoring_paths = paths["scoring"]
    dirs_to_create.append(Path(scoring_paths["individual_dir"]))
    dirs_to_create.append(Path(scoring_paths["reports_dir"]))

    # Create all directories
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ensured {len(dirs_to_create)} data directories exist")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    try:
        config = merge_configs()
        ensure_data_directories(config)
    except Exception as e:
        logger.error(f"Failed to initialize data directories: {e}")

# Global state
process_state = {
    "data_collected": False,
    "preprocessed": False,
    "models_trained": False,
    "patterns_interpreted": False,
    "patterns_identified": False,
    "all_engineers_scored": False,
    "current_task": None,
    "last_error": None,
}

# Training control
_training_stop_signal = threading.Event()
_training_thread: threading.Thread | None = None



def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if not os.path.exists(config_path):
        # Try loading from config directory
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
        if os.path.exists(config_dir):
            config = {}
            for config_file in Path(config_dir).glob("*.yaml"):
                with open(config_file, "r") as f:
                    config.update(yaml.safe_load(f))
            return config
        raise FileNotFoundError("No config.yaml or config/ directory found")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs():
    """Merge all config files from config/ directory"""
    config_dir = Path(__file__).parent / "config"
    merged = {}

    for config_file in sorted(config_dir.glob("*.yaml")):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            if config:
                merged.update(config)

    return merged


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the main frontend page"""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Behavioral Pattern Discovery API", "status": "running"}


@app.get("/population_viewer")
async def population_viewer():
    """Serve the population viewer page"""
    viewer_path = os.path.join(frontend_dir, "population_viewer.html")
    if os.path.exists(viewer_path):
        return FileResponse(viewer_path)
    raise HTTPException(status_code=404, detail="Population viewer not found")


@app.get("/api/status")
async def get_status():
    """Get current processing status"""
    config = merge_configs()
    paths = config["paths"]
    pi_paths = paths["pattern_identification"]

    # Check file existence based on config paths
    activities_exists = Path(paths["data"]["collection"]["activities_csv"]).exists()
    message_db_exists = Path(paths["data"]["processing"]["message_database"]).exists()
    model_exists = Path(paths["training"]["checkpoint"]).exists()

    # Pattern identification file checks
    activations_exist = Path(pi_paths["scoring"]["activations"]).exists()
    message_examples_exist = Path(pi_paths["messages"]["examples"]).exists()
    population_stats_exist = Path(pi_paths["scoring"]["population_stats"]).exists()
    hierarchical_weights_exist = Path(pi_paths["shap"]["hierarchical_weights"]).exists()
    pattern_names_exist = Path(pi_paths["naming"]["pattern_names"]).exists()

    # Update state based on file existence
    process_state["data_collected"] = activities_exists
    process_state["preprocessed"] = message_db_exists
    process_state["models_trained"] = model_exists
    process_state["batch_scored"] = activations_exist and message_examples_exist and population_stats_exist
    process_state["patterns_interpreted"] = hierarchical_weights_exist
    process_state["patterns_identified"] = pattern_names_exist
    process_state["all_engineers_scored"] = pattern_names_exist

    can_run = {
        "preprocess": activities_exists and process_state["current_task"] is None,
        "train_vae": message_db_exists and process_state["current_task"] is None,
        # Pattern identification - dependency chain: batch_score -> shap_analysis -> identify_patterns
        "batch_score": model_exists and process_state["current_task"] is None,
        "shap_analysis": activations_exist and process_state["current_task"] is None,
        "identify_patterns": (
            message_examples_exist and
            hierarchical_weights_exist and
            process_state["current_task"] is None
        ),
        "population_viewer": population_stats_exist,
        "score_engineer": pattern_names_exist and process_state["current_task"] is None,
    }

    return {
        "status": process_state,
        "can_run": can_run,
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    try:
        config = merge_configs()
        return {"status": "success", "config": config}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def build_model_config(config: dict) -> dict:
    """Build model config from merged configs.

    The model uses ModelDimensions to compute derived values internally.
    Each section has its own layer_config for activation/normalization.
    """
    return {
        "input": config["input"],
        "encoder": config["encoder"],
        "unification": config["unification"],
        "decoder": config["decoder"],
        "distribution": config["distribution"],
        "mixture_prior": config["mixture_prior"],
        "discriminators": config["discriminators"],
    }


def build_trainer_config(config: dict, model_config: dict) -> dict:
    """Build trainer config from merged configs."""
    return {
        "training": config["training"],
        "batching": config["batching"],
        "cluster_separation": config["cluster_separation"],
        "discriminator": config["discriminator"],
        "decoder_training": config["decoder_training"],
        "decoder_validation": config["decoder_validation"],
        "loss_weights": config["loss_weights"],
        "capacity": config["capacity"],
        "focal_loss": config["focal_loss"],
        "hoyer": config["hoyer"],
        "iwo": config["iwo"],
        "beta_controller": config["beta_controller"],
        "range_regularization": config["range_regularization"],
        "contrastive_memory_loss": config["contrastive_memory_loss"],
        "entropy_uniformity_loss": config["entropy_uniformity_loss"],
        "performance": config["performance"],
        "vae_optimizer": config["vae_optimizer"],
        "discriminator_optimizer": config["discriminator_optimizer"],
        "scheduler": config["scheduler"],
        "logging": config["logging"],
        "model": model_config,
    }


@app.post("/api/train_vae")
async def train_model(background_tasks: BackgroundTasks):
    """Start VAE training"""
    global _training_thread, _training_stop_signal

    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task already running: {process_state['current_task']}")

    process_state["current_task"] = "training_vae"
    process_state["last_error"] = None
    _training_stop_signal.clear()

    def train_task():
        try:
            logger.info("=" * 80)
            logger.info("STARTING VAE TRAINING")
            logger.info("=" * 80)

            # Import here to avoid circular imports
            import pickle
            from src.model.vae import MultiEncoderVAE
            from src.training import Trainer
            from src.data import (
                create_dataset,
                engineer_collate_fn,
            )
            from src.data.processing import prepare_training_data
            from torch.utils.data import DataLoader

            config = merge_configs()
            paths = config["paths"]

            # Load preprocessed data from config paths
            message_database_path = Path(paths["data"]["processing"]["message_database"])

            if not message_database_path.exists():
                raise FileNotFoundError(
                    f"Message database not found at {message_database_path}. "
                    "Please run preprocessing first."
                )

            # Prepare training input data (combines embeddings + aux_features based on config)
            # This creates cached files for fast switching between aux_features enabled/disabled
            logger.info("Preparing training input data...")
            training_input_path, dimensions = prepare_training_data(config)
            embedding_dim = dimensions["embedding_dim"]
            aux_dim = dimensions["aux_features_dim"]
            input_dim = dimensions["input_dim"]
            aux_enabled = config["input"]["aux_features"]["enabled"]

            logger.info(f"  Embedding dim: {embedding_dim}")
            logger.info(f"  Aux features dim: {aux_dim}")
            logger.info(f"  Aux features enabled: {aux_enabled}")
            logger.info(f"  Input dim (model): {input_dim}")
            logger.info(f"  Training input: {training_input_path}")

            # Load message database to get engineer list
            logger.info("Loading message database for engineer list...")
            with open(message_database_path, "rb") as f:
                data = pickle.load(f)

            # Handle both old format (list) and new format (dict with messages/metadata)
            if isinstance(data, dict) and "messages" in data:
                messages = data["messages"]
            else:
                messages = data

            # Get unique engineers
            all_engineers = list(set(msg["engineer_id"] for msg in messages))
            logger.info(f"Found {len(all_engineers)} unique engineers")

            # Split into train/validation
            validation_split = config["training"]["validation_split"]
            n_val = int(len(all_engineers) * validation_split)
            n_train = len(all_engineers) - n_val

            np.random.seed(42)
            np.random.shuffle(all_engineers)
            train_engineers = all_engineers[:n_train]
            val_engineers = all_engineers[n_train:] if n_val > 0 else []
            logger.info(f"Split: {len(train_engineers)} train, {len(val_engineers)} validation engineers")

            # Update base dimensions from actual data (ModelDimensions computes derived values)
            config["input"]["embedding_dim"] = embedding_dim
            config["input"]["aux_features_dim"] = aux_dim

            # Build model config
            model_config = build_model_config(config)

            # Device selection
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            if device == "cuda":
                logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

            # Build model (model logs its own architecture)
            logger.info("-" * 80)
            model = MultiEncoderVAE(model_config, training_config=config)

            # Build trainer config and create trainer (trainer logs its settings)
            trainer_config = build_trainer_config(config, model_config)
            # Pass num_engineers for DiversityDiscriminator (if enabled)
            num_engineers = len(train_engineers)
            trainer = Trainer(
                model, trainer_config, device=device,
                num_engineers=num_engineers,
                stop_signal=_training_stop_signal,
            )

            # Create datasets using factory pattern
            max_messages = config["training"]["max_messages_per_engineer"]
            batching_config = config["batching"]
            batching_mode = batching_config["mode"]
            batch_size = batching_config.get("random_batch_size", 512)
            messages_per_engineer = batching_config.get("messages_per_engineer", 32)

            logger.info(f"Batching mode: {batching_mode}")

            # Create training dataset via factory
            train_dataset = create_dataset(
                mode=batching_mode,
                message_database_path=message_database_path,
                engineer_ids=train_engineers,
                batch_size=batch_size,
                max_messages=max_messages,
                messages_per_engineer=messages_per_engineer,
                training_input_path=training_input_path,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=1,  # All dataset types return pre-built batches
                shuffle=True,
                collate_fn=engineer_collate_fn,
                num_workers=config["performance"]["num_workers"],
                pin_memory=config["performance"]["pin_memory"],
            )

            # Create validation dataset via factory
            val_dataloader = None
            if val_engineers:
                val_dataset = create_dataset(
                    mode=batching_mode,
                    message_database_path=message_database_path,
                    engineer_ids=val_engineers,
                    batch_size=batch_size,
                    max_messages=max_messages,
                    messages_per_engineer=messages_per_engineer,
                    training_input_path=training_input_path,
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=engineer_collate_fn,
                    num_workers=config["performance"]["num_workers"],
                    pin_memory=config["performance"]["pin_memory"],
                )

            logger.info(f"  Batches per epoch: {len(train_dataloader)}")

            # Training loop (trainer logs per-epoch metrics)
            epochs = config["training"]["epochs"]
            logger.info("-" * 80)
            logger.info(f"Starting training for {epochs} epochs...")
            logger.info(f"  Training engineers: {len(train_engineers)}")
            logger.info(f"  Validation engineers: {len(val_engineers)}")
            if batching_mode == "multi_engineer":
                logger.info(f"  Batching: multi-engineer (max {messages_per_engineer} per engineer per batch)")
                logger.info(f"  Batch size: ~{batch_size}, Batches per epoch: {len(train_dataloader)}")
            elif batching_mode == "random":
                logger.info(f"  Batching: random messages, batch_size={batch_size}")
                logger.info(f"  Batches per epoch: {len(train_dataloader)}")
            else:
                logger.info(f"  Batching: engineer mode (1 engineer per batch)")
                logger.info(f"  Max messages per engineer: {max_messages}")
            logger.info("-" * 80)

            for epoch in range(epochs):
                train_metrics, val_metrics, should_stop = trainer.train_epoch(
                    train_dataloader,
                    val_dataloader=val_dataloader,
                    total_epochs=epochs,
                )
                if should_stop:
                    logger.info("Early stopping triggered")
                    break

                # Check for user-requested stop
                if _training_stop_signal.is_set():
                    logger.info("Training stopped by user request")
                    break

                # Rebuild batches for next epoch (multi_engineer and random modes)
                if hasattr(train_dataset, "on_epoch_end"):
                    train_dataset.on_epoch_end()

            # Restore best model if early stopping with save_best
            if trainer.save_best and trainer.best_model_state is not None:
                trainer.restore_best_model()

            # Save checkpoint to config path
            checkpoint_dir = Path(paths["training"]["checkpoints_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = Path(paths["training"]["checkpoint"])
            trainer.save_checkpoint(checkpoint_path)

            # Finish any logging sessions (e.g., wandb)
            trainer.finish_logging()

            logger.info("=" * 80)
            logger.info("TRAINING COMPLETE")
            logger.info(f"Checkpoint saved to: {checkpoint_path}")
            logger.info("=" * 80)
            process_state["models_trained"] = True

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(train_task)
    return {"status": "started", "message": "Training started in background"}


@app.post("/api/train/stop")
async def stop_training():
    """Request early termination of training.

    Sets a stop signal that the training loop checks after each epoch.
    Training will complete the current epoch before stopping.
    """
    global _training_stop_signal

    if process_state["current_task"] not in ("training_vae", "training_vae_from_existing"):
        raise HTTPException(
            status_code=400,
            detail="No training in progress"
        )

    _training_stop_signal.set()
    logger.info("Stop signal sent - training will stop after current epoch")
    return {"status": "stop_requested", "message": "Training will stop after current epoch"}


@app.post("/api/train_vae_from_existing")
async def train_model_from_existing(request: dict, background_tasks: BackgroundTasks):
    """Start VAE training using an existing trained VAE as starting point."""
    global _training_thread, _training_stop_signal

    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task already running: {process_state['current_task']}")

    model_path = request.get("model_path")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")

    model_path = Path(model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

    process_state["current_task"] = "training_vae_from_existing"
    process_state["last_error"] = None
    _training_stop_signal.clear()

    def train_task():
        try:
            logger.info("=" * 80)
            logger.info("STARTING VAE TRAINING FROM EXISTING MODEL")
            logger.info(f"Loading base model from: {model_path}")
            logger.info("=" * 80)

            import pickle
            from src.model.vae import MultiEncoderVAE
            from src.training import Trainer
            from src.data import (
                create_dataset,
                engineer_collate_fn,
            )
            from src.data.processing import TrainingDataPreparer
            from torch.utils.data import DataLoader

            # Load existing checkpoint to get saved config
            checkpoint = torch.load(model_path, map_location="cpu")
            saved_config = checkpoint["config"]
            logger.info("Loaded config from existing model checkpoint")

            # Load current config for paths and training settings
            current_config = merge_configs()
            paths = current_config["paths"]

            # Load preprocessed data from config paths
            message_database_path = Path(paths["data"]["processing"]["message_database"])

            if not message_database_path.exists():
                raise FileNotFoundError(
                    f"Message database not found at {message_database_path}. "
                    "Please run preprocessing first."
                )

            # Get aux_features setting from saved model config
            saved_aux_enabled = saved_config["model"]["input"]["aux_features"]["enabled"]

            # Prepare training input data using saved model's aux_features setting
            logger.info("Preparing training input data...")
            preparer = TrainingDataPreparer(current_config)
            training_input_path = preparer.prepare(saved_aux_enabled)
            dimensions = preparer.get_dimensions(saved_aux_enabled)
            embedding_dim = dimensions["embedding_dim"]
            aux_dim = dimensions["aux_features_dim"]
            input_dim = dimensions["input_dim"]

            logger.info(f"  Embedding dim: {embedding_dim}")
            logger.info(f"  Aux features dim: {aux_dim}")
            logger.info(f"  Aux features enabled (from saved model): {saved_aux_enabled}")
            logger.info(f"  Input dim (model): {input_dim}")
            logger.info(f"  Training input: {training_input_path}")

            # Load message database to get engineer list
            logger.info("Loading message database for engineer list...")
            with open(message_database_path, "rb") as f:
                data = pickle.load(f)

            # Handle both old format (list) and new format (dict with messages/metadata)
            if isinstance(data, dict) and "messages" in data:
                messages = data["messages"]
            else:
                messages = data

            # Get unique engineers
            all_engineers = list(set(msg["engineer_id"] for msg in messages))
            logger.info(f"Found {len(all_engineers)} unique engineers")

            # Verify dimensions match the saved model
            saved_embedding_dim = saved_config["model"]["input"]["embedding_dim"]
            saved_aux_dim = saved_config["model"]["input"]["aux_features_dim"]
            if embedding_dim != saved_embedding_dim or aux_dim != saved_aux_dim:
                raise ValueError(
                    f"Data dimensions mismatch! "
                    f"Saved model expects embedding_dim={saved_embedding_dim}, aux_dim={saved_aux_dim}, "
                    f"but current data has embedding_dim={embedding_dim}, aux_dim={aux_dim}. "
                    f"Ensure both datasets use the same embedding model and aux features."
                )

            # Split into train/validation using current config settings
            validation_split = current_config["training"]["validation_split"]
            n_val = int(len(all_engineers) * validation_split)
            n_train = len(all_engineers) - n_val

            np.random.seed(42)
            np.random.shuffle(all_engineers)
            train_engineers = all_engineers[:n_train]
            val_engineers = all_engineers[n_train:] if n_val > 0 else []
            logger.info(f"Split: {len(train_engineers)} train, {len(val_engineers)} validation engineers")

            # Use saved config for model architecture
            model_config = saved_config["model"]
            logger.info("Using model architecture from saved checkpoint")

            # Device selection
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            if device == "cuda":
                logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

            # Build model using saved config (with current training settings for decoder)
            logger.info("-" * 80)
            model = MultiEncoderVAE(model_config, training_config=current_config)

            # Load weights from checkpoint
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model weights from checkpoint")

            # Build trainer config using current training settings but saved model config
            trainer_config = build_trainer_config(current_config, model_config)
            num_engineers = len(train_engineers)
            trainer = Trainer(
                model, trainer_config, device=device,
                num_engineers=num_engineers,
                stop_signal=_training_stop_signal,
            )

            # Load discriminator weights if present
            if checkpoint.get("tc_intra_discriminators"):
                for k, state_dict in checkpoint["tc_intra_discriminators"].items():
                    if k in trainer.tc_intra_discriminators:
                        trainer.tc_intra_discriminators[k].load_state_dict(state_dict)
                logger.info("Loaded TC intra discriminator weights")

            if checkpoint.get("pc_inter_discriminators"):
                for k, state_dict in checkpoint["pc_inter_discriminators"].items():
                    if k in trainer.pc_inter_discriminators:
                        trainer.pc_inter_discriminators[k].load_state_dict(state_dict)
                logger.info("Loaded PC inter discriminator weights")

            if checkpoint.get("tc_unified_discriminator") and trainer.tc_unified_discriminator:
                trainer.tc_unified_discriminator.load_state_dict(checkpoint["tc_unified_discriminator"])
                logger.info("Loaded TC unified discriminator weights")

            if checkpoint.get("decoder_ema") and trainer.decoder_ema:
                trainer.decoder_ema.load_state_dict(checkpoint["decoder_ema"])
                logger.info("Loaded decoder EMA weights")

            # Create datasets using factory pattern
            max_messages = current_config["training"]["max_messages_per_engineer"]
            batching_config = current_config["batching"]
            batching_mode = batching_config["mode"]
            batch_size = batching_config.get("random_batch_size", 512)
            messages_per_engineer = batching_config.get("messages_per_engineer", 32)

            logger.info(f"Batching mode: {batching_mode}")

            # Create training dataset via factory
            train_dataset = create_dataset(
                mode=batching_mode,
                message_database_path=message_database_path,
                engineer_ids=train_engineers,
                batch_size=batch_size,
                max_messages=max_messages,
                messages_per_engineer=messages_per_engineer,
                training_input_path=training_input_path,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                collate_fn=engineer_collate_fn,
                num_workers=current_config["performance"]["num_workers"],
                pin_memory=current_config["performance"]["pin_memory"],
            )

            # Create validation dataset via factory
            val_dataloader = None
            if val_engineers:
                val_dataset = create_dataset(
                    mode=batching_mode,
                    message_database_path=message_database_path,
                    engineer_ids=val_engineers,
                    batch_size=batch_size,
                    max_messages=max_messages,
                    messages_per_engineer=messages_per_engineer,
                    training_input_path=training_input_path,
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=engineer_collate_fn,
                    num_workers=current_config["performance"]["num_workers"],
                    pin_memory=current_config["performance"]["pin_memory"],
                )

            logger.info(f"  Batches per epoch: {len(train_dataloader)}")

            # Training loop
            epochs = current_config["training"]["epochs"]
            logger.info("-" * 80)
            logger.info(f"Starting training for {epochs} epochs (fine-tuning from existing model)...")
            logger.info(f"  Training engineers: {len(train_engineers)}")
            logger.info(f"  Validation engineers: {len(val_engineers)}")
            if batching_mode == "multi_engineer":
                logger.info(f"  Batching: multi-engineer (max {messages_per_engineer} per engineer per batch)")
                logger.info(f"  Batch size: ~{batch_size}, Batches per epoch: {len(train_dataloader)}")
            elif batching_mode == "random":
                logger.info(f"  Batching: random messages, batch_size={batch_size}")
                logger.info(f"  Batches per epoch: {len(train_dataloader)}")
            else:
                logger.info(f"  Batching: engineer mode (1 engineer per batch)")
                logger.info(f"  Max messages per engineer: {max_messages}")
            logger.info("-" * 80)

            for epoch in range(epochs):
                train_metrics, val_metrics, should_stop = trainer.train_epoch(
                    train_dataloader,
                    val_dataloader=val_dataloader,
                    total_epochs=epochs,
                )
                if should_stop:
                    logger.info("Early stopping triggered")
                    break

                # Check for user-requested stop
                if _training_stop_signal.is_set():
                    logger.info("Training stopped by user request")
                    break

                # Rebuild batches for next epoch
                if hasattr(train_dataset, "on_epoch_end"):
                    train_dataset.on_epoch_end()

            # Restore best model if early stopping with save_best
            if trainer.save_best and trainer.best_model_state is not None:
                trainer.restore_best_model()

            # Save checkpoint to config path
            checkpoint_dir = Path(paths["training"]["checkpoints_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = Path(paths["training"]["checkpoint"])
            trainer.save_checkpoint(checkpoint_path)

            # Finish any logging sessions
            trainer.finish_logging()

            logger.info("=" * 80)
            logger.info("TRAINING FROM EXISTING MODEL COMPLETE")
            logger.info(f"Checkpoint saved to: {checkpoint_path}")
            logger.info("=" * 80)
            process_state["models_trained"] = True

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(train_task)
    return {"status": "started", "message": "Training from existing model started in background"}


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Get recent log entries"""
    log_file = Path("data/logs/application.log")
    if not log_file.exists():
        return {"logs": []}

    with open(log_file, "r") as f:
        all_lines = f.readlines()
        recent_lines = all_lines[-lines:]

    return {"logs": [line.strip() for line in recent_lines]}


@app.get("/api/check-preprocessed")
async def check_preprocessed():
    """Check if preprocessed data exists"""
    config = merge_configs()
    paths = config["paths"]["data"]["processing"]

    files = {
        "message_database": Path(paths["message_database"]).exists(),
        "train_features": Path(paths["train_features"]).exists(),
        "train_aux_vars": Path(paths["train_aux_vars"]).exists(),
    }

    has_data = files["message_database"]

    return {
        "status": "success",
        "has_preprocessed_data": has_data,
        "files": files,
    }


@app.get("/api/check-model")
async def check_model():
    """Check if trained model exists"""
    config = merge_configs()
    model_path = Path(config["paths"]["training"]["checkpoint"])

    return {
        "status": "success",
        "has_trained_model": model_path.exists(),
        "model_path": str(model_path) if model_path.exists() else None,
    }


# ============================================================================
# Data Collection Endpoints
# ============================================================================

@app.post("/api/fetch_mongodb")
async def fetch_mongodb(request: dict, background_tasks: BackgroundTasks):
    """Fetch activities from MongoDB."""
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    process_state["current_task"] = "fetching_mongodb"
    process_state["last_error"] = None

    async def fetch_task():
        try:
            from src.data import MongoDBLoader

            config = merge_configs()
            loader = MongoDBLoader(config)

            databases = [request["database"]] if request.get("database") else None
            result = loader.fetch_activities(
                databases=databases,
                append=request.get("append", False),
                max_engineers=request.get("max_engineers"),
                max_activities_per_engineer=request.get("max_activities_per_engineer"),
            )

            process_state["data_collected"] = True
            logger.info(f"Fetched {result['count']} activities from {result['engineers']} engineers")

        except Exception as e:
            logger.error(f"MongoDB fetch failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(fetch_task)
    return {"status": "started"}


@app.post("/api/fetch_ndjson")
async def fetch_ndjson(request: dict, background_tasks: BackgroundTasks):
    """
    Fetch activities from NDJSON files in a folder.

    Request body:
    {
        "base_path": "/path/to/data/folder",
        "date_range": {"start": "2025-01-01", "end": "2025-06-01"}
    }

    The folder should contain:
    - adoIdentities.ndjson (identity file for bot detection)
    - *.ndjson.gz (data files)
    """
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    process_state["current_task"] = "fetching_ndjson"
    process_state["last_error"] = None

    async def fetch_task():
        try:
            import pandas as pd
            from src.data.collection import NDJSONLoader

            config = merge_configs()
            paths = config["paths"]
            ndjson_config = config["collection"]["ndjson"]

            # Get parameters from request
            base_path = Path(request["base_path"])
            date_range = request.get("date_range")  # Optional

            # Identity file is expected in the base folder
            identity_file = base_path / "adoIdentities.ndjson"

            loader_config = {
                "input_path": str(base_path),
                "identity_file": str(identity_file),
                "date_range": date_range,
                "excluded_sections": ndjson_config.get("excluded_sections", []),
                "parser_configs": config["collection"]["parsers"],
                "text_cleanup": config["collection"]["text_cleanup"],
            }

            logger.info(f"Loading NDJSON from: {base_path}")
            logger.info(f"Using identity file: {identity_file}")
            if date_range:
                logger.info(f"Date range filter: {date_range['start']} to {date_range['end']}")
            else:
                logger.info("Loading all files (no date filter)")

            loader = NDJSONLoader(loader_config)
            df = loader.load()

            # Save to activities.csv
            activities_path = Path(paths["data"]["collection"]["activities_csv"])
            activities_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(activities_path, index=False)

            # Aggregate engineer metadata for Engineer Viewer
            agg_dict = {
                "is_bot": "first",
                "project": lambda x: ",".join(sorted(set(p for p in x if p))),
            }

            # Include is_internal if present
            if "is_internal" in df.columns:
                agg_dict["is_internal"] = "first"

            engineer_meta = df.groupby("engineer_id").agg(agg_dict).reset_index()
            engineer_meta = engineer_meta.rename(columns={"project": "projects"})

            meta_path = activities_path.parent / "engineer_metadata.csv"
            engineer_meta.to_csv(meta_path, index=False)

            logger.info(f"Saved {len(df)} activities to {activities_path}")
            logger.info(f"Saved metadata for {len(engineer_meta)} engineers to {meta_path}")
            logger.info(f"  Bots: {engineer_meta['is_bot'].sum()}, Humans: {(~engineer_meta['is_bot']).sum()}")
            if "is_internal" in df.columns:
                logger.info(f"  Internal: {engineer_meta['is_internal'].sum()}, External: {(~engineer_meta['is_internal']).sum()}")

            process_state["data_collected"] = True

        except Exception as e:
            logger.error(f"NDJSON fetch failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(fetch_task)
    return {"status": "started"}


@app.get("/api/engineers")
async def get_engineers():
    """Get engineer summary (list, counts, splits)."""
    from src.data import EngineerManager

    config = merge_configs()
    manager = EngineerManager(config)
    return manager.get_summary()


@app.post("/api/engineers/set_split")
async def set_engineer_split(request: dict):
    """Set train/validation split for engineers."""
    from src.data import EngineerManager

    config = merge_configs()
    manager = EngineerManager(config)
    return manager.set_split(request["engineer_ids"], request["split"])


@app.post("/api/engineers/remove")
async def remove_engineers(request: dict):
    """Remove engineers from activities."""
    from src.data import EngineerManager

    config = merge_configs()
    manager = EngineerManager(config)
    return manager.remove_engineers(request["engineer_ids"])


@app.post("/api/engineers/merge")
async def merge_engineers(request: dict):
    """Merge engineers into one canonical ID."""
    from src.data import EngineerManager

    config = merge_configs()
    manager = EngineerManager(config)
    return manager.merge_engineers(request["source_ids"], request["target_id"])


# ============================================================================
# Synthetic Profile Endpoints
# ============================================================================

@app.post("/api/synthetic/generate")
async def generate_synthetic(request: dict, background_tasks: BackgroundTasks):
    """Generate synthetic profiles."""
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    process_state["current_task"] = "generating_synthetic"
    process_state["last_error"] = None

    async def generate_task():
        try:
            from src.data import SyntheticProfileGenerator

            config = merge_configs()
            generator = SyntheticProfileGenerator(config)
            result = generator.generate(request.get("copies_per_profile"))

            logger.info(f"Generated {result['generated']} profiles")
            if result["errors"]:
                logger.warning(f"Generation errors: {result['errors']}")

        except Exception as e:
            logger.error(f"Synthetic generation failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(generate_task)
    return {"status": "started"}


@app.get("/api/synthetic/summary")
async def get_synthetic_summary():
    """Get summary of synthetic profiles."""
    from src.data import SyntheticProfileGenerator

    config = merge_configs()
    generator = SyntheticProfileGenerator(config)
    return generator.get_summary()


@app.post("/api/synthetic/add_to_activities")
async def add_synthetic_to_activities(request: dict):
    """Add synthetic profiles to activities.csv."""
    from src.data import SyntheticProfileGenerator

    config = merge_configs()
    generator = SyntheticProfileGenerator(config)
    return generator.add_to_activities(request["split"])


# ============================================================================
# Preprocessing Endpoint
# ============================================================================

@app.post("/api/preprocess")
async def preprocess_data(
    background_tasks: BackgroundTasks,
    max_activities_per_engineer: Optional[int] = None,
):
    """Run preprocessing pipeline.

    Args:
        max_activities_per_engineer: Optional override for max activities per engineer.
            If provided, enables sampling and uses this value instead of config.
    """
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    process_state["current_task"] = "preprocessing"
    process_state["last_error"] = None

    # Capture the override value for the async task
    max_activities_override = max_activities_per_engineer

    async def preprocess_task():
        try:
            from src.data import DataPreprocessor

            config = merge_configs()

            # Apply frontend override if provided
            if max_activities_override is not None:
                if "sampling" not in config["processing"]:
                    config["processing"]["sampling"] = {}
                config["processing"]["sampling"]["enabled"] = True
                config["processing"]["sampling"]["max_activities_per_engineer"] = max_activities_override
                logger.info(f"Using frontend override: max_activities_per_engineer={max_activities_override}")

            preprocessor = DataPreprocessor(config)
            result = preprocessor.preprocess()

            process_state["preprocessed"] = True
            logger.info(f"Preprocessing complete: {result['num_messages']} messages, {result['num_engineers']} engineers")

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(preprocess_task)
    return {"status": "started"}


@app.post("/api/normalization")
async def apply_normalization(background_tasks: BackgroundTasks):
    """Apply normalization to existing preprocessed data.

    Renames original files to *-base and creates new normalized files.
    """
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    process_state["current_task"] = "normalization"
    process_state["last_error"] = None

    async def normalization_task():
        try:
            import pickle
            from datetime import datetime
            from src.data.processing.normalizer import NormalizationPipeline

            logger.info("=" * 80)
            logger.info("APPLYING NORMALIZATION TO EXISTING DATA")
            logger.info("=" * 80)

            config = merge_configs()
            data_paths = config["paths"]["data"]["processing"]
            norm_config = config["processing"]["normalization"]

            pipeline_str = norm_config["pipeline"]
            logger.info(f"Normalization pipeline: {pipeline_str}")

            # Define file paths
            embeddings_path = Path(data_paths["train_features"])
            message_db_path = Path(data_paths["message_database"])

            if not embeddings_path.exists():
                raise FileNotFoundError(
                    f"Embeddings not found at {embeddings_path}. "
                    "Run preprocessing first."
                )
            if not message_db_path.exists():
                raise FileNotFoundError(
                    f"Message database not found at {message_db_path}. "
                    "Run preprocessing first."
                )

            # Load existing data
            logger.info("Loading existing embeddings...")
            embeddings = np.load(embeddings_path)
            logger.info(f"  Shape: {embeddings.shape}")
            logger.info(f"  Variance before: {embeddings.var():.6f}")

            logger.info("Loading existing message database...")
            with open(message_db_path, "rb") as f:
                db_data = pickle.load(f)

            # Handle both old and new format
            if isinstance(db_data, dict) and "messages" in db_data:
                messages = db_data["messages"]
                metadata = db_data["metadata"]
            else:
                messages = db_data
                metadata = {
                    "embedding_dim": messages[0]["embedding"].shape[0] if messages else 0,
                    "aux_features_dim": messages[0]["aux_features"].shape[0] if messages else 0,
                    "num_messages": len(messages),
                }

            # Check if already normalized
            if metadata.get("normalization_params") is not None:
                logger.warning("Data already has normalization applied!")
                logger.warning("Proceeding will apply normalization on top of existing.")

            # Rename original files to *-base
            base_embeddings_path = embeddings_path.parent / f"{embeddings_path.stem}-base{embeddings_path.suffix}"
            base_message_db_path = message_db_path.parent / f"{message_db_path.stem}-base{message_db_path.suffix}"

            # Only rename if base doesn't already exist
            if not base_embeddings_path.exists():
                logger.info(f"Backing up embeddings to {base_embeddings_path}")
                embeddings_path.rename(base_embeddings_path)
            else:
                logger.info(f"Base embeddings already exist at {base_embeddings_path}")

            if not base_message_db_path.exists():
                logger.info(f"Backing up message database to {base_message_db_path}")
                message_db_path.rename(base_message_db_path)
            else:
                logger.info(f"Base message database already exists at {base_message_db_path}")

            # Fit and apply normalization
            logger.info(f"Fitting normalization pipeline: {pipeline_str}...")
            pipeline = NormalizationPipeline(config)
            normalized_embeddings = pipeline.fit_transform(embeddings)
            normalization_params = pipeline.get_params()

            logger.info(f"  Variance after: {normalized_embeddings.var():.6f}")
            logger.info(f"  Variance ratio: {normalized_embeddings.var() / embeddings.var():.2f}x")

            # Update message embeddings
            logger.info("Updating message embeddings...")
            for idx, msg in enumerate(messages):
                msg["embedding"] = normalized_embeddings[idx]

            # Update metadata
            metadata["normalization_params"] = normalization_params
            metadata["normalization_applied_at"] = datetime.now().isoformat()
            metadata["base_files"] = {
                "embeddings": str(base_embeddings_path),
                "message_database": str(base_message_db_path),
            }

            # Save normalized data
            logger.info("Saving normalized embeddings...")
            np.save(embeddings_path, normalized_embeddings)

            logger.info("Saving updated message database...")
            output_data = {
                "messages": messages,
                "metadata": metadata,
            }
            with open(message_db_path, "wb") as f:
                pickle.dump(output_data, f)

            logger.info("=" * 80)
            logger.info("NORMALIZATION COMPLETE")
            logger.info(f"  Pipeline: {pipeline_str}")
            logger.info(f"  Variance: {embeddings.var():.6f} → {normalized_embeddings.var():.6f}")
            logger.info(f"  Original files backed up with -base suffix")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Normalization failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(normalization_task)
    return {"status": "started"}


# ============================================================================
# Pattern Identification Endpoints
# ============================================================================

def load_trained_vae(config: dict):
    """Load trained VAE model from checkpoint.

    Uses model config saved in checkpoint to ensure architecture matches,
    regardless of current config settings.
    """
    from src.model.vae import MultiEncoderVAE

    checkpoint_path = Path(config["paths"]["training"]["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Load checkpoint first to get saved config
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Use model config from checkpoint if available, else fall back to current config
    if "config" in checkpoint and "model" in checkpoint["config"]:
        model_config = checkpoint["config"]["model"]
        logger.info("Using model config from checkpoint")
    else:
        model_config = build_model_config(config)
        logger.warning("Checkpoint missing saved config, using current config")

    # Create model with saved architecture
    # Pass config for training settings (used for decoder validation config if available)
    model = MultiEncoderVAE(model_config, training_config=config)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def load_message_database(config: dict) -> tuple[list[dict], dict]:
    """
    Load message database from pickle file.

    Returns:
        Tuple of (messages list, metadata dict)
    """
    import pickle

    message_db_path = Path(config["paths"]["data"]["processing"]["message_database"])
    if not message_db_path.exists():
        raise FileNotFoundError(f"Message database not found: {message_db_path}")

    with open(message_db_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old format (list) and new format (dict with messages/metadata)
    if isinstance(data, dict) and "messages" in data:
        return data["messages"], data["metadata"]
    else:
        # Old format: just a list of messages
        if data:
            metadata = {
                "embedding_dim": data[0]["embedding"].shape[0],
                "aux_features_dim": data[0]["aux_features"].shape[0],
            }
        else:
            metadata = {}
        return data, metadata


@app.post("/api/batch_score")
async def batch_score(background_tasks: BackgroundTasks):
    """Score all messages and store activations at all levels."""
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    process_state["current_task"] = "batch_scoring"
    process_state["last_error"] = None

    async def score_task():
        try:
            from src.pattern_identification import BatchScorer, MessageAssigner, PopulationStats, WordAttributor

            logger.info("=" * 80)
            logger.info("STARTING BATCH SCORING")
            logger.info("=" * 80)

            config = merge_configs()

            # Load model and data
            logger.info("Loading trained VAE model...")
            vae = load_trained_vae(config)

            logger.info("Loading message database...")
            message_database, _ = load_message_database(config)
            logger.info(f"Loaded {len(message_database)} messages")

            # Score all messages
            logger.info("Scoring all messages...")
            scorer = BatchScorer(config)
            activations = scorer.score_all(vae, message_database)

            # Assign message examples
            logger.info("Assigning message examples to patterns...")
            assigner = MessageAssigner(config)
            assigner.assign_all(activations, message_database)

            # Run word attribution if enabled
            if config["word_attribution"]["enabled"]:
                logger.info("Computing word attributions...")
                # Load the message_examples that were just saved
                examples_path = Path(config["paths"]["pattern_identification"]["messages"]["examples"])
                with open(examples_path, "r") as f:
                    message_examples = json.load(f)

                attributor = WordAttributor(config)
                attributor.compute_attributions(
                    vae=vae,
                    message_database=message_database,
                    message_examples=message_examples,
                    activations=activations,
                )
            else:
                logger.info("Word attribution disabled, skipping")

            # Compute population statistics (enables population viewer)
            logger.info("Computing population statistics...")
            pop_stats = PopulationStats(config)
            engineer_scores = pop_stats.compute_engineer_scores(activations, message_database)
            pop_stats.save(engineer_scores)

            logger.info("=" * 80)
            logger.info("BATCH SCORING COMPLETE")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Batch scoring failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(score_task)
    return {"status": "started"}


@app.post("/api/shap_analysis")
async def shap_analysis(background_tasks: BackgroundTasks):
    """Run SHAP analysis to extract hierarchical weights."""
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    process_state["current_task"] = "shap_analysis"
    process_state["last_error"] = None

    async def shap_task():
        try:
            from src.pattern_identification import BatchScorer, SHAPAnalyzer

            logger.info("=" * 80)
            logger.info("STARTING SHAP ANALYSIS")
            logger.info("=" * 80)

            config = merge_configs()
            pi_paths = config["paths"]["pattern_identification"]

            # Load model and activations
            logger.info("Loading trained VAE model...")
            vae = load_trained_vae(config)

            logger.info("Loading activations...")
            activations = BatchScorer.load_activations(pi_paths["scoring"]["activations"])

            # Run SHAP analysis
            logger.info("Running SHAP analysis...")
            analyzer = SHAPAnalyzer(config)
            analyzer.analyze(vae, activations)

            logger.info("=" * 80)
            logger.info("SHAP ANALYSIS COMPLETE")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(shap_task)
    return {"status": "started"}


@app.post("/api/identify_patterns")
async def identify_patterns(background_tasks: BackgroundTasks):
    """Run LLM pattern naming."""
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    process_state["current_task"] = "pattern_naming"
    process_state["last_error"] = None

    async def naming_task():
        try:
            from src.pattern_identification import MessageAssigner, SHAPAnalyzer, PatternNamer

            logger.info("=" * 80)
            logger.info("STARTING PATTERN NAMING")
            logger.info("=" * 80)

            config = merge_configs()
            pi_paths = config["paths"]["pattern_identification"]

            # Load prerequisites
            logger.info("Loading message examples...")
            message_examples = MessageAssigner.load_examples(pi_paths["messages"]["examples"])

            logger.info("Loading hierarchical weights...")
            hierarchical_weights = SHAPAnalyzer.load_weights(pi_paths["shap"]["hierarchical_weights"])

            logger.info("Loading message database...")
            message_database, _ = load_message_database(config)

            # Name patterns
            logger.info("Naming patterns with LLM...")
            namer = PatternNamer(config)
            namer.name_all_patterns(message_examples, hierarchical_weights, message_database)

            logger.info("=" * 80)
            logger.info("PATTERN NAMING COMPLETE")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Pattern naming failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(naming_task)
    return {"status": "started"}


# ============================================================================
# Population Viewer Endpoints
# ============================================================================

@app.get("/api/population_viewer_data")
async def get_population_viewer_data(engineer_ids: str = None):
    """
    Get data for population viewer.

    Works with or without pattern names - uses placeholder dimension names
    if patterns haven't been named yet.

    Args:
        engineer_ids: Optional comma-separated list of engineer IDs to filter
    """
    import json

    config = merge_configs()
    pi_paths = config["paths"]["pattern_identification"]

    # Check if population stats exist
    pop_stats_path = Path(pi_paths["scoring"]["population_stats"])
    if not pop_stats_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Population stats not found. Run batch scoring first."
        )

    # Load population stats
    with open(pop_stats_path, "r") as f:
        pop_stats = json.load(f)

    # Try to load pattern names (optional)
    pattern_names_path = Path(pi_paths["naming"]["pattern_names"])
    pattern_names = None
    if pattern_names_path.exists():
        with open(pattern_names_path, "r") as f:
            pattern_names = json.load(f)

    # Build pattern info with names or placeholders
    # Frontend expects: patterns[model_key][level] = [{name: "...", idx: ...}, ...]
    patterns = {
        "model_1": {"bottom": [], "mid": [], "top": []},
        "model_2": {"bottom": [], "mid": [], "top": []},
        "model_3": {"bottom": [], "mid": [], "top": []},
        "unified": {"top": []},
    }

    # Map backend keys to frontend keys
    level_mapping = {
        "enc1_bottom": ("model_1", "bottom"),
        "enc1_mid": ("model_1", "mid"),
        "enc1_top": ("model_1", "top"),
        "enc2_bottom": ("model_2", "bottom"),
        "enc2_mid": ("model_2", "mid"),
        "enc2_top": ("model_2", "top"),
        "enc3_bottom": ("model_3", "bottom"),
        "enc3_mid": ("model_3", "mid"),
        "enc3_top": ("model_3", "top"),
        "unified": ("unified", "top"),
    }

    # Get dimension counts from population stats and build pattern list
    for backend_key, (model_key, level) in level_mapping.items():
        if backend_key not in pop_stats:
            continue

        pop_mean = pop_stats[backend_key]["population_mean"]
        n_dims = len(pop_mean)

        for dim_idx in range(n_dims):
            # Try to get LLM-generated name
            name = None
            if pattern_names:
                # Pattern names use keys like "enc1_bottom", "enc1_mid", etc.
                names_key = backend_key
                if names_key in pattern_names:
                    # Names are stored as {"bottom_0": {"name": "...", "description": "..."}, ...}
                    # For unified patterns, the key is "unified_X", not "top_X"
                    if backend_key == "unified":
                        dim_key = f"unified_{dim_idx}"
                    else:
                        dim_key = f"{level}_{dim_idx}"
                    if dim_key in pattern_names[names_key]:
                        name = pattern_names[names_key][dim_key].get("name")

            # Use placeholder if no name
            if not name:
                name = f"{level}_{dim_idx}"

            patterns[model_key][level].append({
                "name": name,
                "idx": dim_idx,
            })

    # Build engineer scores
    # Frontend expects: engineers[engineer_id][model_key][level][idx] = {score, percentile}
    all_engineer_ids = set()
    for level_data in pop_stats.values():
        if "engineers" in level_data:
            all_engineer_ids.update(level_data["engineers"].keys())

    all_engineer_ids = sorted(all_engineer_ids)

    # Filter if specific engineers requested
    if engineer_ids:
        requested_ids = [e.strip() for e in engineer_ids.split(",")]
        filtered_ids = [e for e in requested_ids if e in all_engineer_ids]
        if not filtered_ids:
            raise HTTPException(
                status_code=404,
                detail=f"None of the requested engineers found in population stats"
            )
        selected_engineer_ids = filtered_ids
    else:
        selected_engineer_ids = all_engineer_ids

    engineers = {}
    for eng_id in selected_engineer_ids:
        engineers[eng_id] = {
            "model_1": {"bottom": [], "mid": [], "top": []},
            "model_2": {"bottom": [], "mid": [], "top": []},
            "model_3": {"bottom": [], "mid": [], "top": []},
            "unified": {"top": []},
        }

        for backend_key, (model_key, level) in level_mapping.items():
            if backend_key not in pop_stats:
                continue

            level_data = pop_stats[backend_key]
            if eng_id not in level_data["engineers"]:
                continue

            eng_data = level_data["engineers"][eng_id]
            posterior_means = eng_data["posterior_mean"]
            percentiles = eng_data["percentiles"]

            for dim_idx in range(len(posterior_means)):
                engineers[eng_id][model_key][level].append({
                    "score": posterior_means[dim_idx],
                    "percentile": percentiles[dim_idx],
                })

    return {
        "patterns": patterns,
        "engineers": engineers,
        "available_engineers": all_engineer_ids,
        "has_pattern_names": pattern_names is not None,
    }


@app.get("/api/message_distribution_data")
async def get_message_distribution_data(
    pattern_key: str,
    pattern_idx: int,
    engineer_ids: str = None
):
    """
    Get message-level activation data for a specific pattern.

    Args:
        pattern_key: Backend key like "enc1_bottom", "enc2_mid", "unified"
        pattern_idx: Dimension index within that level
        engineer_ids: Optional comma-separated list of engineer IDs to filter
    """
    import json
    import pickle
    import h5py

    config = merge_configs()
    pi_paths = config["paths"]["pattern_identification"]
    data_paths = config["paths"]["data"]

    # Load activations
    activations_path = Path(pi_paths["scoring"]["activations"])
    if not activations_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Activations not found. Run batch scoring first."
        )

    # Load message database
    message_db_path = Path(data_paths["processing"]["message_database"])
    if not message_db_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Message database not found. Run preprocessing first."
        )

    with open(message_db_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old format (list) and new format (dict with messages/metadata)
    if isinstance(data, dict) and "messages" in data:
        message_database = data["messages"]
    else:
        message_database = data

    # Load activations for the requested pattern
    with h5py.File(activations_path, "r") as f:
        if pattern_key not in f:
            raise HTTPException(
                status_code=404,
                detail=f"Pattern key '{pattern_key}' not found in activations"
            )
        activations = f[pattern_key][:]

    if pattern_idx < 0 or pattern_idx >= activations.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Pattern index {pattern_idx} out of range (0-{activations.shape[1]-1})"
        )

    # Get scores for this specific pattern dimension
    pattern_scores = activations[:, pattern_idx]

    # Filter by engineer IDs if specified
    requested_ids = None
    if engineer_ids:
        requested_ids = set(e.strip() for e in engineer_ids.split(","))

    # Build message data
    messages = []
    for idx, msg in enumerate(message_database):
        eng_id = msg["engineer_id"]

        if requested_ids and eng_id not in requested_ids:
            continue

        messages.append({
            "engineer_id": eng_id,
            "score": float(pattern_scores[idx]),
            "text": msg.get("text", "")[:500],
            "source": msg.get("source", ""),
            "activity_type": msg.get("activity_type", ""),
        })

    # Compute statistics
    all_scores = [m["score"] for m in messages]
    if not all_scores:
        raise HTTPException(
            status_code=404,
            detail="No messages found for the specified engineers"
        )

    return {
        "messages": messages,
        "stats": {
            "count": len(messages),
            "min": min(all_scores),
            "max": max(all_scores),
            "mean": sum(all_scores) / len(all_scores),
        }
    }


# ============================================================================
# Individual Scoring Endpoints
# ============================================================================

@app.get("/api/check_score_exists")
async def check_score_exists(engineer_id: str):
    """Check if individual scores and report exist for an engineer."""
    from src.scoring import IndividualScorer, ReportGenerator

    config = merge_configs()
    scorer = IndividualScorer(config)
    report_generator = ReportGenerator(config)

    return {
        "exists": scorer.check_score_exists(engineer_id),
        "report_exists": report_generator.check_report_exists(engineer_id),
        "engineer_id": engineer_id,
    }


@app.post("/api/score_engineer")
async def score_engineer(request: dict, background_tasks: BackgroundTasks):
    """Score an individual engineer."""
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    engineer_id = request.get("engineer_id")
    if not engineer_id:
        raise HTTPException(status_code=400, detail="engineer_id is required")

    process_state["current_task"] = "scoring_engineer"
    process_state["last_error"] = None

    async def score_task():
        try:
            from src.scoring import IndividualScorer

            logger.info(f"Starting individual scoring for {engineer_id}")

            config = merge_configs()
            pi_paths = config["paths"]["pattern_identification"]

            # Load prerequisites
            vae = load_trained_vae(config)
            message_database, _ = load_message_database(config)

            with open(pi_paths["naming"]["pattern_names"], "r") as f:
                pattern_names = json.load(f)

            # Score engineer
            scorer = IndividualScorer(config)
            scorer.score_engineer(
                engineer_id=engineer_id,
                vae=vae,
                message_database=message_database,
                pattern_names=pattern_names,
            )

            logger.info(f"Individual scoring complete for {engineer_id}")

        except Exception as e:
            logger.error(f"Individual scoring failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(score_task)
    return {"status": "started", "engineer_id": engineer_id}


@app.get("/api/get_scores")
async def get_scores(engineer_id: str):
    """Get individual scores for an engineer."""
    from src.scoring import IndividualScorer

    config = merge_configs()
    scorer = IndividualScorer(config)

    scores = scorer.load_scores(engineer_id)
    if scores is None:
        raise HTTPException(status_code=404, detail=f"Scores not found for {engineer_id}")

    return scores


# ============================================================================
# Report Endpoints
# ============================================================================

@app.get("/api/check_report_exists")
async def check_report_exists(engineer_id: str):
    """Check if report exists for an engineer."""
    from src.scoring import ReportGenerator

    config = merge_configs()
    generator = ReportGenerator(config)

    return {
        "exists": generator.check_report_exists(engineer_id),
        "engineer_id": engineer_id,
    }


@app.post("/api/generate_report")
async def generate_report(request: dict, background_tasks: BackgroundTasks):
    """Generate performance report for an engineer."""
    if process_state["current_task"]:
        raise HTTPException(status_code=409, detail=f"Task running: {process_state['current_task']}")

    engineer_id = request.get("engineer_id")
    if not engineer_id:
        raise HTTPException(status_code=400, detail="engineer_id is required")

    process_state["current_task"] = "generating_report"
    process_state["last_error"] = None

    async def report_task():
        try:
            from src.scoring import IndividualScorer, ReportGenerator

            logger.info(f"Generating report for {engineer_id}")

            config = merge_configs()
            pi_paths = config["paths"]["pattern_identification"]

            # Load scores
            scorer = IndividualScorer(config)
            scores = scorer.load_scores(engineer_id)

            if scores is None:
                raise ValueError(f"Scores not found for {engineer_id}. Run scoring first.")

            # Load pattern data
            with open(pi_paths["naming"]["pattern_names"], "r") as f:
                pattern_names = json.load(f)

            with open(pi_paths["messages"]["examples"], "r") as f:
                message_examples = json.load(f)

            # Generate report
            generator = ReportGenerator(config)
            generator.generate_report(
                engineer_id=engineer_id,
                scores=scores,
                pattern_names=pattern_names,
                message_examples=message_examples,
            )

            logger.info(f"Report generation complete for {engineer_id}")

        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            process_state["last_error"] = str(e)
        finally:
            process_state["current_task"] = None

    background_tasks.add_task(report_task)
    return {"status": "started", "engineer_id": engineer_id}


@app.get("/api/get_report")
async def get_report(engineer_id: str):
    """Get report for an engineer."""
    from src.scoring import ReportGenerator

    config = merge_configs()
    generator = ReportGenerator(config)

    report = generator.load_report(engineer_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report not found for {engineer_id}")

    return report


@app.get("/api/pattern_names")
async def get_pattern_names():
    """Get pattern names from pattern identification."""
    config = merge_configs()
    pi_paths = config["paths"]["pattern_identification"]
    pattern_names_path = Path(pi_paths["naming"]["pattern_names"])

    if not pattern_names_path.exists():
        raise HTTPException(status_code=404, detail="Pattern names not found. Run pattern identification first.")

    with open(pattern_names_path, "r") as f:
        return json.load(f)


@app.get("/api/hierarchical_weights")
async def get_hierarchical_weights():
    """Get hierarchical weights from SHAP analysis."""
    config = merge_configs()
    pi_paths = config["paths"]["pattern_identification"]
    weights_path = Path(pi_paths["shap"]["hierarchical_weights"])

    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Hierarchical weights not found. Run SHAP analysis first.")

    with open(weights_path, "r") as f:
        return json.load(f)


# ============================================================================
# Explanation Endpoints
# ============================================================================

@app.post("/api/explain_pattern")
async def explain_pattern(request: dict):
    """Generate explanation for a specific pattern score."""
    engineer_id = request.get("engineer_id")
    pattern_id = request.get("pattern_id")

    if not engineer_id or not pattern_id:
        raise HTTPException(status_code=400, detail="engineer_id and pattern_id are required")

    try:
        from src.scoring import IndividualScorer, ExplanationGenerator

        config = merge_configs()

        # Load scores
        scorer = IndividualScorer(config)
        scores = scorer.load_scores(engineer_id)

        if scores is None:
            raise HTTPException(status_code=404, detail=f"Scores not found for {engineer_id}")

        message_database, _ = load_message_database(config)

        # Generate explanation
        generator = ExplanationGenerator(config)
        explanation = generator.explain_pattern(
            engineer_id=engineer_id,
            pattern_id=pattern_id,
            scores=scores,
            message_database=message_database,
        )

        return explanation

    except Exception as e:
        logger.error(f"Explanation generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/explanations/{engineer_id}")
async def get_explanations(engineer_id: str):
    """Get all explanations for an engineer."""
    from src.scoring import ExplanationGenerator

    config = merge_configs()
    generator = ExplanationGenerator(config)

    pattern_ids = generator.list_explanations(engineer_id)
    explanations = {}

    for pattern_id in pattern_ids:
        exp = generator.load_explanation(engineer_id, pattern_id)
        if exp:
            explanations[pattern_id] = exp

    return {
        "engineer_id": engineer_id,
        "explanations": explanations,
    }


# ============================================================================
# Report Viewer Page
# ============================================================================

@app.get("/report_viewer")
async def report_viewer():
    """Serve the report viewer page."""
    viewer_path = os.path.join(frontend_dir, "report_viewer.html")
    if os.path.exists(viewer_path):
        return FileResponse(viewer_path)
    raise HTTPException(status_code=404, detail="Report viewer not found")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
