"""Modal app for VAE training, scoring, and analysis."""
import modal

from cloud.modal_apps.common.config import create_ml_image, get_headers, get_hetzner_url
from cloud.modal_apps.common.data_transfer import (
    ProgressCallback,
    compress_and_upload,
    decompress_file,
    download_and_decompress,
    download_file_streaming,
    upload_file,
)

app = modal.App("bpd-vae")

image = (
    create_ml_image()
    .pip_install(
        "pandas>=2.0.0",
    )
    .add_local_dir("src", "/app/src")
    .add_local_dir("config", "/app/config")
)

training_volume = modal.Volume.from_name("bpd-training", create_if_missing=True)


# =============================================================================
# TRAINING
# =============================================================================


@app.function(
    image=image,
    gpu="A100",
    volumes={"/training": training_volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
def train_vae(
    project_id: str,
    job_id: str,
    config: dict,
) -> dict:
    """Train VAE model."""
    import os
    import pickle
    import sys
    import tarfile
    from pathlib import Path

    import numpy as np
    import torch
    import wandb

    sys.path.insert(0, "/app")

    hetzner_url = get_hetzner_url()
    headers = get_headers()
    callback = ProgressCallback(hetzner_url, job_id, headers, section="training")

    wandb_enabled = os.environ.get("WANDB_API_KEY") is not None
    if wandb_enabled:
        wandb.init(
            project="behavioral-pattern-discovery",
            name=f"train-{project_id[:8]}",
            config=config,
            tags=["modal", "cloud"],
        )

    try:
        callback.status("Downloading training data...")

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Download tarball (streaming)
            tar_path = tmpdir / "data.tar"
            download_file_streaming(
                f"{hetzner_url}/internal/projects/{project_id}/training-data",
                headers,
                tar_path,
                timeout=600,
            )

            # Extract tarball
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(tmpdir)

            # Decompress files
            train_input_zst = tmpdir / "train_input.npy.zst"
            train_input_npy = tmpdir / "train_input.npy"
            decompress_file(train_input_zst, train_input_npy)

            msg_db_zst = tmpdir / "message_database.pkl.zst"
            msg_db_pkl = tmpdir / "message_database.pkl"
            decompress_file(msg_db_zst, msg_db_pkl)

            # Load data
            train_input = np.load(train_input_npy)
            with open(msg_db_pkl, "rb") as f:
                message_db = pickle.load(f)

            callback.status("Initializing model...")

            from src.core.config import ModelDimensions
            from src.model.vae import MultiEncoderVAE
            from src.training.trainer import Trainer

            metadata = message_db.get("metadata", {})
            dims = ModelDimensions.from_config(config["model"], metadata)

            model = MultiEncoderVAE(config["model"], dims)
            model = model.to("cuda")

            callback.status("Starting training...")

            def on_epoch_end(epoch: int, metrics: dict, is_best: bool):
                callback("epoch", {
                    "epoch": epoch,
                    "metrics": {k: float(v) for k, v in metrics.items()},
                    "is_best": is_best,
                })
                if wandb_enabled:
                    wandb.log({k: float(v) for k, v in metrics.items()}, step=epoch)

            trainer = Trainer(
                model=model,
                config=config["training"],
                dims=dims,
                on_epoch_end=on_epoch_end,
                metadata=metadata,
            )

            checkpoint_dir = Path(f"/training/{project_id}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "best_model.pt"

            trainer.train(
                train_data=train_input,
                message_db=message_db,
                checkpoint_path=str(checkpoint_path),
            )

            training_volume.commit()

            callback.status("Uploading checkpoint...")

            # Compress and upload checkpoint (streaming)
            compress_and_upload(
                f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
                headers,
                checkpoint_path,
                timeout=600,
            )

            callback.completed("Training completed successfully")

            if wandb_enabled:
                wandb.finish()

            return {"status": "completed", "checkpoint_path": str(checkpoint_path)}

    except Exception as e:
        callback.failed(str(e))
        if wandb_enabled:
            wandb.finish(exit_code=1)
        raise


# =============================================================================
# BATCH SCORING
# =============================================================================


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
def batch_score(
    project_id: str,
    job_id: str,
    config: dict,
) -> dict:
    """Score all messages through trained VAE encoder."""
    import pickle
    import sys
    import tempfile
    from pathlib import Path

    import h5py
    import requests
    import torch

    sys.path.insert(0, "/app")

    hetzner_url = get_hetzner_url()
    headers = get_headers()
    callback = ProgressCallback(hetzner_url, job_id, headers, section="scoring")

    try:
        callback.status("Loading checkpoint...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Download and decompress checkpoint
            checkpoint_path = tmpdir / "best_model.pt"
            download_and_decompress(
                f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
                headers,
                checkpoint_path,
            )

            checkpoint = torch.load(checkpoint_path, map_location="cuda")

            from src.core.config import ModelDimensions
            from src.model.vae import MultiEncoderVAE

            model_config = checkpoint["config"]
            metadata = checkpoint["metadata"]
            dims = ModelDimensions.from_config(model_config, metadata)

            model = MultiEncoderVAE(model_config, dims)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to("cuda")
            model.eval()

            callback.status("Loading message data...")

            # Download and decompress message database
            msg_db_path = tmpdir / "message_database.pkl"
            download_and_decompress(
                f"{hetzner_url}/internal/projects/{project_id}/messages",
                headers,
                msg_db_path,
            )

            with open(msg_db_path, "rb") as f:
                message_db = pickle.load(f)

            callback.status("Scoring messages...")

            from src.pattern_identification.batch_scorer import BatchScorer

            scorer = BatchScorer(model, config)

            def progress_fn(p):
                callback.progress(p)

            activations, population_stats = scorer.score_all(
                message_db,
                progress_callback=progress_fn,
            )

            callback.status("Uploading results...")

            # Save activations to HDF5
            activations_path = tmpdir / "activations.h5"
            with h5py.File(activations_path, "w") as f:
                for key, value in activations.items():
                    f.create_dataset(key, data=value, compression="gzip")

            # Compress and upload activations
            compress_and_upload(
                f"{hetzner_url}/internal/projects/{project_id}/activations",
                headers,
                activations_path,
            )

            # Upload population stats (small JSON)
            requests.post(
                f"{hetzner_url}/internal/projects/{project_id}/population-stats",
                headers=headers,
                json=population_stats,
                timeout=30,
            ).raise_for_status()

            callback.completed("Batch scoring completed")

            return {"status": "completed", "num_messages": len(message_db["messages"])}

    except Exception as e:
        callback.failed(str(e))
        raise


# =============================================================================
# SHAP ANALYSIS
# =============================================================================


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
def shap_analyze(
    project_id: str,
    job_id: str,
    config: dict,
) -> dict:
    """Run SHAP analysis to extract hierarchical weights."""
    import sys
    import tempfile
    from pathlib import Path

    import h5py
    import torch

    sys.path.insert(0, "/app")

    hetzner_url = get_hetzner_url()
    headers = get_headers()
    callback = ProgressCallback(hetzner_url, job_id, headers, section="shap")

    try:
        callback.status("Loading checkpoint...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Download and decompress checkpoint
            checkpoint_path = tmpdir / "best_model.pt"
            download_and_decompress(
                f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
                headers,
                checkpoint_path,
            )

            checkpoint = torch.load(checkpoint_path, map_location="cuda")

            from src.core.config import ModelDimensions
            from src.model.vae import MultiEncoderVAE

            model_config = checkpoint["config"]
            metadata = checkpoint["metadata"]
            dims = ModelDimensions.from_config(model_config, metadata)

            model = MultiEncoderVAE(model_config, dims)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to("cuda")
            model.eval()

            callback.status("Loading activations...")

            # Download and decompress activations
            activations_path = tmpdir / "activations.h5"
            download_and_decompress(
                f"{hetzner_url}/internal/projects/{project_id}/activations",
                headers,
                activations_path,
            )

            activations = {}
            with h5py.File(activations_path, "r") as f:
                for key in f.keys():
                    activations[key] = f[key][:]

            callback.status("Running SHAP analysis...")

            from src.pattern_identification.shap_analysis import SHAPAnalyzer

            analyzer = SHAPAnalyzer(model, config)

            def progress_fn(p):
                callback.progress(p)

            hierarchical_weights = analyzer.extract_hierarchical_weights(
                activations,
                progress_callback=progress_fn,
            )

            callback.status("Uploading results...")

            upload_file(
                f"{hetzner_url}/internal/projects/{project_id}/shap-weights",
                headers,
                json_data=hierarchical_weights,
            )

            callback.completed("SHAP analysis completed")

            return {"status": "completed"}

    except Exception as e:
        callback.failed(str(e))
        raise


# =============================================================================
# INDIVIDUAL SCORING
# =============================================================================


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
def score_individual(
    project_id: str,
    engineer_id: str,
    messages: list[dict],
    population_stats: dict,
) -> dict:
    """Score a single engineer."""
    import sys
    import tempfile
    from pathlib import Path

    import torch

    sys.path.insert(0, "/app")

    hetzner_url = get_hetzner_url()
    headers = get_headers()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Download and decompress checkpoint
        checkpoint_path = tmpdir / "best_model.pt"
        download_and_decompress(
            f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
            headers,
            checkpoint_path,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        from src.core.config import ModelDimensions
        from src.model.vae import MultiEncoderVAE
        from src.scoring.individual_scorer import IndividualScorer

        model_config = checkpoint["config"]
        metadata = checkpoint["metadata"]
        dims = ModelDimensions.from_config(model_config, metadata)

        model = MultiEncoderVAE(model_config, dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to("cuda")
        model.eval()

        scorer = IndividualScorer(model, population_stats)
        scores = scorer.score_engineer(messages)

        return {
            "engineer_id": engineer_id,
            "scores": scores,
        }


# =============================================================================
# WARM SCORING SERVICE
# =============================================================================


@app.cls(
    image=image,
    gpu="A100",
    scaledown_window=60,
    timeout=300,
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
class ScoringService:
    """Warm service for individual scoring requests."""

    def __init__(self):
        self.models: dict = {}
        self.dims: dict = {}

    def _load_model(self, project_id: str):
        """Load and cache model for project."""
        import sys
        import tempfile
        from pathlib import Path

        import torch

        sys.path.insert(0, "/app")

        if project_id in self.models:
            return

        hetzner_url = get_hetzner_url()
        headers = get_headers()

        # Download and decompress to temp file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            checkpoint_path = Path(tmp.name)

        download_and_decompress(
            f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
            headers,
            checkpoint_path,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        checkpoint_path.unlink()

        from src.core.config import ModelDimensions
        from src.model.vae import MultiEncoderVAE

        model_config = checkpoint["config"]
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
    ) -> dict:
        """Score individual engineer with cached model."""
        import sys

        sys.path.insert(0, "/app")

        self._load_model(project_id)
        model = self.models[project_id]

        from src.scoring.individual_scorer import IndividualScorer

        scorer = IndividualScorer(model, population_stats)
        scores = scorer.score_engineer(messages)

        return {"engineer_id": engineer_id, "scores": scores}

    @modal.fastapi_endpoint(method="POST")
    def score_endpoint(self, request: dict) -> dict:
        """HTTP endpoint for scoring."""
        return self.score(
            project_id=request["project_id"],
            engineer_id=request["engineer_id"],
            messages=request["messages"],
            population_stats=request["population_stats"],
        )


# =============================================================================
# LOCAL TESTING
# =============================================================================


@app.local_entrypoint()
def test():
    """Test VAE functions locally."""
    print("VAE Modal app loaded successfully")
    print("Available functions: train_vae, batch_score, shap_analyze, score_individual")
    print("Available services: ScoringService")
