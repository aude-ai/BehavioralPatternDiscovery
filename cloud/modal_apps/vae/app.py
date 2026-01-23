"""Modal app for VAE training, scoring, and analysis."""
import modal

from cloud.modal_apps.common.config import create_ml_image, get_headers, get_hetzner_url
from cloud.modal_apps.common.data_transfer import (
    ProgressCallback,
    compress_numpy,
    decompress_numpy,
    download_file,
    upload_file,
)

app = modal.App("bpd-vae")

# Image with ML dependencies + source code
image = (
    create_ml_image()
    .pip_install(
        "pandas>=2.0.0",
    )
    .copy_local_dir("src", "/app/src")
    .copy_local_dir("config", "/app/config")
)

# Volume for training checkpoints
training_volume = modal.Volume.from_name("bpd-training", create_if_missing=True)


# =============================================================================
# TRAINING
# =============================================================================


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/training": training_volume},
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
def train_vae(
    project_id: str,
    job_id: str,
    config: dict,
) -> dict:
    """
    Train VAE model.

    Long-running function that downloads training data from Hetzner,
    trains the model, and uploads the checkpoint when complete.
    """
    import json
    import os
    import pickle
    import sys
    import tarfile
    import tempfile

    import numpy as np
    import torch

    sys.path.insert(0, "/app")

    hetzner_url = get_hetzner_url()
    headers = get_headers()
    callback = ProgressCallback(hetzner_url, job_id, headers)

    try:
        callback.status("Downloading training data...")

        # Download training data bundle
        data = download_file(
            f"{hetzner_url}/internal/projects/{project_id}/training-data",
            headers,
            timeout=600,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract tarball
            tar_path = os.path.join(tmpdir, "data.tar.gz")
            with open(tar_path, "wb") as f:
                f.write(data)

            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(tmpdir)

            # Load data
            train_input = np.load(os.path.join(tmpdir, "train_input.npy"))
            with open(os.path.join(tmpdir, "message_database.pkl"), "rb") as f:
                message_db = pickle.load(f)

            callback.status("Initializing model...")

            from src.core.config import ModelDimensions
            from src.model.vae import MultiEncoderVAE
            from src.training.trainer import Trainer

            # Build dimensions from data metadata
            metadata = message_db.get("metadata", {})
            dims = ModelDimensions.from_config(config["model"], metadata)

            # Create model
            model = MultiEncoderVAE(config["model"], dims)
            model = model.to("cuda")

            callback.status("Starting training...")

            # Training callback for epoch progress
            def on_epoch_end(epoch: int, metrics: dict, is_best: bool):
                callback("epoch", {
                    "epoch": epoch,
                    "metrics": {k: float(v) for k, v in metrics.items()},
                    "is_best": is_best,
                })

            # Create trainer
            trainer = Trainer(
                model=model,
                config=config["training"],
                dims=dims,
                on_epoch_end=on_epoch_end,
            )

            # Prepare checkpoint path
            checkpoint_dir = f"/training/{project_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

            # Train
            trainer.train(
                train_data=train_input,
                message_db=message_db,
                checkpoint_path=checkpoint_path,
            )

            # Commit checkpoint to volume
            training_volume.commit()

            callback.status("Uploading checkpoint...")

            # Upload checkpoint to Hetzner
            with open(checkpoint_path, "rb") as f:
                upload_file(
                    f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
                    headers,
                    files={"checkpoint": ("best_model.pt", f)},
                )

            callback.completed("Training completed successfully")

            return {"status": "completed", "checkpoint_path": checkpoint_path}

    except Exception as e:
        callback.failed(str(e))
        raise


# =============================================================================
# BATCH SCORING
# =============================================================================


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
def batch_score(
    project_id: str,
    job_id: str,
    config: dict,
) -> dict:
    """
    Score all messages through trained VAE encoder.

    Downloads checkpoint and message data, computes activations
    for all messages, and uploads results.
    """
    import io
    import json
    import pickle
    import sys

    import h5py
    import numpy as np
    import torch

    sys.path.insert(0, "/app")

    hetzner_url = get_hetzner_url()
    headers = get_headers()
    callback = ProgressCallback(hetzner_url, job_id, headers)

    try:
        callback.status("Loading checkpoint...")

        # Download checkpoint
        checkpoint_data = download_file(
            f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
            headers,
        )
        checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cuda")

        # Reconstruct model
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

        # Download message database
        msg_data = download_file(
            f"{hetzner_url}/internal/projects/{project_id}/messages",
            headers,
        )
        message_db = pickle.loads(msg_data)

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

        # Save activations to HDF5 in memory
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as f:
            for key, value in activations.items():
                f.create_dataset(key, data=value, compression="gzip")
        buffer.seek(0)

        upload_file(
            f"{hetzner_url}/internal/projects/{project_id}/activations",
            headers,
            files={"activations": ("activations.h5", buffer.read())},
            data={"population_stats": json.dumps(population_stats)},
        )

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
    gpu="A10G",
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
def shap_analyze(
    project_id: str,
    job_id: str,
    config: dict,
) -> dict:
    """
    Run SHAP analysis to extract hierarchical weights.

    Computes how patterns at each level compose into higher levels.
    """
    import io
    import json
    import sys

    import h5py
    import torch

    sys.path.insert(0, "/app")

    hetzner_url = get_hetzner_url()
    headers = get_headers()
    callback = ProgressCallback(hetzner_url, job_id, headers)

    try:
        callback.status("Loading checkpoint...")

        # Download checkpoint
        checkpoint_data = download_file(
            f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
            headers,
        )
        checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cuda")

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

        # Download activations
        act_data = download_file(
            f"{hetzner_url}/internal/projects/{project_id}/activations",
            headers,
        )

        activations = {}
        with h5py.File(io.BytesIO(act_data), "r") as f:
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
    gpu="T4",
    timeout=600,  # 10 minutes
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
def score_individual(
    project_id: str,
    engineer_id: str,
    messages: list[dict],
    population_stats: dict,
) -> dict:
    """
    Score a single engineer.

    Lighter weight function for on-demand individual scoring.
    """
    import io
    import sys

    import torch

    sys.path.insert(0, "/app")

    hetzner_url = get_hetzner_url()
    headers = get_headers()

    # Download checkpoint
    checkpoint_data = download_file(
        f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
        headers,
    )
    checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cuda")

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
# WARM SCORING SERVICE (for low-latency individual scoring)
# =============================================================================


@app.cls(
    image=image,
    gpu="T4",
    scaledown_window=600,
    timeout=300,
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
class ScoringService:
    """
    Warm service for individual scoring requests.

    Caches loaded models to reduce latency for repeated scoring.
    """

    def __init__(self):
        self.models: dict = {}
        self.dims: dict = {}

    def _load_model(self, project_id: str):
        """Load and cache model for project."""
        import io
        import sys

        import torch

        sys.path.insert(0, "/app")

        if project_id in self.models:
            return

        hetzner_url = get_hetzner_url()
        headers = get_headers()

        checkpoint_data = download_file(
            f"{hetzner_url}/internal/projects/{project_id}/checkpoint",
            headers,
        )
        checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cuda")

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
