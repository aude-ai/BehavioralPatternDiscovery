"""Modal app for text embedding using transformer models."""
import modal

from cloud.modal_apps.common.config import create_embedding_image, get_headers, get_hetzner_url
from cloud.modal_apps.common.data_transfer import (
    ProgressCallback,
    compress_and_upload,
)

app = modal.App("bpd-embedding")

image = (
    create_embedding_image()
    .add_local_dir("src", "/app/src")
    .add_local_dir("config", "/app/config")
)

model_cache = modal.Volume.from_name("bpd-model-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A100",
    volumes={"/cache": model_cache},
    scaledown_window=60,
    timeout=3600,
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
class EmbeddingService:
    """Config-driven embedding service supporting multiple encoder types."""

    def __init__(self):
        self._encoder = None
        self._config_hash = None

    @modal.enter()
    def setup(self):
        """Set up cache directories."""
        import os
        os.environ["HF_HOME"] = "/cache/models"
        os.environ["TRANSFORMERS_CACHE"] = "/cache/models"

    def _config_signature(self, config: dict) -> str:
        """Generate a signature for config to detect changes."""
        import json
        encoder_config = config.get("processing", {}).get("text_encoder", {})
        return json.dumps(encoder_config, sort_keys=True)

    def _ensure_encoder(self, config: dict):
        """Lazy-load encoder based on config."""
        import sys
        sys.path.insert(0, "/app")

        config_hash = self._config_signature(config)
        if self._encoder is not None and self._config_hash == config_hash:
            return

        from src.data.processing.encoders import create_text_encoder

        encoder_type = config["processing"]["text_encoder"]["type"]
        print(f"Loading encoder: {encoder_type}")

        self._encoder = create_text_encoder(config)
        self._config_hash = config_hash

        model_cache.commit()
        print(f"Encoder loaded: {self._encoder.model_name}, dim={self._encoder.embedding_dim}")

    @modal.method()
    def embed_batch(
        self,
        texts: list[str],
        config: dict,
    ) -> dict:
        """Embed a batch of texts using config-specified encoder."""
        self._ensure_encoder(config)

        embeddings = self._encoder.encode(texts)

        return {
            "embeddings": embeddings.tolist(),
            "embedding_dim": embeddings.shape[1],
            "encoder_type": config["processing"]["text_encoder"]["type"],
            "model_name": self._encoder.model_name,
        }

    @modal.fastapi_endpoint(method="POST")
    def embed_endpoint(self, request: dict) -> dict:
        """HTTP endpoint for embedding."""
        texts = request.get("texts", [])
        config = request.get("config", {})

        if not texts:
            return {"error": "No texts provided"}
        if not config:
            return {"error": "No config provided"}

        return self.embed_batch(texts, config)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/cache": model_cache},
    timeout=7200,
    secrets=[modal.Secret.from_name("hetzner-internal-key")],
)
def embed_all_texts(
    project_id: str,
    job_id: str,
    texts: list[str],
    config: dict,
) -> dict:
    """Embed all texts for a project using config-specified encoder."""
    import os
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, "/app")

    # Set cache directories
    os.environ["HF_HOME"] = "/cache/models"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/models"

    hetzner_url = get_hetzner_url()
    headers = get_headers()
    callback = ProgressCallback(hetzner_url, job_id, headers, section="embedding")

    try:
        encoder_config = config["processing"]["text_encoder"]
        encoder_type = encoder_config["type"]
        encoder_settings = encoder_config.get(encoder_type, {})
        batch_size = encoder_settings.get("batch_size", 32)

        callback.status(f"Initializing {encoder_type} embedding model...")

        # Create encoder directly using the registry
        from src.data.processing.encoders import create_text_encoder

        encoder = create_text_encoder(config)
        model_name = encoder.model_name
        embedding_dim = encoder.embedding_dim

        model_cache.commit()

        callback.status(f"Encoding {len(texts)} texts with {encoder_type} (dim={embedding_dim})...")

        # Encode all texts in batches
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = encoder.encode(batch)
            all_embeddings.append(batch_embeddings)

            progress = min((i + len(batch)) / total, 1.0)
            callback.progress(progress, processed=i + len(batch), total=total)

        embeddings_array = np.concatenate(all_embeddings, axis=0)

        callback.status("Uploading embeddings...")

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            np.save(tmp_path, embeddings_array)

        try:
            compress_and_upload(
                f"{hetzner_url}/internal/projects/{project_id}/embeddings",
                headers,
                tmp_path,
                filename="embeddings.npy.zst",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        callback.completed(
            f"Encoded {total} texts to {embedding_dim}-dim embeddings using {encoder_type}"
        )

        return {
            "status": "completed",
            "num_texts": total,
            "embedding_dim": embedding_dim,
            "encoder_type": encoder_type,
            "model_name": model_name,
        }

    except Exception as e:
        callback.failed(str(e))
        raise


@app.local_entrypoint()
def test_embed():
    """Test embedding locally."""
    print("Embedding Modal app loaded successfully")
    print("Requires config with processing.text_encoder section")
    print("Supported encoders: jina, jina_v3, jina_v4, nv_embed, qwen_raw, qwen3_embedding")
