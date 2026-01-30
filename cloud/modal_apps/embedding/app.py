"""Modal app for text embedding using transformer models."""
import modal

from cloud.modal_apps.common.config import create_embedding_image, get_headers, get_hetzner_url
from cloud.modal_apps.common.data_transfer import (
    ProgressCallback,
    compress_and_upload,
)

app = modal.App("bpd-embedding")

image = create_embedding_image()

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
    """Service for encoding text to embeddings."""

    model_name: str = "jinaai/jina-embeddings-v3"

    @modal.enter()
    def load_model(self):
        """Load embedding model when container starts."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        cache_dir = "/cache/models"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        model_cache.commit()

    @modal.method()
    def embed_batch(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
    ) -> dict:
        """Embed a batch of texts."""
        import torch

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                task=task,
                max_length=8192,
            )

        return {
            "embeddings": embeddings.tolist(),
            "embedding_dim": embeddings.shape[1],
        }

    @modal.fastapi_endpoint(method="POST")
    def embed_endpoint(self, request: dict) -> dict:
        """HTTP endpoint for embedding."""
        texts = request.get("texts", [])
        task = request.get("task", "retrieval.passage")

        if not texts:
            return {"error": "No texts provided"}

        return self.embed_batch(texts, task)


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
    task: str = "retrieval.passage",
    batch_size: int = 32,
) -> dict:
    """Embed all texts for a project."""
    import tempfile
    from pathlib import Path

    import numpy as np

    hetzner_url = get_hetzner_url()
    headers = get_headers()
    callback = ProgressCallback(hetzner_url, job_id, headers)

    try:
        callback.status("Initializing embedding model...")

        service = EmbeddingService()
        all_embeddings = []
        total = len(texts)

        callback.status(f"Encoding {total} texts...")

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            result = service.embed_batch.remote(batch, task)
            all_embeddings.extend(result["embeddings"])

            progress = min((i + len(batch)) / total, 1.0)
            callback.progress(progress, processed=i + len(batch), total=total)

        # Save to temp file
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        embedding_dim = embeddings_array.shape[1]

        callback.status("Uploading embeddings...")

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            np.save(tmp_path, embeddings_array)

        try:
            # Compress and upload (streaming)
            compress_and_upload(
                f"{hetzner_url}/internal/projects/{project_id}/embeddings",
                headers,
                tmp_path,
                filename="embeddings.npy.zst",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        callback.completed(f"Encoded {total} texts to {embedding_dim}-dim embeddings")

        return {
            "status": "completed",
            "num_texts": total,
            "embedding_dim": embedding_dim,
        }

    except Exception as e:
        callback.failed(str(e))
        raise


@app.local_entrypoint()
def test_embed():
    """Test embedding locally."""
    texts = ["Hello world", "This is a test"]
    service = EmbeddingService()
    result = service.embed_batch.remote(texts)
    print(f"Embedded {len(texts)} texts to {result['embedding_dim']} dimensions")
