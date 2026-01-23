"""Configuration for Modal apps."""
import os
import modal


def get_hetzner_url() -> str:
    """Get Hetzner base URL from environment (includes /bpd prefix)."""
    return os.environ.get("HETZNER_BASE_URL", "https://yourdomain.com/bpd")


def get_internal_key() -> str:
    """Get internal API key from Modal secret."""
    return os.environ["HETZNER_INTERNAL_KEY"]


def get_headers() -> dict:
    """Get headers for Hetzner API calls."""
    return {"X-Internal-Key": get_internal_key()}


# Shared Modal configurations
CUDA_VERSION = "12.1.0"
PYTHON_VERSION = "3.11"


def create_base_image():
    """Create base Modal image with PyTorch and common dependencies."""
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(
            "torch==2.2.0",
            "numpy>=1.24.0,<2.0.0",
            "requests>=2.31.0",
            "pyyaml>=6.0",
            "fastapi[standard]",
        )
    )


def create_ml_image():
    """Create image with ML dependencies for VAE training."""
    return (
        create_base_image()
        .pip_install(
            "h5py>=3.9.0",
            "shap>=0.44.0",
            "scikit-learn>=1.3.0",
        )
    )


def create_embedding_image():
    """Create image with embedding model dependencies."""
    return (
        create_base_image()
        .pip_install(
            "transformers>=4.36.0",
            "einops>=0.7.0",
            "sentence-transformers>=2.2.0",
        )
    )
