"""
R2 Storage Utilities

Provides upload/download functions for Cloudflare R2 object storage.
Used by Modal functions to persist large files between pipeline steps.
"""

import io
import logging
import os
import pickle
import tempfile
import time
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import boto3
import numpy as np
import zstandard as zstd
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Compression level for zstd (3 = good balance of speed/ratio)
ZSTD_LEVEL = 3

# Multipart upload threshold (50MB)
MULTIPART_THRESHOLD = 50 * 1024 * 1024

T = TypeVar("T")


@lru_cache
def _load_cloud_config() -> dict:
    """Load cloud.yaml config (cached)."""
    import yaml

    config_path = Path("/app/config/cloud.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Cloud config not found at {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_r2_paths_config() -> dict[str, str]:
    """Get R2 paths from cloud.yaml config."""
    config = _load_cloud_config()

    if "r2" not in config:
        raise ValueError("Missing required config section: r2 in cloud.yaml")
    if "paths" not in config["r2"]:
        raise ValueError("Missing required config section: r2.paths in cloud.yaml")

    return config["r2"]["paths"]


def get_r2_retry_config() -> dict:
    """Get R2 retry config from cloud.yaml."""
    config = _load_cloud_config()

    if "r2" not in config:
        raise ValueError("Missing required config section: r2 in cloud.yaml")
    if "retry" not in config["r2"]:
        raise ValueError("Missing required config section: r2.retry in cloud.yaml")

    return config["r2"]["retry"]


def with_r2_retry(operation_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that adds retry logic with logging to R2 operations.

    Uses retry config from cloud.yaml with exponential backoff.
    Logs each attempt with attempt number and any errors.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            retry_config = get_r2_retry_config()
            max_attempts = retry_config["max_attempts"]
            initial_delay = retry_config["initial_delay_seconds"]
            max_delay = retry_config["max_delay_seconds"]
            exponential_base = retry_config["exponential_base"]

            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.info(f"R2 {operation_name}: attempt {attempt}/{max_attempts}")
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(f"R2 {operation_name}: succeeded on attempt {attempt}")
                    return result

                except ClientError as e:
                    last_exception = e
                    error_code = e.response.get("Error", {}).get("Code", "Unknown")

                    # Don't retry 404 errors for existence checks
                    if error_code == "404":
                        raise

                    logger.warning(
                        f"R2 {operation_name}: attempt {attempt}/{max_attempts} failed "
                        f"(error: {error_code}): {e}"
                    )

                    if attempt < max_attempts:
                        delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                        logger.info(f"R2 {operation_name}: retrying in {delay:.1f}s...")
                        time.sleep(delay)

                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"R2 {operation_name}: attempt {attempt}/{max_attempts} failed: {e}"
                    )

                    if attempt < max_attempts:
                        delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                        logger.info(f"R2 {operation_name}: retrying in {delay:.1f}s...")
                        time.sleep(delay)

            logger.error(f"R2 {operation_name}: failed after {max_attempts} attempts")
            raise last_exception

        return wrapper
    return decorator


def get_r2_client():
    """Create R2 client from environment variables."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def get_bucket_name() -> str:
    """Get R2 bucket name from environment."""
    return os.environ["R2_BUCKET_NAME"]


def get_r2_key(project_id: str, file_type: str) -> str:
    """
    Get full R2 key for a file type.

    Args:
        project_id: Project identifier
        file_type: One of: embeddings, aux_features, normalized_embeddings,
                   train_input, message_database, checkpoint, activations

    Returns:
        Full R2 key path
    """
    paths = get_r2_paths_config()

    if file_type not in paths:
        raise ValueError(f"Unknown file type: {file_type}. Valid types: {list(paths.keys())}")

    # Replace {project_id} placeholder in path template
    return paths[file_type].format(project_id=project_id)


@with_r2_retry("HEAD (exists check)")
def r2_file_exists(project_id: str, file_type: str) -> bool:
    """Check if a file exists in R2."""
    r2 = get_r2_client()
    key = get_r2_key(project_id, file_type)

    try:
        r2.head_object(Bucket=get_bucket_name(), Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


@with_r2_retry("HEAD (file info)")
def get_r2_file_info(project_id: str, file_type: str) -> dict | None:
    """
    Get metadata for an R2 file.

    Returns:
        Dict with size_bytes, last_modified, or None if not found
    """
    r2 = get_r2_client()
    key = get_r2_key(project_id, file_type)

    try:
        response = r2.head_object(Bucket=get_bucket_name(), Key=key)
        return {
            "exists": True,
            "size_bytes": response["ContentLength"],
            "last_modified": response["LastModified"].isoformat(),
            "r2_key": key,
        }
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return {"exists": False}
        raise


def get_all_r2_file_info(project_id: str) -> dict[str, dict]:
    """Get metadata for all R2 files for a project."""
    file_types = [
        "embeddings",
        "aux_features",
        "normalized_embeddings",
        "train_input",
        "message_database",
        "checkpoint",
        "activations",
    ]

    return {ft: get_r2_file_info(project_id, ft) for ft in file_types}


# =============================================================================
# UPLOAD FUNCTIONS
# =============================================================================


@with_r2_retry("PUT (numpy)")
def upload_numpy_to_r2(
    arr: np.ndarray,
    project_id: str,
    file_type: str,
    compress: bool = True,
) -> dict:
    """
    Upload numpy array to R2 with optional compression.

    Args:
        arr: Numpy array to upload
        project_id: Project identifier
        file_type: Type of file (determines R2 path)
        compress: Whether to zstd compress (default True)

    Returns:
        Dict with upload info (size_bytes, r2_key)
    """
    r2 = get_r2_client()
    key = get_r2_key(project_id, file_type)

    # Serialize to bytes
    buffer = io.BytesIO()
    np.save(buffer, arr)
    data = buffer.getvalue()

    # Compress if requested
    if compress:
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        data = cctx.compress(data)

    # Upload
    r2.put_object(
        Bucket=get_bucket_name(),
        Key=key,
        Body=data,
    )

    logger.info(f"Uploaded {file_type} to R2: {key} ({len(data):,} bytes)")

    return {
        "size_bytes": len(data),
        "r2_key": key,
    }


@with_r2_retry("PUT (pickle)")
def upload_pickle_to_r2(
    obj: Any,
    project_id: str,
    file_type: str,
    compress: bool = True,
) -> dict:
    """
    Upload pickled object to R2 with optional compression.

    Args:
        obj: Python object to pickle and upload
        project_id: Project identifier
        file_type: Type of file (determines R2 path)
        compress: Whether to zstd compress (default True)

    Returns:
        Dict with upload info (size_bytes, r2_key)
    """
    r2 = get_r2_client()
    key = get_r2_key(project_id, file_type)

    # Serialize
    data = pickle.dumps(obj)

    # Compress if requested
    if compress:
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        data = cctx.compress(data)

    # Upload
    r2.put_object(
        Bucket=get_bucket_name(),
        Key=key,
        Body=data,
    )

    logger.info(f"Uploaded {file_type} to R2: {key} ({len(data):,} bytes)")

    return {
        "size_bytes": len(data),
        "r2_key": key,
    }


@with_r2_retry("PUT (file)")
def upload_file_to_r2(
    local_path: Path,
    project_id: str,
    file_type: str,
    compress: bool = True,
) -> dict:
    """
    Upload a local file to R2 with optional compression.

    For large files, uses multipart upload.

    Args:
        local_path: Path to local file
        project_id: Project identifier
        file_type: Type of file (determines R2 path)
        compress: Whether to zstd compress (default True)

    Returns:
        Dict with upload info (size_bytes, r2_key)
    """
    from boto3.s3.transfer import TransferConfig

    r2 = get_r2_client()
    key = get_r2_key(project_id, file_type)

    if compress:
        # Compress to temp file first
        with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
            with open(local_path, "rb") as f_in:
                with open(tmp_path, "wb") as f_out:
                    cctx.copy_stream(f_in, f_out)

            file_size = tmp_path.stat().st_size

            # Use multipart for large files
            config = TransferConfig(multipart_threshold=MULTIPART_THRESHOLD)
            r2.upload_file(str(tmp_path), get_bucket_name(), key, Config=config)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        file_size = local_path.stat().st_size
        config = TransferConfig(multipart_threshold=MULTIPART_THRESHOLD)
        r2.upload_file(str(local_path), get_bucket_name(), key, Config=config)

    logger.info(f"Uploaded {file_type} to R2: {key} ({file_size:,} bytes)")

    return {
        "size_bytes": file_size,
        "r2_key": key,
    }


def upload_h5_to_r2(
    local_path: Path,
    project_id: str,
    file_type: str = "activations",
) -> dict:
    """
    Upload HDF5 file to R2 with compression.

    Specialized for activations.h5 files.
    """
    return upload_file_to_r2(local_path, project_id, file_type, compress=True)


def upload_checkpoint_to_r2(
    local_path: Path,
    project_id: str,
) -> dict:
    """
    Upload PyTorch checkpoint to R2 with compression.
    """
    return upload_file_to_r2(local_path, project_id, "checkpoint", compress=True)


@with_r2_retry("PUT (json)")
def upload_json_to_r2(
    data: dict,
    project_id: str,
    file_type: str,
) -> dict:
    """
    Upload JSON data to R2 (no compression for small JSON files).

    Args:
        data: Dictionary to upload as JSON
        project_id: Project identifier
        file_type: Type of file (determines R2 path)

    Returns:
        Dict with upload info (size_bytes, r2_key)
    """
    import json

    r2 = get_r2_client()
    key = get_r2_key(project_id, file_type)

    json_bytes = json.dumps(data, indent=2).encode("utf-8")
    r2.put_object(Bucket=get_bucket_name(), Key=key, Body=json_bytes)

    logger.info(f"Uploaded {file_type} to R2: {key} ({len(json_bytes):,} bytes)")

    return {
        "size_bytes": len(json_bytes),
        "r2_key": key,
    }


@with_r2_retry("GET (json)")
def download_json_from_r2(
    project_id: str,
    file_type: str,
    parent_id: str | None = None,
    owned_files: dict | None = None,
) -> dict:
    """
    Download JSON data from R2.

    Args:
        project_id: Project identifier
        file_type: Type of file (determines R2 path)
        parent_id: Parent project ID for inheritance resolution
        owned_files: Dict of {file_key: bool} for inheritance resolution

    Returns:
        Parsed JSON as dict
    """
    import json

    r2 = get_r2_client()
    source_project = resolve_r2_project_id(project_id, file_type, parent_id, owned_files)
    key = get_r2_key(source_project, file_type)

    response = r2.get_object(Bucket=get_bucket_name(), Key=key)
    json_bytes = response["Body"].read()

    logger.info(f"Downloaded {file_type} from R2: {key} ({len(json_bytes):,} bytes)")

    return json.loads(json_bytes.decode("utf-8"))


# =============================================================================
# INHERITANCE RESOLUTION (Phase 5 - Variants)
# =============================================================================


def resolve_r2_project_id(
    project_id: str,
    file_type: str,
    parent_id: str | None = None,
    owned_files: dict | None = None,
) -> str:
    """
    Resolve which project ID to use for downloading a file.

    For variants, checks owned_files to determine if file should come
    from this project or from parent.

    Args:
        project_id: Current project ID
        file_type: Type of file to download
        parent_id: Parent project ID (if this is a variant)
        owned_files: Dict of {file_key: bool} indicating ownership

    Returns:
        Project ID to download from
    """
    owned_files = owned_files or {}

    # If this project owns the file, use its path
    if owned_files.get(file_type, True):  # Default to True for non-variants
        return project_id

    # Not owned - must have a parent
    if parent_id:
        logger.info(f"Resolving {file_type} from parent project {parent_id}")
        return parent_id

    # No parent and not owned - error
    raise FileNotFoundError(
        f"File {file_type} not owned by project {project_id} and no parent to inherit from"
    )


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================


@with_r2_retry("GET (numpy)")
def download_numpy_from_r2(
    project_id: str,
    file_type: str,
    decompress: bool = True,
    parent_id: str | None = None,
    owned_files: dict | None = None,
) -> np.ndarray:
    """
    Download numpy array from R2.

    Args:
        project_id: Project identifier
        file_type: Type of file
        decompress: Whether to zstd decompress (default True)
        parent_id: Parent project ID for inheritance resolution
        owned_files: Dict of {file_key: bool} for inheritance resolution

    Returns:
        Numpy array
    """
    r2 = get_r2_client()
    source_project = resolve_r2_project_id(project_id, file_type, parent_id, owned_files)
    key = get_r2_key(source_project, file_type)

    response = r2.get_object(Bucket=get_bucket_name(), Key=key)
    data = response["Body"].read()

    if decompress:
        dctx = zstd.ZstdDecompressor()
        data = dctx.decompress(data)

    buffer = io.BytesIO(data)
    arr = np.load(buffer)

    logger.info(f"Downloaded {file_type} from R2: {key} (shape: {arr.shape})")

    return arr


@with_r2_retry("GET (pickle)")
def download_pickle_from_r2(
    project_id: str,
    file_type: str,
    decompress: bool = True,
    parent_id: str | None = None,
    owned_files: dict | None = None,
) -> Any:
    """
    Download pickled object from R2.

    Args:
        project_id: Project identifier
        file_type: Type of file
        decompress: Whether to zstd decompress (default True)
        parent_id: Parent project ID for inheritance resolution
        owned_files: Dict of {file_key: bool} for inheritance resolution

    Returns:
        Unpickled Python object
    """
    r2 = get_r2_client()
    source_project = resolve_r2_project_id(project_id, file_type, parent_id, owned_files)
    key = get_r2_key(source_project, file_type)

    response = r2.get_object(Bucket=get_bucket_name(), Key=key)
    data = response["Body"].read()

    if decompress:
        dctx = zstd.ZstdDecompressor()
        data = dctx.decompress(data)

    obj = pickle.loads(data)

    logger.info(f"Downloaded {file_type} from R2: {key}")

    return obj


@with_r2_retry("GET (file)")
def download_file_from_r2(
    project_id: str,
    file_type: str,
    local_path: Path,
    decompress: bool = True,
    parent_id: str | None = None,
    owned_files: dict | None = None,
) -> Path:
    """
    Download file from R2 to local path.

    Args:
        project_id: Project identifier
        file_type: Type of file
        local_path: Where to save the file
        decompress: Whether to zstd decompress (default True)
        parent_id: Parent project ID for inheritance resolution
        owned_files: Dict of {file_key: bool} for inheritance resolution

    Returns:
        Path to downloaded file
    """
    r2 = get_r2_client()
    source_project = resolve_r2_project_id(project_id, file_type, parent_id, owned_files)
    key = get_r2_key(source_project, file_type)

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if decompress:
        # Download to temp, decompress to target
        with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            r2.download_file(get_bucket_name(), key, str(tmp_path))

            dctx = zstd.ZstdDecompressor()
            with open(tmp_path, "rb") as f_in:
                with open(local_path, "wb") as f_out:
                    dctx.copy_stream(f_in, f_out)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        r2.download_file(get_bucket_name(), key, str(local_path))

    logger.info(f"Downloaded {file_type} from R2: {key} -> {local_path}")

    return local_path


def download_checkpoint_from_r2(
    project_id: str,
    local_path: Path,
    parent_id: str | None = None,
    owned_files: dict | None = None,
) -> Path:
    """
    Download PyTorch checkpoint from R2.
    """
    return download_file_from_r2(
        project_id, "checkpoint", local_path, decompress=True,
        parent_id=parent_id, owned_files=owned_files
    )


def download_h5_from_r2(
    project_id: str,
    local_path: Path,
    file_type: str = "activations",
    parent_id: str | None = None,
    owned_files: dict | None = None,
) -> Path:
    """
    Download HDF5 file from R2.
    """
    return download_file_from_r2(
        project_id, file_type, local_path, decompress=True,
        parent_id=parent_id, owned_files=owned_files
    )


# =============================================================================
# DELETE FUNCTIONS
# =============================================================================


def delete_r2_file(project_id: str, file_type: str) -> bool:
    """
    Delete a single file from R2.

    Uses retry logic but returns False on final failure (doesn't raise).
    This allows project deletion to continue even if some R2 files fail to delete.
    """
    r2 = get_r2_client()
    key = get_r2_key(project_id, file_type)

    retry_config = get_r2_retry_config()
    max_attempts = retry_config["max_attempts"]
    initial_delay = retry_config["initial_delay_seconds"]
    max_delay = retry_config["max_delay_seconds"]
    exponential_base = retry_config["exponential_base"]

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"R2 DELETE: attempt {attempt}/{max_attempts} for {key}")
            r2.delete_object(Bucket=get_bucket_name(), Key=key)
            logger.info(f"Deleted {file_type} from R2: {key}")
            return True
        except Exception as e:
            logger.warning(f"R2 DELETE: attempt {attempt}/{max_attempts} failed for {key}: {e}")

            if attempt < max_attempts:
                delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                logger.info(f"R2 DELETE: retrying in {delay:.1f}s...")
                time.sleep(delay)

    logger.error(f"R2 DELETE: failed to delete {key} after {max_attempts} attempts")
    return False


def delete_all_project_files(project_id: str) -> dict[str, bool]:
    """
    Delete all R2 files for a project.

    Returns:
        Dict mapping file_type to success status
    """
    file_types = [
        "embeddings",
        "aux_features",
        "normalized_embeddings",
        "train_input",
        "message_database",
        "checkpoint",
        "activations",
    ]

    results = {}
    for ft in file_types:
        results[ft] = delete_r2_file(project_id, ft)

    return results


# =============================================================================
# VALIDATION
# =============================================================================


def _get_starting_points_config() -> dict:
    """Get starting points config from cloud.yaml."""
    config = _load_cloud_config()

    if "pipeline" not in config:
        raise ValueError("Missing required config section: pipeline in cloud.yaml")
    if "starting_points" not in config["pipeline"]:
        raise ValueError("Missing required config section: pipeline.starting_points in cloud.yaml")

    return config["pipeline"]["starting_points"]


def validate_prerequisites(
    project_id: str,
    starting_step: str,
    hetzner_check_fn=None,
    parent_id: str | None = None,
    owned_files: dict | None = None,
) -> tuple[bool, str]:
    """
    Validate that all prerequisites exist for a starting step.

    Supports inheritance: checks parent project for files not owned by this project.

    Args:
        project_id: Project identifier
        starting_step: Step to start from (e.g., "B.1", "B.5")
        hetzner_check_fn: Optional function to check Hetzner files
        parent_id: Parent project ID for inheritance resolution
        owned_files: Dict of {file_key: bool} for inheritance resolution

    Returns:
        (valid, error_message)
    """
    starting_points = _get_starting_points_config()

    if starting_step not in starting_points:
        return False, f"Unknown starting step: {starting_step}"

    point_config = starting_points[starting_step]
    r2_prereqs = point_config.get("r2_prerequisites", [])
    hetzner_prereqs = point_config.get("hetzner_prerequisites", [])

    owned_files = owned_files or {}

    # Check R2 files (with inheritance)
    for file_type in r2_prereqs:
        # Determine which project should have this file
        source_project = project_id
        if not owned_files.get(file_type, True) and parent_id:
            source_project = parent_id

        if not r2_file_exists(source_project, file_type):
            return False, f"Missing R2 file: {file_type} (checked project {source_project})"

    # Check Hetzner files (if check function provided)
    if hetzner_check_fn:
        for hetzner_file in hetzner_prereqs:
            if not hetzner_check_fn(project_id, hetzner_file):
                return False, f"Missing Hetzner file: {hetzner_file}"

    return True, ""
