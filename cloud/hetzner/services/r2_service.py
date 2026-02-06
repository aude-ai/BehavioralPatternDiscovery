"""
R2 Storage Service for Hetzner

Provides read-only access to R2 file status via HEAD requests.
Does NOT download files - only checks existence and metadata.
"""

import logging
import time
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from ..config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _load_cloud_config() -> dict:
    """Load cloud.yaml config (cached)."""
    import yaml

    config_path = Path(__file__).parent.parent.parent.parent / "config" / "cloud.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Cloud config not found at {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_r2_paths() -> dict[str, str]:
    """Load R2 paths from cloud.yaml config."""
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


def get_r2_file_paths() -> dict[str, str]:
    """Get R2 file paths from config."""
    return _load_r2_paths()


@lru_cache
def get_r2_client():
    """
    Get cached R2 client.

    Returns None if R2 credentials are not configured.
    """
    settings = get_settings()

    if not settings.r2_access_key_id or not settings.r2_endpoint_url:
        logger.warning("R2 credentials not configured - R2 status checks disabled")
        return None

    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


def get_r2_key(project_id: str, file_type: str) -> str:
    """Get full R2 key for a file type."""
    paths = get_r2_file_paths()
    if file_type not in paths:
        raise ValueError(f"Unknown file type: {file_type}. Valid types: {list(paths.keys())}")
    return paths[file_type].format(project_id=project_id)


def get_bucket_name() -> str:
    """Get R2 bucket name."""
    return get_settings().r2_bucket_name


def r2_file_exists(project_id: str, file_type: str) -> bool:
    """Check if a file exists in R2 with retry logic."""
    client = get_r2_client()
    if client is None:
        return False

    key = get_r2_key(project_id, file_type)
    retry_config = get_r2_retry_config()
    max_attempts = retry_config["max_attempts"]
    initial_delay = retry_config["initial_delay_seconds"]
    max_delay = retry_config["max_delay_seconds"]
    exponential_base = retry_config["exponential_base"]

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"R2 HEAD (exists): attempt {attempt}/{max_attempts} for {key}")
            client.head_object(Bucket=get_bucket_name(), Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            logger.warning(f"R2 HEAD: attempt {attempt}/{max_attempts} failed for {key}: {e}")

            if attempt < max_attempts:
                delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                logger.info(f"R2 HEAD: retrying in {delay:.1f}s...")
                time.sleep(delay)

    logger.error(f"R2 HEAD: failed to check {key} after {max_attempts} attempts")
    return False


def get_r2_file_info(project_id: str, file_type: str) -> dict:
    """
    Get metadata for an R2 file with retry logic.

    Returns:
        Dict with exists, size_bytes, last_modified (if exists)
    """
    client = get_r2_client()
    if client is None:
        return {"exists": False, "error": "R2 not configured"}

    key = get_r2_key(project_id, file_type)
    retry_config = get_r2_retry_config()
    max_attempts = retry_config["max_attempts"]
    initial_delay = retry_config["initial_delay_seconds"]
    max_delay = retry_config["max_delay_seconds"]
    exponential_base = retry_config["exponential_base"]

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"R2 HEAD (info): attempt {attempt}/{max_attempts} for {key}")
            response = client.head_object(Bucket=get_bucket_name(), Key=key)
            return {
                "exists": True,
                "size_bytes": response["ContentLength"],
                "last_modified": response["LastModified"].isoformat(),
                "r2_key": key,
            }
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return {"exists": False}
            logger.warning(f"R2 HEAD: attempt {attempt}/{max_attempts} failed for {key}: {e}")

            if attempt < max_attempts:
                delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                logger.info(f"R2 HEAD: retrying in {delay:.1f}s...")
                time.sleep(delay)

    logger.error(f"R2 HEAD: failed to get info for {key} after {max_attempts} attempts")
    return {"exists": False, "error": f"Failed after {max_attempts} attempts"}


def get_all_r2_file_info(project_id: str) -> dict[str, dict]:
    """
    Get metadata for all R2 files for a project.

    Returns:
        Dict mapping file_type to file info
    """
    paths = get_r2_file_paths()
    return {
        file_type: get_r2_file_info(project_id, file_type)
        for file_type in paths.keys()
    }


def delete_r2_file(project_id: str, file_type: str) -> bool:
    """
    Delete a single file from R2 with retry logic.

    R2 deletion errors are logged but do not block project deletion.
    Returns True if deleted (or didn't exist), False if error occurred.
    """
    client = get_r2_client()
    if client is None:
        logger.warning(f"R2 not configured - cannot delete {file_type}")
        return False

    key = get_r2_key(project_id, file_type)
    retry_config = get_r2_retry_config()
    max_attempts = retry_config["max_attempts"]
    initial_delay = retry_config["initial_delay_seconds"]
    max_delay = retry_config["max_delay_seconds"]
    exponential_base = retry_config["exponential_base"]

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"R2 DELETE: attempt {attempt}/{max_attempts} for {key}")
            client.delete_object(Bucket=get_bucket_name(), Key=key)
            logger.info(f"Deleted from R2: {key}")
            return True
        except ClientError as e:
            # 404 means it doesn't exist - that's fine for delete
            if e.response["Error"]["Code"] == "404":
                logger.info(f"R2 DELETE: {key} does not exist (already deleted)")
                return True
            logger.warning(f"R2 DELETE: attempt {attempt}/{max_attempts} failed for {key}: {e}")

            if attempt < max_attempts:
                delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                logger.info(f"R2 DELETE: retrying in {delay:.1f}s...")
                time.sleep(delay)
        except Exception as e:
            logger.warning(f"R2 DELETE: attempt {attempt}/{max_attempts} failed for {key}: {e}")

            if attempt < max_attempts:
                delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                logger.info(f"R2 DELETE: retrying in {delay:.1f}s...")
                time.sleep(delay)

    logger.error(f"R2 DELETE: failed to delete {key} after {max_attempts} attempts")
    return False


def delete_all_project_r2_files(project_id: str) -> dict[str, bool]:
    """
    Delete all R2 files for a project.

    Returns dict mapping file_type to success status.
    Logs each deletion. Errors don't block other deletions.
    """
    logger.info(f"Starting R2 cleanup for project {project_id}")
    paths = get_r2_file_paths()
    results = {}

    for file_type in paths.keys():
        results[file_type] = delete_r2_file(project_id, file_type)

    successful = sum(1 for v in results.values() if v)
    logger.info(
        f"R2 cleanup complete for project {project_id}: "
        f"{successful}/{len(results)} files deleted successfully"
    )

    return results


def _get_starting_points_config() -> dict:
    """Get starting points config from cloud.yaml."""
    import yaml

    config_path = Path(__file__).parent.parent.parent.parent / "config" / "cloud.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Cloud config not found at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "pipeline" not in config:
        raise ValueError("Missing required config section: pipeline in cloud.yaml")
    if "starting_points" not in config["pipeline"]:
        raise ValueError("Missing required config section: pipeline.starting_points in cloud.yaml")

    return config["pipeline"]["starting_points"]


def validate_prerequisites(project_id: str, starting_step: str) -> tuple[bool, str]:
    """
    Validate that R2 prerequisites exist for a starting step.

    Args:
        project_id: Project identifier
        starting_step: Step to start from (B.1, B.5, B.6, B.8)

    Returns:
        (valid, error_message)
    """
    starting_points = _get_starting_points_config()

    if starting_step not in starting_points:
        return False, f"Unknown starting step: {starting_step}"

    r2_prereqs = starting_points[starting_step].get("r2_prerequisites", [])

    for file_type in r2_prereqs:
        if not r2_file_exists(project_id, file_type):
            return False, f"Missing R2 file: {file_type}"

    return True, ""


def download_json_from_r2(project_id: str, file_type: str) -> dict:
    """
    Download JSON data from R2 with retry logic.

    Args:
        project_id: Project identifier
        file_type: Type of file (determines R2 path)

    Returns:
        Parsed JSON as dict

    Raises:
        Exception: If download fails after retries
    """
    import json

    client = get_r2_client()
    if client is None:
        raise RuntimeError("R2 not configured")

    key = get_r2_key(project_id, file_type)
    retry_config = get_r2_retry_config()
    max_attempts = retry_config["max_attempts"]
    initial_delay = retry_config["initial_delay_seconds"]
    max_delay = retry_config["max_delay_seconds"]
    exponential_base = retry_config["exponential_base"]

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"R2 GET (json): attempt {attempt}/{max_attempts} for {key}")
            response = client.get_object(Bucket=get_bucket_name(), Key=key)
            json_bytes = response["Body"].read()
            logger.info(f"Downloaded {file_type} from R2: {key} ({len(json_bytes):,} bytes)")
            return json.loads(json_bytes.decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(f"File not found in R2: {key}")
            logger.warning(f"R2 GET: attempt {attempt}/{max_attempts} failed for {key}: {e}")

            if attempt < max_attempts:
                delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                logger.info(f"R2 GET: retrying in {delay:.1f}s...")
                time.sleep(delay)

    raise RuntimeError(f"Failed to download {key} from R2 after {max_attempts} attempts")


def download_pickle_from_r2(project_id: str, file_type: str) -> list:
    """
    Download pickle data (zstd compressed) from R2 with retry logic.

    Args:
        project_id: Project identifier
        file_type: Type of file (determines R2 path)

    Returns:
        Unpickled data (typically a list)

    Raises:
        Exception: If download fails after retries
    """
    import pickle
    import zstandard as zstd

    client = get_r2_client()
    if client is None:
        raise RuntimeError("R2 not configured")

    key = get_r2_key(project_id, file_type)
    retry_config = get_r2_retry_config()
    max_attempts = retry_config["max_attempts"]
    initial_delay = retry_config["initial_delay_seconds"]
    max_delay = retry_config["max_delay_seconds"]
    exponential_base = retry_config["exponential_base"]

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"R2 GET (pickle): attempt {attempt}/{max_attempts} for {key}")
            response = client.get_object(Bucket=get_bucket_name(), Key=key)
            compressed_bytes = response["Body"].read()
            decompressor = zstd.ZstdDecompressor()
            pickle_bytes = decompressor.decompress(compressed_bytes)
            logger.info(f"Downloaded {file_type} from R2: {key} ({len(compressed_bytes):,} bytes compressed)")
            return pickle.loads(pickle_bytes)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(f"File not found in R2: {key}")
            logger.warning(f"R2 GET: attempt {attempt}/{max_attempts} failed for {key}: {e}")

            if attempt < max_attempts:
                delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
                logger.info(f"R2 GET: retrying in {delay:.1f}s...")
                time.sleep(delay)

    raise RuntimeError(f"Failed to download {key} from R2 after {max_attempts} attempts")
