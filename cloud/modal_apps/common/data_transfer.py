"""Data transfer utilities for Modal <-> Hetzner communication."""
import gzip
import io
import logging
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)


def compress_numpy(arr: np.ndarray) -> bytes:
    """Compress numpy array for network transfer."""
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return gzip.compress(buffer.read())


def decompress_numpy(data: bytes) -> np.ndarray:
    """Decompress numpy array from network transfer."""
    decompressed = gzip.decompress(data)
    buffer = io.BytesIO(decompressed)
    return np.load(buffer)


def download_file(url: str, headers: dict, timeout: int = 300) -> bytes:
    """Download file from Hetzner with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
    raise RuntimeError("Download failed after all retries")


def upload_file(
    url: str,
    headers: dict,
    files: dict | None = None,
    data: dict | None = None,
    json_data: dict | None = None,
    timeout: int = 600,
) -> dict:
    """Upload file to Hetzner."""
    response = requests.post(
        url,
        headers=headers,
        files=files,
        data=data,
        json=json_data,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def send_callback(
    base_url: str,
    job_id: str,
    headers: dict,
    event_type: str,
    data: dict,
) -> None:
    """Send callback event to Hetzner."""
    try:
        requests.post(
            f"{base_url}/internal/jobs/{job_id}/event",
            headers=headers,
            json={"type": event_type, **data},
            timeout=10,
        )
    except requests.RequestException as e:
        logger.warning(f"Callback failed: {e}")


class ProgressCallback:
    """Context manager for sending progress updates."""

    def __init__(self, base_url: str, job_id: str, headers: dict):
        self.base_url = base_url
        self.job_id = job_id
        self.headers = headers

    def __call__(self, event_type: str, data: dict) -> None:
        send_callback(self.base_url, self.job_id, self.headers, event_type, data)

    def status(self, message: str) -> None:
        self.__call__("status", {"message": message})

    def progress(self, progress: float, **extra) -> None:
        self.__call__("progress", {"progress": progress, **extra})

    def completed(self, message: str = "Completed successfully") -> None:
        self.__call__("completed", {"message": message})

    def failed(self, error: str) -> None:
        self.__call__("failed", {"error": error})
