"""Data transfer utilities for Modal <-> Hetzner communication."""
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Chunk size for streaming (64KB)
CHUNK_SIZE = 65536


# =============================================================================
# STREAMING DOWNLOAD
# =============================================================================


def download_file_streaming(
    url: str,
    headers: dict,
    output_path: Path,
    timeout: int = 600,
) -> None:
    """
    Download file from Hetzner using streaming.

    Writes directly to disk in chunks - never loads full file into memory.
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            with requests.get(url, headers=headers, timeout=timeout, stream=True) as response:
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                return

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")

    raise RuntimeError("Download failed after all retries")


# =============================================================================
# STREAMING UPLOAD
# =============================================================================


def upload_file_streaming(
    url: str,
    headers: dict,
    file_path: Path,
    filename: str | None = None,
    timeout: int = 600,
) -> dict:
    """
    Upload file using streaming.

    Streams file content directly - Hetzner receives via request.stream().
    """
    if filename is None:
        filename = file_path.name

    file_size = file_path.stat().st_size

    def file_generator():
        with open(file_path, "rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                yield chunk

    upload_headers = {
        **headers,
        "Content-Type": "application/octet-stream",
        "Content-Length": str(file_size),
    }

    response = requests.post(
        url,
        headers=upload_headers,
        data=file_generator(),
        timeout=timeout,
    )

    response.raise_for_status()
    return response.json()


def upload_json(
    url: str,
    headers: dict,
    data: dict,
    timeout: int = 30,
) -> dict:
    """
    Upload JSON data to Hetzner.

    For small data payloads (population_stats, message_examples, etc.)
    """
    response = requests.post(
        url,
        headers={**headers, "Content-Type": "application/json"},
        json=data,
        timeout=timeout,
    )
    response.raise_for_status()
    logger.info(f"Uploaded JSON to: {url}")
    return response.json() if response.text else {}


# =============================================================================
# CALLBACKS
# =============================================================================


def send_callback(
    base_url: str,
    job_id: str,
    headers: dict,
    event_type: str,
    data: dict,
    section: str = "general",
) -> None:
    """Send callback event to Hetzner."""
    try:
        requests.post(
            f"{base_url}/internal/jobs/{job_id}/event",
            headers=headers,
            json={"type": event_type, "section": section, **data},
            timeout=10,
        )
    except requests.RequestException as e:
        logger.warning(f"Callback failed: {e}")


class ProgressCallback:
    """Helper for sending progress updates with section routing."""

    def __init__(
        self,
        base_url: str,
        job_id: str,
        headers: dict,
        section: str = "general",
    ):
        self.base_url = base_url
        self.job_id = job_id
        self.headers = headers
        self.section = section

    def __call__(self, event_type: str, data: dict) -> None:
        send_callback(
            self.base_url,
            self.job_id,
            self.headers,
            event_type,
            data,
            section=self.section,
        )

    def status(self, message: str) -> None:
        self.__call__("status", {"message": message})

    def progress(self, progress: float, **extra) -> None:
        self.__call__("progress", {"progress": progress, **extra})

    def completed(self, message: str = "Completed successfully") -> None:
        self.__call__("completed", {"message": message})

    def failed(self, error: str) -> None:
        self.__call__("failed", {"error": error})
