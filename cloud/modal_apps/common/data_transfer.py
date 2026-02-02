"""Data transfer utilities for Modal <-> Hetzner communication."""
import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import requests
import zstandard as zstd

logger = logging.getLogger(__name__)

# Chunk size for streaming (64KB)
CHUNK_SIZE = 65536

# Zstd compression level (3 is a good balance of speed/ratio)
ZSTD_LEVEL = 3


# =============================================================================
# COMPRESSION UTILITIES
# =============================================================================


def compress_file(input_path: Path, output_path: Path) -> None:
    """Compress a file using zstd, streaming to avoid memory issues."""
    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    with open(input_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            cctx.copy_stream(f_in, f_out)


def decompress_file(input_path: Path, output_path: Path) -> None:
    """Decompress a zstd file, streaming to avoid memory issues."""
    dctx = zstd.ZstdDecompressor()
    with open(input_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            dctx.copy_stream(f_in, f_out)


def compress_bytes(data: bytes) -> bytes:
    """Compress bytes using zstd (for small data only)."""
    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    return cctx.compress(data)


def decompress_bytes(data: bytes) -> bytes:
    """Decompress zstd bytes (for small data only)."""
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)


def compress_numpy(arr: np.ndarray) -> bytes:
    """Compress numpy array for network transfer."""
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    return compress_bytes(buffer.read())


def decompress_numpy(data: bytes) -> np.ndarray:
    """Decompress numpy array from network transfer."""
    decompressed = decompress_bytes(data)
    buffer = io.BytesIO(decompressed)
    return np.load(buffer)


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


def download_and_decompress(
    url: str,
    headers: dict,
    output_path: Path,
    timeout: int = 600,
) -> None:
    """
    Download compressed file and decompress it.

    Downloads to temp file, then decompresses to final path.
    Streaming throughout - never loads full file into memory.
    """
    with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_file_streaming(url, headers, tmp_path, timeout)
        decompress_file(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


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


def compress_and_upload(
    url: str,
    headers: dict,
    input_path: Path,
    filename: str | None = None,
    timeout: int = 600,
) -> dict:
    """
    Compress file and upload it.

    Compresses to temp file, then uploads.
    Streaming throughout - never loads full file into memory.
    """
    if filename is None:
        filename = input_path.name + ".zst"

    with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        compress_file(input_path, tmp_path)
        return upload_file_streaming(url, headers, tmp_path, filename, timeout)
    finally:
        tmp_path.unlink(missing_ok=True)


# =============================================================================
# LEGACY FUNCTIONS (for small data / backward compatibility)
# =============================================================================


def download_file(url: str, headers: dict, timeout: int = 300) -> bytes:
    """
    Download file and return bytes.

    WARNING: Loads entire file into memory. Only use for small files.
    For large files, use download_file_streaming() instead.
    """
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
    """
    Upload file to Hetzner.

    WARNING: Caller must manage memory. For large files, use
    upload_file_streaming() or compress_and_upload() instead.
    """
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
