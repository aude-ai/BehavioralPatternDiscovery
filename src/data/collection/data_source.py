"""
Data Source Handler

Handles fetching and extracting data from multiple source types:
- Local folder path (existing behavior)
- Local zip file
- URL to zip file (download link or API endpoint)

Provides a context manager that extracts zips to a temp directory
and cleans up afterward.
"""

import logging
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def _is_url(path: str) -> bool:
    """Check if path is a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def _is_zip_file(path: str) -> bool:
    """Check if path points to a zip file."""
    return path.lower().endswith(".zip")


def _download_file(url: str, dest_path: Path, timeout: int = 300) -> None:
    """
    Download file from URL to destination path.

    Args:
        url: URL to download from
        dest_path: Local path to save file
        timeout: Request timeout in seconds
    """
    logger.info(f"Downloading from {url}...")

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and downloaded % (1024 * 1024) == 0:
                pct = (downloaded / total_size) * 100
                logger.info(f"  Downloaded {downloaded / 1024 / 1024:.1f} MB ({pct:.1f}%)")

    logger.info(f"Download complete: {dest_path.stat().st_size / 1024 / 1024:.1f} MB")


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """
    Extract zip file to destination directory.

    Args:
        zip_path: Path to zip file
        dest_dir: Directory to extract to
    """
    logger.info(f"Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Check for nested directory (common in zip files)
        names = zf.namelist()

        # Extract all files
        zf.extractall(dest_dir)

        # If all files are in a single subdirectory, move them up
        extracted_items = list(dest_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            nested_dir = extracted_items[0]
            logger.info(f"  Flattening nested directory: {nested_dir.name}")

            # Move contents up one level
            for item in nested_dir.iterdir():
                shutil.move(str(item), str(dest_dir / item.name))

            # Remove empty nested directory
            nested_dir.rmdir()

    # Log extracted contents
    extracted_files = list(dest_dir.glob("*"))
    logger.info(f"Extracted {len(extracted_files)} items to {dest_dir}")
    for f in extracted_files[:10]:
        logger.info(f"  - {f.name}")
    if len(extracted_files) > 10:
        logger.info(f"  ... and {len(extracted_files) - 10} more")


@contextmanager
def resolve_data_source(source: str) -> Generator[Path, None, None]:
    """
    Resolve a data source to a local directory path.

    Handles:
    - Local folder: returns path directly
    - Local zip file: extracts to temp dir, returns temp path
    - URL to zip: downloads, extracts to temp dir, returns temp path

    Usage:
        with resolve_data_source("/path/to/data") as data_path:
            # data_path is a Path to a directory containing the data
            loader = NDJSONLoader({"input_path": data_path, ...})

    Args:
        source: Path to folder, path to zip file, or URL to zip file

    Yields:
        Path to directory containing the data files
    """
    temp_dir: Optional[Path] = None

    try:
        # Case 1: URL to zip file
        if _is_url(source):
            logger.info(f"Data source is URL: {source}")

            temp_dir = Path(tempfile.mkdtemp(prefix="bpd_data_"))
            zip_path = temp_dir / "download.zip"
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir()

            _download_file(source, zip_path)
            _extract_zip(zip_path, extract_dir)

            # Clean up zip file to save space
            zip_path.unlink()

            yield extract_dir

        # Case 2: Local zip file
        elif _is_zip_file(source):
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Zip file not found: {source}")

            logger.info(f"Data source is local zip: {source_path}")

            temp_dir = Path(tempfile.mkdtemp(prefix="bpd_data_"))
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir()

            _extract_zip(source_path, extract_dir)

            yield extract_dir

        # Case 3: Local folder (existing behavior)
        else:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Data folder not found: {source}")
            if not source_path.is_dir():
                raise ValueError(f"Expected directory, got file: {source}")

            logger.info(f"Data source is local folder: {source_path}")
            yield source_path

    finally:
        # Clean up temp directory if we created one
        if temp_dir and temp_dir.exists():
            logger.info(f"Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
