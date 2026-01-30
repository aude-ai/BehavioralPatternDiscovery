"""Storage service for file operations."""
import io
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import zstandard as zstd

from ..config import get_settings

settings = get_settings()

# Zstd compression level (3 is a good balance of speed/ratio)
ZSTD_LEVEL = 3


class StorageService:
    """Service for managing project file storage."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.base_path = Path(settings.projects_dir) / project_id

    def ensure_directories(self):
        """Create all required directories for a project."""
        dirs = [
            "config",
            "data/collection",
            "data/processing",
            "data/external/input",
            "training/checkpoints",
            "training/logs",
            "pattern_identification/scoring",
            "pattern_identification/messages",
            "pattern_identification/shap",
            "pattern_identification/naming",
            "pattern_identification/naming/debug",
            "scoring/batch",
            "scoring/individual",
            "scoring/reports",
        ]
        for d in dirs:
            (self.base_path / d).mkdir(parents=True, exist_ok=True)

    def delete_project(self):
        """Delete all project files."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)

    # =========================================================================
    # UNCOMPRESSED PATHS (small files, local processing)
    # =========================================================================

    @property
    def activities_path(self) -> Path:
        return self.base_path / "data/collection/activities.csv"

    @property
    def engineer_metadata_path(self) -> Path:
        return self.base_path / "data/collection/engineer_metadata.csv"

    @property
    def train_aux_vars_path(self) -> Path:
        return self.base_path / "data/processing/train_aux_vars.npy"

    @property
    def population_stats_path(self) -> Path:
        return self.base_path / "pattern_identification/scoring/population_stats.json"

    @property
    def message_examples_path(self) -> Path:
        return self.base_path / "pattern_identification/messages/message_examples.json"

    @property
    def hierarchical_weights_path(self) -> Path:
        return self.base_path / "pattern_identification/shap/hierarchical_weights.json"

    @property
    def pattern_names_path(self) -> Path:
        return self.base_path / "pattern_identification/naming/pattern_names.json"

    # =========================================================================
    # COMPRESSED PATHS (large files, transferred to/from Modal)
    # =========================================================================

    @property
    def message_database_path(self) -> Path:
        """Message database - compressed for Modal transfer."""
        return self.base_path / "data/processing/message_database.pkl.zst"

    @property
    def train_features_path(self) -> Path:
        """Training features - compressed for Modal transfer."""
        return self.base_path / "data/processing/train_features.npy.zst"

    @property
    def train_input_path(self) -> Path:
        """Combined training input - compressed for Modal transfer."""
        return self.base_path / "data/processing/train_input_with_aux.npy.zst"

    @property
    def checkpoint_path(self) -> Path:
        """Model checkpoint - compressed for Modal transfer."""
        return self.base_path / "training/checkpoints/best_model.pt.zst"

    @property
    def activations_path(self) -> Path:
        """Activations HDF5 - compressed for Modal transfer."""
        return self.base_path / "pattern_identification/scoring/activations.h5.zst"

    # =========================================================================
    # UNCOMPRESSED FILE OPERATIONS
    # =========================================================================

    def save_json(self, path: Path, data: dict):
        """Save JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(self, path: Path) -> Optional[dict]:
        """Load JSON file."""
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def save_pickle(self, path: Path, data):
        """Save pickle file (uncompressed)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_pickle(self, path: Path):
        """Load pickle file (uncompressed)."""
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_numpy(self, path: Path, arr: np.ndarray):
        """Save numpy array (uncompressed)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    def load_numpy(self, path: Path) -> Optional[np.ndarray]:
        """Load numpy array (uncompressed)."""
        if not path.exists():
            return None
        return np.load(path)

    # =========================================================================
    # COMPRESSED FILE OPERATIONS (for .zst files)
    # =========================================================================

    def save_pickle_compressed(self, path: Path, data):
        """Save pickle file with zstd compression."""
        path.parent.mkdir(parents=True, exist_ok=True)
        pickled = pickle.dumps(data)
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        compressed = cctx.compress(pickled)
        with open(path, "wb") as f:
            f.write(compressed)

    def load_pickle_compressed(self, path: Path):
        """Load zstd-compressed pickle file."""
        if not path.exists():
            return None
        with open(path, "rb") as f:
            compressed = f.read()
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed)
        return pickle.loads(decompressed)

    def save_numpy_compressed(self, path: Path, arr: np.ndarray):
        """Save numpy array with zstd compression."""
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, arr)
        buffer.seek(0)
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        compressed = cctx.compress(buffer.read())
        with open(path, "wb") as f:
            f.write(compressed)

    def load_numpy_compressed(self, path: Path) -> Optional[np.ndarray]:
        """Load zstd-compressed numpy array."""
        if not path.exists():
            return None
        with open(path, "rb") as f:
            compressed = f.read()
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed)
        buffer = io.BytesIO(decompressed)
        return np.load(buffer)

    def decompress_to_temp(self, path: Path) -> Path:
        """
        Decompress a .zst file to a temporary file.

        Returns the path to the temporary file. Caller is responsible
        for deleting it when done.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Create temp file with appropriate suffix
        suffix = path.stem  # e.g., "activations.h5" from "activations.h5.zst"
        with tempfile.NamedTemporaryFile(suffix=f".{suffix.split('.')[-1]}", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f_in:
            with open(tmp_path, "wb") as f_out:
                dctx.copy_stream(f_in, f_out)

        return tmp_path

    def compress_from_temp(self, temp_path: Path, output_path: Path):
        """
        Compress a file to zstd format.

        Reads from temp_path, writes compressed to output_path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        with open(temp_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                cctx.copy_stream(f_in, f_out)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def file_exists(self, path: Path) -> bool:
        """Check if file exists."""
        return path.exists()

    def get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        if not path.exists():
            return 0
        return path.stat().st_size
