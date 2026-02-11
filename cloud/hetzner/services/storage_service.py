"""Storage service for file operations."""
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Mapping from file_key to relative path within project directory
HETZNER_FILE_PATHS = {
    "activities": "data/collection/activities.csv",
    "engineer_metadata": "data/collection/engineer_metadata.csv",
    "train_aux_vars": "data/processing/train_aux_vars.npy",
    "population_stats": "pattern_identification/scoring/population_stats.json",
    "message_scores_index": "pattern_identification/scoring/message_scores_index.json",
    "word_attributions": "pattern_identification/scoring/word_attributions.json",
    "hierarchical_weights": "pattern_identification/shap/hierarchical_weights.json",
    "pattern_names": "pattern_identification/naming/pattern_names.json",
}


class StorageService:
    """Service for managing project file storage.

    Supports variant inheritance: if parent_id and owned_files are provided,
    file reads will fall back to parent storage for files not owned by this project.
    """

    def __init__(
        self,
        project_id: str,
        parent_id: Optional[str] = None,
        owned_files: Optional[dict] = None,
    ):
        self.project_id = project_id
        self.parent_id = parent_id
        self.owned_files = owned_files or {}
        self.base_path = Path(settings.projects_dir) / project_id
        self.parent_path = Path(settings.projects_dir) / parent_id if parent_id else None

    def _resolve_read_path(self, file_key: str, relative_path: str) -> Path:
        """Resolve the path to read a file from, considering inheritance.

        For writes, always use self.base_path directly.
        For reads, check if this project owns the file; if not, use parent.
        """
        if self.owned_files.get(file_key, True):
            # Owned by this project (or no inheritance info - assume owned)
            return self.base_path / relative_path
        elif self.parent_path:
            # Not owned - use parent's path
            logger.info(f"Resolving {file_key} from parent {self.parent_id}")
            return self.parent_path / relative_path
        else:
            # Root project doesn't own it - file doesn't exist
            raise FileNotFoundError(f"File {file_key} not found in project or parent")

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
    # FILE PATH RESOLUTION (with inheritance support)
    # =========================================================================

    def get_read_path(self, file_key: str) -> Path:
        """Get the path to read a file from, resolving inheritance if needed."""
        relative_path = HETZNER_FILE_PATHS.get(file_key)
        if not relative_path:
            raise ValueError(f"Unknown file key: {file_key}")
        return self._resolve_read_path(file_key, relative_path)

    def get_write_path(self, file_key: str) -> Path:
        """Get the path to write a file to (always this project's storage)."""
        relative_path = HETZNER_FILE_PATHS.get(file_key)
        if not relative_path:
            raise ValueError(f"Unknown file key: {file_key}")
        return self.base_path / relative_path

    # =========================================================================
    # UNCOMPRESSED PATHS (small files, local processing)
    # These properties return WRITE paths for backwards compatibility.
    # Use get_read_path(file_key) when reading with inheritance support.
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
    def message_scores_index_path(self) -> Path:
        return self.base_path / "pattern_identification/scoring/message_scores_index.json"

    @property
    def word_attributions_path(self) -> Path:
        return self.base_path / "pattern_identification/scoring/word_attributions.json"

    @property
    def hierarchical_weights_path(self) -> Path:
        return self.base_path / "pattern_identification/shap/hierarchical_weights.json"

    @property
    def pattern_names_path(self) -> Path:
        return self.base_path / "pattern_identification/naming/pattern_names.json"

    # =========================================================================
    # INHERITANCE-AWARE READ METHODS
    # =========================================================================

    def load_json_inherited(self, file_key: str) -> Optional[dict]:
        """Load JSON file with inheritance resolution."""
        try:
            path = self.get_read_path(file_key)
        except FileNotFoundError:
            return None
        return self.load_json(path)

    def load_pickle_inherited(self, file_key: str):
        """Load pickle file with inheritance resolution."""
        try:
            path = self.get_read_path(file_key)
        except FileNotFoundError:
            return None
        return self.load_pickle(path)

    def load_numpy_inherited(self, file_key: str) -> Optional[np.ndarray]:
        """Load numpy file with inheritance resolution."""
        try:
            path = self.get_read_path(file_key)
        except FileNotFoundError:
            return None
        return self.load_numpy(path)

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    # NOTE: Large processing files (embeddings, checkpoints, activations, etc.)
    # are stored in R2, not on local Hetzner storage. Use r2_service for those.

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

    # =========================================================================
    # CACHE METHODS (for large files downloaded from R2)
    # =========================================================================

    def get_cached_h5(self, name: str) -> Path:
        """Get path for cached HDF5 file from R2."""
        cache_dir = self.base_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{name}.h5"

    def invalidate_cache(self, name: str) -> bool:
        """Delete a cached file. Returns True if file was deleted."""
        cache_path = self.get_cached_h5(name)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False
