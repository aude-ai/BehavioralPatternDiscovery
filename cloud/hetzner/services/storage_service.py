"""Storage service for file operations."""
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import get_settings

settings = get_settings()


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

    # Paths
    @property
    def activities_path(self) -> Path:
        return self.base_path / "data/collection/activities.csv"

    @property
    def engineer_metadata_path(self) -> Path:
        return self.base_path / "data/collection/engineer_metadata.csv"

    @property
    def message_database_path(self) -> Path:
        return self.base_path / "data/processing/message_database.pkl"

    @property
    def train_features_path(self) -> Path:
        return self.base_path / "data/processing/train_features.npy"

    @property
    def train_aux_vars_path(self) -> Path:
        return self.base_path / "data/processing/train_aux_vars.npy"

    @property
    def train_input_path(self) -> Path:
        return self.base_path / "data/processing/train_input_with_aux.npy"

    @property
    def checkpoint_path(self) -> Path:
        return self.base_path / "training/checkpoints/best_model.pt"

    @property
    def activations_path(self) -> Path:
        return self.base_path / "pattern_identification/scoring/activations.h5"

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

    # File operations
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
        """Save pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_pickle(self, path: Path):
        """Load pickle file."""
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_numpy(self, path: Path, arr: np.ndarray):
        """Save numpy array."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    def load_numpy(self, path: Path) -> Optional[np.ndarray]:
        """Load numpy array."""
        if not path.exists():
            return None
        return np.load(path)

    def file_exists(self, path: Path) -> bool:
        """Check if file exists."""
        return path.exists()

    def get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        if not path.exists():
            return 0
        return path.stat().st_size
