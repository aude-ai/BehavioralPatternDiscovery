"""
Configuration Loading, Validation, and Derived Value Computation

Provides utilities for loading YAML configuration files and validating
that all required keys are present. Follows the fail-fast philosophy:
missing keys cause immediate errors rather than silent fallbacks.

Also provides ModelDimensions class for computing derived dimensions
that should NOT be stored in config (e.g., input_dim of layer N = output_dim of layer N-1).
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelDimensions:
    """
    Computes all derived model dimensions from base config values.

    This class exists because many dimensions are not independent choices
    but are determined by other dimensions:
    - encoder input = embedding_dim + aux_features_dim (if enabled)
    - each level's input = previous level's output
    - decoder output = embedding_dim (reconstructing embeddings)
    - unification input = last_level_dim * num_encoders

    These values should NOT be in the config because they cannot be
    changed independently - they are computed from true choices.

    Supports dynamic hierarchy levels - the number and names of levels
    are read from config rather than being hardcoded.
    """

    # Base dimensions (true config values)
    embedding_dim: int
    aux_features_dim: int
    aux_features_enabled: bool
    num_encoders: int
    unified_output_dim: int

    # Dynamic level dimensions: ordered list of (name, output_dim) tuples
    # Order determines hierarchy: first = most detailed, last = most abstract
    level_names: tuple[str, ...]
    level_output_dims: dict[str, int]

    @classmethod
    def from_config(
        cls,
        model_config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> "ModelDimensions":
        """
        Create ModelDimensions from config.

        Args:
            model_config: Model configuration dict
            metadata: Optional metadata dict with embedding_dim and aux_features_dim.
                      Can be either:
                      - Raw metadata dict with keys: embedding_dim, aux_features_dim
                      - Data config dict with paths section (for local file access)

        Returns:
            ModelDimensions with all base values extracted
        """
        # Determine dimensions from metadata or model_config
        if metadata is not None:
            if "embedding_dim" in metadata and "aux_features_dim" in metadata:
                # Direct metadata (from Modal/cloud context)
                embedding_dim = metadata["embedding_dim"]
                aux_features_dim = metadata["aux_features_dim"]
            elif "paths" in metadata:
                # Data config with paths (local file access)
                dims = infer_dimensions_from_data(metadata)
                embedding_dim = dims["embedding_dim"]
                aux_features_dim = dims["aux_features_dim"]
            else:
                raise ValueError(
                    "metadata must contain either 'embedding_dim'/'aux_features_dim' "
                    "or 'paths' for local file access"
                )
        else:
            # Backward compatibility: use model config
            input_config = model_config["input"]
            embedding_dim = input_config["embedding_dim"]
            aux_features_dim = input_config["aux_features_dim"]

        # Check aux_features toggle
        input_config = model_config["input"]
        aux_enabled = input_config["aux_features"]["enabled"]

        encoder_config = model_config["encoder"]
        levels_list = encoder_config["hierarchical"]["levels"]
        unification_config = model_config["unification"]
        unification_type = unification_config["type"]

        # Parse level names and output dims from list format
        level_names = tuple(level["name"] for level in levels_list)
        level_output_dims = {level["name"]: level["output_dim"] for level in levels_list}

        return cls(
            embedding_dim=embedding_dim,
            aux_features_dim=aux_features_dim if aux_enabled else 0,
            aux_features_enabled=aux_enabled,
            num_encoders=encoder_config["num_encoders"],
            unified_output_dim=unification_config[unification_type]["output_dim"],
            level_names=level_names,
            level_output_dims=level_output_dims,
        )

    # =========================================================================
    # Level accessors
    # =========================================================================

    @property
    def num_levels(self) -> int:
        """Number of hierarchy levels."""
        return len(self.level_names)

    @property
    def first_level(self) -> str:
        """Name of the first (most detailed) level."""
        return self.level_names[0]

    @property
    def last_level(self) -> str:
        """Name of the last (most abstract) level."""
        return self.level_names[-1]

    def get_level_output_dim(self, level: str) -> int:
        """Get output dimension for a specific level."""
        if level not in self.level_output_dims:
            raise ValueError(f"Unknown level: {level}. Available: {list(self.level_names)}")
        return self.level_output_dims[level]

    def get_previous_level(self, level: str) -> str | None:
        """Get the previous level name, or None if first level."""
        idx = self.level_names.index(level)
        return self.level_names[idx - 1] if idx > 0 else None

    # =========================================================================
    # Computed dimensions (derived, not configurable)
    # =========================================================================

    @property
    def encoder_input_dim(self) -> int:
        """Input to encoder = embedding + aux features (if enabled)."""
        if self.aux_features_enabled:
            return self.embedding_dim + self.aux_features_dim
        return self.embedding_dim

    def get_level_input_dim(self, level: str) -> int:
        """
        Get input dimension for a specific level.

        First level input = encoder_input_dim
        Other levels input = previous level's output_dim
        """
        prev = self.get_previous_level(level)
        if prev is None:
            return self.encoder_input_dim
        return self.level_output_dims[prev]

    @property
    def unification_input_dim(self) -> int:
        """Unification input = last level outputs from all encoders concatenated."""
        return self.level_output_dims[self.last_level] * self.num_encoders

    @property
    def decoder_input_dim(self) -> int:
        """Decoder input = unified latent dimension."""
        return self.unified_output_dim

    @property
    def decoder_output_dim(self) -> int:
        """Decoder output = embedding dimension (reconstructing embeddings)."""
        return self.embedding_dim

    @property
    def latent_dims(self) -> dict[str, int]:
        """Convenience accessor for latent dimensions per level."""
        return dict(self.level_output_dims)

    def get_level_dims(self, level: str) -> tuple[int, int]:
        """
        Get (input_dim, output_dim) for a hierarchical level.

        Args:
            level: Name of the level

        Returns:
            Tuple of (input_dim, output_dim)
        """
        if level not in self.level_output_dims:
            raise ValueError(f"Unknown level: {level}. Available: {list(self.level_names)}")
        return (self.get_level_input_dim(level), self.level_output_dims[level])

    # =========================================================================
    # Backward compatibility properties (for existing code)
    # =========================================================================

    @property
    def bottom_output_dim(self) -> int:
        """Output dimension of bottom level (first level)."""
        return self.level_output_dims[self.first_level]

    @property
    def top_output_dim(self) -> int:
        """Output dimension of top level (last level)."""
        return self.level_output_dims[self.last_level]

    @property
    def bottom_input_dim(self) -> int:
        """Input dimension of bottom level (first level)."""
        return self.encoder_input_dim

    @property
    def top_input_dim(self) -> int:
        """Input dimension of top level (last level)."""
        return self.get_level_input_dim(self.last_level)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a single YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file does not exist
        yaml.YAMLError: If the YAML is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_all_configs(config_dir: str | Path) -> dict[str, Any]:
    """
    Load all configuration files from a directory and merge them.

    Expected files:
        - data.yaml
        - model.yaml
        - training.yaml
        - pattern_identification.yaml
        - scoring.yaml

    Each file's contents are stored under a key matching its name (without extension).

    Args:
        config_dir: Path to the configuration directory

    Returns:
        Dictionary with structure:
        {
            "data": {...},
            "model": {...},
            "training": {...},
            "pattern_identification": {...},
            "scoring": {...}
        }

    Raises:
        FileNotFoundError: If any required config file is missing
    """
    config_dir = Path(config_dir)

    required_files = [
        "data.yaml",
        "model.yaml",
        "training.yaml",
        "pattern_identification.yaml",
        "scoring.yaml",
    ]

    configs: dict[str, Any] = {}

    for filename in required_files:
        config_path = config_dir / filename
        key = filename.replace(".yaml", "").replace("-", "_")
        configs[key] = load_config(config_path)

    logger.info(f"Loaded all configurations from {config_dir}")
    return configs


def validate_required_keys(config: dict[str, Any], required_keys: list[str], context: str) -> None:
    """
    Validate that all required keys are present in the configuration.

    Uses dot notation for nested keys (e.g., "model.encoder.type").

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys in dot notation
        context: Description of what is being validated (for error messages)

    Raises:
        ValueError: If any required keys are missing

    Example:
        validate_required_keys(
            config,
            ["model.encoder.type", "model.decoder.type", "training.learning_rate"],
            "VAE model configuration"
        )
    """
    missing = []

    for key in required_keys:
        parts = key.split(".")
        current = config

        try:
            for part in parts:
                current = current[part]
        except (KeyError, TypeError):
            missing.append(key)

    if missing:
        raise ValueError(
            f"[{context}] Missing required configuration keys:\n"
            + "\n".join(f"  - {key}" for key in missing)
        )


def get_nested(config: dict[str, Any], key: str) -> Any:
    """
    Get a value from a nested configuration using dot notation.

    Args:
        config: Configuration dictionary
        key: Key in dot notation (e.g., "model.encoder.type")

    Returns:
        The value at the specified key

    Raises:
        KeyError: If the key path does not exist
    """
    parts = key.split(".")
    current = config

    for part in parts:
        if not isinstance(current, dict):
            raise KeyError(f"Cannot traverse into non-dict at '{part}' in key '{key}'")
        if part not in current:
            raise KeyError(f"Key '{part}' not found in path '{key}'")
        current = current[part]

    return current


def infer_dimensions_from_data(data_config: dict) -> dict[str, int]:
    """
    Infer embedding and aux_features dimensions from preprocessed data.

    Args:
        data_config: Data configuration with paths section

    Returns:
        Dict with 'embedding_dim' and 'aux_features_dim'

    Raises:
        FileNotFoundError: If preprocessed data doesn't exist
        KeyError: If metadata is missing required fields
    """
    message_db_path = Path(data_config["paths"]["data"]["processing"]["message_database"])

    if not message_db_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {message_db_path}. "
            f"Run preprocessing before model configuration."
        )

    with open(message_db_path, "rb") as f:
        data = pickle.load(f)

    # Handle both old and new format
    if isinstance(data, dict) and "metadata" in data:
        metadata = data["metadata"]
        return {
            "embedding_dim": metadata["embedding_dim"],
            "aux_features_dim": metadata["aux_features_dim"],
        }
    else:
        # Old format - infer from first message
        first_msg = data[0]
        return {
            "embedding_dim": first_msg["embedding"].shape[0],
            "aux_features_dim": first_msg["aux_features"].shape[0],
        }


def validate_training_prerequisites(data_config: dict) -> None:
    """
    Validate that preprocessing has been run before training.

    Args:
        data_config: Data configuration with paths section

    Raises:
        FileNotFoundError: If preprocessed data doesn't exist
    """
    message_db_path = Path(data_config["paths"]["data"]["processing"]["message_database"])
    if not message_db_path.exists():
        raise FileNotFoundError(
            "Cannot start training: preprocessed data not found. "
            "Run the preprocessing step first."
        )


def load_normalization_params(data_config: dict) -> dict | None:
    """
    Load normalization params from preprocessed data.

    This is used by the training script to get the normalization params
    that were fitted during preprocessing, so they can be saved
    in the model checkpoint.

    Args:
        data_config: Data configuration with paths section

    Returns:
        Normalization params dict or None if no normalization was used

    Raises:
        FileNotFoundError: If preprocessed data doesn't exist
    """
    message_db_path = Path(data_config["paths"]["data"]["processing"]["message_database"])

    if not message_db_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {message_db_path}. "
            f"Run preprocessing before training."
        )

    with open(message_db_path, "rb") as f:
        data = pickle.load(f)

    # Get normalization params from metadata
    if isinstance(data, dict) and "metadata" in data:
        return data["metadata"].get("normalization_params")

    # Old format - no normalization params
    return None
