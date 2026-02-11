"""
Pattern Export Module

Exports discovered patterns in a format suitable for the category mapping system.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from src.external.schemas import PatternExport, PatternListResponse

logger = logging.getLogger(__name__)


class PatternExporter:
    """Export discovered patterns for the category mapping system."""

    def __init__(self, data_dir: Path):
        """
        Initialize exporter.

        Args:
            data_dir: Base data directory (e.g., data/)
        """
        self.data_dir = Path(data_dir)
        self.pattern_id_dir = self.data_dir / "pattern_identification"

    def _load_json(self, path: Path) -> Optional[Dict]:
        """Load JSON file if exists."""
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def _get_model_version(self) -> str:
        """Generate model version ID from checkpoint timestamp."""
        checkpoint_dir = self.data_dir / "training" / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                mtime = datetime.fromtimestamp(latest.stat().st_mtime)
                return f"vae_{mtime.strftime('%Y%m%d_%H%M%S')}"
        return f"vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _load_message_examples(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Load message examples (legacy - returns empty, use message_scores.h5 queries)."""
        return {}

    def _load_pattern_names(self) -> Dict[str, Dict[str, Dict]]:
        """Load pattern names from PatternNamer output."""
        names_path = self.pattern_id_dir / "naming" / "pattern_names.json"
        if names_path.exists():
            with open(names_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_shap_weights(self) -> Dict[str, Any]:
        """Load SHAP hierarchical weights."""
        shap_path = self.pattern_id_dir / "shap" / "hierarchical_weights.json"
        if shap_path.exists():
            with open(shap_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_message_database(self) -> Dict[str, Dict]:
        """Load message database for text lookup."""
        db_path = self.pattern_id_dir / "assignment" / "message_database.json"
        if db_path.exists():
            with open(db_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_population_stats(self) -> Dict[str, Any]:
        """Load population statistics for activation stats."""
        stats_path = self.pattern_id_dir / "scoring" / "population_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                return json.load(f)
        return {}

    def _get_activation_stats(
        self,
        population_stats: Dict,
        key: str,
        dim_idx: int
    ) -> Dict[str, Any]:
        """
        Extract activation statistics for a pattern from population stats.

        Returns dict with:
            - activation_count: Messages with significant activation
            - unique_engineers: Engineers exhibiting pattern
            - mean_activation: Mean activation score
            - std_activation: Std of activation scores
        """
        # Default stats if not available
        default_stats = {
            "activation_count": 0,
            "unique_engineers": 0,
            "mean_activation": 0.0,
            "std_activation": 0.0,
        }

        if not population_stats:
            return default_stats

        # Try to find stats for this pattern
        # Population stats structure varies, adapt as needed
        pattern_stats = population_stats.get(key, {})
        if isinstance(pattern_stats, dict):
            dim_stats = pattern_stats.get(str(dim_idx), {})
            if dim_stats:
                return {
                    "activation_count": dim_stats.get("count", 0),
                    "unique_engineers": dim_stats.get("unique_engineers", 0),
                    "mean_activation": dim_stats.get("mean", 0.0),
                    "std_activation": dim_stats.get("std", 0.0),
                }

        return default_stats

    def _extract_composition(
        self,
        shap_weights: Dict,
        key: str,
        dim_idx: int
    ) -> tuple:
        """
        Extract composition info from SHAP weights.

        Returns:
            (composed_from, composition_weights) or (None, None)
        """
        if not shap_weights:
            return None, None

        # SHAP weights structure depends on shap_analysis.py output
        # Look for weights that show what patterns contribute to this one
        weights_key = f"{key}_{dim_idx}"
        pattern_weights = shap_weights.get(weights_key, {})

        if pattern_weights:
            composed_from = []
            composition_weights = {}
            for source_pattern, weight in pattern_weights.items():
                if isinstance(weight, (int, float)) and abs(weight) > 0.01:
                    composed_from.append(source_pattern)
                    composition_weights[source_pattern] = float(weight)

            if composed_from:
                return composed_from, composition_weights

        return None, None

    def export_all_patterns(self, top_k_examples: int = 5) -> PatternListResponse:
        """
        Export all discovered patterns.

        Args:
            top_k_examples: Number of example messages per pattern

        Returns:
            PatternListResponse with all patterns
        """
        model_version = self._get_model_version()
        pattern_names = self._load_pattern_names()
        message_examples = self._load_message_examples()
        shap_weights = self._load_shap_weights()
        message_db = self._load_message_database()
        population_stats = self._load_population_stats()

        if not pattern_names:
            logger.warning("No pattern names found - has pattern naming been run?")
            return PatternListResponse(
                model_version_id=model_version,
                export_timestamp=datetime.now(),
                pattern_count=0,
                patterns=[],
            )

        patterns: List[PatternExport] = []

        for key, names_dict in pattern_names.items():
            # Parse key (e.g., "enc1_bottom", "enc1_mid", "unified")
            if key == "unified":
                encoder = None
                level = "unified"
            else:
                parts = key.split("_", 1)
                encoder = parts[0] if len(parts) > 1 else None
                level = parts[1] if len(parts) > 1 else parts[0]

            for dim_key, name_info in names_dict.items():
                # dim_key is like "bottom_0", "unified_3"
                dim_parts = dim_key.split("_")
                dim_idx = int(dim_parts[-1]) if dim_parts else 0
                pattern_id = f"{key}_{dim_idx}"

                # Get example messages
                key_examples = message_examples.get(key, {})
                dim_examples = key_examples.get(dim_key, [])[:top_k_examples]

                example_ids = []
                example_texts = []
                for ex in dim_examples:
                    msg_idx = ex.get("message_idx")
                    if msg_idx is not None:
                        example_ids.append(str(msg_idx))
                        # Get text from message database
                        msg_data = message_db.get(str(msg_idx), {})
                        text = msg_data.get("text", "")
                        if text:
                            example_texts.append(text)

                # Get composition info from SHAP
                composed_from, composition_weights = self._extract_composition(
                    shap_weights, key, dim_idx
                )

                # Get activation stats
                stats = self._get_activation_stats(population_stats, key, dim_idx)

                pattern = PatternExport(
                    pattern_id=pattern_id,
                    model_version_id=model_version,
                    level=level,
                    encoder=encoder,
                    name=name_info.get("name", f"Pattern {pattern_id}"),
                    description=name_info.get("description", ""),
                    activation_count=stats["activation_count"],
                    unique_engineers=stats["unique_engineers"],
                    mean_activation=stats["mean_activation"],
                    std_activation=stats["std_activation"],
                    example_message_ids=example_ids,
                    example_texts=example_texts,
                    composed_from=composed_from,
                    composition_weights=composition_weights,
                )
                patterns.append(pattern)

        logger.info(f"Exported {len(patterns)} patterns")

        return PatternListResponse(
            model_version_id=model_version,
            export_timestamp=datetime.now(),
            pattern_count=len(patterns),
            patterns=patterns,
        )

    def export_pattern(
        self,
        pattern_id: str,
        top_k_examples: int = 10
    ) -> Optional[PatternExport]:
        """Export a single pattern by ID."""
        all_patterns = self.export_all_patterns(top_k_examples)
        for pattern in all_patterns.patterns:
            if pattern.pattern_id == pattern_id:
                return pattern
        return None

    def save_export(self, output_path: Path, top_k_examples: int = 5) -> None:
        """Save pattern export to JSON file."""
        export = self.export_all_patterns(top_k_examples)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export.model_dump(), f, indent=2, default=str)
        logger.info(f"Exported {export.pattern_count} patterns to {output_path}")
