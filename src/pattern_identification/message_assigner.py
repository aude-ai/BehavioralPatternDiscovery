"""
Message Assigner

Assigns representative messages to patterns at all hierarchical levels
based on actual activation scores (not ideal embedding similarity).
Level names and encoder count are inferred from activation keys.

Supports two scoring modes:
- "positive": Select messages with highest positive activation scores
- "absolute": Select messages with highest absolute activation scores

Supports optional normalization of scores (global or per-dimension).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PatternExamples:
    """Examples for a single pattern."""
    encoder: str
    level: str
    pattern_idx: int
    examples: list[dict]


class MessageAssigner:
    """Assign message examples to patterns at all levels."""

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Config containing message_assignment and normalization sections.
                    Optional: paths.pattern_identification for local file output.
        """
        msg_config = config["message_assignment"]
        norm_config = config["normalization"]

        self.examples_per_pattern = msg_config["examples_per_pattern"]
        self.max_words = msg_config["max_words_per_message"]
        self.scoring_mode = msg_config["scoring_mode"]

        # Output path is optional (not used in cloud context)
        pi_config = config.get("paths", {}).get("pattern_identification", {})
        messages_config = pi_config.get("messages", {})
        output_path_str = messages_config.get("examples")
        self.output_path = Path(output_path_str) if output_path_str else None

        # Ratio constraints for absolute mode
        self.min_positive_ratio = msg_config.get("min_positive_ratio", 0.0)
        self.min_negative_ratio = msg_config.get("min_negative_ratio", 0.0)

        # Normalization config
        self.normalize_scores = norm_config["scores"]["enabled"]
        self.score_mode = norm_config["scores"]["mode"]
        self.score_percentile = norm_config["scores"]["percentile"]

        if self.scoring_mode not in ("positive", "absolute"):
            raise ValueError(
                f"Invalid scoring_mode: '{self.scoring_mode}'. "
                f"Must be 'positive' or 'absolute'."
            )

        if self.min_positive_ratio + self.min_negative_ratio > 1.0:
            raise ValueError(
                f"min_positive_ratio ({self.min_positive_ratio}) + "
                f"min_negative_ratio ({self.min_negative_ratio}) must be <= 1.0"
            )

    def assign_all(
        self,
        activations: dict[str, np.ndarray],
        message_database: list[dict],
    ) -> dict[str, list[PatternExamples]]:
        """
        Assign examples for all patterns at all levels.

        Args:
            activations: Output from BatchScorer.score_all()
            message_database: List of message data with 'text' field

        Returns:
            Dictionary keyed by activation key (e.g., 'enc1_bottom')
        """
        all_examples = {}

        logger.info(f"Assigning message examples using scoring_mode='{self.scoring_mode}'"
                    f", normalize_scores={self.normalize_scores}")

        # Infer encoder and level names from activation keys
        # Keys are like "enc1_bottom", "enc2_mid", "unified"
        encoder_level_keys = [k for k in activations.keys() if k != "unified"]

        # Process per-encoder levels dynamically
        for act_key in encoder_level_keys:
            # Parse key like "enc1_bottom" into encoder="enc1", level="bottom"
            parts = act_key.split("_", 1)
            enc_key = parts[0]
            level = parts[1]

            examples = self._assign_for_level(
                activations[act_key],
                message_database,
                encoder=enc_key,
                level=level,
            )
            all_examples[act_key] = examples
            logger.info(f"Assigned examples for {act_key}: {len(examples)} patterns")

        # Unified level
        if "unified" in activations:
            examples = self._assign_for_level(
                activations["unified"],
                message_database,
                encoder="unified",
                level="unified",
            )
            all_examples["unified"] = examples
            logger.info(f"Assigned examples for unified: {len(examples)} patterns")

        # Apply normalization if enabled
        if self.normalize_scores:
            all_examples = self._normalize_scores(all_examples)

        # Save to JSON (only if output_path configured - skipped in cloud context)
        if self.output_path:
            self._save_examples(all_examples)

        return all_examples

    def _assign_for_level(
        self,
        activations: np.ndarray,
        message_database: list[dict],
        encoder: str,
        level: str,
    ) -> list[PatternExamples]:
        """
        Assign examples for all patterns at a specific level.

        Args:
            activations: (n_messages, n_dims) activation scores
            message_database: List of message data
            encoder: Encoder name
            level: Level name

        Returns:
            List of PatternExamples, one per dimension
        """
        n_dims = activations.shape[1]
        results = []

        for dim_idx in range(n_dims):
            scores = activations[:, dim_idx]

            # Select top indices based on scoring mode
            if self.scoring_mode == "positive":
                # Highest positive scores
                top_indices = np.argsort(scores)[-self.examples_per_pattern:][::-1]
            else:  # absolute
                top_indices = self._select_absolute_with_ratios(scores)

            # Compute percentile ranks for all messages
            percentile_ranks = (np.argsort(np.argsort(scores)) / len(scores)) * 100

            examples = []
            for idx in top_indices:
                text = message_database[idx].get("text", "")
                examples.append({
                    "text": self._truncate_text(text),
                    "score": float(scores[idx]),
                    "percentile": float(percentile_ranks[idx]),
                    "message_idx": int(idx),
                })

            results.append(PatternExamples(
                encoder=encoder,
                level=level,
                pattern_idx=dim_idx,
                examples=examples,
            ))

        return results

    def _select_absolute_with_ratios(self, scores: np.ndarray) -> np.ndarray:
        """
        Select top indices by absolute value while respecting min ratio constraints.

        Ensures minimum representation of both positive and negative examples,
        with remaining slots filled by highest absolute values.

        Args:
            scores: Array of activation scores

        Returns:
            Array of selected indices, sorted by absolute value descending
        """
        n_total = self.examples_per_pattern

        # Calculate minimum counts for each sign
        min_positive_count = int(np.ceil(n_total * self.min_positive_ratio))
        min_negative_count = int(np.ceil(n_total * self.min_negative_ratio))

        # Get indices for positive and negative scores
        positive_mask = scores > 0
        negative_mask = scores < 0

        positive_indices = np.where(positive_mask)[0]
        negative_indices = np.where(negative_mask)[0]

        # Sort positive by value descending, negative by value ascending (most negative first)
        positive_sorted = positive_indices[np.argsort(scores[positive_indices])[::-1]]
        negative_sorted = negative_indices[np.argsort(scores[negative_indices])]

        # Select required minimum from each
        required_positive = positive_sorted[:min_positive_count]
        required_negative = negative_sorted[:min_negative_count]

        # Combine required indices
        selected = set(required_positive.tolist() + required_negative.tolist())

        # Calculate remaining slots
        remaining_slots = n_total - len(selected)

        if remaining_slots > 0:
            # Get all indices not yet selected, sort by absolute value
            all_indices = np.arange(len(scores))
            unselected_mask = ~np.isin(all_indices, list(selected))
            unselected_indices = all_indices[unselected_mask]

            # Sort unselected by absolute value descending
            abs_sorted = unselected_indices[
                np.argsort(np.abs(scores[unselected_indices]))[::-1]
            ]

            # Fill remaining slots
            for idx in abs_sorted[:remaining_slots]:
                selected.add(idx)

        # Convert to array and sort by absolute value descending for final output
        selected_array = np.array(list(selected))
        selected_array = selected_array[
            np.argsort(np.abs(scores[selected_array]))[::-1]
        ]

        return selected_array

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max words."""
        words = text.split()
        if len(words) > self.max_words:
            return " ".join(words[:self.max_words]) + "..."
        return text

    def _save_examples(self, all_examples: dict[str, list[PatternExamples]]) -> None:
        """Save examples to JSON file."""
        if self.output_path is None:
            logger.warning("output_path not configured, skipping save")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        output = self.to_dict(all_examples)

        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved message examples to {self.output_path}")

    @staticmethod
    def to_dict(all_examples: dict[str, list["PatternExamples"]]) -> dict[str, Any]:
        """Convert PatternExamples to JSON-serializable dict."""
        output = {}
        for level_key, examples_list in all_examples.items():
            output[level_key] = {}
            for pe in examples_list:
                pattern_key = f"{pe.level}_{pe.pattern_idx}"
                output[level_key][pattern_key] = {
                    "encoder": pe.encoder,
                    "level": pe.level,
                    "pattern_idx": pe.pattern_idx,
                    "examples": pe.examples,
                }
        return output

    @staticmethod
    def load_examples(path: str | Path) -> dict[str, Any]:
        """Load examples from JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def format_for_llm(
        self,
        pattern_examples: PatternExamples,
        max_examples: int = 10,
    ) -> str:
        """
        Format examples for LLM prompt.

        Args:
            pattern_examples: PatternExamples to format
            max_examples: Maximum examples to include

        Returns:
            Formatted string for LLM prompt
        """
        lines = []
        for i, ex in enumerate(pattern_examples.examples[:max_examples]):
            lines.append(f"{i+1}. (score: {ex['score']:.3f}) {ex['text']}")
        return "\n".join(lines)

    def _normalize_scores(
        self, all_examples: dict[str, list[PatternExamples]]
    ) -> dict[str, list[PatternExamples]]:
        """
        Normalize scores based on configured mode.

        Modes:
        - "global": Normalize across ALL scores in all_examples
        - "layer": Normalize within each layer (enc1_bottom, enc1_mid, unified, etc.)
        - "dimension": Normalize within each pattern independently
        """
        if self.score_mode == "global":
            return self._normalize_scores_globally(all_examples)
        elif self.score_mode == "layer":
            return self._normalize_scores_per_layer(all_examples)
        elif self.score_mode == "dimension":
            return self._normalize_scores_per_dimension(all_examples)
        else:
            raise ValueError(f"Unknown normalization mode: {self.score_mode}")

    def _normalize_scores_globally(
        self, all_examples: dict[str, list[PatternExamples]]
    ) -> dict[str, list[PatternExamples]]:
        """
        Normalize all scores globally across the entire message_examples structure.

        Uses the Nth percentile of absolute values as the scale factor,
        allowing some values to exceed +/-1.0.
        """
        # Collect all scores
        all_scores = []
        for level_key, examples_list in all_examples.items():
            for pe in examples_list:
                for ex in pe.examples:
                    all_scores.append(ex["score"])

        if not all_scores:
            return all_examples

        all_scores = np.array(all_scores)
        scale = np.percentile(np.abs(all_scores), self.score_percentile)

        if scale < 1e-8:
            logger.warning("Score normalization scale near zero, skipping normalization")
            return all_examples

        logger.info(f"Normalizing scores globally: scale={scale:.4f} "
                    f"(p{self.score_percentile} of {len(all_scores)} scores)")

        # Apply normalization
        for level_key, examples_list in all_examples.items():
            for pe in examples_list:
                for ex in pe.examples:
                    ex["score"] = float(ex["score"] / scale)

        return all_examples

    def _normalize_scores_per_layer(
        self, all_examples: dict[str, list[PatternExamples]]
    ) -> dict[str, list[PatternExamples]]:
        """
        Normalize scores within each layer independently.

        Each layer (enc1_bottom, enc1_mid, unified, etc.) gets its own
        scale factor based on the Nth percentile of absolute values.
        """
        for level_key, examples_list in all_examples.items():
            # Collect all scores for this layer
            layer_scores = []
            for pe in examples_list:
                for ex in pe.examples:
                    layer_scores.append(ex["score"])

            if not layer_scores:
                continue

            layer_scores = np.array(layer_scores)
            scale = np.percentile(np.abs(layer_scores), self.score_percentile)

            if scale < 1e-8:
                logger.warning(f"Score normalization scale near zero for {level_key}, skipping")
                continue

            logger.info(f"Normalizing scores for {level_key}: scale={scale:.4f} "
                        f"(p{self.score_percentile} of {len(layer_scores)} scores)")

            # Apply normalization to this layer
            for pe in examples_list:
                for ex in pe.examples:
                    ex["score"] = float(ex["score"] / scale)

        return all_examples

    def _normalize_scores_per_dimension(
        self, all_examples: dict[str, list[PatternExamples]]
    ) -> dict[str, list[PatternExamples]]:
        """
        Normalize scores within each pattern independently.

        Uses the Nth percentile of absolute values within each pattern
        as the scale factor.
        """
        for level_key, examples_list in all_examples.items():
            for pe in examples_list:
                # Collect scores for this pattern
                pattern_scores = [ex["score"] for ex in pe.examples]

                if not pattern_scores:
                    continue

                pattern_scores = np.array(pattern_scores)
                scale = np.percentile(np.abs(pattern_scores), self.score_percentile)

                if scale < 1e-8:
                    continue

                # Apply normalization to this pattern
                for ex in pe.examples:
                    ex["score"] = float(ex["score"] / scale)

        logger.info(f"Normalized scores per-dimension (p{self.score_percentile})")
        return all_examples
