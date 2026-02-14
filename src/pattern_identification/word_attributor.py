"""
Word Attributor

Computes word-level score attributions using leave-one-out analysis.
For each top message per dimension, measures how removing each word
affects activation scores across ALL latent dimensions.

Returns aggregated word attributions per dimension, added to the
message_examples structure passed in by the caller.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.model.base import BaseVAE

logger = logging.getLogger(__name__)

STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "where", "when",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "also", "now", "here", "there",
    "then", "once", "if", "else", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "while", "any", "being", "am", "up", "down", "out", "off",
    "over", "because", "until", "against", "get", "got", "my", "your",
    "his", "her", "our", "their", "me", "him", "us", "them",
])


@dataclass
class AggregatedWordAttribution:
    """Aggregated attribution across multiple messages."""
    word: str
    mean_delta: float
    occurrences: int


class WordAttributor:
    """
    Compute word-level attributions using leave-one-out analysis.

    For each top message per dimension, removes each word and measures
    the score change across all latent dimensions. Outputs aggregated
    attributions per dimension.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: Full merged config
        """
        wa_config = config["word_attribution"]
        data_paths = config["paths"]["data"]["processing"]
        norm_config = config["normalization"]

        self.enabled = wa_config["enabled"]
        self.max_messages_per_dimension = wa_config["max_messages_per_dimension"]
        self.top_words_per_dimension = wa_config["top_words_per_dimension"]
        self.max_words_per_message = wa_config.get("max_words_per_message", 0)
        self.min_word_length = wa_config["min_word_length"]
        self.skip_stopwords = wa_config["skip_stopwords"]
        self.batch_size = wa_config["batch_size"]
        self.scoring_mode = wa_config["scoring_mode"]

        # Ratio constraints for absolute mode
        self.min_positive_ratio = wa_config.get("min_positive_ratio", 0.0)
        self.min_negative_ratio = wa_config.get("min_negative_ratio", 0.0)

        if self.scoring_mode not in ("positive", "absolute"):
            raise ValueError(
                f"Invalid word_attribution.scoring_mode: '{self.scoring_mode}'. "
                f"Must be 'positive' or 'absolute'."
            )

        if self.min_positive_ratio + self.min_negative_ratio > 1.0:
            raise ValueError(
                f"min_positive_ratio ({self.min_positive_ratio}) + "
                f"min_negative_ratio ({self.min_negative_ratio}) must be <= 1.0"
            )

        # Normalization config for word deltas
        self.normalize_deltas = norm_config["word_deltas"]["enabled"]
        self.delta_mode = norm_config["word_deltas"]["mode"]
        self.delta_percentile = norm_config["word_deltas"]["percentile"]

        # Text encoder config for re-encoding
        self.text_encoder_config = config

        # Determine training input path for aux features
        aux_enabled = config["model"]["input"]["aux_features"]["enabled"]
        self._aux_enabled = aux_enabled
        self._aux_path = Path(data_paths["train_aux_vars"])

        self.device = wa_config["device"]

        # Will be set during compute
        self._text_encoder = None
        self._aux_features = None

    def _get_text_encoder(self):
        """Lazy-load text encoder."""
        if self._text_encoder is None:
            from src.data.processing.encoders import text_encoder_registry
            encoder_config = self.text_encoder_config["processing"]["text_encoder"]
            encoder_type = encoder_config["type"]
            self._text_encoder = text_encoder_registry.create(
                encoder_type, config=self.text_encoder_config
            )
        return self._text_encoder

    def _load_aux_features(self) -> np.ndarray | None:
        """Load auxiliary features if enabled."""
        if not self._aux_enabled:
            return None
        if self._aux_path.exists():
            return np.load(self._aux_path)
        return None

    def compute_attributions(
        self,
        vae: BaseVAE,
        message_database: list[dict],
        message_examples: dict[str, Any],
        activations: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """
        Compute word attributions and add to message_examples.

        Args:
            vae: Trained VAE model
            message_database: List of message dicts with 'text' field
            message_examples: Dict of pattern examples (queried dynamically)
            activations: Output from BatchScorer (all activation arrays)

        Returns:
            Updated message_examples dict with aggregated_word_attributions
        """
        if not self.enabled:
            logger.info("Word attribution disabled, skipping")
            return message_examples

        vae.eval()
        vae.to(self.device)

        # Load aux features if needed
        self._aux_features = self._load_aux_features()

        # Collect all unique message indices to process
        message_indices_by_dim = self._collect_message_indices(message_examples)
        all_unique_indices = set()
        for indices in message_indices_by_dim.values():
            all_unique_indices.update(indices)

        logger.info(f"Computing word attributions for {len(all_unique_indices)} unique messages")

        # Process all unique messages once
        message_word_deltas = self._compute_all_word_deltas(
            vae, message_database, list(all_unique_indices), activations
        )

        # Add aggregated attributions to message examples
        # Use ALL processed messages for each dimension (not just dimension-specific samples)
        updated_examples = self._add_aggregated_attributions(
            message_examples, message_word_deltas, list(all_unique_indices)
        )

        # Apply normalization to word deltas if enabled
        if self.normalize_deltas:
            updated_examples = self._normalize_deltas(updated_examples)

        return updated_examples

    def _collect_message_indices(
        self, message_examples: dict[str, Any]
    ) -> dict[tuple[str, int], list[int]]:
        """
        Collect message indices to process per dimension.

        Returns:
            Dict mapping (level_key, pattern_idx) to list of message indices
        """
        indices_by_dim = {}

        for level_key, patterns in message_examples.items():
            for pattern_key, pattern_data in patterns.items():
                examples = pattern_data.get("examples", [])
                # Limit to max_messages_per_dimension
                limited_examples = examples[:self.max_messages_per_dimension]
                dim_key = (level_key, pattern_data["pattern_idx"])
                indices_by_dim[dim_key] = [ex["message_idx"] for ex in limited_examples]

        return indices_by_dim

    def _compute_all_word_deltas(
        self,
        vae: BaseVAE,
        message_database: list[dict],
        message_indices: list[int],
        activations: dict[str, np.ndarray],
    ) -> dict[int, dict[str, dict[str, float]]]:
        """
        Compute word deltas for all unique messages.

        Returns:
            Dict mapping message_idx -> word -> activation_key -> delta_array
        """
        text_encoder = self._get_text_encoder()
        results = {}

        for i, msg_idx in enumerate(tqdm(message_indices, desc="Computing word attributions")):
            text = message_database[msg_idx].get("text", "")
            words = self._tokenize(text)

            if not words:
                results[msg_idx] = {}
                continue

            # Get original activations for this message (from pre-computed)
            original_acts = {
                key: arr[msg_idx] for key, arr in activations.items()
            }

            # Generate all word-removed variants
            variants = []
            valid_words = []
            seen_removed_texts = set()

            for word in words:
                if self._should_skip_word(word):
                    continue
                removed_text = self._remove_word(text, word)
                # Skip if removal had no effect or duplicate
                if removed_text == text or removed_text in seen_removed_texts:
                    continue
                seen_removed_texts.add(removed_text)
                variants.append(removed_text)
                valid_words.append(word)

            if not variants:
                results[msg_idx] = {}
                continue

            # Apply max_words_per_message limit if configured
            original_variant_count = len(variants)
            if self.max_words_per_message > 0 and len(variants) > self.max_words_per_message:
                variants = variants[:self.max_words_per_message]
                valid_words = valid_words[:self.max_words_per_message]

            # Log progress for messages with many variants (potential bottleneck)
            if original_variant_count > 50:
                truncated = f" (truncated from {original_variant_count})" if len(variants) < original_variant_count else ""
                logger.info(
                    f"  Message {i+1}/{len(message_indices)} (idx={msg_idx}): "
                    f"{len(variants)} variants{truncated}, text_len={len(text)}"
                )

            # Batch encode all variants
            variant_embeddings = text_encoder.encode(variants)

            # If aux features enabled, append them
            if self._aux_enabled and self._aux_features is not None:
                aux = self._aux_features[msg_idx]
                aux_repeated = np.tile(aux, (len(variants), 1))
                variant_inputs = np.concatenate([variant_embeddings, aux_repeated], axis=1)
            else:
                variant_inputs = variant_embeddings

            # Get VAE activations for all variants
            variant_acts = self._get_vae_activations(vae, variant_inputs)

            # Compute deltas: original - variant (positive = word increases score)
            word_deltas = {}
            for i, word in enumerate(valid_words):
                word_deltas[word] = {}
                for key in original_acts.keys():
                    # Delta per dimension in this activation array
                    delta = original_acts[key] - variant_acts[key][i]
                    word_deltas[word][key] = delta.tolist()

            results[msg_idx] = word_deltas

        return results

    def _get_vae_activations(
        self, vae: BaseVAE, inputs: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Get VAE activations for input batch."""
        x_tensor = torch.from_numpy(inputs).float().to(self.device)

        with torch.no_grad():
            latent_codes = vae.get_latent_codes(x_tensor, deterministic=True)

        # Convert to numpy
        result = {}
        encoder_names = vae.get_encoder_names()
        level_names = vae.get_level_names()

        for enc_name in encoder_names:
            for level in level_names:
                key = f"{enc_name}_{level}"
                result[key] = latent_codes[enc_name][level].cpu().float().numpy()

        result["unified"] = latent_codes["unified"]["z"].cpu().float().numpy()

        return result

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into unique words."""
        words = re.findall(r'\b\w+\b', text.lower())
        # Return unique words while preserving order
        seen = set()
        unique_words = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)
        return unique_words

    def _should_skip_word(self, word: str) -> bool:
        """Check if word should be skipped."""
        if len(word) < self.min_word_length:
            return True
        if self.skip_stopwords and word.lower() in STOPWORDS:
            return True
        if word.isdigit():
            return True
        return False

    def _remove_word(self, text: str, word: str) -> str:
        """Remove all occurrences of word from text."""
        pattern = r'\b' + re.escape(word) + r'\b'
        result = re.sub(pattern, '', text, flags=re.IGNORECASE)
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def _add_aggregated_attributions(
        self,
        message_examples: dict[str, Any],
        message_word_deltas: dict[int, dict[str, dict[str, float]]],
        all_message_indices: list[int],
    ) -> dict[str, Any]:
        """
        Add aggregated word attributions to each dimension in message_examples.

        Uses word deltas from ALL processed messages (not just dimension-specific samples)
        to get comprehensive word attribution data for each dimension.

        Words are sorted based on scoring_mode:
        - "positive": Sort by mean_delta descending (highest positive first)
        - "absolute": Sort by abs(mean_delta) descending (largest magnitude first)
                      with min ratio constraints for positive/negative representation
        """
        for level_key, patterns in message_examples.items():
            for pattern_key, pattern_data in patterns.items():
                pattern_idx = pattern_data["pattern_idx"]

                # Collect word deltas for this dimension from ALL processed messages
                word_deltas_for_dim = defaultdict(list)

                for msg_idx in all_message_indices:
                    word_deltas = message_word_deltas.get(msg_idx, {})
                    for word, activation_deltas in word_deltas.items():
                        if level_key in activation_deltas:
                            delta_array = activation_deltas[level_key]
                            delta = delta_array[pattern_idx]
                            word_deltas_for_dim[word].append(delta)

                # Compute aggregated attributions
                all_attributions = []

                for word, deltas in word_deltas_for_dim.items():
                    mean_delta = float(np.mean(deltas))
                    occurrences = len(deltas)

                    all_attributions.append(AggregatedWordAttribution(
                        word=word,
                        mean_delta=round(mean_delta, 4),
                        occurrences=occurrences,
                    ))

                # Select and sort based on scoring_mode
                if self.scoring_mode == "positive":
                    # Highest positive values first
                    all_attributions.sort(key=lambda x: x.mean_delta, reverse=True)
                    selected = all_attributions[:self.top_words_per_dimension]
                else:  # absolute
                    selected = self._select_absolute_with_ratios(all_attributions)

                # Add to pattern data as flat list
                pattern_data["aggregated_word_attributions"] = [
                    {"word": a.word, "mean_delta": a.mean_delta, "occurrences": a.occurrences}
                    for a in selected
                ]

        return message_examples

    def _select_absolute_with_ratios(
        self, attributions: list[AggregatedWordAttribution]
    ) -> list[AggregatedWordAttribution]:
        """
        Select top words by absolute value while respecting min ratio constraints.

        Ensures minimum representation of both positive and negative attributions,
        with remaining slots filled by highest absolute values.

        Args:
            attributions: List of all word attributions

        Returns:
            Selected attributions sorted by absolute value descending
        """
        n_total = self.top_words_per_dimension

        if len(attributions) <= n_total:
            # Not enough to be selective, return all sorted by absolute
            return sorted(attributions, key=lambda x: abs(x.mean_delta), reverse=True)

        # Calculate minimum counts for each sign
        min_positive_count = int(np.ceil(n_total * self.min_positive_ratio))
        min_negative_count = int(np.ceil(n_total * self.min_negative_ratio))

        # Separate positive and negative attributions
        positive_attrs = [a for a in attributions if a.mean_delta > 0]
        negative_attrs = [a for a in attributions if a.mean_delta < 0]

        # Sort positive by value descending, negative by value ascending (most negative first)
        positive_sorted = sorted(positive_attrs, key=lambda x: x.mean_delta, reverse=True)
        negative_sorted = sorted(negative_attrs, key=lambda x: x.mean_delta)

        # Select required minimum from each
        selected = []
        selected_set = set()

        for attr in positive_sorted[:min_positive_count]:
            selected.append(attr)
            selected_set.add(attr.word)

        for attr in negative_sorted[:min_negative_count]:
            selected.append(attr)
            selected_set.add(attr.word)

        # Calculate remaining slots
        remaining_slots = n_total - len(selected)

        if remaining_slots > 0:
            # Get all attributions not yet selected, sort by absolute value
            unselected = [a for a in attributions if a.word not in selected_set]
            unselected_sorted = sorted(unselected, key=lambda x: abs(x.mean_delta), reverse=True)

            # Fill remaining slots
            selected.extend(unselected_sorted[:remaining_slots])

        # Sort final list by absolute value descending
        return sorted(selected, key=lambda x: abs(x.mean_delta), reverse=True)

    def _normalize_deltas(self, message_examples: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize mean_delta values based on configured mode.

        Modes:
        - "global": Normalize across ALL deltas in message_examples
        - "layer": Normalize within each layer (enc1_bottom, enc1_mid, unified, etc.)
        - "dimension": Normalize within each pattern independently
        """
        if self.delta_mode == "global":
            return self._normalize_deltas_globally(message_examples)
        elif self.delta_mode == "layer":
            return self._normalize_deltas_per_layer(message_examples)
        elif self.delta_mode == "dimension":
            return self._normalize_deltas_per_dimension(message_examples)
        else:
            raise ValueError(f"Unknown normalization mode: {self.delta_mode}")

    def _normalize_deltas_globally(
        self, message_examples: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Normalize all mean_delta values globally across the entire structure.

        Uses the Nth percentile of absolute values as the scale factor,
        allowing some values to exceed +/-1.0.
        """
        # Collect all mean_delta values
        all_deltas = []
        for level_key, patterns in message_examples.items():
            for pattern_key, pattern_data in patterns.items():
                aggregated = pattern_data.get("aggregated_word_attributions", [])
                for attr in aggregated:
                    all_deltas.append(attr["mean_delta"])

        if not all_deltas:
            return message_examples

        all_deltas = np.array(all_deltas)
        scale = np.percentile(np.abs(all_deltas), self.delta_percentile)

        if scale < 1e-8:
            logger.warning("Delta normalization scale near zero, skipping normalization")
            return message_examples

        logger.info(f"Normalizing word deltas globally: scale={scale:.4f} "
                    f"(p{self.delta_percentile} of {len(all_deltas)} deltas)")

        # Apply normalization
        for level_key, patterns in message_examples.items():
            for pattern_key, pattern_data in patterns.items():
                aggregated = pattern_data.get("aggregated_word_attributions", [])
                for attr in aggregated:
                    attr["mean_delta"] = round(attr["mean_delta"] / scale, 4)

        return message_examples

    def _normalize_deltas_per_layer(
        self, message_examples: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Normalize mean_delta values within each layer independently.

        Each layer (enc1_bottom, enc1_mid, unified, etc.) gets its own
        scale factor based on the Nth percentile of absolute values.
        """
        for level_key, patterns in message_examples.items():
            # Collect all deltas for this layer
            layer_deltas = []
            for pattern_key, pattern_data in patterns.items():
                aggregated = pattern_data.get("aggregated_word_attributions", [])
                for attr in aggregated:
                    layer_deltas.append(attr["mean_delta"])

            if not layer_deltas:
                continue

            layer_deltas = np.array(layer_deltas)
            scale = np.percentile(np.abs(layer_deltas), self.delta_percentile)

            if scale < 1e-8:
                logger.warning(f"Delta normalization scale near zero for {level_key}, skipping")
                continue

            logger.info(f"Normalizing word deltas for {level_key}: scale={scale:.4f} "
                        f"(p{self.delta_percentile} of {len(layer_deltas)} deltas)")

            # Apply normalization to this layer
            for pattern_key, pattern_data in patterns.items():
                aggregated = pattern_data.get("aggregated_word_attributions", [])
                for attr in aggregated:
                    attr["mean_delta"] = round(attr["mean_delta"] / scale, 4)

        return message_examples

    def _normalize_deltas_per_dimension(
        self, message_examples: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Normalize mean_delta values within each pattern independently.

        Uses the Nth percentile of absolute values within each pattern
        as the scale factor.
        """
        for level_key, patterns in message_examples.items():
            for pattern_key, pattern_data in patterns.items():
                aggregated = pattern_data.get("aggregated_word_attributions", [])

                if not aggregated:
                    continue

                # Collect deltas for this pattern
                pattern_deltas = np.array([attr["mean_delta"] for attr in aggregated])
                scale = np.percentile(np.abs(pattern_deltas), self.delta_percentile)

                if scale < 1e-8:
                    continue

                # Apply normalization to this pattern
                for attr in aggregated:
                    attr["mean_delta"] = round(attr["mean_delta"] / scale, 4)

        logger.info(f"Normalized word deltas per-dimension (p{self.delta_percentile})")
        return message_examples
