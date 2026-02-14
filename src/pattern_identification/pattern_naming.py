"""
Pattern Naming

Generates human-readable names for patterns at all hierarchical levels
using LLM analysis of:
- Message examples (queried on-demand from message_scores.h5)
- Hierarchical composition (from SHAPAnalyzer)
- Rich context about the data, model, and disentanglement

Uses modular PromptBuilder for consistent, comprehensive prompts.
Dynamically iterates over encoders and levels from model metadata.

Process order (critical):
1. First level patterns (per encoder): Named from message examples
2. Middle level patterns (per encoder): Named from previous level composition + examples
3. Final level patterns (per encoder): Named from previous level composition + examples
4. Unified patterns: Named from cross-encoder composition + examples
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from src.llm import UnifiedLLMClient

from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class PatternNamer:
    """Generate names for patterns at all levels using modular prompts."""

    def __init__(self, config: dict[str, Any], output_path: str | Path | None = None):
        """
        Args:
            config: Full merged config
            output_path: Path for saving pattern names (checkpoint/resume).
                         If None, checkpointing is disabled.
        """
        naming_config = config["pattern_naming"]

        self.max_retries = naming_config["max_retries"]
        self.output_path = Path(output_path) if output_path else None

        # Debug directory for saving prompts and responses
        if self.output_path:
            self.debug_dir = self.output_path.parent / "debug"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.debug_dir = None

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(naming_config)

        # Initialize LLM client via unified interface
        self.llm_client = UnifiedLLMClient(config, config_key="pattern_naming")

    def name_all_patterns(
        self,
        hierarchical_weights: dict[str, Any],
        message_database: list[dict],
        query_examples_fn: Callable[[str, int, int], list[dict] | dict],
        resume: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Name all patterns at all levels.

        CRITICAL: Must process levels in order (first -> last -> unified)
        because higher levels reference lower level names.

        Supports checkpointing: saves after each successful LLM call.
        On resume, skips already-completed encoder/level combinations.

        Args:
            hierarchical_weights: Output from SHAPAnalyzer (includes metadata)
            message_database: Original message data (for detecting sources)
            query_examples_fn: Function(pattern_key, pattern_idx, limit) that returns
                               either list of example dicts or dict with "examples" and
                               optionally "aggregated_word_attributions"
            resume: If True, load existing partial results and skip completed items
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            Pattern names dictionary
        """
        # Extract metadata from SHAP output
        metadata = hierarchical_weights["metadata"]
        level_names = metadata["level_names"]
        encoder_names = metadata["encoder_names"]
        level_dims = metadata["level_dims"]
        num_levels = len(level_names)

        # Detect data sources from message database
        detected_sources = self._detect_data_sources(message_database)
        logger.info(f"Detected data sources: {detected_sources}")

        logger.info(f"Naming patterns for {len(encoder_names)} encoders, {num_levels} levels")

        # Calculate total steps for progress tracking
        total_steps = len(encoder_names) * num_levels + 1  # +1 for unified
        current_step = 0

        # Load existing partial results if resuming
        all_names = self._load_checkpoint() if resume else {}
        if all_names:
            logger.info(f"Resumed from checkpoint with {len(all_names)} completed items: {list(all_names.keys())}")

        # Process levels in order
        for level_idx, level_name in enumerate(level_names):
            # Determine level type
            if level_idx == 0:
                level_type = "first"
            elif level_idx == num_levels - 1:
                level_type = "final_encoder"
            else:
                level_type = "middle"

            for enc_name in encoder_names:
                examples_key = f"{enc_name}_{level_name}"

                # Skip if already completed
                if examples_key in all_names:
                    logger.info(f"Skipping {examples_key} (already completed)")
                    continue

                logger.info(f"Naming {level_name} patterns for {enc_name} (type: {level_type})...")

                # Query examples on-the-fly for each dimension
                n_dims = level_dims[level_name]
                examples_dict = self._query_examples_for_level(
                    query_examples_fn, examples_key, level_name, n_dims
                )

                if not examples_dict:
                    logger.warning(f"No examples found for {examples_key}, skipping")
                    continue

                # Get composition context for levels 2+
                composition_context = None
                prev_level = None
                prev_level_names = None

                if level_idx > 0:
                    prev_level = level_names[level_idx - 1]
                    transition_key = f"{prev_level}_to_{level_name}"
                    prev_names_key = f"{enc_name}_{prev_level}"

                    composition_context = hierarchical_weights[enc_name].get(transition_key, {})
                    prev_level_names = all_names.get(prev_names_key, {})

                # Build prompt
                prompt = self.prompt_builder.build_prompt(
                    level_type=level_type,
                    level_name=level_name,
                    encoder_name=enc_name,
                    examples=examples_dict,
                    model_metadata=metadata,
                    detected_sources=detected_sources,
                    composition_context=composition_context,
                    prev_level=prev_level,
                    prev_level_names=prev_level_names,
                )

                # Call LLM - pass expected keys for better missing detection
                expected_keys = list(examples_dict.keys())
                names = self._call_llm_for_names(
                    prompt,
                    len(examples_dict),
                    debug_key=examples_key,
                    expected_keys=expected_keys,
                )
                all_names[examples_key] = names
                logger.info(f"Named {len(names)} {level_name} patterns for {enc_name}")

                # Checkpoint after each successful naming
                self._save_names(all_names)
                current_step += 1
                logger.info(f"Checkpoint saved ({len(all_names)} items completed)")

                # Report progress
                if progress_callback:
                    progress_callback(
                        current_step,
                        total_steps,
                        f"Named {enc_name} {level_name} patterns ({current_step}/{total_steps})"
                    )

        # Unified level
        if "unified" not in all_names:
            logger.info("Naming unified patterns...")
            final_level = level_names[-1]

            # Gather all final level names for composition lookup
            all_final_names = {}
            for enc_name in encoder_names:
                all_final_names[f"{enc_name}_{final_level}"] = all_names.get(f"{enc_name}_{final_level}", {})

            # Build flattened names for unified level composition
            flattened_final_names = self._flatten_final_names(
                all_final_names,
                encoder_names,
                final_level,
                level_dims[final_level],
            )

            # Query unified examples on-the-fly
            unified_dims = level_dims.get("unified", level_dims[final_level])
            unified_examples = self._query_examples_for_level(
                query_examples_fn, "unified", "unified", unified_dims
            )

            if unified_examples:
                prompt = self.prompt_builder.build_prompt(
                    level_type="unified",
                    level_name="unified",
                    encoder_name=None,
                    examples=unified_examples,
                    model_metadata=metadata,
                    detected_sources=detected_sources,
                    composition_context=hierarchical_weights.get("final_to_unified", {}),
                    prev_level=final_level,
                    prev_level_names=flattened_final_names,
                )

                unified_expected_keys = list(unified_examples.keys())
                unified_names = self._call_llm_for_names(
                    prompt,
                    len(unified_examples),
                    debug_key="unified",
                    expected_keys=unified_expected_keys,
                )
                all_names["unified"] = unified_names
                logger.info(f"Named {len(unified_names)} unified patterns")

                # Final checkpoint
                self._save_names(all_names)

                # Report final progress
                if progress_callback:
                    progress_callback(total_steps, total_steps, "Named unified patterns")
            else:
                logger.warning("No unified examples found, skipping unified naming")
        else:
            logger.info("Skipping unified (already completed)")

        logger.info(f"Pattern naming complete: {len(all_names)} total items")
        return all_names

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load existing partial results if checkpoint file exists."""
        if self.output_path and self.output_path.exists():
            try:
                with open(self.output_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                return {}
        return {}

    def _query_examples_for_level(
        self,
        query_fn: Callable[[str, int, int], list[dict] | dict],
        pattern_key: str,
        level_name: str,
        n_dims: int,
    ) -> dict[str, dict]:
        """
        Query examples for all dimensions in a level.

        Args:
            query_fn: Function(pattern_key, pattern_idx, limit) -> examples
                      Returns either list of examples or dict with "examples" key
                      and optionally "aggregated_word_attributions"
            pattern_key: The key to query (e.g., "enc1_bottom", "unified")
            level_name: The level name for building dim keys (e.g., "bottom", "unified")
            n_dims: Number of dimensions at this level

        Returns:
            Dict mapping dim_key to {"examples": [...], "pattern_idx": int, ...}
        """
        max_examples = self.prompt_builder.max_examples
        examples_dict = {}

        for dim_idx in range(n_dims):
            dim_key = f"{level_name}_{dim_idx}"
            try:
                result = query_fn(pattern_key, dim_idx, max_examples)
                if result:
                    # Handle both list and dict return formats
                    if isinstance(result, list):
                        examples_dict[dim_key] = {
                            "examples": result,
                            "pattern_idx": dim_idx,
                        }
                    elif isinstance(result, dict):
                        examples_dict[dim_key] = {
                            "examples": result.get("examples", []),
                            "pattern_idx": dim_idx,
                            "aggregated_word_attributions": result.get("aggregated_word_attributions", []),
                        }
            except Exception as e:
                logger.warning(f"Failed to query examples for {pattern_key}[{dim_idx}]: {e}")

        return examples_dict

    def _detect_data_sources(self, message_database: list[dict]) -> list[str]:
        """Detect which data sources are present in the message database."""
        sources = set()
        for msg in message_database:
            source = msg.get("source", "").lower()
            if source:
                # Normalize source names
                if "github" in source:
                    sources.add("github")
                elif "slack" in source:
                    sources.add("slack")
                elif "jira" in source:
                    sources.add("jira")
                elif "confluence" in source:
                    sources.add("confluence")
                elif "trello" in source:
                    sources.add("trello")
                else:
                    sources.add(source)
        return sorted(list(sources))

    def _flatten_final_names(
        self,
        all_final_names: dict[str, dict],
        encoder_names: list[str],
        final_level: str,
        final_dim: int,
    ) -> dict[str, dict[str, str]]:
        """
        Flatten final level names for unified level composition lookup.

        The unified level receives concatenated final latents from all encoders.
        SHAP weights reference global dimension indices (0 to num_encoders * final_dim - 1).
        This function maps those global indices back to encoder-specific pattern names.

        Args:
            all_final_names: Dict mapping encoder_level keys to pattern names
            encoder_names: List of encoder names
            final_level: Name of final level
            final_dim: Dimension of final level per encoder

        Returns:
            Dict mapping global dim keys (e.g., "dim_5") to pattern info
        """
        flattened = {}
        for enc_idx, enc_name in enumerate(encoder_names):
            enc_key = f"{enc_name}_{final_level}"
            enc_names = all_final_names.get(enc_key, {})

            for local_dim in range(final_dim):
                global_dim = enc_idx * final_dim + local_dim
                local_key = f"{final_level}_{local_dim}"

                if local_key in enc_names:
                    # Map global dim to the local pattern info, with encoder annotation
                    pattern_info = enc_names[local_key].copy()
                    pattern_info["encoder"] = enc_name
                    flattened[f"{final_level}_{global_dim}"] = pattern_info

        return flattened

    def _call_llm_for_names(
        self,
        prompt: str,
        expected_count: int,
        debug_key: str = "unknown",
        expected_keys: list[str] | None = None,
    ) -> dict[str, dict[str, str]]:
        """
        Call LLM and parse response. Saves prompt and response to debug folder.

        Supports partial retries: if some patterns are named but others missing,
        subsequent retries only request the missing patterns. If zero patterns
        are returned (catastrophic failure), retries from scratch.

        On restart, checks for partial results file and resumes with follow-up prompt.

        Args:
            prompt: The full prompt for naming all patterns
            expected_count: Number of patterns expected
            debug_key: Key for debug file naming (e.g., "enc1_bottom")
            expected_keys: List of expected pattern keys (e.g., ["bottom_0", "bottom_1", ...])

        Returns:
            Dictionary mapping pattern keys to {"name": str, "description": str}

        Raises:
            RuntimeError: If naming fails after max_retries attempts
        """
        # Save original prompt for debugging
        if self.debug_dir:
            prompt_path = self.debug_dir / f"{debug_key}_prompt.txt"
            with open(prompt_path, "w") as f:
                f.write(prompt)

        # Check for partial results from previous run
        partial_path = self.debug_dir / f"{debug_key}_partial.json" if self.debug_dir else None
        accumulated_names = self._load_partial_results(partial_path) if partial_path else {}

        if accumulated_names:
            logger.info(f"Loaded {len(accumulated_names)} partial results for {debug_key}")
            # Start with follow-up prompt for missing patterns
            missing_keys = self._find_missing_keys(accumulated_names, expected_keys, expected_count)
            if not missing_keys:
                # All patterns already named
                return self._validate_names(accumulated_names, expected_keys)
            current_prompt = self._build_retry_prompt(accumulated_names, missing_keys)
            logger.info(f"Resuming with follow-up prompt for {len(missing_keys)} missing patterns")
        else:
            current_prompt = prompt

        for attempt in range(self.max_retries):
            try:
                # Call LLM with JSON mode enabled
                result = self.llm_client.generate_content_json_mode(current_prompt)
                text = result["text"]

                # Save raw response for debugging (with attempt number if retry)
                if self.debug_dir:
                    suffix = "" if attempt == 0 else f"_retry{attempt}"
                    response_path = self.debug_dir / f"{debug_key}_response{suffix}.txt"
                    with open(response_path, "w") as f:
                        f.write(text)

                # Parse JSON using unified client's robust parser
                names = self.llm_client._parse_json(text)
                if names is None:
                    names = {}

                # Merge new names into accumulated results
                accumulated_names.update(names)

                # Save partial results for potential resume
                if partial_path:
                    self._save_partial_results(partial_path, accumulated_names)

                # Validate all patterns have name and description
                validated = self._validate_names(accumulated_names, expected_keys)

                if len(validated) >= expected_count:
                    # Success - clean up partial file
                    if partial_path and partial_path.exists():
                        partial_path.unlink()
                    return validated

                # Check if this was a catastrophic failure (zero new names)
                if len(names) == 0:
                    logger.warning(
                        f"LLM returned 0 names (catastrophic failure). "
                        f"Retrying from scratch. Attempt {attempt + 1}/{self.max_retries}"
                    )
                    # Reset and retry from scratch
                    accumulated_names = {}
                    current_prompt = prompt
                    continue

                # Partial success - build follow-up prompt for missing patterns only
                missing_keys = self._find_missing_keys(validated, expected_keys, expected_count)
                logger.warning(
                    f"LLM returned {len(names)} names, have {len(validated)}/{expected_count} valid. "
                    f"Missing: {missing_keys}. Attempt {attempt + 1}/{self.max_retries}"
                )

                # Build retry prompt for missing patterns only
                current_prompt = self._build_retry_prompt(validated, missing_keys)

                # Save retry prompt for debugging
                if self.debug_dir:
                    retry_prompt_path = self.debug_dir / f"{debug_key}_retry{attempt + 1}_prompt.txt"
                    with open(retry_prompt_path, "w") as f:
                        f.write(current_prompt)

            except Exception as e:
                logger.warning(f"LLM call failed: {e}. Attempt {attempt + 1}/{self.max_retries}")
                # On exception, retry from scratch
                accumulated_names = {}
                current_prompt = prompt

        raise RuntimeError(
            f"Failed to get all pattern names for {debug_key} after {self.max_retries} attempts. "
            f"Got {len(accumulated_names)}/{expected_count}. Partial results saved to {partial_path}. "
            f"Restart to resume from partial results."
        )

    def _load_partial_results(self, path: Path) -> dict[str, dict[str, str]]:
        """Load partial naming results if they exist."""
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load partial results: {e}")
        return {}

    def _save_partial_results(self, path: Path, names: dict[str, dict[str, str]]) -> None:
        """Save partial naming results for potential resume."""
        with open(path, "w") as f:
            json.dump(names, f, indent=2)

    def _validate_names(
        self,
        names: dict[str, dict[str, str]],
        expected_keys: list[str] | None,
    ) -> dict[str, dict[str, str]]:
        """
        Validate that each pattern has both name and description.

        Returns only valid entries (those with both name and description).
        """
        valid = {}
        for key, info in names.items():
            if not isinstance(info, dict):
                logger.warning(f"Pattern {key} has invalid format: {info}")
                continue
            name = info.get("name", "").strip()
            description = info.get("description", "").strip()
            if name and description:
                valid[key] = {"name": name, "description": description}
            else:
                logger.warning(f"Pattern {key} missing name or description: name='{name}', desc='{description[:50] if description else ''}'")
        return valid

    def _find_missing_keys(
        self,
        current_names: dict[str, Any],
        expected_keys: list[str] | None,
        expected_count: int,
    ) -> list[str]:
        """Find which pattern keys are missing from current results."""
        if expected_keys:
            return [k for k in expected_keys if k not in current_names]
        # Infer missing keys from count
        # Assumes keys are like "level_0", "level_1", etc.
        all_keys = set(current_names.keys())
        missing = []
        for i in range(expected_count):
            # Try common key patterns
            for prefix in ["bottom", "mid", "top", "unified"]:
                key = f"{prefix}_{i}"
                if key not in all_keys:
                    missing.append(key)
                    break
        return missing

    def _build_retry_prompt(
        self,
        existing_names: dict[str, dict[str, str]],
        missing_keys: list[str],
    ) -> str:
        """Build a follow-up prompt requesting only missing patterns."""
        lines = [
            "You previously provided names for some patterns, but the following are still missing.",
            "Please provide names and descriptions for ONLY the missing patterns listed below.",
            "",
            "## Already Named (for context):",
            "",
        ]

        for key, info in existing_names.items():
            name = info.get("name", "Unknown")
            desc = info.get("description", "")[:100]
            lines.append(f"- {key}: \"{name}\" - {desc}")

        lines.extend([
            "",
            "## Missing Patterns (please name these):",
            "",
        ])

        for key in missing_keys:
            lines.append(f"- {key}")

        lines.extend([
            "",
            "## Output Format (JSON):",
            "Return ONLY valid JSON with no markdown formatting or code blocks.",
            "{",
        ])

        for key in missing_keys[:3]:
            lines.append(f'  "{key}": {{"name": "...", "description": "..."}},')

        lines.extend([
            "  ...",
            "}",
        ])

        return "\n".join(lines)

    def _save_names(self, names: dict[str, Any]) -> None:
        """Save pattern names to JSON (checkpoint)."""
        if not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(names, f, indent=2)
        logger.info(f"Saved pattern names to {self.output_path}")

    @staticmethod
    def load_names(path: str | Path) -> dict[str, Any]:
        """Load pattern names from JSON."""
        with open(path, "r") as f:
            return json.load(f)
