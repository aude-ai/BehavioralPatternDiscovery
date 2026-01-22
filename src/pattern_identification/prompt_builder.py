"""
Prompt Builder for Pattern Naming

Builds modular, composable prompts for LLM pattern naming.
Imports templates from prompt_templates.py and assembles them
based on level type and configuration.
"""

import logging
from typing import Any

from . import prompt_templates as templates

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts for pattern naming from modular components.

    Imports static content from prompt_templates.py and assembles
    complete prompts based on:
    - Level type (first, middle, final_encoder, unified)
    - Model metadata (encoders, levels, dimensions)
    - Detected data sources
    - Configuration options
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: pattern_naming config section
        """
        self.config = config
        self.max_examples = config["max_examples_in_prompt"]
        self.max_words = config["naming_guidelines"]["max_name_words"]

        prompt_config = config["prompts"]
        self.max_contributors = prompt_config["max_composition_contributors"]

        # Which sections to include
        self.include_system = prompt_config["include_system_context"]
        self.include_data_source = prompt_config["include_data_source_context"]
        self.include_team = prompt_config["include_team_context"]
        self.include_disentanglement = prompt_config["include_disentanglement_context"]
        self.include_model_structure = prompt_config["include_model_structure_context"]
        self.include_word_attribution = prompt_config.get("include_word_attribution_context", True)

        # Custom notes per level
        self.custom_notes = prompt_config["custom_level_notes"]

    def build_prompt(
        self,
        level_type: str,
        level_name: str,
        encoder_name: str | None,
        examples: dict[str, Any],
        model_metadata: dict[str, Any],
        detected_sources: list[str],
        composition_context: dict[str, Any] | None = None,
        prev_level: str | None = None,
        prev_level_names: dict[str, dict[str, str]] | None = None,
    ) -> str:
        """
        Build complete prompt from modular components.

        Args:
            level_type: "first", "middle", "final_encoder", or "unified"
            level_name: Name of current level (e.g., "bottom", "mid")
            encoder_name: Name of encoder (e.g., "enc1") or None for unified
            examples: Pattern examples from MessageAssigner
            model_metadata: Metadata from SHAP analysis
            detected_sources: List of data sources found in data
            composition_context: SHAP weights for composition (levels 2+)
            prev_level: Name of previous level (levels 2+)
            prev_level_names: Names assigned to previous level patterns

        Returns:
            Complete prompt string
        """
        parts = []

        # 1. System context
        if self.include_system:
            parts.append(self._build_system_context())

        # 2. Data source context (dynamic)
        if self.include_data_source:
            parts.append(self._build_data_source_context(detected_sources))

        # 3. Team context
        if self.include_team:
            parts.append(self._build_team_context())

        # 4. Disentanglement context
        if self.include_disentanglement:
            parts.append(self._build_disentanglement_context())

        # 5. Model structure context
        if self.include_model_structure:
            parts.append(self._build_model_structure_context(model_metadata))

        # 5.5. Word attribution context (if examples contain word attributions)
        if self.include_word_attribution and self._examples_have_word_attributions(examples):
            parts.append(self._build_word_attribution_context())

        # 6. Level-specific context
        parts.append(self._build_level_context(
            level_type=level_type,
            level_name=level_name,
            encoder_name=encoder_name,
            prev_level=prev_level,
            num_encoders=model_metadata["num_encoders"],
            final_level=model_metadata["level_names"][-1],
        ))

        # 7. Composition context (if not first level)
        if composition_context and prev_level_names and prev_level:
            parts.append(self._build_composition_header(prev_level))

        # 8. Examples with inline composition
        parts.append(self._build_examples_section(
            examples=examples,
            composition_weights=composition_context,
            prev_level_names=prev_level_names,
            prev_level=prev_level,
        ))

        # 9. Naming guidelines
        parts.append(self._build_naming_guidelines())

        # 10. Output format
        parts.append(self._build_output_format(examples))

        return "\n\n".join(filter(None, parts))

    # =========================================================================
    # Component builders - import from templates and format as needed
    # =========================================================================

    def _build_system_context(self) -> str:
        """Return system context from templates."""
        return templates.SYSTEM_CONTEXT

    def _build_data_source_context(self, detected_sources: list[str]) -> str:
        """Build data source context dynamically from detected sources."""
        lines = [templates.DATA_SOURCE_HEADER]

        for source in detected_sources:
            source_lower = source.lower()
            if source_lower in templates.DATA_SOURCE_DESCRIPTIONS:
                info = templates.DATA_SOURCE_DESCRIPTIONS[source_lower]
                lines.append(f"### {info['name']}")
                for msg_type in info["types"]:
                    lines.append(f"- {msg_type}")
                lines.append("")

        lines.append(templates.MESSAGE_CHARACTERISTICS)
        return "\n".join(lines)

    def _build_team_context(self) -> str:
        """Return team context from templates."""
        return templates.TEAM_CONTEXT

    def _build_disentanglement_context(self) -> str:
        """Return disentanglement context from templates."""
        return templates.DISENTANGLEMENT_CONTEXT

    def _build_word_attribution_context(self) -> str:
        """Return word attribution context from templates."""
        return templates.WORD_ATTRIBUTION_CONTEXT

    def _examples_have_word_attributions(self, examples: dict) -> bool:
        """Check if any examples contain word attributions."""
        for pattern_key, pattern_data in examples.items():
            if pattern_data.get("aggregated_word_attributions"):
                return True
        return False

    def _build_model_structure_context(self, metadata: dict) -> str:
        """Build model structure context with actual dimensions."""
        level_names = metadata["level_names"]
        encoder_names = metadata["encoder_names"]
        level_dims = metadata["level_dims"]
        unified_dim = metadata["unified_dim"]
        num_encoders = len(encoder_names)
        num_levels = len(level_names)

        lines = [
            templates.MODEL_STRUCTURE_HEADER.format(
                num_encoders=num_encoders,
                num_levels=num_levels,
            ),
            "",
            "### Dimension Counts",
            "",
        ]

        for level in level_names:
            lines.append(f"- **{level}**: {level_dims[level]} dimensions per encoder")
        lines.append(f"- **unified**: {unified_dim} dimensions (combines all {num_encoders} encoders)")

        return "\n".join(lines)

    def _build_level_context(
        self,
        level_type: str,
        level_name: str,
        encoder_name: str | None,
        prev_level: str | None,
        num_encoders: int,
        final_level: str,
    ) -> str:
        """Build level-specific context from templates with abstraction info."""
        template = templates.LEVEL_CONTEXTS[level_type]

        # Get abstraction info for current level
        current_abstraction = templates.get_abstraction_for_level(level_name)

        # Get abstraction info for previous level (if applicable)
        prev_abstraction = {}
        if prev_level:
            prev_abstraction = templates.get_abstraction_for_level(prev_level)

        # Format examples as bullet list
        examples_list = current_abstraction["examples"]
        if examples_list:
            examples_formatted = "\n".join(f"- {ex}" for ex in examples_list)
        else:
            examples_formatted = "(see guidelines below)"

        context = template.format(
            level_name=level_name,
            encoder_name=encoder_name if encoder_name else "all encoders",
            prev_level=prev_level if prev_level else "",
            num_encoders=num_encoders,
            final_level=final_level,
            # Current level abstraction
            current_abstraction_degree=current_abstraction["degree"],
            current_abstraction_description=current_abstraction["description"],
            current_abstraction_detail=current_abstraction["detail"],
            current_abstraction_examples=examples_formatted,
            # Previous level abstraction (empty strings if first level)
            prev_abstraction_degree=prev_abstraction.get("degree", ""),
            prev_abstraction_description=prev_abstraction.get("description", ""),
        )

        # Append custom notes if configured
        custom = self.custom_notes.get(level_type, "")
        if custom:
            context += f"\n\n### Additional Notes\n{custom}"

        return context

    def _build_composition_header(self, prev_level: str) -> str:
        """Return composition header from templates."""
        return templates.COMPOSITION_HEADER.format(prev_level=prev_level)

    def _build_examples_section(
        self,
        examples: dict[str, Any],
        composition_weights: dict | None,
        prev_level_names: dict | None,
        prev_level: str | None,
    ) -> str:
        """Build examples section with inline composition context."""
        lines = ["## Patterns to Name", ""]

        for pattern_key, pattern_data in examples.items():
            pattern_idx = pattern_data["pattern_idx"]
            lines.append(f"### {pattern_key}")
            lines.append("")

            # Add composition if available (for levels 2+)
            if composition_weights and prev_level_names and prev_level:
                comp = self._format_composition(
                    composition_weights,
                    prev_level_names,
                    prev_level,
                    pattern_idx,
                )
                lines.append(comp)
                lines.append("")

            # Add message examples
            lines.append("**Top activating messages:**")
            lines.append("")
            for i, ex in enumerate(pattern_data["examples"][:self.max_examples]):
                score = ex.get("score", 0)
                percentile = ex.get("percentile", 0)
                text = ex.get("text", "")
                lines.append(f"{i+1}. [score: {score:.2f}, top {100-percentile:.1f}%] {text}")
            lines.append("")

            # Add word attributions if present
            aggregated = pattern_data.get("aggregated_word_attributions")
            if aggregated:
                lines.append(self._format_word_attributions(
                    aggregated,
                    len(pattern_data["examples"][:self.max_examples])
                ))
                lines.append("")

        return "\n".join(lines)

    def _format_word_attributions(
        self,
        aggregated: list[dict],
        num_examples: int,
    ) -> str:
        """Format aggregated word attributions for prompt.

        Args:
            aggregated: List of word attribution dicts with word, mean_delta, occurrences
            num_examples: Number of message examples shown
        """
        if not aggregated:
            return ""

        lines = ["**Word Attributions (aggregated across top messages):**", ""]

        for w in aggregated[:8]:  # Show top 8 words
            delta = w["mean_delta"]
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"- \"{w['word']}\" ({sign}{delta:.3f}, "
                f"in {w['occurrences']}/{num_examples} messages)"
            )

        return "\n".join(lines)

    def _format_composition(
        self,
        weights: dict,
        prev_level_names: dict,
        prev_level: str,
        pattern_idx: int,
    ) -> str:
        """Format composition showing which lower patterns contribute."""
        lines = [f"**Composed from {prev_level} patterns:**"]

        weight_key = f"dim_{pattern_idx}"
        if weight_key not in weights:
            lines.append("  (composition data not available)")
            return "\n".join(lines)

        for contrib in weights[weight_key][:self.max_contributors]:
            source_dim = list(contrib.keys())[0]
            weight_value = list(contrib.values())[0]
            dim_num = source_dim.split("_")[1]

            # Look up the name for this dimension
            prev_pattern_key = f"{prev_level}_{dim_num}"
            prev_info = prev_level_names.get(prev_pattern_key, {})
            prev_name = prev_info.get("name", f"Pattern {dim_num}")
            prev_desc = prev_info.get("description", "")

            if prev_desc:
                lines.append(f"  - **{prev_name}** (weight: {weight_value:.3f}): {prev_desc[:80]}...")
            else:
                lines.append(f"  - **{prev_name}** (weight: {weight_value:.3f})")

        return "\n".join(lines)

    def _build_naming_guidelines(self) -> str:
        """Return naming guidelines with configured max_words."""
        return templates.NAMING_GUIDELINES_TEMPLATE.format(max_words=self.max_words)

    def _build_output_format(self, examples: dict) -> str:
        """Build output format with actual pattern keys."""
        pattern_keys = list(examples.keys())[:3]

        lines = [templates.OUTPUT_FORMAT_HEADER]
        for key in pattern_keys:
            lines.append(f'  "{key}": {{"name": "Your Pattern Name", "description": "What this pattern captures..."}},')
        lines.append("  ...")
        lines.append("}")
        lines.append(templates.OUTPUT_FORMAT_FOOTER)

        return "\n".join(lines)
