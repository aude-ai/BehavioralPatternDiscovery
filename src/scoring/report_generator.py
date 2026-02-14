"""
Report Generator

Generates LLM-based performance reports for individual engineers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.llm import UnifiedLLMClient

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate performance reports using LLM."""

    def __init__(self, config: dict[str, Any], debug_dir: Path | str | None = None):
        """
        Initialize report generator.

        Args:
            config: Full merged configuration
            debug_dir: Directory for saving LLM prompts and responses
        """
        self.config = config
        self.report_config = config["report"]

        self.llm_client = UnifiedLLMClient(config, config_key="report", debug_dir=debug_dir)

        self.strength_threshold = self.report_config["content"]["strength_threshold"]
        self.weakness_threshold = self.report_config["content"]["weakness_threshold"]
        self.max_examples_per_pattern = self.report_config["content"].get("max_examples_per_pattern", 5)

    def generate_report(
        self,
        engineer_id: str,
        scores: dict[str, Any],
        pattern_names: dict[str, Any],
        query_examples_fn: Any = None,
    ) -> dict[str, Any]:
        """
        Generate performance report for an engineer.

        Args:
            engineer_id: Engineer being reported on
            scores: Individual scores from IndividualScorer
            pattern_names: LLM-generated pattern names
            query_examples_fn: Optional function(pattern_key, pattern_idx, limit) -> examples

        Returns:
            Report dict with summary
        """
        logger.info(f"Generating report for {engineer_id}")

        # Categorize patterns by percentile
        patterns = scores["patterns"]
        strengths = [p for p in patterns if p["percentile"] >= self.strength_threshold]
        weaknesses = [p for p in patterns if p["percentile"] <= self.weakness_threshold]

        # Sort by percentile
        strengths.sort(key=lambda x: x["percentile"], reverse=True)
        weaknesses.sort(key=lambda x: x["percentile"])

        # Query message examples for top strengths and weaknesses
        strength_examples = {}
        weakness_examples = {}
        if query_examples_fn:
            for p in strengths[:5]:
                examples = self._query_pattern_examples(query_examples_fn, p)
                if examples:
                    strength_examples[p["name"]] = examples

            for p in weaknesses[:5]:
                examples = self._query_pattern_examples(query_examples_fn, p)
                if examples:
                    weakness_examples[p["name"]] = examples

            logger.info(
                f"Queried examples: {len(strength_examples)} strengths, "
                f"{len(weakness_examples)} weaknesses"
            )

        # Build prompt
        prompt = self._build_report_prompt(
            engineer_id=engineer_id,
            strengths=strengths[:10],
            weaknesses=weaknesses[:10],
            n_messages=scores["n_messages"],
            strength_examples=strength_examples,
            weakness_examples=weakness_examples,
        )

        # Generate report
        result = self.llm_client.generate_json_content(
            prompt=prompt,
            max_retries=self.report_config["max_retries"],
            log_name=f"report_{engineer_id}",
        )

        if not result["success"]:
            logger.error(f"Report generation failed: {result.get('error')}")
            report = {
                "engineer_id": engineer_id,
                "overall_summary": "Report generation failed. Please try again.",
                "error": result.get("error"),
            }
        else:
            report = result["data"]
            report["engineer_id"] = engineer_id

        return report

    def _query_pattern_examples(
        self,
        query_fn: Any,
        pattern: dict,
    ) -> list[dict]:
        """Query message examples for a pattern, handling errors gracefully."""
        try:
            result = query_fn(pattern["level"], pattern["dim"], self.max_examples_per_pattern)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return result.get("examples", [])
            return []
        except Exception as e:
            logger.warning(f"Failed to query examples for {pattern['name']}: {e}")
            return []

    def _build_report_prompt(
        self,
        engineer_id: str,
        strengths: list[dict],
        weaknesses: list[dict],
        n_messages: int,
        strength_examples: dict[str, list[dict]] | None = None,
        weakness_examples: dict[str, list[dict]] | None = None,
    ) -> str:
        """Build the prompt for report generation."""
        min_sentences = self.report_config["content"]["summary_sentences_min"]
        max_sentences = self.report_config["content"]["summary_sentences_max"]

        prompt = f"""You are generating a professional performance report for an engineer based on behavioral pattern analysis of their work communications.

## Engineer Profile
- **Engineer ID**: {engineer_id}
- **Messages Analyzed**: {n_messages}

## Pattern Analysis Summary

Patterns are behavioral traits detected in the engineer's work communications.
Percentiles indicate how the engineer compares to the population (100 = top performer).

### Strengths (Top Patterns, >= {self.strength_threshold}th percentile)
"""
        if strengths:
            for p in strengths:
                prompt += f"- **{p['name']}**: {p['percentile']}th percentile\n"
                # Add example messages if available
                examples = (strength_examples or {}).get(p["name"], [])
                if examples:
                    prompt += self._format_examples(examples)
        else:
            prompt += "- No standout strengths identified\n"

        prompt += f"""
### Areas for Growth (<= {self.weakness_threshold}th percentile)
"""
        if weaknesses:
            for p in weaknesses:
                prompt += f"- **{p['name']}**: {p['percentile']}th percentile\n"
                # Add example messages if available
                examples = (weakness_examples or {}).get(p["name"], [])
                if examples:
                    prompt += self._format_examples(examples)
        else:
            prompt += "- No significant weaknesses identified\n"

        prompt += f"""
## Instructions

Generate a performance report with the following structure. Return valid JSON.

{{
  "overall_summary": "Markdown-formatted summary with ## Overview, ## Strengths, ## Growth Areas, ## Recommendations sections. {min_sentences}-{max_sentences} sentences total."
}}

Guidelines:
- Be specific and actionable — reference concrete behaviors from the example messages
- Reference the pattern names naturally
- When citing evidence, describe the behavior you observe in the messages, don't quote them verbatim
- Focus on professional development
- Maintain a constructive, supportive tone
- Use markdown formatting for readability
"""

        return prompt

    def _format_examples(self, examples: list[dict]) -> str:
        """Format message examples for inclusion in the report prompt."""
        lines = ["  Example messages from this engineer:\n"]
        for i, ex in enumerate(examples, 1):
            text = ex.get("text", "")[:200]
            source = ex.get("source", "")
            activity_type = ex.get("activity_type", "")
            source_tag = f"({source}/{activity_type}) " if source else ""
            lines.append(f"  {i}. {source_tag}{text}\n")
        return "".join(lines)
