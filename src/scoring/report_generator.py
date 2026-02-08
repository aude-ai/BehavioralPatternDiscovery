"""
Report Generator

Generates LLM-based performance reports for individual engineers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.llm import UnifiedLLMClient

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate performance reports using LLM."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize report generator.

        Args:
            config: Full merged configuration
        """
        self.config = config
        self.paths = config["paths"]
        self.report_config = config["report"]

        self.output_dir = Path(self.paths["scoring"]["reports_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llm_client = UnifiedLLMClient(config, config_key="report")

        self.strength_threshold = self.report_config["content"]["strength_threshold"]
        self.weakness_threshold = self.report_config["content"]["weakness_threshold"]

    def generate_report(
        self,
        engineer_id: str,
        scores: dict[str, Any],
        pattern_names: dict[str, Any],
        message_examples: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate performance report for an engineer.

        Args:
            engineer_id: Engineer being reported on
            scores: Individual scores from IndividualScorer
            pattern_names: LLM-generated pattern names
            message_examples: Message examples from pattern identification

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

        # Build prompt
        prompt = self._build_report_prompt(
            engineer_id=engineer_id,
            strengths=strengths[:10],
            weaknesses=weaknesses[:10],
            n_messages=scores["n_messages"],
        )

        # Generate report
        result = self.llm_client.generate_json_content(
            prompt=prompt,
            max_retries=self.report_config["max_retries"],
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

        # Save report
        output_path = self.output_dir / f"report_{engineer_id}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        # Save LLM interaction for debugging
        self.llm_client.save_interaction(
            output_dir=self.output_dir,
            name=f"report_{engineer_id}",
            prompt=prompt,
            response=result,
        )

        logger.info(f"Saved report to {output_path}")

        return report

    def _build_report_prompt(
        self,
        engineer_id: str,
        strengths: list[dict],
        weaknesses: list[dict],
        n_messages: int,
    ) -> str:
        """Build the prompt for report generation."""
        min_sentences = self.report_config["content"]["summary_sentences_min"]
        max_sentences = self.report_config["content"]["summary_sentences_max"]

        prompt = f"""You are generating a professional performance report for an engineer.

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
        else:
            prompt += "- No standout strengths identified\n"

        prompt += f"""
### Areas for Growth (<= {self.weakness_threshold}th percentile)
"""
        if weaknesses:
            for p in weaknesses:
                prompt += f"- **{p['name']}**: {p['percentile']}th percentile\n"
        else:
            prompt += "- No significant weaknesses identified\n"

        prompt += f"""
## Instructions

Generate a performance report with the following structure. Return valid JSON.

{{
  "overall_summary": "Markdown-formatted summary with ## Overview, ## Strengths, ## Growth Areas, ## Recommendations sections. {min_sentences}-{max_sentences} sentences total."
}}

Guidelines:
- Be specific and actionable
- Reference the pattern names naturally
- Focus on professional development
- Maintain a constructive, supportive tone
- Use markdown formatting for readability
"""

        return prompt

    def check_report_exists(self, engineer_id: str) -> bool:
        """Check if report exists for an engineer."""
        output_path = self.output_dir / f"report_{engineer_id}.json"
        return output_path.exists()

    def load_report(self, engineer_id: str) -> dict[str, Any] | None:
        """Load existing report for an engineer."""
        output_path = self.output_dir / f"report_{engineer_id}.json"
        if not output_path.exists():
            return None

        with open(output_path, "r") as f:
            return json.load(f)
