"""
Pattern Explanation Generator

Generates detailed explanations for individual pattern scores on-demand.
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.llm import UnifiedLLMClient

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generate explanations for pattern scores."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize explanation generator.

        Args:
            config: Full merged configuration
        """
        self.config = config
        self.paths = config["paths"]
        self.explanation_config = config["explanation"]

        self.output_dir = Path(self.paths["scoring"]["reports_dir"]) / "explanations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llm_client = UnifiedLLMClient(config, config_key="explanation")

    def explain_pattern(
        self,
        engineer_id: str,
        pattern_id: str,
        scores: dict[str, Any],
        message_database: list[dict],
    ) -> dict[str, Any]:
        """
        Generate explanation for a specific pattern score.

        Args:
            engineer_id: Engineer being explained
            pattern_id: Pattern to explain (e.g., "enc1_bottom_0")
            scores: Individual scores
            message_database: Full message database

        Returns:
            Explanation dict
        """
        logger.info(f"Generating explanation for {engineer_id} pattern {pattern_id}")

        # Find the pattern in scores
        pattern = None
        for p in scores["patterns"]:
            if p["id"] == pattern_id:
                pattern = p
                break

        if pattern is None:
            raise ValueError(f"Pattern {pattern_id} not found in scores")

        # Get engineer's messages
        engineer_messages = [m for m in message_database if m["engineer_id"] == engineer_id]
        n_examples = self.explanation_config["engineer_examples"]

        # Build prompt
        prompt = self._build_explanation_prompt(
            pattern=pattern,
            engineer_messages=engineer_messages[:n_examples],
        )

        # Generate explanation
        result = self.llm_client.generate_content(prompt=prompt)

        explanation = {
            "engineer_id": engineer_id,
            "pattern_id": pattern_id,
            "pattern_name": pattern["name"],
            "percentile": pattern["percentile"],
            "explanation": result["text"],
        }

        # Save explanation
        engineer_dir = self.output_dir / engineer_id
        engineer_dir.mkdir(parents=True, exist_ok=True)

        output_path = engineer_dir / f"{pattern_id}.json"
        with open(output_path, "w") as f:
            json.dump(explanation, f, indent=2)

        logger.info(f"Saved explanation to {output_path}")

        return explanation

    def _build_explanation_prompt(
        self,
        pattern: dict[str, Any],
        engineer_messages: list[dict],
    ) -> str:
        """Build prompt for pattern explanation."""
        prompt = f"""Explain why this engineer scored at the {pattern['percentile']}th percentile for the "{pattern['name']}" behavioral pattern.

## Pattern Details
- **Name**: {pattern['name']}
- **Percentile**: {pattern['percentile']}th (compared to population)

## Engineer's Recent Messages
"""
        for i, msg in enumerate(engineer_messages[:5], 1):
            text = msg.get("text", "")[:200]
            prompt += f"{i}. {text}...\n\n"

        prompt += """
## Instructions

Provide a clear, actionable explanation:
1. What this pattern measures
2. Why the engineer scored at this percentile
3. What behaviors contributed to this score
4. Specific suggestions for improvement (if below 50th percentile)

Keep the explanation concise (3-5 paragraphs) and professional.
"""
        return prompt

    def load_explanation(self, engineer_id: str, pattern_id: str) -> dict[str, Any] | None:
        """Load existing explanation."""
        output_path = self.output_dir / engineer_id / f"{pattern_id}.json"
        if not output_path.exists():
            return None

        with open(output_path, "r") as f:
            return json.load(f)

    def list_explanations(self, engineer_id: str) -> list[str]:
        """List all explanation IDs for an engineer."""
        engineer_dir = self.output_dir / engineer_id
        if not engineer_dir.exists():
            return []

        return [f.stem for f in engineer_dir.glob("*.json")]
