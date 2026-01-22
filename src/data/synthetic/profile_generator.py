"""
Synthetic Profile Generator

Generates synthetic engineer profiles using LLM based on template profiles.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticProfileGenerator:
    """Generate synthetic engineer profiles using LLM."""

    def __init__(self, config: dict):
        """
        Initialize synthetic profile generator.

        Args:
            config: Full config dict with synthetic and paths sections
        """
        self.llm_provider = config["synthetic"]["llm_provider"]
        self.llm_model = config["synthetic"]["llm_model"]
        self.copies_per_template = config["synthetic"]["copies_per_template"]
        self.templates_dir = Path(config["paths"]["data"]["synthetic"]["templates_dir"])
        self.generated_dir = Path(config["paths"]["data"]["synthetic"]["generated_dir"])
        self.activities_path = Path(config["paths"]["data"]["collection"]["activities_csv"])

    def get_summary(self) -> dict:
        """
        Get summary of templates and generated profiles.

        Returns:
            Dict with template_count, total_profiles, and profile_details
        """
        if not self.templates_dir.exists():
            return {
                "template_count": 0,
                "total_profiles": 0,
                "profile_details": []
            }

        templates = list(self.templates_dir.glob("*.json"))

        profile_details = []
        total_profiles = 0

        for template_path in templates:
            template_name = template_path.stem
            gen_dir = self.generated_dir / template_name
            if gen_dir.exists():
                profile_count = len(list(gen_dir.glob("*.json")))
            else:
                profile_count = 0
            profile_details.append({
                "template_name": template_name,
                "profile_count": profile_count
            })
            total_profiles += profile_count

        return {
            "template_count": len(templates),
            "total_profiles": total_profiles,
            "profile_details": profile_details
        }

    def generate(self, copies_per_profile: int | None = None) -> dict:
        """
        Generate synthetic profiles from all templates.

        Args:
            copies_per_profile: Number of copies per template (overrides config)

        Returns:
            Dict with generated count and any errors
        """
        copies = copies_per_profile or self.copies_per_template
        templates = self._load_templates()

        if not templates:
            logger.warning(f"No templates found in {self.templates_dir}")
            return {"generated": 0, "errors": ["No templates found"]}

        generated_count = 0
        errors = []

        for template in templates:
            template_name = template.get("engineer_id", template.get("name", "unknown"))
            gen_dir = self.generated_dir / template_name
            gen_dir.mkdir(parents=True, exist_ok=True)

            for i in range(copies):
                try:
                    profile = self._generate_profile(template, i)
                    output_path = gen_dir / f"v{i+1:03d}.json"
                    with open(output_path, "w") as f:
                        json.dump(profile, f, indent=2)
                    generated_count += 1
                    logger.info(f"Generated profile: {output_path}")
                except Exception as e:
                    error_msg = f"{template_name} copy {i+1}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"Failed to generate: {error_msg}")

        return {"generated": generated_count, "errors": errors}

    def add_to_activities(self, split: str) -> dict:
        """
        Convert generated profiles to activities and add to activities.csv.

        Args:
            split: Target split for synthetic data ('train' or 'validation')

        Returns:
            Dict with profiles_added, activities_added, and errors
        """
        profiles = self._load_all_generated_profiles()

        if not profiles:
            return {
                "profiles_added": 0,
                "activities_added": 0,
                "errors": ["No generated profiles found"]
            }

        activities = self._profiles_to_activities(profiles, split)

        if not activities:
            return {
                "profiles_added": len(profiles),
                "activities_added": 0,
                "errors": ["No activities extracted from profiles"]
            }

        # Create DataFrame
        df = pd.DataFrame(activities)

        # Append to activities.csv
        self.activities_path.parent.mkdir(parents=True, exist_ok=True)

        if self.activities_path.exists():
            existing = pd.read_csv(self.activities_path)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(self.activities_path, index=False)

        logger.info(f"Added {len(activities)} activities from {len(profiles)} synthetic profiles")

        return {
            "profiles_added": len(profiles),
            "activities_added": len(activities),
            "errors": []
        }

    def _load_templates(self) -> list[dict]:
        """Load template profiles from templates directory."""
        if not self.templates_dir.exists():
            return []

        templates = []
        for path in self.templates_dir.glob("*.json"):
            try:
                with open(path) as f:
                    template = json.load(f)
                    templates.append(template)
                    logger.info(f"Loaded template: {path.name}")
            except Exception as e:
                logger.error(f"Failed to load template {path}: {e}")

        return templates

    def _load_all_generated_profiles(self) -> list[dict]:
        """Load all generated profiles from generated directory."""
        if not self.generated_dir.exists():
            return []

        profiles = []
        for template_dir in self.generated_dir.iterdir():
            if not template_dir.is_dir():
                continue
            for profile_path in template_dir.glob("*.json"):
                try:
                    with open(profile_path) as f:
                        profile = json.load(f)
                        profiles.append(profile)
                except Exception as e:
                    logger.error(f"Failed to load profile {profile_path}: {e}")

        return profiles

    def _generate_profile(self, template: dict, copy_index: int) -> dict:
        """
        Generate one synthetic profile using LLM.

        Args:
            template: Template profile dict
            copy_index: Index of this copy (for variation)

        Returns:
            Generated profile dict
        """
        if self.llm_provider == "gemini":
            return self._generate_with_gemini(template, copy_index)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _generate_with_gemini(self, template: dict, copy_index: int) -> dict:
        """Generate profile using Gemini API."""
        from google import genai

        client = genai.Client()

        prompt = self._build_generation_prompt(template, copy_index)

        response = client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
        )
        response_text = response.text

        # Extract JSON from response
        profile = self._parse_json_response(response_text)

        # Ensure required fields
        if "engineer_id" not in profile:
            base_id = template.get("engineer_id", "synthetic")
            profile["engineer_id"] = f"{base_id}_v{copy_index + 1:03d}"

        return profile

    def _build_generation_prompt(self, template: dict, copy_index: int) -> str:
        """Build prompt for synthetic profile generation."""
        template_json = json.dumps(template, indent=2)

        return f"""You are an expert at generating realistic software engineer activity profiles.

Given this template profile of a software engineer:

{template_json}

Generate a NEW synthetic engineer profile with DIFFERENT but similar characteristics.
This is variation #{copy_index + 1}, so make it distinct from the template but maintain the same general style and role.

Requirements:
1. Generate realistic commit messages, PR titles/descriptions, code review comments
2. Maintain consistent personality and coding style throughout
3. Include a mix of activity types (commits, PRs, reviews, issues)
4. Use realistic timestamps spanning several months
5. Keep the same general skill level and domain expertise
6. Change the name/email to be unique

Return ONLY a valid JSON object with this structure:
{{
    "engineer_id": "unique_email@example.com",
    "name": "Unique Name",
    "commits": [
        {{"message": "...", "date": "ISO8601", "repo": "repo-name", "additions": N, "deletions": N}}
    ],
    "prs": [
        {{"title": "...", "description": "...", "date": "ISO8601", "repo": "repo-name"}}
    ],
    "reviews": [
        {{"body": "...", "date": "ISO8601", "pr_number": N}}
    ],
    "issues": [
        {{"title": "...", "body": "...", "date": "ISO8601", "repo": "repo-name"}}
    ]
}}

Return ONLY the JSON, no markdown code blocks or explanation."""

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from LLM response."""
        # Try direct parse first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {response_text[:500]}")

    def _profiles_to_activities(self, profiles: list[dict], split: str) -> list[dict]:
        """
        Convert profile JSON to activity rows.

        Args:
            profiles: List of profile dicts
            split: Target split ('train' or 'validation')

        Returns:
            List of activity dicts for DataFrame
        """
        activities = []

        for profile in profiles:
            eng_id = profile.get("engineer_id", "unknown")

            # Process commits
            for commit in profile.get("commits", []):
                activities.append({
                    "engineer_id": eng_id,
                    "text": commit.get("message", ""),
                    "source": "github",
                    "activity_type": "commit",
                    "timestamp": commit.get("date"),
                    "split": split,
                    "meta_synthetic": True,
                    "meta_repository": commit.get("repo"),
                    "meta_additions": commit.get("additions"),
                    "meta_deletions": commit.get("deletions"),
                })

            # Process PRs
            for pr in profile.get("prs", []):
                title = pr.get("title", "")
                desc = pr.get("description", "") or ""
                text = f"{title}\n{desc}".strip()
                activities.append({
                    "engineer_id": eng_id,
                    "text": text,
                    "source": "github",
                    "activity_type": "pull_request",
                    "timestamp": pr.get("date"),
                    "split": split,
                    "meta_synthetic": True,
                    "meta_repository": pr.get("repo"),
                })

            # Process reviews
            for review in profile.get("reviews", []):
                activities.append({
                    "engineer_id": eng_id,
                    "text": review.get("body", ""),
                    "source": "github",
                    "activity_type": "review",
                    "timestamp": review.get("date"),
                    "split": split,
                    "meta_synthetic": True,
                    "meta_pr_number": review.get("pr_number"),
                })

            # Process issues
            for issue in profile.get("issues", []):
                title = issue.get("title", "")
                body = issue.get("body", "") or ""
                text = f"{title}\n{body}".strip()
                activities.append({
                    "engineer_id": eng_id,
                    "text": text,
                    "source": "github",
                    "activity_type": "issue",
                    "timestamp": issue.get("date"),
                    "split": split,
                    "meta_synthetic": True,
                    "meta_repository": issue.get("repo"),
                })

            # Process Slack messages if present
            for msg in profile.get("slack_messages", []):
                activities.append({
                    "engineer_id": eng_id,
                    "text": msg.get("text", ""),
                    "source": "slack",
                    "activity_type": "message",
                    "timestamp": msg.get("date") or msg.get("timestamp"),
                    "split": split,
                    "meta_synthetic": True,
                    "meta_channel": msg.get("channel"),
                })

        return activities
