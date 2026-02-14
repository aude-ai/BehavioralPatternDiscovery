"""
Unified LLM Client

Auto-detects provider based on model name and provides consistent interface.

Supports:
- Gemini (gemini-*)
- OpenAI (gpt-*, o1-*, o3-*)
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """Unified client that auto-detects and uses the correct LLM provider."""

    def __init__(
        self,
        config: dict[str, Any],
        config_key: str,
        debug_dir: Path | str | None = None,
    ):
        """
        Initialize unified LLM client.

        Args:
            config: Full merged configuration
            config_key: Config section key (e.g., "report", "explanation")
            debug_dir: Directory for saving prompts and responses. If set,
                       every LLM call is automatically logged.
        """
        self.config = config
        self.config_key = config_key
        self.llm_config = config[config_key]
        self.model_name = self.llm_config["llm_model"]
        self.provider = self._detect_provider(self.model_name)
        self._call_counter = 0

        # Auto-save directory
        if debug_dir:
            self.debug_dir = Path(debug_dir) / "llm_logs"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.debug_dir = None

        logger.info(f"[{config_key}] Provider: {self.provider}, Model: {self.model_name}")

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _detect_provider(self, model_name: str) -> str:
        """Detect provider from model name."""
        model_lower = model_name.lower()

        if "gemini" in model_lower:
            return "gemini"

        if any(p in model_lower for p in ["gpt-", "o1-", "o3-", "o4-"]):
            return "openai"

        raise ValueError(f"Could not detect provider for model: {model_name}")

    def _init_gemini(self):
        """Initialize Gemini client."""
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)

    def _init_openai(self):
        """Initialize OpenAI client."""
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=api_key)

    def generate_content(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = None,
        log_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate content using the configured provider.

        Args:
            prompt: The prompt to send
            max_tokens: Override max tokens
            temperature: Override temperature
            thinking_budget: Thinking budget for Gemini
            log_name: Name for debug log files (e.g., "enc1_bottom").
                      If None and debug_dir is set, uses auto-incrementing counter.

        Returns:
            Dict with 'text', 'usage', 'finish_reason'
        """
        if self.provider == "gemini":
            result = self._generate_gemini(prompt, max_tokens, temperature, thinking_budget)
        elif self.provider == "openai":
            result = self._generate_openai(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self._auto_save(prompt, result, log_name)
        return result

    def _generate_gemini(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        """Generate content using Gemini."""
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        config_kwargs = {
            "temperature": temperature or self.llm_config["temperature"],
            "max_output_tokens": max_tokens or self.llm_config["max_completion_tokens"],
        }

        # Use thinking_level if configured (Gemini 3+), otherwise thinking_budget (Gemini 2.x)
        thinking_level = self.llm_config.get("thinking_level")
        if thinking_level:
            # Gemini 3+ uses thinking_budget=-1 for dynamic with include_thoughts
            config_kwargs["thinking_config"] = ThinkingConfig(
                thinking_budget=-1,
                include_thoughts=True,
            )
        else:
            budget = thinking_budget if thinking_budget is not None else self.llm_config.get("thinking_budget", 0)
            if budget != 0:
                config_kwargs["thinking_config"] = ThinkingConfig(
                    thinking_budget=budget,
                    include_thoughts=True,
                )

        gen_config = GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=gen_config,
        )

        usage = {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
        }

        finish_reason = "STOP"
        if hasattr(response, "candidates") and response.candidates:
            if hasattr(response.candidates[0], "finish_reason"):
                finish_reason = str(response.candidates[0].finish_reason)

        return {
            "text": response.text or "",
            "usage": usage,
            "finish_reason": finish_reason,
        }

    def _generate_openai(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate content using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.llm_config["temperature"],
            max_completion_tokens=max_tokens or self.llm_config["max_completion_tokens"],
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return {
            "text": response.choices[0].message.content or "",
            "usage": usage,
            "finish_reason": response.choices[0].finish_reason,
        }

    def generate_content_json_mode(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        log_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate content with JSON response mode enabled.

        For Gemini, uses response_mime_type="application/json".
        For OpenAI, uses response_format={"type": "json_object"}.

        Args:
            prompt: The prompt to send
            max_tokens: Override max tokens
            temperature: Override temperature
            log_name: Name for debug log files. If None and debug_dir is set,
                      uses auto-incrementing counter.

        Returns:
            Dict with 'text', 'usage', 'finish_reason'
        """
        if self.provider == "gemini":
            result = self._generate_gemini_json_mode(prompt, max_tokens, temperature)
        elif self.provider == "openai":
            result = self._generate_openai_json_mode(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self._auto_save(prompt, result, log_name)
        return result

    def _generate_gemini_json_mode(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate content using Gemini with JSON response mode."""
        from google.genai.types import GenerateContentConfig, ThinkingConfig

        config_kwargs = {
            "response_mime_type": "application/json",
            "max_output_tokens": max_tokens or self.llm_config.get("max_output_tokens") or self.llm_config.get("max_completion_tokens", 65536),
        }

        # Temperature is optional for JSON mode (some models don't support it with thinking)
        temp = temperature or self.llm_config.get("temperature")
        if temp is not None:
            config_kwargs["temperature"] = temp

        # Use thinking_level if configured (Gemini 3+), otherwise thinking_budget
        thinking_level = self.llm_config.get("thinking_level")
        if thinking_level:
            config_kwargs["thinking_config"] = ThinkingConfig(
                thinking_budget=-1,
                include_thoughts=True,
            )
        else:
            budget = self.llm_config.get("thinking_budget", 0)
            if budget != 0:
                config_kwargs["thinking_config"] = ThinkingConfig(
                    thinking_budget=budget,
                    include_thoughts=True,
                )

        gen_config = GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=gen_config,
        )

        usage = {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
        }

        finish_reason = "STOP"
        if hasattr(response, "candidates") and response.candidates:
            if hasattr(response.candidates[0], "finish_reason"):
                finish_reason = str(response.candidates[0].finish_reason)

        return {
            "text": response.text or "",
            "usage": usage,
            "finish_reason": finish_reason,
        }

    def _generate_openai_json_mode(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate content using OpenAI with JSON response mode."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.llm_config.get("temperature", 0.7),
            max_completion_tokens=max_tokens or self.llm_config.get("max_completion_tokens", 8192),
            response_format={"type": "json_object"},
        )

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return {
            "text": response.choices[0].message.content or "",
            "usage": usage,
            "finish_reason": response.choices[0].finish_reason,
        }

    def generate_json_content(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
        log_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate JSON content with parsing and retry logic.

        Args:
            prompt: The prompt to send
            max_tokens: Override max tokens
            temperature: Override temperature
            max_retries: Number of retries on parse failure
            log_name: Name for debug log files

        Returns:
            Dict with 'success', 'data', 'raw_text', 'usage', 'finish_reason'
        """
        result = None
        raw_text = ""

        for attempt in range(max_retries):
            attempt_name = f"{log_name}_attempt{attempt + 1}" if log_name else None
            result = self.generate_content(prompt, max_tokens, temperature, log_name=attempt_name)
            raw_text = result["text"]

            parsed = self._parse_json(raw_text)
            if parsed is not None:
                return {
                    "success": True,
                    "data": parsed,
                    "raw_text": raw_text,
                    "usage": result["usage"],
                    "finish_reason": result["finish_reason"],
                }

            logger.warning(f"[{self.config_key}] JSON parse failed, attempt {attempt + 1}/{max_retries}")

        return {
            "success": False,
            "data": None,
            "raw_text": raw_text,
            "usage": result["usage"] if result else {},
            "finish_reason": result["finish_reason"] if result else "",
            "error": "JSON parse failed after retries",
        }

    def _parse_json(self, text: str) -> dict | None:
        """Parse JSON from response text with repair for common LLM errors."""
        # Try to extract JSON from code blocks or raw text
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
        else:
            json_match = re.search(r"\{[\s\S]*\}", text)
            if not json_match:
                logger.error(f"[{self.config_key}] No JSON found in response")
                return None
            json_str = json_match.group()

        # Try parsing as-is first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"[{self.config_key}] Initial JSON parse failed: {e}. Attempting repair...")

        # Attempt to repair common JSON errors from LLMs
        repaired = self._repair_json(json_str)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.error(f"[{self.config_key}] JSON repair failed: {e}")
            return None

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair truncated/malformed JSON from LLM responses.

        Strategy:
        1. Progressively truncate from end looking for valid structure
        2. Skip mid-string truncations (odd quote count)
        3. Remove incomplete key-value pairs (lines ending with :)
        4. Remove trailing commas
        5. Add missing closing brackets
        """
        start_idx = json_str.find('{')
        if start_idx == -1:
            return json_str

        # Try progressively truncating from the end
        for end_idx in range(len(json_str) - 1, start_idx, -1):
            truncated = json_str[start_idx:end_idx + 1].rstrip()

            # Skip if mid-string (odd number of quotes)
            if truncated.count('"') % 2 != 0:
                continue

            # Remove incomplete key-value pairs
            lines = truncated.split('\n')
            cleaned_lines = []
            for line in lines:
                stripped = line.rstrip()
                if stripped.endswith(':'):
                    break
                cleaned_lines.append(line)

            truncated = '\n'.join(cleaned_lines).rstrip()

            # Remove trailing commas
            while truncated.rstrip().endswith(','):
                truncated = truncated.rstrip()[:-1].rstrip()

            # Add missing closing brackets
            truncated += ']' * (truncated.count('[') - truncated.count(']'))
            truncated += '}' * (truncated.count('{') - truncated.count('}'))

            # Try to parse
            try:
                json.loads(truncated)
                logger.info(f"[{self.config_key}] Repaired JSON (truncated to {len(truncated)} from {len(json_str)} chars)")
                return truncated
            except json.JSONDecodeError:
                continue

        # Fallback: simple fixes only
        repaired = re.sub(r',(\s*[}\]])', r'\1', json_str)
        repaired += '}' * (repaired.count('{') - repaired.count('}'))
        return repaired

    def _auto_save(
        self,
        prompt: str,
        result: dict[str, Any],
        log_name: str | None,
    ) -> None:
        """Automatically save prompt and response if debug_dir is set."""
        if not self.debug_dir:
            return

        self._call_counter += 1
        name = log_name or f"{self.config_key}_{self._call_counter:03d}"

        self._save_to_dir(self.debug_dir, name, prompt, result)

    def _save_to_dir(
        self,
        directory: Path,
        name: str,
        prompt: str,
        response: dict[str, Any],
    ) -> None:
        """Save prompt and response to a directory."""
        directory.mkdir(parents=True, exist_ok=True)

        prompt_path = directory / f"{name}_prompt.txt"
        with open(prompt_path, "w") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Provider: {self.provider}\n\n")
            f.write("=== PROMPT ===\n\n")
            f.write(prompt)

        response_path = directory / f"{name}_response.json"
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "usage": response.get("usage", {}),
            "finish_reason": response.get("finish_reason", ""),
            "text": response.get("text", ""),
        }
        with open(response_path, "w") as f:
            json.dump(response_data, f, indent=2)
