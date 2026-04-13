from __future__ import annotations

import json
import os
import re
from typing import Protocol, runtime_checkable

from dotenv import load_dotenv
from google import genai


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Minimal interface required by IdeaEngine.

    Any object that implements ask(prompt) -> str is a valid LLM client.
    Use this protocol to create test stubs without needing a real API key.
    """

    def ask(self, prompt: str) -> str:
        ...


class LLMClient:
    """Wraps the Gemini API for sending prompts and parsing responses."""

    def __init__(self, model_name: str | None = None):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Set it in your .env file or as an environment variable."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-pro"

    def ask(self, prompt: str) -> str:
        """Send a prompt to Gemini and return the raw text response."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    @staticmethod
    def _extract_json_candidates(text: str) -> list[str]:
        """Return all top-level {...} substrings from text using brace counting."""
        candidates = []
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])
                    start = None
        return candidates

    @staticmethod
    def _clean_json(json_str: str) -> str:
        """Remove trailing commas that LLMs commonly emit."""
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        return json_str

    @staticmethod
    def parse_response(response: str) -> tuple[str, str, str | None]:
        """Parse LLM JSON response to extract (name, description, type).

        Tries fenced JSON blocks first, then scans for all top-level {...}
        objects and picks the first one that contains 'name' and 'description'.

        Returns:
            Tuple of (name, description, type_value) where type_value may be
            None if the model omitted the 'type' field.

        Raises:
            ValueError: If no valid JSON with required fields is found.
        """
        # 1. Prefer fenced JSON block when present
        fence_match = re.search(r"```json?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fence_match:
            candidates = [fence_match.group(1)]
        else:
            # 2. Scan for all top-level {...} objects — no greedy cross-object matching
            candidates = LLMClient._extract_json_candidates(response)
            if not candidates:
                raise ValueError(f"Could not find JSON in LLM response:\n{response}")

        for json_str in candidates:
            json_str = LLMClient._clean_json(json_str)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            name = data.get("name")
            description = data.get("description")
            if name and description:
                return str(name).strip(), str(description).strip(), data.get("type")

        raise ValueError(
            f"No JSON object with 'name' and 'description' found in LLM response:\n{response}"
        )