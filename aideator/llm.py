from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from google import genai


class LLMClient:
    """Wraps the Gemini API for sending prompts and parsing responses."""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Set it in your .env file or as an environment variable."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def ask(self, prompt: str) -> str:
        """Send a prompt to Gemini and return the raw text response."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    @staticmethod
    def parse_response(response: str) -> tuple[str, str]:
        """Parse LLM JSON response to extract (name, description).

        Handles JSON wrapped in ```json ... ``` fences, raw JSON,
        and common LLM issues like trailing commas.

        Returns:
            Tuple of (name, description).

        Raises:
            ValueError: If parsing fails.
        """
        # Try to extract JSON from markdown code fences first
        fence_match = re.search(r"```json?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1)
        else:
            # Fall back to finding raw JSON object
            raw_match = re.search(r"\{.*\}", response, re.DOTALL)
            if raw_match:
                json_str = raw_match.group(0)
            else:
                raise ValueError(
                    f"Could not find JSON in LLM response:\n{response}"
                )

        # Clean trailing commas before closing braces (common LLM issue)
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from LLM response: {e}\nExtracted: {json_str}"
            ) from e

        name = data.get("name")
        description = data.get("description")
        if not name or not description:
            raise ValueError(
                f"JSON missing 'name' or 'description' fields: {data}"
            )

        return str(name).strip(), str(description).strip()
