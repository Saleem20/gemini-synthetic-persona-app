"""Thin wrapper around the OpenAI API for synthetic persona simulation.

Replaces gemini_client.py / claude_client.py. Same public interface
(constructor + generate_persona_response) so test_engine.py only needs
the import line changed.

Uses OpenAI's JSON mode for reliable structured output - no regex needed.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

# Models in order of cost (cheapest first). Pick via constructor or sidebar.
AVAILABLE_MODELS = [
    "gpt-4o-mini",        # Cheapest, fast, fine for screening
    "gpt-4o",             # Balanced quality and cost
    "gpt-4.1-mini",       # Newer cheap option (if your account has access)
    "gpt-4.1",            # Newer flagship
    "gpt-5-mini",         # If available on your account
    "gpt-5",              # Highest quality
]

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024


class OpenAIClient:
    """Minimal wrapper around openai.OpenAI for persona simulation."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Missing OpenAI API key. The app owner needs to add OPENAI_API_KEY "
                "to Streamlit Cloud secrets or the local environment."
            )
        self.client = OpenAI(api_key=resolved_key)
        self.model = model

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_persona_response(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Call OpenAI and return the parsed JSON payload.

        Uses response_format={"type": "json_object"} so the model returns
        valid JSON without code fences. Falls back to tolerant parsing
        on rare malformed output.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        text = (response.choices[0].message.content or "").strip()
        return _parse_json_payload(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_payload(text: str) -> Dict[str, Any]:
    """Tolerant JSON parser.

    JSON mode normally returns clean JSON. This fallback only triggers
    if a model variant ignores the format hint.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(candidate)
        if not match:
            raise RuntimeError(f"Could not find a JSON object in model output:\n{text}")
        return json.loads(match.group(0))
