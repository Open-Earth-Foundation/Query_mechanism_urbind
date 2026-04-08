"""JSON parsing helpers for LLM response extraction.

Replicates and centralizes the extraction logic originally in
``backend/api/services/assumptions_review.py``.
"""

from __future__ import annotations

import re
from typing import Any

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_message_text(content: Any) -> str:
    """Extract plain text content from OpenAI chat message payload variants."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()
    return str(content).strip()


def extract_json_candidate(raw_text: str) -> str:
    """Extract best JSON candidate from model response text.

    Priority: fenced JSON block → outermost braces → outermost brackets → raw.
    """
    stripped = raw_text.strip()
    if not stripped:
        return "{}"

    fence_match = _JSON_FENCE_PATTERN.search(stripped)
    if fence_match:
        fenced = fence_match.group(1).strip()
        if fenced:
            return fenced

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]

    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket >= 0 and last_bracket > first_bracket:
        return stripped[first_bracket : last_bracket + 1]

    return stripped


__all__ = ["extract_message_text", "extract_json_candidate"]
