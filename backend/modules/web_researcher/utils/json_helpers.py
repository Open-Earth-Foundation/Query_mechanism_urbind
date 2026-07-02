"""JSON parsing helpers for LLM response extraction.

Replicates and centralizes the extraction logic originally in
``backend/api/services/assumptions_review.py``.
"""

from __future__ import annotations

import json
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

    Priority: fenced JSON block, earliest outer JSON container, then raw text.
    """
    stripped = raw_text.strip()
    if not stripped:
        return "{}"

    fence_match = _JSON_FENCE_PATTERN.search(stripped)
    if fence_match:
        fenced = fence_match.group(1).strip()
        if fenced:
            return fenced

    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    candidates: list[tuple[int, str]] = []
    if first_bracket >= 0 and last_bracket > first_bracket:
        candidates.append((first_bracket, stripped[first_bracket : last_bracket + 1]))
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append((first_brace, stripped[first_brace : last_brace + 1]))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    return stripped


def parse_json_array_candidate(raw_text: str) -> list[Any]:
    """Parse a model response expected to contain a JSON array.

    Some chat models emit adjacent JSON objects instead of wrapping them in an
    array. When normal parsing fails with trailing data, decode those objects
    one-by-one so callers can still consume a complete classification list. A
    single JSON object is treated as a one-item list.
    """
    candidate = extract_json_candidate(raw_text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = _parse_adjacent_json_values(candidate)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _parse_adjacent_json_values(candidate: str) -> list[Any]:
    """Decode adjacent JSON values such as ``{} , {}`` into a list."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    length = len(candidate)
    while index < length:
        while index < length and candidate[index].isspace():
            index += 1
        if index < length and candidate[index] == ",":
            index += 1
            continue
        if index >= length:
            break
        value, end_index = decoder.raw_decode(candidate, index)
        values.append(value)
        index = end_index
    return values


__all__ = [
    "extract_message_text",
    "extract_json_candidate",
    "parse_json_array_candidate",
]
