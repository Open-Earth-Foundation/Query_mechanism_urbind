"""Helpers for serializing structured LLM inputs as TOON."""

from __future__ import annotations

import dataclasses
from typing import Any

from toon_format import decode, encode

from backend.utils.tokenization import count_tokens


def to_llm_serializable(value: Any) -> Any:
    """Normalize arbitrary runtime values into TOON-safe Python primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): to_llm_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_llm_serializable(item) for item in value]
    if dataclasses.is_dataclass(value):
        try:
            return to_llm_serializable(dataclasses.asdict(value))
        except Exception:  # noqa: BLE001
            return str(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return to_llm_serializable(model_dump(mode="json"))
        except TypeError:
            return to_llm_serializable(model_dump())
        except Exception:  # noqa: BLE001
            return str(value)
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return to_llm_serializable(to_dict())
        except Exception:  # noqa: BLE001
            return str(value)
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        filtered = {
            key: item for key, item in value_dict.items() if not key.startswith("_")
        }
        return to_llm_serializable(filtered)
    return str(value)


def serialize_for_llm(value: Any) -> str:
    """Serialize structured LLM input payloads as TOON."""
    return encode(to_llm_serializable(value))


def count_serialized_tokens_for_llm(value: Any) -> int:
    """Count prompt tokens for the TOON form actually sent to the LLM."""
    return count_tokens(serialize_for_llm(value))


def render_toon_block(title: str, value: Any) -> str:
    """Render a labeled TOON code block for prompt text builders."""
    return f"{title}:\n~~~toon\n{serialize_for_llm(value)}\n~~~"


def parse_llm_serialized(text: str) -> Any:
    """Decode TOON text back into Python primitives for tests and debugging."""
    return decode(text)


__all__ = [
    "count_serialized_tokens_for_llm",
    "parse_llm_serialized",
    "render_toon_block",
    "serialize_for_llm",
    "to_llm_serializable",
]
