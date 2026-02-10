"""Formatting and utility helpers for markdown researcher."""

from pathlib import Path


def extract_city_name(document: dict[str, str]) -> str:
    """Extract city name from document dict."""
    city_name = document.get("city_name")
    if city_name:
        return str(city_name)
    path_value = document.get("path", "")
    if path_value:
        return Path(str(path_value)).stem
    return ""


def format_batch_failure(city_name: str, batch_index: int, reason: str) -> str:
    """Build a compact failure marker for a city batch."""
    return f"{city_name}#batch{batch_index}: {reason}"


__all__ = [
    "extract_city_name",
    "format_batch_failure",
]
