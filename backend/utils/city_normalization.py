from __future__ import annotations

import re


_CITY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def normalize_city_key(value: str) -> str:
    """Return a stable normalized city key for matching and filtering."""
    return value.strip().casefold()


def normalize_city_keys(values: list[str] | None) -> list[str]:
    """Normalize and de-duplicate city keys while preserving order."""
    if not values:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_city_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def format_city_stem(value: str) -> str:
    """Title-case city stem while preserving separators such as `_` and `-`."""
    cleaned = value.strip()
    if not cleaned:
        return ""

    def _capitalize(match: re.Match[str]) -> str:
        token = match.group(0)
        return token[:1].upper() + token[1:].lower()

    return _CITY_TOKEN_PATTERN.sub(_capitalize, cleaned)


def format_city_display_name(value: str) -> str:
    """Render a human-readable city display name using spaces."""
    cleaned = value.strip().replace("_", " ").replace("-", " ")
    parts = [part for part in cleaned.split() if part]
    if not parts:
        return ""
    return " ".join(part[:1].upper() + part[1:].lower() for part in parts)


__all__ = [
    "normalize_city_key",
    "normalize_city_keys",
    "format_city_stem",
    "format_city_display_name",
]
