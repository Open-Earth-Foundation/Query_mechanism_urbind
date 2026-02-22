from __future__ import annotations


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


__all__ = ["normalize_city_key", "normalize_city_keys"]
