"""Shared JSON file helpers for persisted backend artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable


def read_json(
    path: Path,
    *,
    logger: logging.Logger | None = None,
    error_prefix: str | None = None,
) -> object | None:
    """Read one JSON value from disk, returning None on missing or invalid files."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        if logger is not None and error_prefix:
            logger.exception("%s at %s", error_prefix, path)
        return None


def read_json_object(
    path: Path,
    *,
    logger: logging.Logger | None = None,
    error_prefix: str | None = None,
) -> dict[str, Any] | None:
    """Read one JSON object from disk, returning None for non-object payloads."""
    payload = read_json(path, logger=logger, error_prefix=error_prefix)
    if isinstance(payload, dict):
        return payload
    return None


def write_json(
    path: Path,
    payload: object,
    *,
    ensure_ascii: bool = True,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Write one JSON value to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii, default=default),
        encoding="utf-8",
    )


__all__ = ["read_json", "read_json_object", "write_json"]
