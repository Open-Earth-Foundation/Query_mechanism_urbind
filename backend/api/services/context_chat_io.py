"""I/O helpers for context chat bundles and evidence cache payloads."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.utils.json_io import read_json_object, write_json

logger = logging.getLogger("backend.api.services.context_chat")


def load_context_bundle(path: Path) -> dict[str, Any]:
    """Load a stored context bundle JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Context bundle at {path} is not a JSON object.")


def load_final_document(path: Path) -> str:
    """Load a stored final markdown document from disk."""
    return path.read_text(encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, object] | None:
    """Read a JSON object from disk for the evidence-cache compatibility path."""
    return read_json_object(
        path,
        logger=logger,
        error_prefix="Failed to read JSON object",
    )


def _write_json_object(path: Path, payload: dict[str, object]) -> None:
    """Persist a JSON object for the evidence-cache compatibility path."""
    write_json(path, payload, ensure_ascii=False)


__all__ = ["load_context_bundle", "load_final_document"]
