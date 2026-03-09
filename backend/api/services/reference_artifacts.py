"""Helpers for loading persisted markdown reference artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.api.models import RunReferenceItem
from backend.modules.orchestrator.utils.references import build_markdown_references

logger = logging.getLogger(__name__)


def parse_excerpt_index(value: object) -> int:
    """Parse excerpt index into a non-negative integer."""
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str):
        try:
            return max(int(value.strip()), 0)
        except ValueError:
            return 0
    return 0


def normalize_source_chunk_ids(value: object) -> list[str]:
    """Normalize source chunk ids to a compact string list."""
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def build_reference_item(record: dict[str, object], include_quote: bool) -> RunReferenceItem:
    """Normalize one persisted reference record into API response shape."""
    item = RunReferenceItem(
        ref_id=str(record.get("ref_id", "")).strip(),
        excerpt_index=parse_excerpt_index(record.get("excerpt_index")),
        city_name=str(record.get("city_name", "")).strip(),
    )
    if include_quote:
        item.quote = str(record.get("quote", ""))
        item.partial_answer = str(record.get("partial_answer", ""))
        item.source_chunk_ids = normalize_source_chunk_ids(record.get("source_chunk_ids"))
    return item


def load_reference_records(artifact_dir: Path, source_id: str) -> list[dict[str, object]]:
    """Load reference records from persisted references or excerpt artifacts."""
    references_path = artifact_dir / "markdown" / "references.json"
    if references_path.exists():
        payload = _load_json_object(references_path)
        if payload is not None:
            records = coerce_reference_records(payload.get("references"))
            if records:
                return records

    excerpts_path = artifact_dir / "markdown" / "excerpts.json"
    payload = _load_json_object(excerpts_path)
    if payload is None:
        return []
    excerpt_records = coerce_reference_records(payload.get("excerpts"))
    if not excerpt_records:
        return []
    _enriched_excerpts, references_payload = build_markdown_references(
        run_id=source_id,
        excerpts=excerpt_records,
    )
    return coerce_reference_records(references_payload.get("references"))


def coerce_reference_records(value: object) -> list[dict[str, object]]:
    """Coerce a raw reference payload into dict records."""
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def _load_json_object(path: Path) -> dict[str, object] | None:
    """Load one JSON object from disk with safe fallback logging."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to parse reference artifact at %s", path)
        return None
    if isinstance(payload, dict):
        return payload
    return None


__all__ = [
    "build_reference_item",
    "coerce_reference_records",
    "load_reference_records",
    "normalize_source_chunk_ids",
    "parse_excerpt_index",
]
