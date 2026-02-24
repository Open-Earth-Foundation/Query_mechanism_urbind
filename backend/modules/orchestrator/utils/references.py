"""Reference mapping helpers for markdown excerpts."""

from __future__ import annotations

import re
from collections.abc import Mapping

REF_ID_PATTERN = re.compile(r"^ref_[1-9]\d*$")


def _coerce_source_chunk_ids(value: object) -> list[str]:
    """Normalize source chunk ids to a compact string list."""
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if not candidate:
            continue
        normalized.append(candidate)
    return normalized


def is_valid_ref_id(ref_id: str) -> bool:
    """Return True when the reference id matches the expected ``ref_n`` format."""
    return bool(REF_ID_PATTERN.fullmatch(ref_id))


def build_markdown_references(
    run_id: str, excerpts: list[Mapping[str, object]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """
    Assign deterministic ``ref_n`` ids to excerpts and build a references payload.

    Reference ids are assigned sequentially by excerpt order within the run.
    """
    updated_excerpts: list[dict[str, object]] = []
    references: list[dict[str, object]] = []

    for excerpt_index, excerpt in enumerate(excerpts):
        ref_id = f"ref_{excerpt_index + 1}"
        source_chunk_ids = _coerce_source_chunk_ids(excerpt.get("source_chunk_ids"))
        city_name = str(excerpt.get("city_name", ""))
        quote = str(excerpt.get("quote", ""))
        partial_answer = str(excerpt.get("partial_answer", ""))

        enriched_excerpt = dict(excerpt)
        enriched_excerpt["ref_id"] = ref_id
        enriched_excerpt["source_chunk_ids"] = source_chunk_ids
        updated_excerpts.append(enriched_excerpt)

        references.append(
            {
                "ref_id": ref_id,
                "excerpt_index": excerpt_index,
                "city_name": city_name,
                "quote": quote,
                "partial_answer": partial_answer,
                "source_chunk_ids": source_chunk_ids,
            }
        )

    payload: dict[str, object] = {
        "run_id": run_id,
        "reference_count": len(references),
        "references": references,
    }
    return updated_excerpts, payload


__all__ = ["build_markdown_references", "is_valid_ref_id", "REF_ID_PATTERN"]
