"""Helpers for reading persisted run context bundles and writer-safe exports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from fastapi import HTTPException, status

from backend.modules.writer.utils.markdown_helpers import (
    city_display_name,
    extract_markdown_bundle,
    extract_markdown_excerpts,
    extract_selected_city_names,
)
from backend.modules.writer.utils.multi_pass import build_writer_context_bundle


def load_run_context_bundle(context_path: Path, run_id: str) -> dict[str, object]:
    """Load one persisted run context bundle from disk."""
    try:
        context_bundle = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read context bundle for run `{run_id}`: {exc}",
        ) from exc

    if not isinstance(context_bundle, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context bundle for run `{run_id}` is not a JSON object.",
        )
    return context_bundle


def build_writer_export_context(
    context_bundle: Mapping[str, object],
) -> dict[str, object]:
    """Return the exact writer-safe context subset used for final answer generation."""
    normalized_context_bundle = dict(context_bundle)
    markdown_bundle = extract_markdown_bundle(normalized_context_bundle)
    excerpts = extract_markdown_excerpts(markdown_bundle)
    selected_city_names = extract_selected_city_names(
        normalized_context_bundle,
        markdown_bundle,
    )
    return build_writer_context_bundle(
        context_bundle=normalized_context_bundle,
        excerpts=excerpts,
        city_names=selected_city_names,
    )


def render_writer_export_markdown(context_bundle: Mapping[str, object]) -> str:
    """Render the exact writer context bundle as a readable Markdown export."""
    writer_context_bundle = build_writer_export_context(context_bundle)
    markdown_bundle = extract_markdown_bundle(writer_context_bundle)
    excerpts = extract_markdown_excerpts(markdown_bundle)
    selected_cities = _read_string_list(writer_context_bundle.get("selected_city_names"))
    if not selected_cities:
        selected_cities = _read_string_list(writer_context_bundle.get("selected_cities"))
    research_question = _read_string(writer_context_bundle.get("research_question"))
    analysis_mode = _read_string(writer_context_bundle.get("analysis_mode")) or "aggregate"

    lines = [
        "# Writer Context Export",
        "",
        "This file contains the writer-safe context bundle used to draft the answer.",
        "",
        f"- Research question: {research_question or '(missing)'}",
        f"- Analysis mode: {analysis_mode}",
        (
            "- Selected cities: "
            + (", ".join(selected_cities) if selected_cities else "(none recorded)")
        ),
        f"- Excerpt count: {len(excerpts)}",
        "",
    ]

    if not excerpts:
        lines.extend(
            [
                "## Excerpts",
                "",
                "_No accepted markdown excerpts were present in the writer bundle._",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    for index, excerpt in enumerate(excerpts, start=1):
        city_name = city_display_name(_read_string(excerpt.get("city_name"))) or f"City {index}"
        ref_id = _read_string(excerpt.get("ref_id"))
        partial_answer = _read_string(excerpt.get("partial_answer"))
        quote = _read_string(excerpt.get("quote"))
        source_chunk_ids = _read_string_list(excerpt.get("source_chunk_ids"))
        heading = f"## Excerpt {index} - {city_name}"
        if ref_id:
            heading += f" (`{ref_id}`)"
        lines.extend([heading, ""])
        if partial_answer:
            lines.extend(["**Partial answer**", "", partial_answer, ""])
        if quote:
            lines.extend(["**Quote**", "", _render_blockquote(quote), ""])
        if source_chunk_ids:
            rendered_ids = ", ".join(f"`{chunk_id}`" for chunk_id in source_chunk_ids)
            lines.extend([f"**Source chunk ids:** {rendered_ids}", ""])

    return "\n".join(lines).strip() + "\n"


def _read_string(value: object) -> str:
    """Return a stripped string value or an empty string."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _read_string_list(value: object) -> list[str]:
    """Return a compact list of non-empty string values."""
    if not isinstance(value, list):
        return []
    values: list[str] = []
    for item in value:
        candidate = _read_string(item)
        if candidate:
            values.append(candidate)
    return values


def _render_blockquote(text: str) -> str:
    """Render one string as a Markdown blockquote."""
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


__all__ = [
    "build_writer_export_context",
    "load_run_context_bundle",
    "render_writer_export_markdown",
]
