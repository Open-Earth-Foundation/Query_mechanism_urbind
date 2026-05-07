"""Writer multi-pass batching helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from backend.modules.writer.models import WriterMultiPassBatch, WriterMultiPassPlan
from backend.modules.writer.utils.markdown_helpers import (
    city_display_name,
    city_key,
    dedupe_city_names,
    extract_markdown_excerpts,
)
from backend.utils.tokenization import count_tokens

_WRITER_ENRICHMENT_KEYS = (
    "gap_manifest",
    "enriched_fields",
    "external_evidence",
    "external_resolutions",
    "external_no_evidence",
    "assumptions",
    "non_estimable",
    "web_findings",
    "freshness_results",
    "saturation_warning",
    "meta",
)


@dataclass(frozen=True)
class WriterBatch:
    """One prepared writer batch plus its token estimate."""

    batch_index: int
    city_names: list[str]
    excerpt_count: int
    payload_tokens: int
    context_bundle: dict[str, object]


@dataclass(frozen=True)
class _CityExcerptUnit:
    """Grouped excerpts for one city-sized writer batching unit."""

    city_names: list[str]
    excerpts: list[dict[str, object]]


def build_writer_payload(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    selected_city_names: list[str],
    reconsideration: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the runtime writer payload for token counting and execution."""
    payload: dict[str, object] = {
        "question": question,
        "context_bundle": context_bundle,
        "analysis_mode": analysis_mode,
        "selected_cities": selected_city_names,
    }
    if reconsideration:
        payload["reconsideration"] = reconsideration
    return payload


def count_writer_payload_tokens(payload: dict[str, object]) -> int:
    """Count tokens for a serialized writer payload."""
    return count_tokens(json.dumps(payload, ensure_ascii=False))


def plan_writer_multi_pass(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    selected_city_names: list[str],
    threshold_tokens: int,
    chunk_tokens: int,
    max_input_tokens: int | None = None,
) -> tuple[WriterMultiPassPlan | None, list[WriterBatch]]:
    """Split oversized writer context into city-scoped batches when needed."""
    base_payload = build_writer_payload(
        question=question,
        context_bundle=context_bundle,
        analysis_mode=analysis_mode,
        selected_city_names=selected_city_names,
    )
    payload_tokens = count_writer_payload_tokens(base_payload)
    effective_chunk_tokens = max(chunk_tokens, 1)
    if max_input_tokens is not None and payload_tokens > max_input_tokens and payload_tokens <= threshold_tokens:
        raise ValueError(
            "Writer payload exceeds the configured LLM input token limit before multi-pass "
            f"can trigger: payload_tokens={payload_tokens}, limit={max_input_tokens}, "
            f"threshold={threshold_tokens}."
        )
    if payload_tokens <= threshold_tokens:
        return None, []

    units = _build_city_units(context_bundle)
    if not units:
        return None, []

    expanded_units = _expand_oversized_units(
        question=question,
        context_bundle=context_bundle,
        analysis_mode=analysis_mode,
        chunk_tokens=effective_chunk_tokens,
        units=units,
    )
    batches = _batch_units(
        question=question,
        context_bundle=context_bundle,
        analysis_mode=analysis_mode,
        chunk_tokens=effective_chunk_tokens,
        units=expanded_units,
    )
    if max_input_tokens is not None:
        largest_batch_tokens = max(
            (batch.payload_tokens for batch in batches),
            default=payload_tokens,
        )
        if largest_batch_tokens > max_input_tokens:
            raise ValueError(
                "Writer payload remains above the configured LLM input token limit after "
                f"batching: largest_batch_tokens={largest_batch_tokens}, limit={max_input_tokens}, "
                f"threshold={threshold_tokens}, chunk_target={effective_chunk_tokens}, "
                f"batch_count={len(batches)}."
            )
    if len(batches) <= 1:
        return None, []

    batch_summaries = [
        WriterMultiPassBatch(
            batch_index=batch.batch_index,
            city_names=batch.city_names,
            excerpt_count=batch.excerpt_count,
            payload_tokens=batch.payload_tokens,
        )
        for batch in batches
    ]
    plan = WriterMultiPassPlan(
        analysis_mode=analysis_mode,
        payload_tokens=payload_tokens,
        threshold_tokens=threshold_tokens,
        batch_count=len(batch_summaries),
        batches=batch_summaries,
    )
    return plan, batches


def build_writer_batch_drafts_payload(
    *,
    batches: list[WriterBatch],
    drafts: list[str],
) -> list[dict[str, object]]:
    """Return structured batch draft metadata for persistence and debugging."""
    payloads: list[dict[str, object]] = []
    for batch, draft in zip(batches, drafts, strict=False):
        payloads.append(
            {
                "batch_index": batch.batch_index,
                "city_names": batch.city_names,
                "excerpt_count": batch.excerpt_count,
                "payload_tokens": batch.payload_tokens,
                "draft_length_chars": len(draft),
                "draft_content": draft,
            }
        )
    return payloads


def _build_city_units(context_bundle: dict[str, object]) -> list[_CityExcerptUnit]:
    """Group writer excerpts by city while preserving their original order."""
    markdown_bundle = context_bundle.get("markdown")
    if not isinstance(markdown_bundle, dict):
        return []

    ordered_units: list[_CityExcerptUnit] = []
    unit_by_key: dict[str, _CityExcerptUnit] = {}
    for index, excerpt in enumerate(extract_markdown_excerpts(markdown_bundle)):
        raw_city_name = str(excerpt.get("city_name", "")).strip()
        display_name = city_display_name(raw_city_name) or f"City {index + 1}"
        normalized_key = city_key(display_name) or f"__city_{index}"
        existing = unit_by_key.get(normalized_key)
        if existing is None:
            existing = _CityExcerptUnit(city_names=[display_name], excerpts=[])
            unit_by_key[normalized_key] = existing
            ordered_units.append(existing)
        existing.excerpts.append(excerpt)
    return ordered_units


def _expand_oversized_units(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    chunk_tokens: int,
    units: list[_CityExcerptUnit],
) -> list[_CityExcerptUnit]:
    """Split one oversized city unit into smaller excerpt slices when needed."""
    expanded: list[_CityExcerptUnit] = []
    for unit in units:
        if _unit_payload_tokens(
            question=question,
            context_bundle=context_bundle,
            analysis_mode=analysis_mode,
            unit=unit,
        ) <= chunk_tokens:
            expanded.append(unit)
            continue
        expanded.extend(
            _split_city_unit(
                question=question,
                context_bundle=context_bundle,
                analysis_mode=analysis_mode,
                chunk_tokens=chunk_tokens,
                unit=unit,
            )
        )
    return expanded


def _split_city_unit(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    chunk_tokens: int,
    unit: _CityExcerptUnit,
) -> list[_CityExcerptUnit]:
    """Split one city unit by excerpt order when it exceeds the batch token cap."""
    batches: list[_CityExcerptUnit] = []
    current_excerpts: list[dict[str, object]] = []
    for excerpt in unit.excerpts:
        candidate_excerpts = [*current_excerpts, excerpt]
        candidate_unit = _CityExcerptUnit(city_names=unit.city_names, excerpts=candidate_excerpts)
        candidate_tokens = _unit_payload_tokens(
            question=question,
            context_bundle=context_bundle,
            analysis_mode=analysis_mode,
            unit=candidate_unit,
        )
        if current_excerpts and candidate_tokens > chunk_tokens:
            batches.append(_CityExcerptUnit(city_names=unit.city_names, excerpts=current_excerpts))
            current_excerpts = [excerpt]
            continue
        current_excerpts = candidate_excerpts
    if current_excerpts:
        batches.append(_CityExcerptUnit(city_names=unit.city_names, excerpts=current_excerpts))
    return batches or [unit]


def _unit_payload_tokens(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    unit: _CityExcerptUnit,
) -> int:
    """Count tokens for a single batching unit when sent to the writer."""
    batch_context = build_writer_context_bundle(
        context_bundle=context_bundle,
        excerpts=unit.excerpts,
        city_names=unit.city_names,
    )
    payload = build_writer_payload(
        question=question,
        context_bundle=batch_context,
        analysis_mode=analysis_mode,
        selected_city_names=unit.city_names,
    )
    return count_writer_payload_tokens(payload)


def _batch_units(
    *,
    question: str,
    context_bundle: dict[str, object],
    analysis_mode: str,
    chunk_tokens: int,
    units: list[_CityExcerptUnit],
) -> list[WriterBatch]:
    """Pack city units into writer batches under the configured token cap."""
    grouped_units: list[list[_CityExcerptUnit]] = []
    current_units: list[_CityExcerptUnit] = []

    for unit in units:
        candidate_units = [*current_units, unit]
        candidate_context = build_writer_context_bundle(
            context_bundle=context_bundle,
            excerpts=_flatten_unit_excerpts(candidate_units),
            city_names=_flatten_unit_city_names(candidate_units),
        )
        candidate_payload = build_writer_payload(
            question=question,
            context_bundle=candidate_context,
            analysis_mode=analysis_mode,
            selected_city_names=_flatten_unit_city_names(candidate_units),
        )
        candidate_tokens = count_writer_payload_tokens(candidate_payload)
        if current_units and candidate_tokens > chunk_tokens:
            grouped_units.append(current_units)
            current_units = [unit]
            continue
        current_units = candidate_units

    if current_units:
        grouped_units.append(current_units)

    batches: list[WriterBatch] = []
    for index, batch_units in enumerate(grouped_units, start=1):
        city_names = _flatten_unit_city_names(batch_units)
        excerpts = _flatten_unit_excerpts(batch_units)
        batch_context = build_writer_context_bundle(
            context_bundle=context_bundle,
            excerpts=excerpts,
            city_names=city_names,
        )
        batch_payload = build_writer_payload(
            question=question,
            context_bundle=batch_context,
            analysis_mode=analysis_mode,
            selected_city_names=city_names,
        )
        batches.append(
            WriterBatch(
                batch_index=index,
                city_names=city_names,
                excerpt_count=len(excerpts),
                payload_tokens=count_writer_payload_tokens(batch_payload),
                context_bundle=batch_context,
            )
        )
    return batches


def build_writer_context_bundle(
    *,
    context_bundle: dict[str, object],
    excerpts: list[dict[str, object]],
    city_names: list[str],
) -> dict[str, object]:
    """Build one writer-safe context bundle subset for the provided excerpts."""
    selected_city_names = dedupe_city_names(city_names)
    if not selected_city_names:
        excerpt_city_names = [
            str(excerpt.get("city_name", ""))
            for excerpt in excerpts
            if isinstance(excerpt.get("city_name"), str)
        ]
        selected_city_names = dedupe_city_names(excerpt_city_names)
    city_keys = [city_key(name) for name in selected_city_names]
    normalized_city_keys = [value for value in city_keys if value]

    markdown_bundle = context_bundle.get("markdown")
    markdown_analysis_mode = (
        markdown_bundle.get("analysis_mode")
        if isinstance(markdown_bundle, dict)
        else None
    )
    analysis_mode = context_bundle.get("analysis_mode")

    writer_context: dict[str, object] = {
        "research_question": context_bundle.get("research_question"),
        "analysis_mode": analysis_mode,
        "selected_cities": selected_city_names,
        "markdown": {
            "status": markdown_bundle.get("status", "success")
            if isinstance(markdown_bundle, dict)
            else "success",
            "analysis_mode": markdown_analysis_mode
            if isinstance(markdown_analysis_mode, str) and markdown_analysis_mode.strip()
            else analysis_mode,
            "excerpt_count": len(excerpts),
            "excerpts": excerpts,
            "selected_city_names": selected_city_names,
            "inspected_city_names": selected_city_names,
            "selected_cities": normalized_city_keys,
            "inspected_cities": normalized_city_keys,
        },
    }
    enrichment = context_bundle.get("enrichment")
    if isinstance(enrichment, dict):
        writer_enrichment = _build_writer_enrichment(enrichment)
        if writer_enrichment:
            writer_context["enrichment"] = writer_enrichment
    return writer_context


def _build_writer_enrichment(enrichment: dict[str, object]) -> dict[str, object]:
    """Return only enrichment fields that the writer prompts consume."""
    return {
        key: enrichment[key]
        for key in _WRITER_ENRICHMENT_KEYS
        if key in enrichment and enrichment[key] is not None
    }


def _flatten_unit_city_names(units: list[_CityExcerptUnit]) -> list[str]:
    """Return de-duplicated city names across batching units."""
    city_names: list[str] = []
    for unit in units:
        city_names.extend(unit.city_names)
    return dedupe_city_names(city_names)


def _flatten_unit_excerpts(units: list[_CityExcerptUnit]) -> list[dict[str, object]]:
    """Return excerpts across batching units in stable order."""
    excerpts: list[dict[str, object]] = []
    for unit in units:
        excerpts.extend(unit.excerpts)
    return excerpts


__all__ = [
    "WriterBatch",
    "build_writer_batch_drafts_payload",
    "build_writer_context_bundle",
    "build_writer_payload",
    "count_writer_payload_tokens",
    "plan_writer_multi_pass",
]
