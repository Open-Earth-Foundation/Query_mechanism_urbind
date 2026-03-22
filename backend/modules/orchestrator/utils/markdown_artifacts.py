"""Helpers for building persisted markdown pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from backend.modules.markdown_researcher.models import MarkdownResearchResult
from backend.modules.orchestrator.utils.references import build_markdown_references
from backend.utils.city_normalization import format_city_stem, normalize_city_key


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return unique non-empty values while preserving input order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


@dataclass(frozen=True)
class MarkdownArtifacts:
    """Persistable markdown artifact payloads for one markdown result."""

    excerpts_payload: dict[str, object]
    accepted_payload: dict[str, object]
    rejected_payload: dict[str, object]
    decision_audit_payload: dict[str, object]
    references_payload: dict[str, object]


def collect_markdown_decision_artifacts(
    markdown_chunks: list[dict[str, object]],
    markdown_result: MarkdownResearchResult,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build accepted, rejected, and decision-audit payloads for markdown output."""
    batch_failures_payload = [
        failure.model_dump() for failure in markdown_result.batch_failures
    ]

    retrieved_ids = _dedupe_preserve_order(
        [str(document.get("chunk_id", "")).strip() for document in markdown_chunks]
    )
    retrieved_set = set(retrieved_ids)

    accepted_ids = _dedupe_preserve_order(markdown_result.accepted_chunk_ids)
    rejected_ids = _dedupe_preserve_order(markdown_result.rejected_chunk_ids)
    unresolved_ids = _dedupe_preserve_order(markdown_result.unresolved_chunk_ids)
    accepted_set = set(accepted_ids)
    rejected_set = set(rejected_ids)
    unresolved_set = set(unresolved_ids)

    overlap_decision_ids = {
        chunk_id
        for chunk_id in accepted_set
        if chunk_id in rejected_set or chunk_id in unresolved_set
    } | {chunk_id for chunk_id in rejected_set if chunk_id in unresolved_set}

    unknown_decision_ids = _dedupe_preserve_order(
        [
            chunk_id
            for chunk_id in accepted_ids + rejected_ids + unresolved_ids
            if chunk_id not in retrieved_set
        ]
    )
    unknown_decision_set = set(unknown_decision_ids)

    excerpt_source_ids: list[str] = []
    for excerpt in markdown_result.excerpts:
        excerpt_source_ids.extend(
            [
                source_chunk_id.strip()
                for source_chunk_id in excerpt.source_chunk_ids
                if source_chunk_id.strip()
            ]
        )
    unknown_excerpt_source_ids = _dedupe_preserve_order(
        [
            source_chunk_id
            for source_chunk_id in excerpt_source_ids
            if source_chunk_id not in accepted_set
        ]
    )

    decided_valid_ids = {
        chunk_id
        for chunk_id in accepted_set | rejected_set | unresolved_set
        if chunk_id in retrieved_set and chunk_id not in unknown_decision_set
    }
    missing_chunk_ids = [
        chunk_id for chunk_id in retrieved_ids if chunk_id not in decided_valid_ids
    ]

    city_by_chunk_id: dict[str, str] = {}
    for chunk in markdown_chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        city_key = str(chunk.get("city_key", "")).strip()
        if not city_key:
            city_name = str(chunk.get("city_name", "")).strip()
            city_key = normalize_city_key(city_name) if city_name else ""
        city_by_chunk_id[chunk_id] = city_key or "unknown"

    accepted_by_city: dict[str, list[str]] = {}
    for chunk_id in accepted_ids:
        city_key = city_by_chunk_id.get(chunk_id, "unknown")
        accepted_by_city.setdefault(city_key, []).append(chunk_id)

    rejected_by_city: dict[str, list[str]] = {}
    for chunk_id in rejected_ids:
        city_key = city_by_chunk_id.get(chunk_id, "unknown")
        rejected_by_city.setdefault(city_key, []).append(chunk_id)

    invariant_ok = not (
        overlap_decision_ids
        or unknown_decision_ids
        or missing_chunk_ids
        or unknown_excerpt_source_ids
    )
    artifact_status = (
        "complete"
        if invariant_ok and not markdown_result.batch_failures and not unresolved_ids
        else "partial"
    )

    accepted_artifact = {
        "status": artifact_status,
        "accepted_chunk_ids": accepted_ids,
        "accepted_by_city": accepted_by_city,
        "counts": {
            "accepted": len(accepted_ids),
        },
    }
    rejected_artifact = {
        "status": artifact_status,
        "rejected_chunk_ids": rejected_ids,
        "rejected_by_city": rejected_by_city,
        "counts": {
            "rejected": len(rejected_ids),
        },
    }
    audit_artifact = {
        "retrieved_total": len(retrieved_ids),
        "accepted_total": len(accepted_ids),
        "rejected_total": len(rejected_ids),
        "unresolved_total": len(unresolved_ids),
        "invariant_ok": invariant_ok,
        "missing_chunk_ids": missing_chunk_ids,
        "unknown_decision_ids": unknown_decision_ids,
        "unknown_excerpt_source_ids": unknown_excerpt_source_ids,
        "overlap_decision_ids": sorted(overlap_decision_ids),
        "batch_failures": batch_failures_payload,
    }
    return accepted_artifact, rejected_artifact, audit_artifact


def build_markdown_artifacts(
    *,
    markdown_chunks: list[dict[str, object]],
    markdown_result: MarkdownResearchResult,
    run_id: str,
    markdown_source_mode: str,
    analysis_mode: str,
    retrieval_queries: list[str] | None = None,
    selected_cities: list[str] | None = None,
) -> MarkdownArtifacts:
    """Build the persisted markdown bundle plus its companion artifacts."""
    (
        accepted_artifact,
        rejected_artifact,
        decision_audit_artifact,
    ) = collect_markdown_decision_artifacts(markdown_chunks, markdown_result)

    markdown_bundle = markdown_result.model_dump()
    markdown_bundle["decision_audit"] = {
        "accepted_total": decision_audit_artifact["accepted_total"],
        "rejected_total": decision_audit_artifact["rejected_total"],
        "unresolved_total": decision_audit_artifact["unresolved_total"],
        "invariant_ok": decision_audit_artifact["invariant_ok"],
        "status": accepted_artifact["status"],
    }
    markdown_bundle["retrieval_mode"] = markdown_source_mode
    markdown_bundle["analysis_mode"] = analysis_mode
    if markdown_source_mode == "vector_store_retrieval":
        markdown_bundle["retrieval_queries"] = list(retrieval_queries or [])

    inspected_cities = sorted(
        {
            normalize_city_key(str(document.get("city_key", "")).strip())
            for document in markdown_chunks
            if normalize_city_key(str(document.get("city_key", "")).strip())
        }
    )
    markdown_bundle["inspected_cities"] = inspected_cities

    key_to_name: dict[str, str] = {}
    for document in markdown_chunks:
        key = normalize_city_key(str(document.get("city_key", "")).strip())
        name = document.get("city_name")
        if key and key not in key_to_name:
            key_to_name[key] = (
                format_city_stem(str(name).strip()) if name else format_city_stem(key)
            )
    markdown_bundle["inspected_city_names"] = [
        key_to_name[key] for key in inspected_cities if key in key_to_name
    ]

    selected_city_keys = sorted(
        {
            normalize_city_key(city)
            for city in (selected_cities or [])
            if isinstance(city, str) and city.strip()
        }
    )
    markdown_bundle["selected_cities"] = selected_city_keys
    if selected_city_keys:
        markdown_bundle["selected_city_names"] = [
            key_to_name.get(key, format_city_stem(key)) for key in selected_city_keys
        ]
    else:
        markdown_bundle["selected_city_names"] = markdown_bundle["inspected_city_names"]

    excerpts = markdown_bundle.get("excerpts", [])
    if isinstance(excerpts, list):
        excerpt_entries = [
            excerpt for excerpt in excerpts if isinstance(excerpt, dict)
        ]
        enriched_excerpts, references_payload = build_markdown_references(
            run_id=run_id,
            excerpts=excerpt_entries,
        )
        markdown_bundle["excerpts"] = enriched_excerpts
        markdown_bundle["excerpt_count"] = len(enriched_excerpts)
    else:
        references_payload = {"references": []}
        markdown_bundle["excerpts"] = []
        markdown_bundle["excerpt_count"] = 0

    return MarkdownArtifacts(
        excerpts_payload=markdown_bundle,
        accepted_payload=accepted_artifact,
        rejected_payload=rejected_artifact,
        decision_audit_payload=decision_audit_artifact,
        references_payload=references_payload,
    )


__all__ = [
    "MarkdownArtifacts",
    "build_markdown_artifacts",
    "collect_markdown_decision_artifacts",
]
