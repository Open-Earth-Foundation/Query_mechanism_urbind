"""Artifact shaping helpers for orchestrator stage logging."""

from __future__ import annotations

from typing import Any

from backend.utils.city_normalization import format_city_stem, normalize_city_key


def _default_city_summary_entry(city_name: str) -> dict[str, object]:
    """Return the default summary shape for one city."""
    return {
        "city_name": city_name,
        "batch_count": 0,
        "chunk_count": 0,
        "accepted_chunk_count": 0,
        "rejected_chunk_count": 0,
        "unresolved_chunk_count": 0,
        "excerpt_count": 0,
        "status": "success",
        "error": None,
    }


def _get_city_summary_entry(
    city_meta: dict[str, dict[str, object]],
    city_key: str,
    city_name: str,
) -> dict[str, object]:
    """Return one mutable city summary entry, creating it when needed."""
    return city_meta.setdefault(city_key, _default_city_summary_entry(city_name))


def percentile(values: list[float], percentile_rank: float) -> float | None:
    """Return a simple nearest-rank percentile for artifact metrics."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        max(round((len(ordered) - 1) * percentile_rank), 0),
        len(ordered) - 1,
    )
    return ordered[index]


def build_retrieval_metrics(retrieval_payload: dict[str, object]) -> dict[str, object]:
    """Summarize retrieval payload counts and distances for observability."""
    chunks = retrieval_payload.get("chunks")
    seed_chunks = retrieval_payload.get("seed_chunks")
    meta = retrieval_payload.get("meta")
    chunk_entries: list[dict[str, object]] = []
    if isinstance(chunks, list):
        chunk_entries = [item for item in chunks if isinstance(item, dict)]
    seed_entries = (
        [item for item in seed_chunks if isinstance(item, dict)]
        if isinstance(seed_chunks, list)
        else []
    )
    distances = [
        float(item["distance"])
        for item in chunk_entries
        if isinstance(item.get("distance"), (int, float))
    ]
    meta_payload = meta if isinstance(meta, dict) else {}
    distance_metric = str(meta_payload.get("distance_metric", "distance")).strip() or "distance"
    return {
        "retrieval_distance_metric": distance_metric,
        "retrieval_total_chunks": len(chunk_entries),
        "retrieval_seed_chunks": len(seed_entries),
        "retrieval_distance_qualified_chunks": meta_payload.get(
            "distance_qualified_total_chunks"
        ),
        "retrieval_fallback_top_up_chunks": meta_payload.get(
            "fallback_top_up_total_chunks"
        ),
        "retrieval_neighbor_context_chunks": meta_payload.get(
            "neighbor_expanded_total_chunks"
        ),
        "retrieval_cosine_distance_min": min(distances) if distances else None,
        "retrieval_cosine_distance_p50": percentile(distances, 0.50),
        "retrieval_cosine_distance_p95": percentile(distances, 0.95),
        "retrieval_cosine_distance_max": max(distances) if distances else None,
        "selected_city_count": len(retrieval_payload.get("selected_cities", []) or []),
        "retrieval_query_count": len(retrieval_payload.get("queries", []) or []),
    }


def build_source_chunk_index(batches_payload: dict[str, object]) -> dict[str, object]:
    """Build a dedicated chunk lookup index from markdown batch artifacts."""
    source_chunks: list[dict[str, object]] = []
    batches = batches_payload.get("batches")
    if isinstance(batches, list):
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            chunks = batch.get("chunks")
            if not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                source_chunks.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "path": chunk.get("path"),
                        "chunk_index": chunk.get("chunk_index"),
                        "city_name": batch.get("city_name"),
                        "batch_index": batch.get("batch_index"),
                        "distance": chunk.get("distance"),
                    }
                )
    return {
        "source_chunk_count": len(source_chunks),
        "source_chunks": source_chunks,
    }


def build_markdown_metrics(
    *,
    markdown_chunks: list[dict[str, object]],
    markdown_bundle: dict[str, Any],
    rejected_artifact: dict[str, object],
    city_summary_artifact: dict[str, object],
    decision_audit_artifact: dict[str, object],
) -> dict[str, object]:
    """Summarize markdown extraction quality and mismatch signals."""
    accepted_total = int(decision_audit_artifact.get("accepted_total") or 0)
    rejected_total = int(decision_audit_artifact.get("rejected_total") or 0)
    total_decisions = accepted_total + rejected_total
    retrieval_summary = decision_audit_artifact.get("retrieval_summary")
    accepted_retrieval = (
        retrieval_summary.get("accepted")
        if isinstance(retrieval_summary, dict)
        else None
    )
    rejected_retrieval = (
        retrieval_summary.get("rejected")
        if isinstance(retrieval_summary, dict)
        else None
    )
    distance_metric = (
        str(decision_audit_artifact.get("distance_metric", "distance")).strip()
        or "distance"
    )
    return {
        "markdown_chunk_count": len(markdown_chunks),
        "markdown_accepted_count": accepted_total,
        "markdown_rejected_count": rejected_total,
        "markdown_unresolved_count": decision_audit_artifact.get("unresolved_total"),
        "markdown_acceptance_ratio": (
            accepted_total / total_decisions if total_decisions else None
        ),
        "markdown_distance_metric": distance_metric,
        "markdown_accepted_cosine_distance_min": (
            accepted_retrieval.get("distance_min")
            if isinstance(accepted_retrieval, dict)
            else None
        ),
        "markdown_accepted_cosine_distance_p50": (
            accepted_retrieval.get("distance_p50")
            if isinstance(accepted_retrieval, dict)
            else None
        ),
        "markdown_accepted_cosine_distance_p95": (
            accepted_retrieval.get("distance_p95")
            if isinstance(accepted_retrieval, dict)
            else None
        ),
        "markdown_accepted_cosine_distance_max": (
            accepted_retrieval.get("distance_max")
            if isinstance(accepted_retrieval, dict)
            else None
        ),
        "markdown_rejected_cosine_distance_min": (
            rejected_retrieval.get("distance_min")
            if isinstance(rejected_retrieval, dict)
            else None
        ),
        "markdown_rejected_cosine_distance_p50": (
            rejected_retrieval.get("distance_p50")
            if isinstance(rejected_retrieval, dict)
            else None
        ),
        "markdown_rejected_cosine_distance_p95": (
            rejected_retrieval.get("distance_p95")
            if isinstance(rejected_retrieval, dict)
            else None
        ),
        "markdown_rejected_cosine_distance_max": (
            rejected_retrieval.get("distance_max")
            if isinstance(rejected_retrieval, dict)
            else None
        ),
        "markdown_excerpt_count": markdown_bundle.get("excerpt_count", 0),
        "markdown_decision_invariant_ok": decision_audit_artifact.get("invariant_ok"),
        "accepted_status": decision_audit_artifact.get("status"),
        "rejected_status": rejected_artifact.get("status"),
        "markdown_city_count": len(city_summary_artifact.get("cities", []) or []),
        "markdown_cities_with_excerpts_count": len(
            city_summary_artifact.get("cities_with_excerpts", []) or []
        ),
        "markdown_cities_without_excerpts_count": len(
            city_summary_artifact.get("cities_without_excerpts", []) or []
        ),
        "markdown_cities_with_failures_count": len(
            city_summary_artifact.get("cities_with_failures", []) or []
        ),
    }


def build_markdown_city_summary(
    *,
    markdown_chunks: list[dict[str, object]],
    markdown_bundle: dict[str, Any],
    decision_audit_artifact: dict[str, object],
    batches_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one city-level markdown extraction summary artifact."""
    city_meta: dict[str, dict[str, object]] = {}
    chunk_city_by_id: dict[str, str] = {}
    unresolved_ids_by_city: dict[str, set[str]] = {}

    def add_unresolved_chunk_ids(chunk_ids: object) -> None:
        """Count unresolved chunk ids once per city across all artifact sources."""
        if not isinstance(chunk_ids, list):
            return
        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str):
                continue
            city_key = chunk_city_by_id.get(chunk_id.strip())
            if city_key is None:
                continue
            city_unresolved_ids = unresolved_ids_by_city.setdefault(city_key, set())
            if chunk_id in city_unresolved_ids:
                continue
            city_unresolved_ids.add(chunk_id)
            city_meta[city_key]["unresolved_chunk_count"] = (
                int(city_meta[city_key]["unresolved_chunk_count"]) + 1
            )

    for chunk in markdown_chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        raw_city_key = str(chunk.get("city_key", "")).strip()
        raw_city_name = str(chunk.get("city_name", "")).strip()
        city_key = normalize_city_key(raw_city_key or raw_city_name)
        if not city_key:
            city_key = "unknown"
        city_name = format_city_stem(raw_city_name or city_key)
        city_entry = _get_city_summary_entry(city_meta, city_key, city_name)
        city_entry["chunk_count"] = int(city_entry["chunk_count"]) + 1
        if raw_city_name and not str(city_entry["city_name"]).strip():
            city_entry["city_name"] = format_city_stem(raw_city_name)
        if chunk_id:
            chunk_city_by_id[chunk_id] = city_key

    batch_entries = (
        batches_payload.get("batches") if isinstance(batches_payload, dict) else None
    )
    if isinstance(batch_entries, list):
        for batch in batch_entries:
            if not isinstance(batch, dict):
                continue
            city_key = normalize_city_key(str(batch.get("city_name", "")).strip())
            if not city_key:
                continue
            city_entry = _get_city_summary_entry(
                city_meta, city_key, format_city_stem(city_key)
            )
            city_entry["batch_count"] = int(city_entry["batch_count"]) + 1

    for chunk_id in markdown_bundle.get("accepted_chunk_ids", []) or []:
        if not isinstance(chunk_id, str):
            continue
        city_key = chunk_city_by_id.get(chunk_id.strip())
        if city_key is None:
            continue
        city_meta[city_key]["accepted_chunk_count"] = (
            int(city_meta[city_key]["accepted_chunk_count"]) + 1
        )

    for chunk_id in markdown_bundle.get("rejected_chunk_ids", []) or []:
        if not isinstance(chunk_id, str):
            continue
        city_key = chunk_city_by_id.get(chunk_id.strip())
        if city_key is None:
            continue
        city_meta[city_key]["rejected_chunk_count"] = (
            int(city_meta[city_key]["rejected_chunk_count"]) + 1
        )

    add_unresolved_chunk_ids(decision_audit_artifact.get("missing_chunk_ids", []) or [])
    add_unresolved_chunk_ids(markdown_bundle.get("unresolved_chunk_ids", []) or [])

    for excerpt in markdown_bundle.get("excerpts", []) or []:
        if not isinstance(excerpt, dict):
            continue
        city_key = normalize_city_key(str(excerpt.get("city_key", "")).strip())
        if not city_key:
            city_key = normalize_city_key(str(excerpt.get("city_name", "")).strip())
        if not city_key:
            continue
        city_entry = _get_city_summary_entry(
            city_meta,
            city_key,
            format_city_stem(str(excerpt.get("city_name", "")).strip() or city_key),
        )
        city_entry["excerpt_count"] = int(city_entry["excerpt_count"]) + 1

    for failure in decision_audit_artifact.get("batch_failures", []) or []:
        if not isinstance(failure, dict):
            continue
        city_key = normalize_city_key(str(failure.get("city_name", "")).strip())
        if not city_key:
            city_key = "unknown"
        city_entry = _get_city_summary_entry(
            city_meta,
            city_key,
            format_city_stem(str(failure.get("city_name", "")).strip() or city_key),
        )
        city_entry["status"] = "partial"
        error_payload = city_entry.get("error")
        reasons = (
            []
            if not isinstance(error_payload, dict)
            else list(error_payload.get("reasons", []) or [])
        )
        reason = str(failure.get("reason", "")).strip()
        if reason and reason not in reasons:
            reasons.append(reason)
        add_unresolved_chunk_ids(failure.get("unresolved_chunk_ids"))
        city_entry["error"] = {"reasons": reasons}

    cities = []
    for city_key in sorted(city_meta.keys()):
        city_entry = dict(city_meta[city_key])
        city_entry["city_key"] = city_key
        cities.append(city_entry)

    cities_with_excerpts = [
        city["city_key"] for city in cities if int(city.get("excerpt_count") or 0) > 0
    ]
    cities_without_excerpts = [
        city["city_key"] for city in cities if int(city.get("excerpt_count") or 0) == 0
    ]
    cities_with_failures = [
        city["city_key"] for city in cities if str(city.get("status", "")) != "success"
    ]
    return {
        "cities": cities,
        "cities_with_excerpts": cities_with_excerpts,
        "cities_without_excerpts": cities_without_excerpts,
        "cities_with_failures": cities_with_failures,
    }


__all__ = [
    "build_markdown_city_summary",
    "build_markdown_metrics",
    "build_retrieval_metrics",
    "build_source_chunk_index",
    "percentile",
]
