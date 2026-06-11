"""Artifact shaping helpers for orchestrator stage logging."""

from __future__ import annotations

from typing import Any


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
    return {
        "retrieval_total_chunks": len(chunk_entries),
        "retrieval_seed_chunks": len(seed_entries),
        "retrieval_neighbor_chunks": meta_payload.get("neighbor_expanded_total_chunks"),
        "retrieval_fallback_chunks": meta_payload.get("fallback_top_up_total_chunks"),
        "retrieval_distance_min": min(distances) if distances else None,
        "retrieval_distance_p50": percentile(distances, 0.50),
        "retrieval_distance_p95": percentile(distances, 0.95),
        "retrieval_distance_max": max(distances) if distances else None,
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
    accepted_artifact: dict[str, object],
    rejected_artifact: dict[str, object],
    decision_audit_artifact: dict[str, object],
) -> dict[str, object]:
    """Summarize markdown extraction quality and mismatch signals."""
    accepted_total = int(decision_audit_artifact.get("accepted_total") or 0)
    rejected_total = int(decision_audit_artifact.get("rejected_total") or 0)
    total_decisions = accepted_total + rejected_total
    return {
        "markdown_chunk_count": len(markdown_chunks),
        "markdown_accepted_count": accepted_total,
        "markdown_rejected_count": rejected_total,
        "markdown_unresolved_count": decision_audit_artifact.get("unresolved_total"),
        "markdown_acceptance_ratio": (
            accepted_total / total_decisions if total_decisions else None
        ),
        "markdown_excerpt_count": markdown_bundle.get("excerpt_count", 0),
        "markdown_decision_invariant_ok": decision_audit_artifact.get("invariant_ok"),
        "accepted_status": accepted_artifact.get("status"),
        "rejected_status": rejected_artifact.get("status"),
    }


__all__ = [
    "build_markdown_metrics",
    "build_retrieval_metrics",
    "build_source_chunk_index",
    "percentile",
]
