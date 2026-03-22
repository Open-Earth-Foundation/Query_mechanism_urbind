from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from backend.modules.vector_store.chroma_store import ChromaStore
from backend.modules.vector_store.indexer import OpenAIEmbeddingProvider
from backend.modules.vector_store.manifest import load_manifest
from backend.modules.vector_store.models import (
    RetrievedChunk,
    RetrievedChunkProvenance,
)
from backend.utils.city_normalization import (
    format_city_stem,
    normalize_city_key,
    normalize_city_keys,
)
from backend.utils.config import AppConfig
from backend.utils.retry import RetrySettings, call_with_retries


def _normalize_queries(queries: list[str]) -> list[str]:
    """Normalize and de-duplicate retrieval queries while preserving order."""
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        candidate = query.strip()
        if not candidate:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)
    return normalized


def _normalize_cities(cities: list[str] | None) -> list[str]:
    """Normalize and de-duplicate city names to internal city keys."""
    return normalize_city_keys(cities)


def _load_manifest_cities(config: AppConfig) -> dict[str, str]:
    """Load and validate city keys from the vector-store manifest."""
    manifest_path = config.vector_store.index_manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Vector store manifest not found at "
            f"{manifest_path}. Build or update the index before retrieval."
        )
    manifest = load_manifest(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(
            "Vector store manifest has no indexed files. "
            "Build or update the index before retrieval."
        )
    city_display_by_key: dict[str, str] = {}
    for source_path in files:
        display_name = format_city_stem(Path(str(source_path)).stem)
        city_key = normalize_city_key(display_name)
        if not city_key:
            continue
        city_display_by_key.setdefault(city_key, display_name)
    if not city_display_by_key:
        raise ValueError(
            "Vector store manifest does not include any city files. "
            "Build or update the index before retrieval."
        )
    return city_display_by_key


def list_indexed_city_names(config: AppConfig) -> list[str]:
    """Return display names for cities currently present in the vector-store manifest."""
    manifest_city_display_by_key = _load_manifest_cities(config)
    return sorted(manifest_city_display_by_key.values(), key=str.casefold)


def _embed_queries(queries: list[str], config: AppConfig) -> dict[str, list[float]]:
    """Embed retrieval queries in one batch using a reusable provider."""
    provider = OpenAIEmbeddingProvider(
        model=config.vector_store.embedding_model,
        base_url=config.openrouter_base_url,
        max_input_tokens=config.vector_store.embedding_max_input_tokens,
    )
    embeddings = provider.embed_texts(queries)
    result: dict[str, list[float]] = {}
    for query_text, embedding in zip(queries, embeddings, strict=True):
        if embedding is None:
            raise ValueError(f"Failed to embed query: {query_text!r}")
        result[query_text] = embedding
    return result


def _extract_query_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Chroma query payload to row dictionaries with distance."""
    ids = payload.get("ids", [[]])[0] if payload.get("ids") else []
    metadatas = payload.get("metadatas", [[]])[0] if payload.get("metadatas") else []
    distances = payload.get("distances", [[]])[0] if payload.get("distances") else []
    rows: list[dict[str, Any]] = []
    for chunk_id, metadata, distance in zip(ids, metadatas, distances, strict=False):
        if not isinstance(metadata, dict):
            continue
        try:
            distance_value = float(distance)
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "chunk_id": str(chunk_id),
                "metadata": metadata,
                "distance": distance_value,
            }
        )
    return rows


def _dedupe_strings(values: list[str]) -> list[str]:
    """Return unique, non-empty strings while preserving input order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = str(value).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _copy_provenance(provenance: dict[str, Any] | None) -> dict[str, Any]:
    """Copy one internal provenance mapping with normalized list values."""
    payload = provenance if isinstance(provenance, dict) else {}
    origin = str(payload.get("origin", "seed")).strip() or "seed"
    selection_mode = (
        str(payload.get("selection_mode", "distance_qualified")).strip()
        or "distance_qualified"
    )
    raw_seed_rank = payload.get("seed_rank")
    seed_rank = raw_seed_rank if isinstance(raw_seed_rank, int) else None
    return {
        "origin": origin,
        "selection_mode": selection_mode,
        "seed_rank": seed_rank,
        "seed_query_ids": _dedupe_strings(
            [
                str(item)
                for item in payload.get("seed_query_ids", [])
                if isinstance(item, str)
            ]
        ),
        "expanded_from_chunk_ids": _dedupe_strings(
            [
                str(item)
                for item in payload.get("expanded_from_chunk_ids", [])
                if isinstance(item, str)
            ]
        ),
    }


def _build_seed_row(
    row: dict[str, Any],
    *,
    query_id: str,
    selection_mode: str,
) -> dict[str, Any]:
    """Return one retrieved row annotated as a direct seed hit."""
    enriched = dict(row)
    enriched["provenance"] = {
        "origin": "seed",
        "selection_mode": selection_mode,
        "seed_rank": None,
        "seed_query_ids": [query_id],
        "expanded_from_chunk_ids": [],
    }
    return enriched


def _selection_mode_priority(selection_mode: str) -> int:
    """Return priority used when a chunk is seen in multiple direct-query modes."""
    if selection_mode == "distance_qualified":
        return 2
    if selection_mode == "fallback_top_up":
        return 1
    return 0


def _assign_seed_ranks(rows_by_id: dict[str, dict[str, Any]]) -> None:
    """Assign deterministic 1-based ranks to direct seed rows."""
    ranked_rows = sorted(
        rows_by_id.values(),
        key=lambda row: (
            float(row.get("distance", float("inf"))),
            str(row.get("chunk_id", "")),
        ),
    )
    for index, row in enumerate(ranked_rows, start=1):
        provenance = _copy_provenance(row.get("provenance"))
        provenance["seed_rank"] = index
        row["provenance"] = provenance


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    """Convert one query row into RetrievedChunk."""
    metadata = row["metadata"]
    distance_value = float(row["distance"])
    city_name = str(metadata.get("city_name", "")).strip()
    raw_chunk_index = metadata.get("chunk_index")
    chunk_index = raw_chunk_index if isinstance(raw_chunk_index, int) else None
    provenance_payload = _copy_provenance(row.get("provenance"))
    return RetrievedChunk(
        city_name=city_name,
        raw_text=str(metadata.get("raw_text", "")),
        source_path=str(metadata.get("source_path", "")),
        heading_path=str(metadata.get("heading_path", "")),
        block_type=str(metadata.get("block_type", "")),
        distance=distance_value,
        chunk_id=str(row["chunk_id"]),
        chunk_index=chunk_index,
        metadata={
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        provenance=RetrievedChunkProvenance(
            origin=str(provenance_payload["origin"]),
            selection_mode=str(provenance_payload["selection_mode"]),
            seed_rank=provenance_payload["seed_rank"],
            seed_query_ids=list(provenance_payload["seed_query_ids"]),
            expanded_from_chunk_ids=list(
                provenance_payload["expanded_from_chunk_ids"]
            ),
        ),
    )


def _resolve_city_list(
    config: AppConfig,
    selected_cities: list[str] | None,
) -> tuple[list[str], dict[str, str]]:
    """Resolve retrieval city keys with manifest-backed fail-fast validation."""
    manifest_city_display_by_key = _load_manifest_cities(config)
    normalized = _normalize_cities(selected_cities)
    if not normalized:
        city_keys = sorted(manifest_city_display_by_key.keys())
        return city_keys, manifest_city_display_by_key

    manifest_keys = set(manifest_city_display_by_key.keys())
    missing_cities = [
        city for city in normalized if city not in manifest_keys
    ]
    if missing_cities:
        raise ValueError(
            "Selected cities are not indexed in vector store manifest: "
            f"{', '.join(missing_cities)}. Build/update the index or adjust selected cities."
        )
    return normalized, manifest_city_display_by_key


def _passes_distance_threshold(distance: float, max_distance: float | None) -> bool:
    """Check whether a row distance passes the configured threshold."""
    if max_distance is None:
        return True
    return distance <= max_distance


def _retrieve_for_city_query(
    store: ChromaStore,
    query_embedding: list[float],
    city_key: str,
    query_id: str,
    fallback_min_chunks_per_city_query: int,
    max_chunks_per_city_query: int,
    max_distance: float | None,
    retry_settings: RetrySettings,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve rows for one city and one query with distance-first + top-up."""
    candidate_n_results = max(max_chunks_per_city_query, 1)
    payload = call_with_retries(
        lambda: store.query_by_embedding(
            query_embeddings=[query_embedding],
            n_results=candidate_n_results,
            where={"city_key": city_key},
        ),
        operation="vector_retrieval.query_by_embedding",
        retry_settings=retry_settings,
        should_retry=lambda _exc: True,
        run_id=run_id,
        context={
            "city_key": city_key,
            "query_id": query_id,
            "n_results": candidate_n_results,
        },
    )
    rows = _extract_query_rows(payload)
    passing = [
        _build_seed_row(
            row,
            query_id=query_id,
            selection_mode="distance_qualified",
        )
        for row in rows
        if _passes_distance_threshold(float(row["distance"]), max_distance)
    ]
    min_n = max(fallback_min_chunks_per_city_query, 1)
    if len(passing) >= min_n:
        return passing

    selected: list[dict[str, Any]] = list(passing)
    selected_ids = {str(row["chunk_id"]) for row in selected}
    for row in rows:
        if len(selected) >= min_n:
            break
        chunk_id = str(row["chunk_id"])
        if chunk_id in selected_ids:
            continue
        selected_ids.add(chunk_id)
        selected.append(
            _build_seed_row(
                row,
                query_id=query_id,
                selection_mode="fallback_top_up",
            )
        )
    return selected


def _merge_rows_best_distance(
    target: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    """Merge rows by chunk_id keeping the smallest distance."""
    for row in rows:
        chunk_id = str(row["chunk_id"])
        current = target.get(chunk_id)
        incoming_provenance = _copy_provenance(row.get("provenance"))
        if current is None:
            target[chunk_id] = dict(row)
            continue

        current_provenance = _copy_provenance(current.get("provenance"))
        replace_current = float(row["distance"]) < float(current["distance"])
        merged = dict(row) if replace_current else dict(current)
        merged_provenance = (
            incoming_provenance if replace_current else current_provenance
        )
        merged_provenance["seed_query_ids"] = _dedupe_strings(
            list(current_provenance["seed_query_ids"])
            + list(incoming_provenance["seed_query_ids"])
        )
        current_mode = str(current_provenance["selection_mode"])
        incoming_mode = str(incoming_provenance["selection_mode"])
        if _selection_mode_priority(incoming_mode) > _selection_mode_priority(
            current_mode
        ):
            merged_provenance["selection_mode"] = incoming_mode
        merged["provenance"] = merged_provenance
        target[chunk_id] = merged


def _parse_get_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Chroma get payload into row dictionaries."""
    ids = payload.get("ids", []) or []
    metadatas = payload.get("metadatas", []) or []
    rows: list[dict[str, Any]] = []
    for chunk_id, metadata in zip(ids, metadatas, strict=False):
        if not isinstance(metadata, dict):
            continue
        rows.append({"chunk_id": str(chunk_id), "metadata": metadata})
    return rows


def _expand_neighbors(
    store: ChromaStore,
    rows_by_id: dict[str, dict[str, Any]],
    context_window_chunks: int,
    table_context_window_chunks: int,
    retry_settings: RetrySettings,
    run_id: str | None = None,
) -> None:
    """Expand retrieval context by adding neighboring chunks from same file.

    Uses one batched Chroma ``get`` per (city_key, source_path) group.
    """
    seed_rows = list(rows_by_id.values())
    seeds_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        source_path = str(metadata.get("source_path", "")).strip()
        city_key = str(metadata.get("city_key", "")).strip()
        raw_index = metadata.get("chunk_index")
        if not source_path or not city_key or not isinstance(raw_index, int):
            continue

        group_key = (city_key, source_path)
        seeds_by_group.setdefault(group_key, []).append(row)

    for (city_key, source_path), grouped_seeds in seeds_by_group.items():
        required_indices: set[int] = set()
        for seed in grouped_seeds:
            seed_metadata = seed.get("metadata", {})
            if not isinstance(seed_metadata, dict):
                continue
            raw_index = seed_metadata.get("chunk_index")
            if not isinstance(raw_index, int):
                continue
            is_table = str(seed_metadata.get("block_type", "")) == "table"
            window = table_context_window_chunks if is_table else context_window_chunks
            for candidate_index in range(raw_index - window, raw_index + window + 1):
                if candidate_index >= 0:
                    required_indices.add(candidate_index)
        if not required_indices:
            continue

        payload = call_with_retries(
            lambda: store.get(
                where={
                    "$and": [
                        {"city_key": city_key},
                        {"source_path": source_path},
                        {"chunk_index": {"$in": sorted(required_indices)}},
                    ]
                },
                limit=max(len(required_indices), 1),
            ),
            operation="vector_retrieval.get_neighbors",
            retry_settings=retry_settings,
            should_retry=lambda _exc: True,
            run_id=run_id,
            context={
                "city_key": city_key,
                "source_path": source_path,
                "required_indices_count": len(required_indices),
            },
        )
        candidates_by_index: dict[int, list[dict[str, Any]]] = {}
        for candidate in _parse_get_rows(payload):
            metadata_candidate = candidate.get("metadata", {})
            if not isinstance(metadata_candidate, dict):
                continue
            chunk_index = metadata_candidate.get("chunk_index")
            if not isinstance(chunk_index, int):
                continue
            candidates_by_index.setdefault(chunk_index, []).append(candidate)

        for seed in grouped_seeds:
            seed_metadata = seed.get("metadata", {})
            if not isinstance(seed_metadata, dict):
                continue
            raw_index = seed_metadata.get("chunk_index")
            if not isinstance(raw_index, int):
                continue
            is_table = str(seed_metadata.get("block_type", "")) == "table"
            window = table_context_window_chunks if is_table else context_window_chunks
            for candidate_index in range(raw_index - window, raw_index + window + 1):
                if candidate_index < 0:
                    continue
                for candidate in candidates_by_index.get(candidate_index, []):
                    chunk_id = str(candidate["chunk_id"])
                    existing = rows_by_id.get(chunk_id)
                    if existing is not None:
                        existing_provenance = _copy_provenance(
                            existing.get("provenance")
                        )
                        if str(existing_provenance["origin"]) == "neighbor":
                            existing_provenance["expanded_from_chunk_ids"] = (
                                _dedupe_strings(
                                    list(existing_provenance["expanded_from_chunk_ids"])
                                    + [str(seed.get("chunk_id", ""))]
                                )
                            )
                            existing["provenance"] = existing_provenance
                        continue
                    metadata_candidate = candidate["metadata"]
                    if not isinstance(metadata_candidate, dict):
                        continue
                    rows_by_id[chunk_id] = {
                        "chunk_id": chunk_id,
                        "metadata": metadata_candidate,
                        "distance": float(seed["distance"]),
                        "provenance": {
                            "origin": "neighbor",
                            "selection_mode": "neighbor_context",
                            "seed_rank": None,
                            "seed_query_ids": [],
                            "expanded_from_chunk_ids": _dedupe_strings(
                                [str(seed.get("chunk_id", ""))]
                            ),
                        },
                    }


def serialize_retrieved_chunk(chunk: RetrievedChunk) -> dict[str, object]:
    """Serialize one retrieved chunk for persisted retrieval artifacts."""
    return {
        "chunk_id": chunk.chunk_id,
        "city_name": chunk.city_name,
        "city_key": str(chunk.metadata.get("city_key", "")),
        "source_path": chunk.source_path,
        "heading_path": chunk.heading_path,
        "block_type": chunk.block_type,
        "distance": chunk.distance,
        "chunk_index": chunk.chunk_index,
        "provenance": asdict(chunk.provenance),
    }


def build_retrieval_artifact(
    *,
    queries: list[str],
    selected_cities: list[str] | None,
    final_chunks: list[RetrievedChunk],
    retrieval_meta: dict[str, Any],
) -> dict[str, object]:
    """Build the persisted retrieval.json payload."""
    raw_seed_chunks = retrieval_meta.get("seed_chunks", [])
    seed_chunks = [
        chunk for chunk in raw_seed_chunks if isinstance(chunk, RetrievedChunk)
    ]
    serializable_meta = dict(retrieval_meta)
    serializable_meta.pop("seed_chunks", None)
    return {
        "queries": list(queries),
        "selected_cities": list(selected_cities or []),
        "retrieved_count": len(final_chunks),
        "meta": serializable_meta,
        "seed_chunks": [serialize_retrieved_chunk(chunk) for chunk in seed_chunks],
        "chunks": [serialize_retrieved_chunk(chunk) for chunk in final_chunks],
    }


def retrieve_top_k_chunks(
    query: str,
    config: AppConfig,
    city_filter: list[str] | None,
    k: int,
    run_id: str | None = None,
) -> list[RetrievedChunk]:
    """Compatibility helper that runs one-query city-level retrieval."""
    chunks, _meta = retrieve_chunks_for_queries(
        queries=[query],
        config=config,
        docs_dir=config.markdown_dir,
        selected_cities=city_filter,
        run_id=run_id,
    )
    return chunks[: max(k, 1)]


def retrieve_chunks_for_queries(
    queries: list[str],
    config: AppConfig,
    docs_dir: Path,
    selected_cities: list[str] | None,
    run_id: str | None = None,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Retrieve chunks across cities and multiple aligned queries."""
    del docs_dir
    normalized_queries = _normalize_queries(queries)
    city_keys, city_display_by_key = _resolve_city_list(config, selected_cities)
    if not normalized_queries or not city_keys:
        return [], {"queries": normalized_queries, "cities": city_keys, "per_city": []}

    fallback_min_chunks_per_city_query = max(
        config.vector_store.retrieval_fallback_min_chunks_per_city_query, 1
    )
    max_chunks_per_city_query = max(
        config.vector_store.retrieval_max_chunks_per_city_query,
        fallback_min_chunks_per_city_query,
    )

    store = ChromaStore(
        persist_path=config.vector_store.chroma_persist_path,
        collection_name=config.vector_store.chroma_collection_name,
    )
    retry_settings = RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )
    query_embeddings = _embed_queries(normalized_queries, config)

    rows_by_id: dict[str, dict[str, Any]] = {}
    per_city_stats: list[dict[str, Any]] = []
    for city_key in city_keys:
        city_name = city_display_by_key.get(city_key, city_key)
        city_rows: dict[str, dict[str, Any]] = {}
        query_stats: list[dict[str, Any]] = []
        for index, query_text in enumerate(normalized_queries, start=1):
            query_id = f"q{index}"
            rows = _retrieve_for_city_query(
                store=store,
                query_embedding=query_embeddings[query_text],
                city_key=city_key,
                query_id=query_id,
                fallback_min_chunks_per_city_query=fallback_min_chunks_per_city_query,
                max_chunks_per_city_query=max_chunks_per_city_query,
                max_distance=config.vector_store.retrieval_max_distance,
                retry_settings=retry_settings,
                run_id=run_id,
            )
            _merge_rows_best_distance(city_rows, rows)
            query_stats.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "qualified_chunks": len(rows),
                }
            )
        _merge_rows_best_distance(
            rows_by_id,
            list(city_rows.values()),
        )
        per_city_stats.append(
            {
                    "city_key": city_key,
                "city_name": city_name,
                "retrieved_unique_chunks": len(city_rows),
                "query_stats": query_stats,
            }
        )

    _assign_seed_ranks(rows_by_id)
    seed_chunks = [_to_retrieved_chunk(row) for row in rows_by_id.values()]
    seed_chunks.sort(
        key=lambda chunk: (
            chunk.provenance.seed_rank
            if chunk.provenance.seed_rank is not None
            else float("inf"),
            chunk.chunk_id,
        )
    )

    _expand_neighbors(
        store=store,
        rows_by_id=rows_by_id,
        context_window_chunks=max(config.vector_store.context_window_chunks, 0),
        table_context_window_chunks=max(config.vector_store.table_context_window_chunks, 0),
        retry_settings=retry_settings,
        run_id=run_id,
    )

    chunks = [_to_retrieved_chunk(row) for row in rows_by_id.values()]
    chunks.sort(key=lambda chunk: chunk.distance)

    max_chunks_per_city = config.vector_store.retrieval_max_chunks_per_city
    if max_chunks_per_city is not None and max_chunks_per_city > 0:
        selected: list[RetrievedChunk] = []
        counts: dict[str, int] = {}
        for chunk in chunks:
            city_key = str(chunk.metadata.get("city_key", "")).strip()
            current = counts.get(city_key, 0)
            if current >= max_chunks_per_city:
                continue
            counts[city_key] = current + 1
            selected.append(chunk)
        chunks = selected

    return (
        chunks,
        {
            "queries": normalized_queries,
            "cities": city_keys,
            "per_city": per_city_stats,
            "max_distance": config.vector_store.retrieval_max_distance,
            "fallback_min_chunks_per_city_query": fallback_min_chunks_per_city_query,
            "max_chunks_per_city_query": max_chunks_per_city_query,
            "min_chunks_per_city": fallback_min_chunks_per_city_query,
            "context_window_chunks": config.vector_store.context_window_chunks,
            "table_context_window_chunks": config.vector_store.table_context_window_chunks,
            "seed_retrieved_total_chunks": len(seed_chunks),
            "neighbor_expanded_total_chunks": sum(
                1 for chunk in chunks if chunk.provenance.origin == "neighbor"
            ),
            "retrieved_total_chunks": len(chunks),
            "seed_chunks": seed_chunks,
        },
    )


def as_markdown_documents(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    """Map retrieved chunks into markdown researcher input shape.

    Includes ``chunk_index`` when available so downstream batching can preserve file order.
    """
    documents: list[dict[str, object]] = []
    for chunk in chunks:
        chunk_index = chunk.metadata.get("chunk_index")
        resolved_chunk_index = chunk_index if isinstance(chunk_index, int) else None
        city_key = chunk.metadata.get("city_key")
        if not isinstance(city_key, str) or not city_key.strip():
            raise ValueError("Retrieved chunk is missing required metadata.city_key.")
        resolved_city_key = normalize_city_key(str(city_key))
        if not resolved_city_key:
            raise ValueError("Retrieved chunk city_key resolves to an empty value.")
        documents.append(
            {
                "path": chunk.source_path,
                "city_name": chunk.city_name,
                "city_key": resolved_city_key,
                "content": chunk.raw_text,
                "chunk_id": chunk.chunk_id,
                "distance": f"{chunk.distance:.6f}",
                "heading_path": chunk.heading_path,
                "block_type": chunk.block_type,
                "chunk_index": (
                    chunk.chunk_index
                    if chunk.chunk_index is not None
                    else resolved_chunk_index
                ),
            }
        )
    return documents


__all__ = [
    "build_retrieval_artifact",
    "as_markdown_documents",
    "list_indexed_city_names",
    "retrieve_chunks_for_queries",
    "retrieve_top_k_chunks",
    "serialize_retrieved_chunk",
]
