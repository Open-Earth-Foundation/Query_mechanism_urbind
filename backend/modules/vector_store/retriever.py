from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.modules.vector_store.chroma_store import ChromaStore
from backend.modules.vector_store.indexer import OpenAIEmbeddingProvider
from backend.modules.vector_store.manifest import load_manifest
from backend.modules.vector_store.models import RetrievedChunk
from backend.utils.city_normalization import normalize_city_key, normalize_city_keys
from backend.utils.config import AppConfig


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
        display_name = Path(str(source_path)).stem.strip()
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


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    """Convert one query row into RetrievedChunk."""
    metadata = row["metadata"]
    distance_value = float(row["distance"])
    city_name = str(metadata.get("city_name", "")).strip()
    return RetrievedChunk(
        city_name=city_name,
        raw_text=str(metadata.get("raw_text", "")),
        source_path=str(metadata.get("source_path", "")),
        heading_path=str(metadata.get("heading_path", "")),
        block_type=str(metadata.get("block_type", "")),
        distance=distance_value,
        chunk_id=str(row["chunk_id"]),
        metadata={
            key: value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
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
    fallback_min_chunks_per_city_query: int,
    max_chunks_per_city_query: int,
    max_distance: float | None,
) -> list[dict[str, Any]]:
    """Retrieve rows for one city and one query with distance-first + top-up."""
    candidate_n_results = max(max_chunks_per_city_query, 1)
    payload = store.query_by_embedding(
        query_embeddings=[query_embedding],
        n_results=candidate_n_results,
        where={"city_key": city_key},
    )
    rows = _extract_query_rows(payload)
    passing = [
        row for row in rows if _passes_distance_threshold(float(row["distance"]), max_distance)
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
        selected.append(row)
    return selected


def _merge_rows_best_distance(
    target: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    query_id: str,
) -> None:
    """Merge rows by chunk_id keeping the smallest distance."""
    for row in rows:
        chunk_id = str(row["chunk_id"])
        current = target.get(chunk_id)
        if current is None or float(row["distance"]) < float(current["distance"]):
            merged = dict(row)
            merged["retrieval_query_id"] = query_id
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

        payload = store.get(
            where={
                "$and": [
                    {"city_key": city_key},
                    {"source_path": source_path},
                    {"chunk_index": {"$in": sorted(required_indices)}},
                ]
            },
            limit=max(len(required_indices), 1),
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
                    if chunk_id in rows_by_id:
                        continue
                    metadata_candidate = candidate["metadata"]
                    if not isinstance(metadata_candidate, dict):
                        continue
                    rows_by_id[chunk_id] = {
                        "chunk_id": chunk_id,
                        "metadata": metadata_candidate,
                        "distance": float(seed["distance"]),
                        "retrieval_query_id": str(
                            seed.get("retrieval_query_id", "neighbor")
                        ),
                    }


def retrieve_top_k_chunks(
    query: str,
    config: AppConfig,
    city_filter: list[str] | None,
    k: int,
) -> list[RetrievedChunk]:
    """Compatibility helper that runs one-query city-level retrieval."""
    chunks, _meta = retrieve_chunks_for_queries(
        queries=[query],
        config=config,
        docs_dir=config.markdown_dir,
        selected_cities=city_filter,
    )
    return chunks[: max(k, 1)]


def retrieve_chunks_for_queries(
    queries: list[str],
    config: AppConfig,
    docs_dir: Path,
    selected_cities: list[str] | None,
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
                fallback_min_chunks_per_city_query=fallback_min_chunks_per_city_query,
                max_chunks_per_city_query=max_chunks_per_city_query,
                max_distance=config.vector_store.retrieval_max_distance,
            )
            _merge_rows_best_distance(city_rows, rows, query_id)
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
            "city_merge",
        )
        per_city_stats.append(
            {
                    "city_key": city_key,
                "city_name": city_name,
                "retrieved_unique_chunks": len(city_rows),
                "query_stats": query_stats,
            }
        )

    _expand_neighbors(
        store=store,
        rows_by_id=rows_by_id,
        context_window_chunks=max(config.vector_store.context_window_chunks, 0),
        table_context_window_chunks=max(config.vector_store.table_context_window_chunks, 0),
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
            "retrieved_total_chunks": len(chunks),
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
        resolved_city_key = str(city_key).strip()
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
                "chunk_index": resolved_chunk_index,
            }
        )
    return documents


__all__ = [
    "as_markdown_documents",
    "retrieve_chunks_for_queries",
    "retrieve_top_k_chunks",
]
