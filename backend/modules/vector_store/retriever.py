from __future__ import annotations

import os
from typing import Any
from pathlib import Path

from openai import OpenAI

from backend.modules.vector_store.chroma_store import ChromaStore
from backend.modules.vector_store.models import RetrievedChunk
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
    """Normalize and de-duplicate city names while preserving order."""
    if not cities:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for city in cities:
        candidate = city.strip()
        if not candidate:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)
    return normalized


def _embed_query_text(query: str, config: AppConfig) -> list[float]:
    """Embed one retrieval query with configured embedding model."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY or OPENROUTER_API_KEY must be set.")
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or config.openrouter_base_url
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(
        model=config.vector_store.embedding_model,
        input=[query],
    )
    return response.data[0].embedding


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
    return RetrievedChunk(
        city_name=str(metadata.get("city_name", "")),
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


def _resolve_city_list(docs_dir: Path, selected_cities: list[str] | None) -> list[str]:
    """Resolve retrieval city list; always operate at city level."""
    normalized = _normalize_cities(selected_cities)
    if normalized:
        return normalized
    if not docs_dir.exists():
        return []
    return sorted(
        {
            path.stem.strip()
            for path in docs_dir.rglob("*.md")
            if path.stem.strip()
        }
    )


def _passes_distance_threshold(distance: float, max_distance: float | None) -> bool:
    """Check whether a row distance passes the configured threshold."""
    if max_distance is None:
        return True
    return distance <= max_distance


def _retrieve_for_city_query(
    store: ChromaStore,
    query_embedding: list[float],
    city_name: str,
    fallback_min_chunks_per_city_query: int,
    max_chunks_per_city_query: int,
    max_distance: float | None,
) -> list[dict[str, Any]]:
    """Retrieve rows for one city and one query with distance-first + top-up."""
    candidate_n_results = max(max_chunks_per_city_query, 1)
    payload = store.query_by_embedding(
        query_embeddings=[query_embedding],
        n_results=candidate_n_results,
        where={"city_name": city_name},
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
    """Expand retrieval context by adding neighboring chunks from same file."""
    seed_rows = list(rows_by_id.values())
    for row in seed_rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        source_path = str(metadata.get("source_path", "")).strip()
        city_name = str(metadata.get("city_name", "")).strip()
        raw_index = metadata.get("chunk_index")
        if not source_path or not city_name or not isinstance(raw_index, int):
            continue

        is_table = str(metadata.get("block_type", "")) == "table"
        window = table_context_window_chunks if is_table else context_window_chunks
        for candidate_index in range(raw_index - window, raw_index + window + 1):
            if candidate_index < 0:
                continue
            payload = store.get(
                where={
                    "$and": [
                        {"city_name": city_name},
                        {"source_path": source_path},
                        {"chunk_index": candidate_index},
                    ]
                },
                limit=1,
            )
            candidates = _parse_get_rows(payload)
            for candidate in candidates:
                chunk_id = str(candidate["chunk_id"])
                if chunk_id in rows_by_id:
                    continue
                metadata_candidate = candidate["metadata"]
                if not isinstance(metadata_candidate, dict):
                    continue
                rows_by_id[chunk_id] = {
                    "chunk_id": chunk_id,
                    "metadata": metadata_candidate,
                    "distance": float(row["distance"]),
                    "retrieval_query_id": str(row.get("retrieval_query_id", "neighbor")),
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
    normalized_queries = _normalize_queries(queries)
    cities = _resolve_city_list(docs_dir, selected_cities)
    if not normalized_queries or not cities:
        return [], {"queries": normalized_queries, "cities": cities, "per_city": []}

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
    query_embeddings = {
        query_text: _embed_query_text(query_text, config) for query_text in normalized_queries
    }

    rows_by_id: dict[str, dict[str, Any]] = {}
    per_city_stats: list[dict[str, Any]] = []
    for city_name in cities:
        city_rows: dict[str, dict[str, Any]] = {}
        query_stats: list[dict[str, Any]] = []
        for index, query_text in enumerate(normalized_queries, start=1):
            query_id = f"q{index}"
            rows = _retrieve_for_city_query(
                store=store,
                query_embedding=query_embeddings[query_text],
                city_name=city_name,
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
            current = counts.get(chunk.city_name, 0)
            if current >= max_chunks_per_city:
                continue
            counts[chunk.city_name] = current + 1
            selected.append(chunk)
        chunks = selected

    return (
        chunks,
        {
            "queries": normalized_queries,
            "cities": cities,
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


def as_markdown_documents(chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    """Map retrieved chunks into markdown researcher input shape."""
    documents: list[dict[str, str]] = []
    for chunk in chunks:
        documents.append(
            {
                "path": chunk.source_path,
                "city_name": chunk.city_name,
                "content": chunk.raw_text,
                "chunk_id": chunk.chunk_id,
                "distance": f"{chunk.distance:.6f}",
                "heading_path": chunk.heading_path,
                "block_type": chunk.block_type,
            }
        )
    return documents


__all__ = [
    "as_markdown_documents",
    "retrieve_chunks_for_queries",
    "retrieve_top_k_chunks",
]
