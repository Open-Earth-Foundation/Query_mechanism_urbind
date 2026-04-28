"""Retrieval against a benchmark Chroma collection.

Distinct from ``backend.modules.vector_store.retriever`` because:
- Benchmark docs are cross-cutting (national, European, methodology) and
  do not have a per-city filter.
- The collection lives at a separate persist path (e.g. ``.chroma/benchmarks/``)
  and uses its own collection name.

The function returns a small list of ``BenchmarkExcerpt`` objects with
enough metadata to attribute each excerpt back to its upstream source
and tier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.modules.vector_store.chroma_store import ChromaStore
from backend.modules.vector_store.indexer import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_PERSIST_PATH = Path(".chroma/benchmarks")
DEFAULT_BENCHMARK_COLLECTION = "urbind_benchmarks"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


@dataclass(frozen=True)
class BenchmarkExcerpt:
    chunk_id: str
    raw_text: str
    distance: float
    source_id: str
    ingestion_id: str
    source_path: str
    doc_slug: str
    tier: str
    heading_path: str
    block_type: str
    chunk_index: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _row_to_excerpt(
    chunk_id: str,
    metadata: dict[str, Any],
    distance: float,
) -> BenchmarkExcerpt:
    raw_index = metadata.get("chunk_index")
    chunk_index = raw_index if isinstance(raw_index, int) else None
    return BenchmarkExcerpt(
        chunk_id=chunk_id,
        raw_text=str(metadata.get("raw_text", "")),
        distance=distance,
        source_id=str(metadata.get("source_id", "")),
        ingestion_id=str(metadata.get("ingestion_id", "")),
        source_path=str(metadata.get("source_path", "")),
        doc_slug=str(metadata.get("doc_slug", "")),
        tier=str(metadata.get("tier", "")),
        heading_path=str(metadata.get("heading_path", "")),
        block_type=str(metadata.get("block_type", "")),
        chunk_index=chunk_index,
        extra={
            key: value
            for key, value in metadata.items()
            if key not in {
                "raw_text",
                "source_id",
                "ingestion_id",
                "source_path",
                "doc_slug",
                "tier",
                "heading_path",
                "block_type",
                "chunk_index",
                "chunk_id",
            }
        },
    )


def _embed(
    queries: list[str],
    *,
    embedding_model: str,
    base_url: str | None,
) -> dict[str, list[float]]:
    provider = OpenAIEmbeddingProvider(model=embedding_model, base_url=base_url)
    embeddings = provider.embed_texts(queries)
    out: dict[str, list[float]] = {}
    for text, embedding in zip(queries, embeddings, strict=True):
        if embedding is None:
            raise ValueError(f"benchmark_retriever: failed to embed query {text!r}")
        out[text] = embedding
    return out


def retrieve_benchmark_excerpts(
    queries: list[str],
    *,
    k: int = 5,
    persist_path: Path | None = None,
    collection_name: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str | None = None,
    max_distance: float | None = None,
    embed_fn: callable | None = None,
) -> list[BenchmarkExcerpt]:
    """Return the top-k chunks across ``queries`` from the benchmark collection.

    Args:
        queries: One or more retrieval queries; results are merged across them
            keeping the smallest distance per chunk.
        k: Maximum number of distinct chunks to return.
        persist_path: Override the default benchmark persist path.
        collection_name: Override the default collection name.
        embedding_model: Embedding model to use.
        base_url: OpenAI/OpenRouter base URL override.
        max_distance: If set, drop chunks with distance > max_distance.
        embed_fn: Test seam — replaces the live embedding call.
    """
    cleaned_queries = [q.strip() for q in queries if q and q.strip()]
    if not cleaned_queries:
        return []

    persist = (persist_path or DEFAULT_BENCHMARK_PERSIST_PATH).resolve()
    if not persist.exists():
        raise FileNotFoundError(
            f"Benchmark Chroma persist path not found: {persist}. "
            "Run `uv run python -m backend.scripts.sources_ingest urbind_benchmarks` first."
        )

    collection = collection_name or DEFAULT_BENCHMARK_COLLECTION
    store = ChromaStore(persist_path=persist, collection_name=collection)

    if embed_fn is not None:
        embedded = embed_fn(cleaned_queries)
    else:
        embedded = _embed(
            cleaned_queries,
            embedding_model=embedding_model,
            base_url=base_url,
        )

    rows_by_id: dict[str, tuple[dict[str, Any], float]] = {}
    n_results = max(k, 1)
    for query in cleaned_queries:
        embedding = embedded[query]
        payload = store.query_by_embedding(
            query_embeddings=[embedding],
            n_results=n_results,
        )
        ids = payload.get("ids", [[]])[0] if payload.get("ids") else []
        metadatas = payload.get("metadatas", [[]])[0] if payload.get("metadatas") else []
        distances = payload.get("distances", [[]])[0] if payload.get("distances") else []
        for chunk_id, metadata, distance in zip(ids, metadatas, distances, strict=False):
            if not isinstance(metadata, dict):
                continue
            try:
                distance_value = float(distance)
            except (TypeError, ValueError):
                continue
            if max_distance is not None and distance_value > max_distance:
                continue
            existing = rows_by_id.get(str(chunk_id))
            if existing is None or distance_value < existing[1]:
                rows_by_id[str(chunk_id)] = (metadata, distance_value)

    sorted_rows = sorted(rows_by_id.items(), key=lambda item: item[1][1])
    sliced = sorted_rows[: max(k, 1)]
    return [
        _row_to_excerpt(chunk_id, metadata, distance)
        for chunk_id, (metadata, distance) in sliced
    ]


__all__ = [
    "BenchmarkExcerpt",
    "DEFAULT_BENCHMARK_COLLECTION",
    "DEFAULT_BENCHMARK_PERSIST_PATH",
    "retrieve_benchmark_excerpts",
]
