from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.utils.config import AppConfig


@dataclass(frozen=True)
class VectorStoreSettings:
    enabled: bool
    persist_path: Path
    collection_name: str
    embedding_model: str
    embedding_batch_size: int
    embedding_max_retries: int
    embedding_retry_base_seconds: float
    embedding_retry_max_seconds: float
    chunk_tokens: int
    chunk_overlap_tokens: int
    table_row_group_max_rows: int
    retrieval_max_distance: float | None
    retrieval_fallback_min_chunks_per_city_query: int
    retrieval_max_chunks_per_city_query: int
    retrieval_max_chunks_per_city: int | None
    context_window_chunks: int
    table_context_window_chunks: int
    auto_update_on_run: bool
    manifest_path: Path


def get_vector_store_settings(config: AppConfig) -> VectorStoreSettings:
    """Map app config into vector store settings dataclass."""
    vector_store = config.vector_store
    return VectorStoreSettings(
        enabled=vector_store.enabled,
        persist_path=vector_store.chroma_persist_path,
        collection_name=vector_store.chroma_collection_name,
        embedding_model=vector_store.embedding_model,
        embedding_batch_size=vector_store.embedding_batch_size,
        embedding_max_retries=vector_store.embedding_max_retries,
        embedding_retry_base_seconds=vector_store.embedding_retry_base_seconds,
        embedding_retry_max_seconds=vector_store.embedding_retry_max_seconds,
        chunk_tokens=vector_store.embedding_chunk_tokens,
        chunk_overlap_tokens=vector_store.embedding_chunk_overlap_tokens,
        table_row_group_max_rows=vector_store.table_row_group_max_rows,
        retrieval_max_distance=vector_store.retrieval_max_distance,
        retrieval_fallback_min_chunks_per_city_query=vector_store.retrieval_fallback_min_chunks_per_city_query,
        retrieval_max_chunks_per_city_query=vector_store.retrieval_max_chunks_per_city_query,
        retrieval_max_chunks_per_city=vector_store.retrieval_max_chunks_per_city,
        context_window_chunks=vector_store.context_window_chunks,
        table_context_window_chunks=vector_store.table_context_window_chunks,
        auto_update_on_run=vector_store.auto_update_on_run,
        manifest_path=vector_store.index_manifest_path,
    )


__all__ = ["VectorStoreSettings", "get_vector_store_settings"]
