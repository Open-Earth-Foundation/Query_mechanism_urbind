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
    chunk_tokens: int
    chunk_overlap_tokens: int
    table_row_group_max_rows: int
    manifest_path: Path


def get_vector_store_settings(config: AppConfig) -> VectorStoreSettings:
    """Map app config into vector store settings dataclass."""
    vector_store = config.vector_store
    return VectorStoreSettings(
        enabled=vector_store.enabled,
        persist_path=vector_store.chroma_persist_path,
        collection_name=vector_store.chroma_collection_name,
        embedding_model=vector_store.embedding_model,
        chunk_tokens=vector_store.embedding_chunk_tokens,
        chunk_overlap_tokens=vector_store.embedding_chunk_overlap_tokens,
        table_row_group_max_rows=vector_store.table_row_group_max_rows,
        manifest_path=vector_store.index_manifest_path,
    )


__all__ = ["VectorStoreSettings", "get_vector_store_settings"]
