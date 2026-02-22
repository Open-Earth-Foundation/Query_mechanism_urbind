from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised in environments without chromadb
    chromadb = None  # type: ignore[assignment]

Collection = Any
MAX_UPSERT_BATCH_SIZE = 5000

from backend.modules.vector_store.models import IndexedChunk


def _resolve_collection_not_found_error_types() -> tuple[type[BaseException], ...]:
    """Resolve Chroma error types that represent a missing collection."""
    if chromadb is None:
        return ()
    errors_module = getattr(chromadb, "errors", None)
    if errors_module is None:
        return ()
    types: list[type[BaseException]] = []
    for name in ("NotFoundError", "CollectionNotFoundError", "InvalidCollectionException"):
        candidate = getattr(errors_module, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            types.append(candidate)
    return tuple(types)


COLLECTION_NOT_FOUND_ERROR_TYPES = _resolve_collection_not_found_error_types()


def _is_collection_not_found_error(error: Exception) -> bool:
    """Return True when delete failure is a collection-not-found condition."""
    if COLLECTION_NOT_FOUND_ERROR_TYPES and isinstance(
        error, COLLECTION_NOT_FOUND_ERROR_TYPES
    ):
        return True
    message = str(error).strip().lower()
    return (
        "collection" in message
        and ("not found" in message or "does not exist" in message)
    )


def get_client(persist_path: Path):
    """Create a persistent Chroma client for a local path."""
    if chromadb is None:
        raise ImportError(
            "chromadb is not installed. Install dependencies to use vector store features."
        )
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path))


class ChromaStore:
    """Thin wrapper around Chroma collection operations."""

    def __init__(self, persist_path: Path, collection_name: str) -> None:
        self._client = get_client(persist_path)
        self._collection_name = collection_name

    def get_collection(self) -> Collection:
        """Get or create underlying Chroma collection."""
        return self._client.get_or_create_collection(name=self._collection_name)

    def reset_collection(self) -> None:
        """Delete and recreate collection for full rebuild."""
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception as exc:  # noqa: BLE001
            if not _is_collection_not_found_error(exc):
                raise
        self._client.get_or_create_collection(name=self._collection_name)

    def upsert(self, chunks: list[IndexedChunk]) -> None:
        """Upsert indexed chunks into Chroma collection in safe batches."""
        if not chunks:
            return
        collection = self.get_collection()
        for start in range(0, len(chunks), MAX_UPSERT_BATCH_SIZE):
            batch = chunks[start : start + MAX_UPSERT_BATCH_SIZE]
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.document for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            embeddings = [chunk.embedding for chunk in batch if chunk.embedding is not None]
            payload: dict[str, Any] = {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
            }
            if len(embeddings) == len(batch):
                payload["embeddings"] = embeddings
            collection.upsert(**payload)

    def delete(self, ids: list[str]) -> None:
        """Delete chunks by id."""
        if not ids:
            return
        self.get_collection().delete(ids=ids)

    def count(self) -> int:
        """Return number of stored vectors."""
        return self.get_collection().count()

    def peek(self, n: int = 5) -> dict[str, Any]:
        """Peek first N vectors for quick inspection."""
        return self.get_collection().peek(limit=n)

    def get(
        self,
        where: dict[str, Any] | None = None,
        limit: int = 20,
        ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch vectors by optional metadata filters or ids."""
        return self.get_collection().get(where=where, limit=limit, ids=ids)

    def query(
        self,
        query_texts: list[str],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run similarity query and return raw Chroma response."""
        return self.get_collection().query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
        )

    def query_by_embedding(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run similarity query using precomputed query embeddings."""
        return self.get_collection().query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
        )


__all__ = ["ChromaStore", "get_client"]
