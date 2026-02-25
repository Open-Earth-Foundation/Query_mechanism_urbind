from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

from openai import OpenAI

from backend.modules.vector_store.chroma_store import ChromaStore
from backend.modules.vector_store.chunk_packer import pack_blocks
from backend.modules.vector_store.config import VectorStoreSettings, get_vector_store_settings
from backend.modules.vector_store.manifest import (
    build_chunk_id,
    compute_content_hash,
    compute_file_hash,
    load_manifest,
    mark_manifest_updated,
    save_manifest,
)
from backend.modules.vector_store.markdown_blocks import parse_markdown_blocks
from backend.modules.vector_store.models import EmbeddingProvider, IndexedChunk
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import AppConfig, load_config
from backend.utils.tokenization import chunk_text, count_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexStats:
    files_indexed: int
    files_changed: int
    files_unchanged: int
    files_deleted: int
    chunks_created: int
    table_chunks: int
    min_tokens: int
    avg_tokens: float
    max_tokens: int
    dry_run: bool


class EmbeddingIndexingError(RuntimeError):
    """Raised when one or more chunks fail to embed during index operations."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by OpenAI-compatible embeddings API."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_base_seconds: float = 0.8,
        retry_max_seconds: float = 8.0,
        max_input_tokens: int | None = 8000,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY or OPENROUTER_API_KEY must be set.")
        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or None
        )
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=resolved_base_url)
        self._batch_size = max(batch_size, 1)
        self._max_retries = max(max_retries, 0)
        self._retry_base_seconds = max(retry_base_seconds, 0.0)
        self._retry_max_seconds = max(retry_max_seconds, self._retry_base_seconds)
        self._max_input_tokens = (
            max_input_tokens if max_input_tokens is not None and max_input_tokens > 0 else None
        )

    def _trim_text_to_input_limit(self, text: str) -> str:
        """Trim one text to provider input limit when configured."""
        if self._max_input_tokens is None:
            return text
        token_count = count_tokens(text)
        if token_count <= self._max_input_tokens:
            return text
        truncated = chunk_text(text, max_tokens=self._max_input_tokens, overlap_tokens=0)[0]
        logger.warning(
            "Embedding input exceeded token limit; truncating model=%s original_tokens=%d "
            "truncated_tokens=%d original_chars=%d truncated_chars=%d max_input_tokens=%d",
            self._model,
            token_count,
            count_tokens(truncated),
            len(text),
            len(truncated),
            self._max_input_tokens,
        )
        return truncated

    def _embed_batch_once(self, texts: list[str]) -> list[list[float]]:
        """Send one embeddings request and validate response length."""
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:
            if "No embedding data" in str(exc):
                text_lengths = [len(text) for text in texts]
                token_lengths = [count_tokens(text) for text in texts]
                try:
                    raw = self._client.with_raw_response.embeddings.create(model=self._model, input=texts)
                    msg = (
                        "No embedding data received. "
                        f"Provider error: {raw.http_response.text}. "
                        f"Texts lengths: {text_lengths}. Texts token lengths: {token_lengths}"
                    )
                except Exception as e2:
                    msg = (
                        "No embedding data received. "
                        f"Texts lengths: {text_lengths}. Texts token lengths: {token_lengths}. "
                        f"Also failed to get raw response: {e2}"
                    )
                raise ValueError(msg) from exc
            raise
        embeddings = [item.embedding for item in response.data]
        if len(embeddings) != len(texts):
            raise ValueError(
                "Embedding response length mismatch: "
                f"requested={len(texts)} received={len(embeddings)}"
            )
        return embeddings

    def _embed_batch_with_retries(self, texts: list[str]) -> list[list[float]]:
        """Retry one batch with exponential backoff for transient provider failures."""
        last_error: Exception | None = None
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._embed_batch_once(texts)
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                sleep_seconds = min(
                    self._retry_base_seconds * (2 ** (attempt - 1)),
                    self._retry_max_seconds,
                )
                logger.warning(
                    "Embedding request failed for batch_size=%d (attempt %d/%d): %s",
                    len(texts),
                    attempt,
                    attempts,
                    exc,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        assert last_error is not None
        raise last_error

    def _embed_batch_one_by_one(self, texts: list[str]) -> list[list[float] | None]:
        """Fallback path: embed texts individually to isolate bad batch payloads.

        Returns ``None`` for any text that permanently fails after all retries
        so the caller can decide whether to skip or raise.
        """
        vectors: list[list[float] | None] = []
        for text in texts:
            try:
                vectors.extend(self._embed_batch_with_retries([text]))
            except Exception as exc:
                logger.error(
                    "Permanently skipping text (char_len=%d) after all retries: %s",
                    len(text),
                    exc,
                )
                vectors.append(None)
        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Embed input texts with retries and per-item fallback.

        Returns one entry per input text. ``None`` means the text permanently
        failed to embed and the caller should skip that item.
        """
        if not texts:
            return []
        prepared_texts = [self._trim_text_to_input_limit(text) for text in texts]
        vectors: list[list[float] | None] = []
        for start in range(0, len(prepared_texts), self._batch_size):
            batch = prepared_texts[start : start + self._batch_size]
            try:
                vectors.extend(self._embed_batch_with_retries(batch))
            except Exception as exc:
                logger.warning(
                    "Embedding batch failed after retries; retrying per item for batch_size=%d: %s",
                    len(batch),
                    exc,
                )
                vectors.extend(self._embed_batch_one_by_one(batch))
        return vectors


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _iter_markdown_files(
    docs_dir: Path,
    selected_cities: list[str] | None = None,
) -> list[Path]:
    """List markdown files optionally filtered by city stem."""
    files = sorted(docs_dir.rglob("*.md"))
    if not selected_cities:
        return files
    selected = {city.strip().casefold() for city in selected_cities if city.strip()}
    return [path for path in files if path.stem.casefold() in selected]


def _source_path(path: Path, project_root: Path) -> str:
    """Render project-relative source path for metadata."""
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _build_indexed_chunks_for_file(
    file_path: Path,
    settings: VectorStoreSettings,
    project_root: Path,
) -> tuple[str, list[IndexedChunk]]:
    """Parse, chunk, and shape one markdown file into indexed chunks."""
    raw_content = file_path.read_text(encoding="utf-8")
    file_hash = compute_file_hash(raw_content)
    blocks = parse_markdown_blocks(raw_content)
    packed_chunks = pack_blocks(
        blocks=blocks,
        max_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        table_row_group_max_rows=settings.table_row_group_max_rows,
    )
    source_path = _source_path(file_path, project_root)
    city_name = file_path.stem.strip()
    city_key = normalize_city_key(city_name)
    timestamp = _now_iso()
    indexed: list[IndexedChunk] = []

    for packed in packed_chunks:
        content_hash = compute_content_hash(packed.raw_text)
        chunk_id = build_chunk_id(
            source_path=source_path,
            chunk_index=packed.chunk_index,
            content_hash=content_hash,
        )
        metadata: dict[str, str | int | float | bool | None] = {
            "city_name": city_name,
            "city_key": city_key,
            "source_path": source_path,
            "block_type": packed.block_type,
            "heading_path": packed.heading_path,
            "chunk_index": packed.chunk_index,
            "token_count": packed.token_count,
            "content_hash": content_hash,
            "file_hash": file_hash,
            "raw_text": packed.raw_text,
            "created_at": timestamp,
            "updated_at": timestamp,
            "start_line": packed.start_line,
            "end_line": packed.end_line,
            "table_id": packed.table_id,
            "row_group_index": packed.row_group_index,
            "table_title": packed.table_title,
            "chunk_id": chunk_id,
        }
        indexed.append(
            IndexedChunk(
                chunk_id=chunk_id,
                document=packed.embedding_text,
                metadata=metadata,
            )
        )

    return file_hash, indexed


def _collect_token_stats(chunks: list[IndexedChunk]) -> tuple[int, float, int]:
    """Compute min/avg/max token statistics for chunk metadata."""
    if not chunks:
        return 0, 0.0, 0
    token_counts = [
        int(chunk.metadata.get("token_count", 0))
        for chunk in chunks
        if isinstance(chunk.metadata.get("token_count", 0), int)
    ]
    if not token_counts:
        return 0, 0.0, 0
    return min(token_counts), sum(token_counts) / len(token_counts), max(token_counts)


def _embed_chunks(
    chunks: list[IndexedChunk],
    provider: EmbeddingProvider,
    operation_name: str,
) -> list[IndexedChunk]:
    """Attach embeddings to chunk objects using provider output.

    Any permanently failed embedding (``None``) aborts the operation to avoid
    partial index state and manifest/vector drift.
    """
    if not chunks:
        return []
    embeddings = provider.embed_texts([chunk.document for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise EmbeddingIndexingError(
            f"{operation_name} aborted due to embedding response size mismatch: "
            f"chunks={len(chunks)} embeddings={len(embeddings)}"
        )

    failed_chunks: list[IndexedChunk] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        if embedding is None:
            failed_chunks.append(chunk)
    if failed_chunks:
        sample = ", ".join(
            f"{chunk.chunk_id}@{chunk.metadata.get('source_path', '<unknown>')}"
            for chunk in failed_chunks[:5]
        )
        logger.error(
            "%s aborted due to embedding failures failed=%d total=%d sample=%s",
            operation_name,
            len(failed_chunks),
            len(chunks),
            sample,
        )
        raise EmbeddingIndexingError(
            f"{operation_name} aborted due to embedding failures "
            f"(failed={len(failed_chunks)}, total={len(chunks)})."
        )

    embedded: list[IndexedChunk] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        embedded.append(
            IndexedChunk(
                chunk_id=chunk.chunk_id,
                document=chunk.document,
                metadata=chunk.metadata,
                embedding=embedding,
            )
        )
    return embedded


def _apply_manifest_file_entry(
    manifest: dict,
    source_path: str,
    file_hash: str,
    chunk_ids: list[str],
) -> None:
    """Upsert one file entry in manifest files map."""
    files = manifest.setdefault("files", {})
    files[source_path] = {
        "file_hash": file_hash,
        "chunk_ids": chunk_ids,
    }


def _source_city_key(source_path: str) -> str:
    """Return normalized city key derived from a manifest source path."""
    return normalize_city_key(Path(str(source_path)).stem)


def build_markdown_index(
    config: AppConfig,
    docs_dir: Path,
    selected_cities: list[str] | None = None,
    dry_run: bool = False,
    chunks_dump_path: Path | None = None,
) -> IndexStats:
    """Build a full markdown index from scratch."""
    settings = get_vector_store_settings(config)
    project_root = Path.cwd()
    files = _iter_markdown_files(docs_dir, selected_cities=selected_cities)

    manifest = {"files": {}}
    mark_manifest_updated(
        manifest,
        embedding_model=settings.embedding_model,
        embedding_chunk_tokens=settings.chunk_tokens,
        embedding_chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    all_chunks: list[IndexedChunk] = []
    files_indexed = 0

    for file_path in files:
        file_hash, chunks = _build_indexed_chunks_for_file(file_path, settings, project_root)
        source_path = _source_path(file_path, project_root)
        _apply_manifest_file_entry(
            manifest=manifest,
            source_path=source_path,
            file_hash=file_hash,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )
        all_chunks.extend(chunks)
        files_indexed += 1

    if chunks_dump_path is not None and all_chunks:
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "metadata": chunk.metadata,
            }
            for chunk in all_chunks
        ]
        chunks_dump_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_dump_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    embedded_chunks: list[IndexedChunk] = []
    if all_chunks and not dry_run:
        provider = OpenAIEmbeddingProvider(
            model=settings.embedding_model,
            base_url=config.openrouter_base_url,
            batch_size=settings.embedding_batch_size,
            max_retries=settings.embedding_max_retries,
            retry_base_seconds=settings.embedding_retry_base_seconds,
            retry_max_seconds=settings.embedding_retry_max_seconds,
            max_input_tokens=settings.embedding_max_input_tokens,
        )
        embedded_chunks = _embed_chunks(
            all_chunks,
            provider,
            operation_name="Index build",
        )
    if not dry_run:
        store = ChromaStore(settings.persist_path, settings.collection_name)
        store.reset_collection()
        if embedded_chunks:
            store.upsert(embedded_chunks)
        save_manifest(settings.manifest_path, manifest)

    min_tokens, avg_tokens, max_tokens = _collect_token_stats(all_chunks)
    table_chunks = len(
        [chunk for chunk in all_chunks if chunk.metadata.get("block_type") == "table"]
    )
    return IndexStats(
        files_indexed=files_indexed,
        files_changed=files_indexed,
        files_unchanged=0,
        files_deleted=0,
        chunks_created=len(all_chunks),
        table_chunks=table_chunks,
        min_tokens=min_tokens,
        avg_tokens=avg_tokens,
        max_tokens=max_tokens,
        dry_run=dry_run,
    )


def update_markdown_index(
    config: AppConfig,
    docs_dir: Path,
    selected_cities: list[str] | None = None,
    dry_run: bool = False,
) -> IndexStats:
    """Incrementally update markdown index from manifest state."""
    settings = get_vector_store_settings(config)
    project_root = Path.cwd()
    manifest = load_manifest(settings.manifest_path)
    mark_manifest_updated(
        manifest,
        embedding_model=settings.embedding_model,
        embedding_chunk_tokens=settings.chunk_tokens,
        embedding_chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    files_section: dict[str, dict] = manifest.setdefault("files", {})

    current_files = _iter_markdown_files(docs_dir, selected_cities=selected_cities)
    current_source_map = {_source_path(path, project_root): path for path in current_files}

    changed_chunks: list[IndexedChunk] = []
    files_changed = 0
    files_unchanged = 0
    files_deleted = 0
    changed_entries: dict[str, tuple[str, list[str]]] = {}
    previous_ids_by_source: dict[str, list[str]] = {}

    for source_path, file_path in current_source_map.items():
        content = file_path.read_text(encoding="utf-8")
        current_hash = compute_file_hash(content)
        previous = files_section.get(source_path)
        if previous and previous.get("file_hash") == current_hash:
            files_unchanged += 1
            continue

        previous_chunk_ids = (
            previous.get("chunk_ids", []) if isinstance(previous, dict) else []
        )

        file_hash, chunks = _build_indexed_chunks_for_file(file_path, settings, project_root)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        changed_entries[source_path] = (file_hash, chunk_ids)
        previous_ids_by_source[source_path] = [str(chunk_id) for chunk_id in previous_chunk_ids]
        changed_chunks.extend(chunks)
        files_changed += 1

    current_source_keys = set(current_source_map.keys())
    if selected_cities:
        selected_city_keys = {
            city.strip().casefold() for city in selected_cities if city.strip()
        }
        manifest_sources_in_scope = {
            source_path
            for source_path in files_section.keys()
            if _source_city_key(source_path) in selected_city_keys
        }
        removed_sources = sorted(manifest_sources_in_scope - current_source_keys)
    else:
        removed_sources = sorted(set(files_section.keys()) - current_source_keys)
    removed_ids_by_source: dict[str, list[str]] = {
        source_path: [
            str(chunk_id)
            for chunk_id in files_section[source_path].get("chunk_ids", [])
        ]
        for source_path in removed_sources
    }
    files_deleted = len(removed_sources)

    embedded_changed_chunks: list[IndexedChunk] = []
    if changed_chunks and not dry_run:
        provider = OpenAIEmbeddingProvider(
            model=settings.embedding_model,
            base_url=config.openrouter_base_url,
            batch_size=settings.embedding_batch_size,
            max_retries=settings.embedding_max_retries,
            retry_base_seconds=settings.embedding_retry_base_seconds,
            retry_max_seconds=settings.embedding_retry_max_seconds,
            max_input_tokens=settings.embedding_max_input_tokens,
        )
        embedded_changed_chunks = _embed_chunks(
            changed_chunks,
            provider,
            operation_name="Index update",
        )
    if not dry_run:
        store = ChromaStore(settings.persist_path, settings.collection_name)
        if embedded_changed_chunks:
            store.upsert(embedded_changed_chunks)
        for source_path, chunk_ids in previous_ids_by_source.items():
            file_hash, new_chunk_ids = changed_entries[source_path]
            new_chunk_id_set = set(new_chunk_ids)
            stale_chunk_ids = [
                chunk_id for chunk_id in chunk_ids if chunk_id not in new_chunk_id_set
            ]
            if stale_chunk_ids:
                store.delete(stale_chunk_ids)
            files_section[source_path] = {
                "file_hash": file_hash,
                "chunk_ids": new_chunk_ids,
            }
        for source_path in removed_sources:
            chunk_ids = removed_ids_by_source[source_path]
            if chunk_ids:
                store.delete(chunk_ids)
            files_section.pop(source_path, None)
        save_manifest(settings.manifest_path, manifest)

    min_tokens, avg_tokens, max_tokens = _collect_token_stats(changed_chunks)
    table_chunks = len(
        [chunk for chunk in changed_chunks if chunk.metadata.get("block_type") == "table"]
    )
    return IndexStats(
        files_indexed=len(current_files),
        files_changed=files_changed,
        files_unchanged=files_unchanged,
        files_deleted=files_deleted,
        chunks_created=len(changed_chunks),
        table_chunks=table_chunks,
        min_tokens=min_tokens,
        avg_tokens=avg_tokens,
        max_tokens=max_tokens,
        dry_run=dry_run,
    )


def ensure_index_up_to_date(docs_dir: str) -> None:
    """Ensure markdown index is incrementally updated for docs dir."""
    config = load_config()
    if not config.vector_store.enabled:
        logger.info("Vector store disabled; skipping ensure_index_up_to_date.")
        return
    stats = update_markdown_index(config=config, docs_dir=Path(docs_dir))
    logger.info(
        "Index update complete changed=%d unchanged=%d deleted=%d chunks=%d",
        stats.files_changed,
        stats.files_unchanged,
        stats.files_deleted,
        stats.chunks_created,
    )


__all__ = [
    "IndexStats",
    "build_markdown_index",
    "ensure_index_up_to_date",
    "update_markdown_index",
]
