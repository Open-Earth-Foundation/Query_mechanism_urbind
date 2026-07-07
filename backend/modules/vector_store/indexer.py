from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import json
from pathlib import PurePosixPath

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
from backend.modules.vector_store.update_lock import acquire_vector_store_update_lock
from backend.utils.city_normalization import format_city_stem, normalize_city_key
from backend.utils.config import AppConfig, load_config
from backend.utils.markdown_files import list_markdown_files
from backend.utils.retry import RetrySettings, call_with_retries
from backend.utils.tokenization import chunk_text, count_tokens

logger = logging.getLogger(__name__)
INDEX_SETTINGS_VERSION = 2


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
    update_mode: str = "incremental_update"
    changed_files: list[dict[str, object]] = field(default_factory=list)
    deleted_files: list[dict[str, object]] = field(default_factory=list)
    lock_details: dict[str, object] = field(default_factory=dict)


class EmbeddingIndexingError(RuntimeError):
    """Raised when one or more chunks fail to embed during index operations."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by OpenAI-compatible embeddings API."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_base_seconds: float = 0.8,
        retry_max_seconds: float = 8.0,
        max_input_tokens: int | None = 8000,
    ) -> None:
        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL")
            or None
        )
        key_env_order = self._api_key_env_order(resolved_base_url, api_key_env)
        api_key = None
        for env_name in key_env_order:
            value = os.getenv(env_name)
            if value:
                api_key = value
                break
        if not api_key:
            expected = " or ".join(key_env_order)
            raise EnvironmentError(f"{expected} must be set.")
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=resolved_base_url)
        self._batch_size = max(batch_size, 1)
        self._max_retries = max(max_retries, 0)
        self._retry_base_seconds = max(retry_base_seconds, 0.0)
        self._retry_max_seconds = max(retry_max_seconds, self._retry_base_seconds)
        self._max_input_tokens = (
            max_input_tokens if max_input_tokens is not None and max_input_tokens > 0 else None
        )

    @staticmethod
    def _api_key_env_order(
        base_url: str | None,
        api_key_env: str | None,
    ) -> tuple[str, ...]:
        """Return API key env vars in provider-appropriate preference order."""
        if api_key_env:
            return (api_key_env,)
        if base_url and "openrouter.ai" in base_url.lower():
            return ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
        return ("OPENAI_API_KEY", "OPENROUTER_API_KEY")

    def _trim_text_to_input_limit(self, text: str) -> str:
        """Trim one text to provider input limit when configured."""
        if self._max_input_tokens is None:
            return text
        token_count = count_tokens(text)
        if token_count <= self._max_input_tokens:
            return text
        chunks = chunk_text(text, max_tokens=self._max_input_tokens, overlap_tokens=0)
        if not chunks:
            logger.warning(
                "Embedding input exceeded token limit but chunk_text returned no chunks; "
                "model=%s original_tokens=%d original_chars=%d max_input_tokens=%d",
                self._model,
                token_count,
                len(text),
                self._max_input_tokens,
            )
            return ""
        truncated = chunks[0]
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
        retry_settings = RetrySettings.bounded(
            max_attempts=self._max_retries + 1,
            backoff_base_seconds=self._retry_base_seconds,
            backoff_max_seconds=self._retry_max_seconds,
        )
        return call_with_retries(
            lambda: self._embed_batch_once(texts),
            operation="vector_embedding.batch",
            retry_settings=retry_settings,
            should_retry=lambda _exc: True,
            context={"batch_size": len(texts)},
        )

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
    """List top-level markdown files optionally filtered by city stem."""
    files = list_markdown_files(docs_dir)
    logger.info(
        "Markdown discovery docs_dir=%s resolved=%s selected_cities=%s file_count=%d",
        docs_dir,
        docs_dir.resolve(),
        selected_cities or [],
        len(files),
    )
    if not selected_cities:
        return files
    selected = {
        normalize_city_key(city)
        for city in selected_cities
        if isinstance(city, str) and city.strip()
    }
    return [path for path in files if normalize_city_key(path.stem) in selected]


def _source_path(path: Path, docs_dir: Path, project_root: Path) -> str:
    """Render one stable source path for manifest keys and chunk metadata."""
    try:
        docs_relative_path = path.relative_to(docs_dir)
    except ValueError:
        docs_relative_path = None
    if docs_relative_path is not None:
        return (Path(docs_dir.name) / docs_relative_path).as_posix()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_manifest_source_path(source_path: str, docs_dir: Path) -> str:
    """Map legacy absolute manifest keys back to one docs-relative source path."""
    normalized_path = source_path.replace("\\", "/")
    parts = PurePosixPath(normalized_path).parts
    try:
        docs_index = max(
            index for index, part in enumerate(parts) if part == docs_dir.name
        )
    except ValueError:
        return normalized_path
    return Path(*parts[docs_index:]).as_posix()


def _normalize_manifest_files_section(
    files_section: dict[str, dict],
    docs_dir: Path,
) -> dict[str, dict]:
    """Return manifest files keyed by stable docs-relative paths."""
    normalized_files: dict[str, dict] = {}
    for source_path, payload in files_section.items():
        normalized_files[_normalize_manifest_source_path(source_path, docs_dir)] = payload
    return normalized_files


def _build_indexed_chunks_for_file(
    file_path: Path,
    docs_dir: Path,
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
    source_path = _source_path(file_path, docs_dir, project_root)
    city_name = format_city_stem(file_path.stem)
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


def _selected_city_metadata(selected_cities: list[str] | None) -> list[str]:
    """Return a JSON-safe copy of requested city filters."""
    if not selected_cities:
        return []
    return [str(city) for city in selected_cities if str(city).strip()]


def _index_settings_payload(settings: VectorStoreSettings) -> dict[str, object]:
    """Return the persisted index settings that shape stored chunks and vectors."""
    return {
        "version": INDEX_SETTINGS_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_base_url": settings.embedding_base_url,
        "embedding_api_key_env": settings.embedding_api_key_env,
        "embedding_max_input_tokens": settings.embedding_max_input_tokens,
        "distance_metric": settings.distance_metric,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "table_row_group_max_rows": settings.table_row_group_max_rows,
    }


def _index_settings_signature(payload: dict[str, object]) -> str:
    """Return a stable hash for persisted index-shaping settings."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _apply_manifest_index_settings(
    manifest: dict[str, object],
    settings_payload: dict[str, object],
) -> None:
    """Persist current index-shaping settings on the manifest."""
    manifest["index_settings"] = settings_payload
    manifest["index_settings_signature"] = _index_settings_signature(settings_payload)


def _requires_full_rebuild(
    manifest: dict[str, object],
    settings_payload: dict[str, object],
) -> bool:
    """Return true when manifest metadata cannot prove the current index is still valid."""
    current_signature = _index_settings_signature(settings_payload)
    manifest_signature = manifest.get("index_settings_signature")
    if isinstance(manifest_signature, str) and manifest_signature == current_signature:
        return False
    manifest_settings = manifest.get("index_settings")
    if isinstance(manifest_settings, dict) and manifest_settings == settings_payload:
        return False
    return True


def _resolve_index_scope(
    *,
    selected_cities: list[str] | None,
    dry_run: bool,
    operation_name: str,
) -> list[str] | None:
    """Resolve city scope for one index operation.

    The shared vector store and manifest always represent the full corpus.
    Persisted writes therefore ignore city filters and rebuild/check the full
    documents set. Dry-run full builds may still use a selected-city subset for
    inspection.
    """
    if not selected_cities:
        return None
    if dry_run and operation_name == "Index build":
        return selected_cities
    logger.info(
        "%s ignoring selected_cities for shared vector-store scope selected_city_count=%d",
        operation_name,
        len(selected_cities),
    )
    return None


def _guard_against_empty_manifest_wipe(
    *,
    manifest_path: Path,
    discovered_files: list[Path],
    dry_run: bool,
    operation_label: str,
) -> None:
    """Abort a real write when zero discovered files would wipe persisted state."""
    if dry_run or discovered_files:
        return
    existing_manifest = load_manifest(manifest_path)
    existing_files = existing_manifest.get("files", {})
    existing_file_count = len(existing_files) if isinstance(existing_files, dict) else 0
    if existing_file_count == 0:
        return
    raise RuntimeError(
        f"Refusing to {operation_label} vector store with zero discovered markdown files because "
        f"the existing manifest at {manifest_path} still tracks {existing_file_count} files. "
        "Check docs_dir or city filters before rebuilding."
    )


def build_markdown_index(
    config: AppConfig,
    docs_dir: Path,
    selected_cities: list[str] | None = None,
    dry_run: bool = False,
    chunks_dump_path: Path | None = None,
) -> IndexStats:
    """Build the persisted markdown index from the full documents corpus."""
    settings = get_vector_store_settings(config)
    if dry_run:
        return _build_markdown_index_impl(
            config=config,
            docs_dir=docs_dir,
            selected_cities=selected_cities,
            dry_run=dry_run,
            chunks_dump_path=chunks_dump_path,
            settings=settings,
        )
    with acquire_vector_store_update_lock(
        settings.persist_path,
        operation="build_markdown_index",
    ) as lock_handle:
        return _build_markdown_index_impl(
            config=config,
            docs_dir=docs_dir,
            selected_cities=selected_cities,
            dry_run=dry_run,
            chunks_dump_path=chunks_dump_path,
            settings=settings,
            lock_details={
                "lock_path": str(lock_handle.path),
                "operation": lock_handle.operation,
                "acquired_after_seconds": lock_handle.acquired_after_seconds,
                "waited_for_holder": lock_handle.waited_for_holder,
            },
        )


def _build_markdown_index_impl(
    *,
    config: AppConfig,
    docs_dir: Path,
    selected_cities: list[str] | None,
    dry_run: bool,
    chunks_dump_path: Path | None,
    settings: VectorStoreSettings,
    lock_details: dict[str, object] | None = None,
) -> IndexStats:
    """Build a full markdown index from scratch after any required lock is held."""
    settings_payload = _index_settings_payload(settings)
    project_root = Path.cwd()
    effective_selected_cities = _resolve_index_scope(
        selected_cities=selected_cities,
        dry_run=dry_run,
        operation_name="Index build",
    )
    files = _iter_markdown_files(docs_dir, selected_cities=effective_selected_cities)
    total_files = len(files)
    logger.info(
        "Index build started docs_total=%d docs_dir=%s dry_run=%s",
        total_files,
        docs_dir,
        dry_run,
    )
    _guard_against_empty_manifest_wipe(
        manifest_path=settings.manifest_path,
        discovered_files=files,
        dry_run=dry_run,
        operation_label="rebuild",
    )

    manifest = {"files": {}}
    mark_manifest_updated(
        manifest,
        embedding_model=settings.embedding_model,
        embedding_chunk_tokens=settings.chunk_tokens,
        embedding_chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    _apply_manifest_index_settings(manifest, settings_payload)
    all_chunks: list[IndexedChunk] = []
    changed_files: list[dict[str, object]] = []
    files_indexed = 0

    for index, file_path in enumerate(files, start=1):
        file_hash, chunks = _build_indexed_chunks_for_file(
            file_path,
            docs_dir,
            settings,
            project_root,
        )
        source_path = _source_path(file_path, docs_dir, project_root)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        _apply_manifest_file_entry(
            manifest=manifest,
            source_path=source_path,
            file_hash=file_hash,
            chunk_ids=chunk_ids,
        )
        all_chunks.extend(chunks)
        changed_files.append(
            {
                "source_path": source_path,
                "status": "indexed",
                "previous_file_hash": None,
                "current_file_hash": file_hash,
                "previous_chunk_count": 0,
                "current_chunk_count": len(chunk_ids),
                "removed_previous_chunk_count": 0,
            }
        )
        files_indexed += 1
        logger.info(
            "Index build progress documents=%d/%d source=%s chunks=%d",
            index,
            total_files,
            source_path,
            len(chunks),
        )

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
        logger.info(
            "Index build embedding started chunks_total=%d",
            len(all_chunks),
        )
        provider = OpenAIEmbeddingProvider(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url or config.openrouter_base_url,
            api_key_env=settings.embedding_api_key_env,
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
        logger.info(
            "Index build embedding finished chunks_total=%d",
            len(embedded_chunks),
        )
    if not dry_run:
        logger.info(
            "Index build persist started collection=%s persist_path=%s",
            settings.collection_name,
            settings.persist_path,
        )
        store = ChromaStore(
            settings.persist_path,
            settings.collection_name,
            distance_metric=settings.distance_metric,
        )
        store.reset_collection()
        if embedded_chunks:
            store.upsert(embedded_chunks)
        save_manifest(
            settings.manifest_path,
            manifest,
            reason="build_markdown_index",
            docs_dir=docs_dir,
            metadata={
                "dry_run": dry_run,
                "files_indexed": files_indexed,
                "selected_cities_requested": _selected_city_metadata(selected_cities),
                "selected_cities_effective": effective_selected_cities or [],
                "docs_dir": str(docs_dir),
                "docs_dir_resolved": str(docs_dir.resolve(strict=False)),
                "persist_path": str(settings.persist_path),
                "collection_name": settings.collection_name,
                "update_mode": "full_rebuild",
            },
        )
        logger.info(
            "Index build persist finished manifest_path=%s",
            settings.manifest_path,
        )

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
        update_mode="full_rebuild",
        changed_files=changed_files,
        lock_details=lock_details or {},
    )


def update_markdown_index(
    config: AppConfig,
    docs_dir: Path,
    selected_cities: list[str] | None = None,
    dry_run: bool = False,
) -> IndexStats:
    """Incrementally update the persisted full-corpus markdown index."""
    settings = get_vector_store_settings(config)
    if dry_run:
        return _update_markdown_index_impl(
            config=config,
            docs_dir=docs_dir,
            selected_cities=selected_cities,
            dry_run=dry_run,
            settings=settings,
        )
    with acquire_vector_store_update_lock(
        settings.persist_path,
        operation="update_markdown_index",
    ) as lock_handle:
        return _update_markdown_index_impl(
            config=config,
            docs_dir=docs_dir,
            selected_cities=selected_cities,
            dry_run=dry_run,
            settings=settings,
            lock_details={
                "lock_path": str(lock_handle.path),
                "operation": lock_handle.operation,
                "acquired_after_seconds": lock_handle.acquired_after_seconds,
                "waited_for_holder": lock_handle.waited_for_holder,
            },
        )


def _update_markdown_index_impl(
    *,
    config: AppConfig,
    docs_dir: Path,
    selected_cities: list[str] | None,
    dry_run: bool,
    settings: VectorStoreSettings,
    lock_details: dict[str, object] | None = None,
) -> IndexStats:
    """Incrementally update markdown index after any required lock is held."""
    settings_payload = _index_settings_payload(settings)
    project_root = Path.cwd()
    effective_selected_cities = _resolve_index_scope(
        selected_cities=selected_cities,
        dry_run=dry_run,
        operation_name="Index update",
    )
    manifest = load_manifest(settings.manifest_path)
    if _requires_full_rebuild(manifest, settings_payload):
        logger.warning(
            "Index settings changed or are missing from manifest; forcing full rebuild "
            "docs_dir=%s selected_cities=%s manifest_path=%s",
            docs_dir,
            selected_cities,
            settings.manifest_path,
        )
        stats = _build_markdown_index_impl(
            config=config,
            docs_dir=docs_dir,
            selected_cities=None,
            dry_run=dry_run,
            chunks_dump_path=None,
            settings=settings,
            lock_details=lock_details,
        )
        if isinstance(stats, IndexStats):
            return replace(stats, update_mode="index_settings_changed_or_missing")
        return stats
    mark_manifest_updated(
        manifest,
        embedding_model=settings.embedding_model,
        embedding_chunk_tokens=settings.chunk_tokens,
        embedding_chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    _apply_manifest_index_settings(manifest, settings_payload)
    files_payload = manifest.setdefault("files", {})
    files_section = _normalize_manifest_files_section(
        files_payload if isinstance(files_payload, dict) else {},
        docs_dir,
    )
    manifest["files"] = files_section

    current_files = _iter_markdown_files(docs_dir, selected_cities=effective_selected_cities)
    _guard_against_empty_manifest_wipe(
        manifest_path=settings.manifest_path,
        discovered_files=current_files,
        dry_run=dry_run,
        operation_label="update",
    )
    current_source_map = {_source_path(path, docs_dir, project_root): path for path in current_files}
    total_files = len(current_source_map)
    if dry_run:
        logger.debug(
            "Index update dry-run started docs_total=%d docs_dir=%s",
            total_files,
            docs_dir,
        )
    else:
        logger.info(
            "Index update started docs_total=%d docs_dir=%s",
            total_files,
            docs_dir,
        )

    changed_chunks: list[IndexedChunk] = []
    files_changed = 0
    files_unchanged = 0
    files_deleted = 0
    changed_entries: dict[str, tuple[str, list[str]]] = {}
    previous_ids_by_source: dict[str, list[str]] = {}
    changed_files: list[dict[str, object]] = []

    for index, (source_path, file_path) in enumerate(current_source_map.items(), start=1):
        content = file_path.read_text(encoding="utf-8")
        current_hash = compute_file_hash(content)
        previous = files_section.get(source_path)
        if previous and previous.get("file_hash") == current_hash:
            files_unchanged += 1
            continue

        previous_chunk_ids = (
            previous.get("chunk_ids", []) if isinstance(previous, dict) else []
        )

        file_hash, chunks = _build_indexed_chunks_for_file(
            file_path,
            docs_dir,
            settings,
            project_root,
        )
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        changed_entries[source_path] = (file_hash, chunk_ids)
        previous_chunk_id_list = [str(chunk_id) for chunk_id in previous_chunk_ids]
        previous_ids_by_source[source_path] = previous_chunk_id_list
        removed_previous_chunk_count = len(set(previous_chunk_id_list) - set(chunk_ids))
        changed_files.append(
            {
                "source_path": source_path,
                "status": "modified" if previous else "added",
                "previous_file_hash": previous.get("file_hash") if isinstance(previous, dict) else None,
                "current_file_hash": file_hash,
                "previous_chunk_count": len(previous_chunk_id_list),
                "current_chunk_count": len(chunk_ids),
                "removed_previous_chunk_count": removed_previous_chunk_count,
            }
        )
        changed_chunks.extend(chunks)
        files_changed += 1
        if not dry_run:
            logger.info(
                "Index update applying documents=%d/%d source=%s status=%s chunks=%d",
                index,
                total_files,
                source_path,
                "modified" if previous else "added",
                len(chunks),
            )

    current_source_keys = set(current_source_map.keys())
    if effective_selected_cities:
        selected_city_keys = {
            normalize_city_key(city)
            for city in effective_selected_cities
            if isinstance(city, str) and city.strip()
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
    deleted_files = [
        {
            "source_path": source_path,
            "status": "deleted",
            "previous_chunk_count": len(chunk_ids),
        }
        for source_path, chunk_ids in removed_ids_by_source.items()
    ]
    files_deleted = len(removed_sources)
    if files_deleted:
        logger.info(
            "Index update detected removed manifest files deleted=%d sample=%s",
            files_deleted,
            removed_sources[:5],
        )

    embedded_changed_chunks: list[IndexedChunk] = []
    if changed_chunks and not dry_run:
        provider = OpenAIEmbeddingProvider(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url or config.openrouter_base_url,
            api_key_env=settings.embedding_api_key_env,
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
        store = ChromaStore(
            settings.persist_path,
            settings.collection_name,
            distance_metric=settings.distance_metric,
        )
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
        save_manifest(
            settings.manifest_path,
            manifest,
            reason="update_markdown_index",
            docs_dir=docs_dir,
            metadata={
                "dry_run": dry_run,
                "files_indexed": len(current_files),
                "files_changed": files_changed,
                "files_deleted": files_deleted,
                "selected_cities_requested": _selected_city_metadata(selected_cities),
                "selected_cities_effective": effective_selected_cities or [],
                "docs_dir": str(docs_dir),
                "docs_dir_resolved": str(docs_dir.resolve(strict=False)),
                "persist_path": str(settings.persist_path),
                "collection_name": settings.collection_name,
                "update_mode": "incremental_update",
            },
        )
        logger.info(
            "Index update persist finished manifest_path=%s changed=%d unchanged=%d "
            "deleted=%d chunks=%d",
            settings.manifest_path,
            files_changed,
            files_unchanged,
            files_deleted,
            len(changed_chunks),
        )
    else:
        logger.debug(
            "Index update dry-run summary docs_total=%d changed=%d unchanged=%d "
            "deleted=%d chunks=%d",
            len(current_files),
            files_changed,
            files_unchanged,
            files_deleted,
            len(changed_chunks),
        )

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
        update_mode="incremental_update",
        changed_files=changed_files,
        deleted_files=deleted_files,
        lock_details=lock_details or {},
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
