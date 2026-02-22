from __future__ import annotations

import logging
import os
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
from backend.utils.config import AppConfig, load_config

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


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by OpenAI-compatible embeddings API."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend([item.embedding for item in response.data])
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
    city_name = file_path.stem
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


def _embed_chunks(chunks: list[IndexedChunk], provider: EmbeddingProvider) -> list[IndexedChunk]:
    """Attach embeddings to chunk objects using provider output."""
    if not chunks:
        return []
    embeddings = provider.embed_texts([chunk.document for chunk in chunks])
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
    return Path(str(source_path)).stem.strip().casefold()


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
    store = ChromaStore(settings.persist_path, settings.collection_name)
    if not dry_run:
        store.reset_collection()

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

    if all_chunks and not dry_run:
        provider = OpenAIEmbeddingProvider(settings.embedding_model, config.openrouter_base_url)
        store.upsert(_embed_chunks(all_chunks, provider))
    if not dry_run:
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

    store = ChromaStore(settings.persist_path, settings.collection_name)
    current_files = _iter_markdown_files(docs_dir, selected_cities=selected_cities)
    current_source_map = {_source_path(path, project_root): path for path in current_files}

    changed_chunks: list[IndexedChunk] = []
    files_changed = 0
    files_unchanged = 0
    files_deleted = 0

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
        if previous_chunk_ids and not dry_run:
            store.delete([str(chunk_id) for chunk_id in previous_chunk_ids])

        file_hash, chunks = _build_indexed_chunks_for_file(file_path, settings, project_root)
        _apply_manifest_file_entry(
            manifest=manifest,
            source_path=source_path,
            file_hash=file_hash,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )
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
    for source_path in removed_sources:
        chunk_ids = files_section[source_path].get("chunk_ids", [])
        if chunk_ids and not dry_run:
            store.delete([str(chunk_id) for chunk_id in chunk_ids])
        files_section.pop(source_path, None)
        files_deleted += 1

    if changed_chunks and not dry_run:
        provider = OpenAIEmbeddingProvider(settings.embedding_model, config.openrouter_base_url)
        store.upsert(_embed_chunks(changed_chunks, provider))
    if not dry_run:
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
