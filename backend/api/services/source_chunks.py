"""Resolve persisted markdown chunk ids into full chunk content."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.api.models import SourceChunkItem
from backend.modules.markdown_researcher.services import build_markdown_chunks_for_file
from backend.modules.vector_store.chroma_store import ChromaStore
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def normalize_chunk_ids(chunk_ids: list[str] | None) -> list[str]:
    """Trim, de-duplicate, and preserve order for requested chunk ids."""
    normalized: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids or []:
        candidate = chunk_id.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def load_source_chunks(
    run_dir: Path,
    markdown_dir: Path,
    config: AppConfig,
    chunk_ids: list[str],
) -> list[SourceChunkItem]:
    """Resolve full chunk content for the requested ids or raise when missing."""
    normalized_chunk_ids = normalize_chunk_ids(chunk_ids)
    if not normalized_chunk_ids:
        raise ValueError("At least one chunk_id query parameter is required.")

    chunks_by_id = _load_vector_store_chunks(config, normalized_chunk_ids)
    missing_chunk_ids = [
        chunk_id for chunk_id in normalized_chunk_ids if chunk_id not in chunks_by_id
    ]
    if missing_chunk_ids:
        fallback_chunks = _load_markdown_chunks(run_dir, markdown_dir, config, missing_chunk_ids)
        chunks_by_id.update(fallback_chunks)
        missing_chunk_ids = [
            chunk_id for chunk_id in normalized_chunk_ids if chunk_id not in chunks_by_id
        ]

    if missing_chunk_ids:
        joined_ids = ", ".join(missing_chunk_ids)
        raise FileNotFoundError(f"Source chunks were not found: {joined_ids}.")

    return [chunks_by_id[chunk_id] for chunk_id in normalized_chunk_ids]


def _load_vector_store_chunks(
    config: AppConfig,
    chunk_ids: list[str],
) -> dict[str, SourceChunkItem]:
    """Fetch chunk content directly from Chroma when the ids exist in the index."""
    if not config.vector_store.enabled:
        return {}

    try:
        store = ChromaStore(
            persist_path=config.vector_store.chroma_persist_path,
            collection_name=config.vector_store.chroma_collection_name,
        )
        payload = store.get(ids=chunk_ids, limit=max(len(chunk_ids), 1))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read source chunks from vector store: %s", exc)
        return {}

    ids = payload.get("ids")
    if not isinstance(ids, list):
        return {}
    metadatas = payload.get("metadatas")
    documents = payload.get("documents")
    metadata_rows = metadatas if isinstance(metadatas, list) else []
    document_rows = documents if isinstance(documents, list) else []
    resolved: dict[str, SourceChunkItem] = {}

    for chunk_id, metadata, document in zip(ids, metadata_rows, document_rows, strict=False):
        if not isinstance(chunk_id, str):
            continue
        metadata_payload = metadata if isinstance(metadata, dict) else {}
        raw_text = metadata_payload.get("raw_text")
        content = raw_text if isinstance(raw_text, str) else str(document or "")
        resolved[chunk_id] = SourceChunkItem(
            chunk_id=chunk_id,
            content=content,
            city_name=_coerce_optional_string(metadata_payload.get("city_name")),
            source_path=_coerce_optional_string(metadata_payload.get("source_path")),
            heading_path=_coerce_optional_string(metadata_payload.get("heading_path")),
            block_type=_coerce_optional_string(metadata_payload.get("block_type")),
        )

    return resolved


def _load_markdown_chunks(
    run_dir: Path,
    markdown_dir: Path,
    config: AppConfig,
    chunk_ids: list[str],
) -> dict[str, SourceChunkItem]:
    """Rebuild chunks from source markdown files using run-local path hints when available."""
    chunk_paths = _load_chunk_path_hints(run_dir)
    resolved_candidate_paths = (
        _resolve_candidate_path(chunk_paths.get(chunk_id), markdown_dir)
        for chunk_id in chunk_ids
    )
    candidate_paths = [
        path
        for path in resolved_candidate_paths
        if path is not None
    ]
    if not candidate_paths:
        candidate_paths = sorted(markdown_dir.rglob("*.md"))

    resolved: dict[str, SourceChunkItem] = {}
    remaining_ids = set(chunk_ids)
    seen_paths: set[Path] = set()
    for path in candidate_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            chunks = build_markdown_chunks_for_file(path, config.markdown_researcher)
        except FileNotFoundError:
            continue
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if chunk_id not in remaining_ids:
                continue
            resolved[chunk_id] = SourceChunkItem(
                chunk_id=chunk_id,
                content=str(chunk.get("content", "")),
                city_name=_coerce_optional_string(chunk.get("city_name")),
                source_path=_coerce_optional_string(chunk.get("path")),
            )
            remaining_ids.remove(chunk_id)
        if not remaining_ids:
            break

    return resolved


def _load_chunk_path_hints(run_dir: Path) -> dict[str, Path]:
    """Read chunk-to-path hints from run-local markdown batch artifacts."""
    batches_path = run_dir / "markdown" / "batches.json"
    if not batches_path.exists():
        return {}

    try:
        payload = json.loads(batches_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    batches = payload.get("batches")
    if not isinstance(batches, list):
        return {}

    chunk_paths: dict[str, Path] = {}
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        chunks = batch.get("chunks")
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("chunk_id")
            raw_path = chunk.get("path")
            if not isinstance(chunk_id, str) or not isinstance(raw_path, str):
                continue
            candidate = raw_path.strip()
            if not candidate:
                continue
            chunk_paths[chunk_id] = Path(candidate)
    return chunk_paths


def _resolve_candidate_path(raw_path: Path | None, markdown_dir: Path) -> Path | None:
    """Resolve a batch-hinted path to an existing markdown file."""
    if raw_path is None:
        return None
    if raw_path.is_absolute():
        return raw_path if raw_path.exists() else None

    candidates = [
        PROJECT_ROOT / raw_path,
        markdown_dir / raw_path,
        markdown_dir / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _coerce_optional_string(value: object) -> str | None:
    """Return a stripped string or None when the value is blank/non-string."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate or None


__all__ = ["load_source_chunks", "normalize_chunk_ids"]
