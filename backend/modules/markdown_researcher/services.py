from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from backend.modules.vector_store.manifest import build_chunk_id, compute_content_hash
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import AppConfig, MarkdownResearcherConfig
from backend.utils.tokenization import count_tokens, chunk_text, get_max_input_tokens

logger = logging.getLogger(__name__)


def _chunk_source_path(path: Path, project_root: Path) -> str:
    """Return project-relative path for stable chunk identifiers."""
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_chunk_tokens(config: MarkdownResearcherConfig) -> int:
    """Resolve the effective chunk token budget for markdown splitting."""
    max_input_tokens = get_max_input_tokens(
        config.context_window_tokens,
        config.max_output_tokens,
        config.input_token_reserve,
        config.max_input_tokens,
    )
    if config.max_chunk_tokens is not None and max_input_tokens is not None:
        return min(config.max_chunk_tokens, max_input_tokens)
    if config.max_chunk_tokens is not None:
        return config.max_chunk_tokens
    if max_input_tokens is not None and max_input_tokens > 0:
        return max_input_tokens
    return 12000


def split_documents_by_city(
    documents: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Group documents by normalized ``city_key``.

    Note:
    - City identity is intentionally based on filename stem (set by
      ``load_markdown_documents``), not directory structure.
    - Documents from different folders with the same stem are intentionally
      merged into one city bucket.
    """
    by_city: dict[str, list[dict[str, object]]] = {}
    for doc in documents:
        raw_city_key = doc.get("city_key")
        if isinstance(raw_city_key, str) and raw_city_key.strip():
            city_key = normalize_city_key(raw_city_key)
        else:
            raw_city = doc.get("city_name", "unknown")
            city_key = normalize_city_key(str(raw_city)) or "unknown"
        if city_key not in by_city:
            by_city[city_key] = []
        by_city[city_key].append(doc)
    return by_city


def resolve_batch_input_token_limit(config: AppConfig) -> int:
    """Resolve batch input token budget with adaptive defaults."""
    configured_limit = config.markdown_researcher.batch_max_input_tokens
    max_chunks = max(config.markdown_researcher.batch_max_chunks, 1)
    overhead = max(config.markdown_researcher.batch_overhead_tokens, 0)
    adaptive_limit = config.vector_store.embedding_chunk_tokens * max_chunks + overhead
    batch_limit = configured_limit if configured_limit is not None else adaptive_limit

    model_input_limit = get_max_input_tokens(
        config.markdown_researcher.context_window_tokens,
        config.markdown_researcher.max_output_tokens,
        config.markdown_researcher.input_token_reserve,
        config.markdown_researcher.max_input_tokens,
    )
    if model_input_limit is None:
        return max(batch_limit, 1)
    return max(min(batch_limit, model_input_limit), 1)


def load_markdown_documents(
    markdown_dir: Path,
    config: MarkdownResearcherConfig,
    selected_cities: list[str] | None = None,
) -> list[dict[str, object]]:
    """Load and chunk markdown files for the researcher input payload.

    Behavior:
    - Recursively discovers ``*.md`` files under ``markdown_dir``.
    - Optionally filters files by ``selected_cities`` (matched against ``Path.stem``,
      case-insensitive).
    - Assigns ``city_name`` from ``Path.stem`` intentionally, so files with the
      same stem in different subdirectories map to the same logical city.
    - Returns one entry per chunk with ``path``, ``city_name``, ``content``,
      deterministic ``chunk_id``, and file-local ``chunk_index``.
    """
    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {markdown_dir}")

    docs: list[dict[str, object]] = []
    files = sorted(markdown_dir.rglob("*.md"))
    if selected_cities:
        requested = {city.strip().casefold() for city in selected_cities if city.strip()}
        files = [path for path in files if path.stem.casefold() in requested]
    if len(files) > config.max_files:
        files = files[: config.max_files]

    project_root = Path(__file__).resolve().parents[3]
    max_chunk_tokens = _resolve_chunk_tokens(config)
    for path in files:
        size = path.stat().st_size
        if size > config.max_file_bytes:
            logger.warning("Skipping large markdown file: %s", path)
            continue
        city_name = path.stem
        city_key = normalize_city_key(city_name)
        content = path.read_text(encoding="utf-8")
        source_path = _chunk_source_path(path, project_root)
        chunks = chunk_text(content, max_chunk_tokens, config.chunk_overlap_tokens)
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = build_chunk_id(
                source_path=source_path,
                chunk_index=chunk_index,
                content_hash=compute_content_hash(chunk),
            )
            entry = {
                "path": str(path),
                "city_name": city_name,
                "city_key": city_key,
                "content": chunk,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
            }
            docs.append(entry)

    return docs


def _parse_chunk_index(value: object) -> int:
    """Parse chunk index into a sortable int with safe fallback."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 1_000_000_000
    return 1_000_000_000


def _document_sequence_key(document: dict[str, object]) -> tuple[str, int]:
    """Sort key that preserves file order when chunk indices are file-local."""
    path = str(document.get("path", "")).strip()
    chunk_index = _parse_chunk_index(document.get("chunk_index"))
    return path, chunk_index


def build_city_batches(
    documents_by_city: dict[str, list[dict[str, object]]],
    max_batch_input_tokens: int,
    max_batch_chunks: int,
    token_counter: Callable[[str], int] = count_tokens,
) -> list[tuple[str, int, list[dict[str, object]]]]:
    """Build deterministic city-scoped batches under token/chunk limits.

    Documents within each city are ordered by (path, chunk_index) before batching so
    that chunks from the same file are processed in sequence.
    """
    safe_max_chunks = max(max_batch_chunks, 1)
    safe_max_tokens = max(max_batch_input_tokens, 1)
    tasks: list[tuple[str, int, list[dict[str, object]]]] = []

    for city_name, city_documents in sorted(documents_by_city.items()):
        ordered_documents = sorted(city_documents, key=_document_sequence_key)
        batches: list[list[dict[str, object]]] = []
        current_batch: list[dict[str, object]] = []
        current_tokens = 0

        for document in ordered_documents:
            content = str(document.get("content", ""))
            document_tokens = max(token_counter(content), 0)
            exceeds_limits = bool(current_batch) and (
                len(current_batch) >= safe_max_chunks
                or (current_tokens + document_tokens) > safe_max_tokens
            )

            if exceeds_limits:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(document)

            # Keep oversized chunks isolated; do not pair them with neighbors.
            if document_tokens > safe_max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
                continue

            current_tokens += document_tokens

        if current_batch:
            batches.append(current_batch)

        for batch_index, batch in enumerate(batches, start=1):
            tasks.append((city_name, batch_index, batch))
    return tasks


__all__ = [
    "build_city_batches",
    "resolve_batch_input_token_limit",
    "load_markdown_documents",
    "split_documents_by_city",
]
