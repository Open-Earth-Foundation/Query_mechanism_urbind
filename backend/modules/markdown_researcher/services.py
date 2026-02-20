from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from backend.modules.vector_store.manifest import build_chunk_id, compute_content_hash
from backend.utils.config import MarkdownResearcherConfig
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
    documents: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Group documents by the precomputed ``city_name`` key.

    Note:
    - City identity is intentionally based on filename stem (set by
      ``load_markdown_documents``), not directory structure.
    - Documents from different folders with the same stem are intentionally
      merged into one city bucket.
    """
    by_city: dict[str, list[dict[str, str]]] = {}
    for doc in documents:
        city_name = doc.get("city_name", "unknown")
        if city_name not in by_city:
            by_city[city_name] = []
        by_city[city_name].append(doc)
    return by_city


def load_markdown_documents(
    markdown_dir: Path,
    config: MarkdownResearcherConfig,
    selected_cities: list[str] | None = None,
) -> list[dict[str, str]]:
    """Load and chunk markdown files for the researcher input payload.

    Behavior:
    - Recursively discovers ``*.md`` files under ``markdown_dir``.
    - Optionally filters files by ``selected_cities`` (matched against ``Path.stem``,
      case-insensitive).
    - Assigns ``city_name`` from ``Path.stem`` intentionally, so files with the
      same stem in different subdirectories map to the same logical city.
    - Returns one entry per chunk with ``path``, ``city_name``, ``content``, and
      deterministic ``chunk_id``.
    """
    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {markdown_dir}")

    docs: list[dict[str, str]] = []
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
                "content": chunk,
                "chunk_id": chunk_id,
            }
            docs.append(entry)

    return docs


def build_city_batches(
    documents_by_city: dict[str, list[dict[str, str]]],
    max_batch_input_tokens: int,
    max_batch_chunks: int,
    token_counter: Callable[[str], int] = count_tokens,
) -> list[tuple[str, int, list[dict[str, str]]]]:
    """Build deterministic city-scoped batches under token/chunk limits."""
    safe_max_chunks = max(max_batch_chunks, 1)
    safe_max_tokens = max(max_batch_input_tokens, 1)
    tasks: list[tuple[str, int, list[dict[str, str]]]] = []

    for city_name, city_documents in sorted(documents_by_city.items()):
        batches: list[list[dict[str, str]]] = []
        current_batch: list[dict[str, str]] = []
        current_tokens = 0

        for document in city_documents:
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
    "load_markdown_documents",
    "split_documents_by_city",
]
