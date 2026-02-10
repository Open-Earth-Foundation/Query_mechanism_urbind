from __future__ import annotations

import logging
from pathlib import Path

from app.utils.config import MarkdownResearcherConfig
from app.utils.tokenization import chunk_text, get_max_input_tokens

logger = logging.getLogger(__name__)


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
    - Returns one entry per chunk with ``path``, ``city_name``, and ``content``.
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

    max_chunk_tokens = _resolve_chunk_tokens(config)
    for path in files:
        size = path.stat().st_size
        if size > config.max_file_bytes:
            logger.warning("Skipping large markdown file: %s", path)
            continue
        city_name = path.stem
        content = path.read_text(encoding="utf-8")
        chunks = chunk_text(content, max_chunk_tokens, config.chunk_overlap_tokens)
        for chunk in chunks:
            entry = {
                "path": str(path),
                "city_name": city_name,
                "content": chunk,
            }
            docs.append(entry)

    return docs


__all__ = [
    "load_markdown_documents",
    "split_documents_by_city",
]
