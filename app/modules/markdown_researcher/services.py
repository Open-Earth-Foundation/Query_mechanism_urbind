from __future__ import annotations

import json
import logging
from pathlib import Path

from app.utils.config import MarkdownResearcherConfig
from app.utils.tokenization import chunk_text, count_tokens, get_max_input_tokens

logger = logging.getLogger(__name__)


def _resolve_chunk_tokens(config: MarkdownResearcherConfig) -> int:
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


def _document_token_count(document: dict[str, str]) -> int:
    return count_tokens(json.dumps(document, ensure_ascii=True))


def split_documents_by_city(
    documents: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Group documents by city name."""
    by_city: dict[str, list[dict[str, str]]] = {}
    for doc in documents:
        city_name = doc.get("city_name", "unknown")
        if city_name not in by_city:
            by_city[city_name] = []
        by_city[city_name].append(doc)
    return by_city


def split_documents_by_token_budget(
    documents: list[dict[str, str]],
    max_input_tokens: int | None,
) -> list[list[dict[str, str]]]:
    if not documents:
        return [[]]
    if max_input_tokens is None or max_input_tokens <= 0:
        return [documents]

    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_tokens = 0

    for document in documents:
        doc_tokens = _document_token_count(document)
        if current and current_tokens + doc_tokens > max_input_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        if doc_tokens > max_input_tokens:
            logger.warning("Skipping markdown chunk that exceeds token budget.")
            continue
        current.append(document)
        current_tokens += doc_tokens

    if current:
        batches.append(current)

    return batches


def load_markdown_documents(
    markdown_dir: Path,
    config: MarkdownResearcherConfig,
) -> list[dict[str, str]]:
    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {markdown_dir}")

    docs: list[dict[str, str]] = []
    files = sorted(markdown_dir.rglob("*.md"))
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
        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            entry = {
                "path": str(path),
                "city_name": city_name,
                "content": chunk,
                "chunk_index": idx,
                "chunk_count": total_chunks,
            }
            docs.append(entry)

    return docs


__all__ = [
    "load_markdown_documents",
    "split_documents_by_token_budget",
    "split_documents_by_city",
]
