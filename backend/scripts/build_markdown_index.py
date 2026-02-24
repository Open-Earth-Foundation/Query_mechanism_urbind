"""
Brief: Build a full Chroma markdown vector index from scratch.

Inputs:
- CLI args:
  - --docs-dir: Directory containing markdown files to index (default: documents).
  - --persist-path: Chroma persistence directory override.
  - --collection: Chroma collection name override.
  - --city: Optional city stem filter; repeatable.
  - --dry-run: Parse/chunk only, do not embed or persist to Chroma/manifest.
  - --write-chunks-json: Optional path to write raw chunk payloads as JSON (works with or without --dry-run).
  - --config: Path to llm_config.yaml (default: llm_config.yaml).
- Env vars:
  - OPENAI_API_KEY or OPENROUTER_API_KEY: key for embeddings when not in dry-run mode.
  - ANONYMIZED_TELEMETRY (optional, default false): disables Chroma telemetry when set to false.
  - EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_RETRIES, EMBEDDING_RETRY_BASE_SECONDS,
    EMBEDDING_RETRY_MAX_SECONDS (optional): embedding request batching and retry behavior.
  - CHROMA_PERSIST_PATH, INDEX_MANIFEST_PATH (optional): in Kubernetes the build-vector-index Job
    uses the backend ConfigMap; typically /data/chroma and /data/chroma/index_manifest.json.

Outputs:
- Chroma collection persisted to disk (unless --dry-run).
- Manifest file at vector store manifest path (unless --dry-run).
- Log output with file/chunk/token statistics.

Usage (from project root):
- python -m backend.scripts.build_markdown_index --docs-dir documents
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.modules.vector_store.indexer import build_markdown_index
from backend.utils.config import load_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Build markdown vector index.")
    parser.add_argument("--docs-dir", default="documents", help="Markdown docs directory.")
    parser.add_argument(
        "--persist-path",
        help="Override vector store persistence path.",
    )
    parser.add_argument(
        "--collection",
        help="Override Chroma collection name.",
    )
    parser.add_argument(
        "--city",
        action="append",
        help="Optional city stem filter (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; skip embedding/Chroma/manifest writes.",
    )
    parser.add_argument(
        "--write-chunks-json",
        help=(
            "Optional path to write raw chunk payloads as JSON for inspection. "
            "Works with or without --dry-run."
        ),
    )
    parser.add_argument("--config", default="llm_config.yaml", help="Path to llm config.")
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    config = load_config(Path(args.config))
    if args.persist_path:
        config.vector_store.chroma_persist_path = Path(args.persist_path)
    if args.collection:
        config.vector_store.chroma_collection_name = args.collection
    chunks_dump_path = Path(args.write_chunks_json) if args.write_chunks_json else None
    stats = build_markdown_index(
        config=config,
        docs_dir=Path(args.docs_dir),
        selected_cities=args.city,
        dry_run=args.dry_run,
        chunks_dump_path=chunks_dump_path,
    )
    logger.info(
        "Build complete files=%d chunks=%d table_chunks=%d token_min=%d token_avg=%.2f token_max=%d dry_run=%s",
        stats.files_indexed,
        stats.chunks_created,
        stats.table_chunks,
        stats.min_tokens,
        stats.avg_tokens,
        stats.max_tokens,
        stats.dry_run,
    )


if __name__ == "__main__":
    main()
