"""
Brief: Incrementally update Chroma markdown vector index from manifest state.

Inputs:
- CLI args:
  - --docs-dir: Directory containing markdown files to index (default: documents).
  - --persist-path: Chroma persistence directory override.
  - --collection: Chroma collection name override.
  - --city: Optional city stem filter; repeatable.
  - --dry-run: Parse/chunk only, do not embed/store or write manifest.
  - --config: Path to llm_config.yaml (default: llm_config.yaml).
- Env vars:
  - OPENAI_API_KEY or OPENROUTER_API_KEY: key for embeddings when not in dry-run mode.

Outputs:
- Upserts changed/new files and deletes removed-file chunks in Chroma.
- Updates manifest file (unless --dry-run).
- Log output with changed/unchanged/deleted file stats.
- Non-zero exit on embedding failures; no delete/upsert/manifest write is committed.

Usage (from project root):
- python -m backend.scripts.update_markdown_index --docs-dir documents
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.modules.vector_store.indexer import update_markdown_index
from backend.utils.config import load_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Update markdown vector index.")
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
        help="Parse and chunk only; skip embedding/persistence.",
    )
    parser.add_argument("--config", default="llm_config.yaml", help="Path to llm config.")
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    config = load_config(Path(args.config))
    if args.persist_path:
        manifest_default = Path(".chroma/index_manifest.json")
        config.vector_store.chroma_persist_path = Path(args.persist_path)
        if config.vector_store.index_manifest_path == manifest_default:
            config.vector_store.index_manifest_path = (
                config.vector_store.chroma_persist_path / "index_manifest.json"
            )
    if args.collection:
        config.vector_store.chroma_collection_name = args.collection
    stats = update_markdown_index(
        config=config,
        docs_dir=Path(args.docs_dir),
        selected_cities=args.city,
        dry_run=args.dry_run,
    )
    logger.info(
        "Update complete indexed=%d changed=%d unchanged=%d deleted=%d chunks=%d dry_run=%s",
        stats.files_indexed,
        stats.files_changed,
        stats.files_unchanged,
        stats.files_deleted,
        stats.chunks_created,
        stats.dry_run,
    )


if __name__ == "__main__":
    main()
