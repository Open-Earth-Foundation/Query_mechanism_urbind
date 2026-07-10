"""
Brief: Build a full Chroma markdown vector index from scratch.

Inputs:
- CLI args:
  - --docs-dir: Optional markdown directory override. Defaults to the resolved
    `markdown_dir` from `llm_config.yaml` / `MARKDOWN_DIR`.
  - --persist-path: Chroma persistence directory override.
  - --collection: Chroma collection name override.
  - --city: Optional city stem filter. Dry runs inspect only those cities; persisted builds ignore the filter and rebuild the full shared index.
  - --dry-run: Parse/chunk only, do not embed or persist to Chroma/manifest.
  - --write-chunks-json: Optional path to write raw chunk payloads as JSON (works with or without --dry-run).
  - --config: Path to llm_config.yaml (default: llm_config.yaml).
- Env vars:
  - OPENAI_API_KEY or OPENROUTER_API_KEY: key for embeddings when not in dry-run mode.
  - VECTOR_STORE_EMBEDDING_BASE_URL (optional): OpenAI-compatible embedding endpoint override.
  - VECTOR_STORE_EMBEDDING_API_KEY_ENV (optional): env var name to use exclusively for embedding API auth.
  - ANONYMIZED_TELEMETRY (optional, default false): disables Chroma telemetry when set to false.
  - CHROMA_PERSIST_PATH (optional): override Chroma persistence root (manifest path follows this root when defaulted).
- Config file (`llm_config.yaml`):
  - `vector_store.*` controls embedding and retrieval/index settings (model, chunking,
    retries, distance metric, distance cutoffs, context windows, auto-update, manifest path).

Outputs:
- Chroma collection persisted to disk (unless --dry-run).
- Manifest file at vector store manifest path (unless --dry-run).
- Log output with file/chunk/token statistics.
- Non-zero exit on embedding failures; no collection reset or manifest write is committed.

Usage (from project root):
- python -m backend.scripts.build_markdown_index
- python -m backend.scripts.build_markdown_index --docs-dir documents
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.modules.vector_store.indexer import build_markdown_index
from backend.utils.config import load_config, resolve_path_relative_to_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Build markdown vector index.")
    parser.add_argument("--docs-dir", help="Markdown docs directory override.")
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
        help=(
            "Optional city stem filter. Dry runs inspect only those cities; "
            "persisted builds ignore the filter."
        ),
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
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.persist_path:
        manifest_default = config.vector_store.chroma_persist_path / "index_manifest.json"
        config.vector_store.chroma_persist_path = resolve_path_relative_to_config(
            config_path,
            Path(args.persist_path),
        )
        if config.vector_store.index_manifest_path == manifest_default:
            config.vector_store.index_manifest_path = (
                config.vector_store.chroma_persist_path / "index_manifest.json"
            )
    if args.collection:
        config.vector_store.chroma_collection_name = args.collection
    chunks_dump_path = Path(args.write_chunks_json) if args.write_chunks_json else None
    stats = build_markdown_index(
        config=config,
        docs_dir=(
            resolve_path_relative_to_config(config_path, Path(args.docs_dir))
            if args.docs_dir
            else config.markdown_dir
        ),
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
