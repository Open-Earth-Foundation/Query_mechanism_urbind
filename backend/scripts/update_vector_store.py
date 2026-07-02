"""
Brief: Refresh the markdown vector store and write shared update status.

Inputs:
- CLI args:
  - --trigger: Reason for this update (`startup`, `run`, or `manual`).
  - --docs-dir: Directory containing markdown files to index (default: documents).
  - --config: Path to llm_config.yaml (default: llm_config.yaml).
- Env vars:
  - OPENAI_API_KEY or OPENROUTER_API_KEY: key for embeddings.
  - CHROMA_PERSIST_PATH (optional): override Chroma persistence root.
  - VECTOR_STORE_UPDATE_MODE (optional): recorded in update_status.json.

Outputs:
- Updates the Chroma collection and index manifest under the configured vector store path.
- Writes update_status.json next to the vector index with running/completed/failed state.
- Logs compact update statistics to stdout/stderr.

Usage (from project root):
- python -m backend.scripts.update_vector_store --trigger startup --docs-dir documents
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.modules.vector_store.indexer import update_markdown_index
from backend.modules.vector_store.update_lock import VectorStoreUpdateLockError
from backend.modules.vector_store.update_status import (
    get_update_status_path,
    now_iso,
    write_update_status,
)
from backend.utils.config import load_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Refresh markdown vector store.")
    parser.add_argument(
        "--trigger",
        choices=["startup", "run", "manual"],
        default="manual",
        help="Reason this vector-store update was started.",
    )
    parser.add_argument("--docs-dir", default="documents", help="Markdown docs directory.")
    parser.add_argument("--config", default="llm_config.yaml", help="Path to llm config.")
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    config = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    status_path = get_update_status_path(config)
    started_at = now_iso()
    write_update_status(
        status_path,
        status="running",
        trigger=args.trigger,
        update_mode=config.vector_store.update_mode,
        message="Vector store update job is running.",
        started_at=started_at,
    )
    try:
        stats = update_markdown_index(
            config=config,
            docs_dir=docs_dir,
            selected_cities=None,
            dry_run=False,
        )
    except VectorStoreUpdateLockError as exc:
        write_update_status(
            status_path,
            status="running",
            trigger=args.trigger,
            update_mode=config.vector_store.update_mode,
            message=str(exc),
            started_at=started_at,
        )
        logger.warning("Vector store update skipped because another process holds the lock: %s", exc)
        raise
    except Exception as exc:
        logger.exception("Vector store update failed")
        write_update_status(
            status_path,
            status="failed",
            trigger=args.trigger,
            update_mode=config.vector_store.update_mode,
            message="Vector store update failed.",
            started_at=started_at,
            completed_at=now_iso(),
            error=str(exc),
        )
        raise

    stats_payload = {
        "files_indexed": stats.files_indexed,
        "files_changed": stats.files_changed,
        "files_unchanged": stats.files_unchanged,
        "files_deleted": stats.files_deleted,
        "chunks_created": stats.chunks_created,
        "table_chunks": stats.table_chunks,
        "dry_run": stats.dry_run,
        "update_mode": stats.update_mode,
        "changed_files": stats.changed_files,
        "deleted_files": stats.deleted_files,
    }
    write_update_status(
        status_path,
        status="completed",
        trigger=args.trigger,
        update_mode=config.vector_store.update_mode,
        message="Vector store is up to date.",
        started_at=started_at,
        completed_at=now_iso(),
        stats=stats_payload,
    )
    logger.info(
        "Vector store update complete changed=%d unchanged=%d deleted=%d chunks=%d",
        stats.files_changed,
        stats.files_unchanged,
        stats.files_deleted,
        stats.chunks_created,
    )


if __name__ == "__main__":
    main()
