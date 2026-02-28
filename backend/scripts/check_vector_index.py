"""
Brief: Inspect local Chroma vector index manifest and database metadata.

Inputs:
- CLI args:
  - --manifest-path: Path to index manifest JSON (default: .chroma/index_manifest.json).
  - --chroma-db-path: Path to Chroma sqlite database file (default: .chroma/chroma.sqlite3).
  - --show-files / --no-show-files: Include or skip per-file chunk counts from the manifest.
- Files/paths:
  - Manifest JSON is expected to include a top-level `files` object keyed by source filename.
- Env vars:
  - LOG_LEVEL (optional): Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

Outputs:
- Log lines with manifest timestamp, indexed file/chunk counts, build metadata, and optional per-file chunk counts.
- Non-zero exit when the manifest is missing, unreadable, or invalid JSON.

Usage (from project root):
- python -m backend.scripts.check_vector_index
- python -m backend.scripts.check_vector_index --manifest-path .chroma/index_manifest.json --no-show-files
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Inspect vector index manifest metadata.")
    parser.add_argument(
        "--manifest-path",
        default=".chroma/index_manifest.json",
        help="Path to index manifest JSON.",
    )
    parser.add_argument(
        "--chroma-db-path",
        default=".chroma/chroma.sqlite3",
        help="Path to Chroma sqlite database.",
    )
    parser.add_argument(
        "--show-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include per-file chunk counts in output.",
    )
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    """Load manifest JSON from disk."""
    if not manifest_path.exists():
        logger.error("Manifest file not found path=%s", manifest_path)
        raise SystemExit(1)
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError:
        logger.exception("Failed reading manifest path=%s", manifest_path)
        raise SystemExit(1) from None
    except json.JSONDecodeError:
        logger.exception("Manifest is not valid JSON path=%s", manifest_path)
        raise SystemExit(1) from None


def _chunk_count(file_payload: object) -> int:
    """Return chunk count from a single manifest file payload."""
    if not isinstance(file_payload, dict):
        return 0
    chunk_ids = file_payload.get("chunk_ids", [])
    if not isinstance(chunk_ids, list):
        return 0
    return len(chunk_ids)


def main() -> None:
    """Script entry point."""
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    manifest = _load_manifest(manifest_path)
    files_payload = manifest.get("files", {})
    if not isinstance(files_payload, dict):
        logger.error("Manifest `files` entry must be an object path=%s", manifest_path)
        raise SystemExit(1)

    files: dict[str, object] = {
        str(file_name): payload for file_name, payload in files_payload.items()
    }
    manifest_mtime = datetime.fromtimestamp(manifest_path.stat().st_mtime)
    total_chunks = sum(_chunk_count(payload) for payload in files.values())

    logger.info("Manifest last modified: %s", manifest_mtime)
    logger.info("Files indexed: %d", len(files))
    logger.info("Total chunks: %d", total_chunks)
    logger.info("Build timestamp: %s", manifest.get("build_timestamp", "N/A"))
    logger.info("Index version: %s", manifest.get("index_version", "N/A"))

    if args.show_files:
        logger.info("Files in index:")
        for filename in sorted(files):
            logger.info("  %s: %d chunks", filename, _chunk_count(files[filename]))

    chroma_db_path = Path(args.chroma_db_path)
    if not chroma_db_path.exists():
        logger.info("Chroma DB not found path=%s", chroma_db_path)
        return

    db_stat = chroma_db_path.stat()
    db_mtime = datetime.fromtimestamp(db_stat.st_mtime)
    db_size_mb = db_stat.st_size / (1024 * 1024)
    logger.info("Chroma DB modified: %s", db_mtime)
    logger.info("Chroma DB size: %.2f MB", db_size_mb)


if __name__ == "__main__":
    setup_logger()
    main()
