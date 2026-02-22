"""
Brief: Inspect chunks stored in the Chroma markdown index.

Inputs:
- CLI args:
  - --persist-path: Chroma persistence directory override.
  - --collection: Chroma collection name override.
  - --city: Filter by city_name.
  - --where: Simple metadata filter key=value.
  - --contains: Client-side text filter against raw_text.
  - --show-id: Show one specific chunk by chunk id.
  - --limit: Maximum records to inspect (default: 20).
  - --config: Path to llm_config.yaml (default: llm_config.yaml).

Outputs:
- Printed chunk summary lines.
- For --show-id, printed full metadata and raw text.

Usage (from project root):
- python -m backend.scripts.inspect_markdown_index --city Aarhus --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from backend.modules.vector_store.chroma_store import ChromaStore
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import load_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Inspect markdown vector index.")
    parser.add_argument("--persist-path", help="Override vector store persistence path.")
    parser.add_argument("--collection", help="Override Chroma collection name.")
    parser.add_argument("--city", help="Filter by city key/name (case-insensitive).")
    parser.add_argument("--where", help="Filter by metadata key=value.")
    parser.add_argument("--contains", help="Client-side raw_text contains filter.")
    parser.add_argument("--show-id", help="Show one specific chunk by id.")
    parser.add_argument("--limit", type=int, default=20, help="Max records to return.")
    parser.add_argument("--config", default="llm_config.yaml", help="Path to llm config.")
    return parser.parse_args()


def _parse_where(where_arg: str | None) -> dict[str, Any] | None:
    if not where_arg:
        return None
    if "=" not in where_arg:
        raise ValueError("--where must be key=value")
    key, value = where_arg.split("=", 1)
    return {key.strip(): value.strip()}


def _iter_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ids = payload.get("ids", [])
    metadatas = payload.get("metadatas", [])
    documents = payload.get("documents", [])
    rows: list[dict[str, Any]] = []
    for chunk_id, metadata, document in zip(ids, metadatas, documents, strict=False):
        if not isinstance(metadata, dict):
            continue
        rows.append({"id": chunk_id, "metadata": metadata, "document": document})
    return rows


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    config = load_config(Path(args.config))
    persist_path = (
        Path(args.persist_path)
        if args.persist_path
        else config.vector_store.chroma_persist_path
    )
    collection = (
        args.collection if args.collection else config.vector_store.chroma_collection_name
    )
    store = ChromaStore(persist_path=persist_path, collection_name=collection)

    if args.show_id:
        payload = store.get(ids=[args.show_id], limit=1)
        rows = _iter_rows(payload)
        if not rows:
            logger.info("Chunk not found: %s", args.show_id)
            return
        row = rows[0]
        metadata = row["metadata"]
        raw_text = metadata.get("raw_text", "")
        logger.info("Chunk %s metadata:\n%s", row["id"], json.dumps(metadata, indent=2))
        logger.info("Chunk %s raw_text:\n%s", row["id"], raw_text)
        return

    where = _parse_where(args.where)
    if args.city:
        where = {"city_key": normalize_city_key(args.city)}
    payload = store.get(where=where, limit=max(args.limit, 1))
    rows = _iter_rows(payload)
    if args.contains:
        needle = args.contains.casefold()
        rows = [
            row
            for row in rows
            if needle in str(row["metadata"].get("raw_text", "")).casefold()
        ]
    rows = rows[: max(args.limit, 1)]
    for row in rows:
        metadata = row["metadata"]
        logger.info(
            "id=%s city=%s city_key=%s type=%s source=%s heading=%s tokens=%s",
            row["id"],
            metadata.get("city_name", ""),
            metadata.get("city_key", ""),
            metadata.get("block_type", ""),
            metadata.get("source_path", ""),
            metadata.get("heading_path", ""),
            metadata.get("token_count", ""),
        )
    logger.info("Total rows shown: %d", len(rows))


if __name__ == "__main__":
    main()
