"""
Brief: Inspect chunk content from retrieval.json outputs.

Inputs:
- CLI args:
  - --retrieval-json: Path to retrieval.json file (required).
  - --chunk-id: Show one specific chunk by chunk_id.
  - --min-distance: Filter chunks by minimum distance.
  - --max-distance: Filter chunks by maximum distance.
  - --city: Filter by normalized city key/name (case-insensitive).
  - --limit: Maximum chunks to show (default: 10).
  - --show-content: Include full raw_text content (default: true).
  - --config: Path to llm_config.yaml (default: llm_config.yaml).

Outputs:
- Printed chunk details including metadata, distance, and optionally raw_text content.

Usage (from project root):
- python -m backend.scripts.inspect_retrieval_chunks --retrieval-json output/20260220_1111/markdown/retrieval.json --chunk-id chunk_2808f03666bfdb9772abd5de
- python -m backend.scripts.inspect_retrieval_chunks --retrieval-json output/20260220_1111/markdown/retrieval.json --max-distance 0.80 --limit 5
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
    parser = argparse.ArgumentParser(description="Inspect chunks from retrieval.json.")
    parser.add_argument("--retrieval-json", required=True, help="Path to retrieval.json file.")
    parser.add_argument("--chunk-id", help="Show one specific chunk by chunk_id.")
    parser.add_argument("--min-distance", type=float, help="Filter by minimum distance.")
    parser.add_argument("--max-distance", type=float, help="Filter by maximum distance.")
    parser.add_argument("--city", help="Filter by city key/name (case-insensitive).")
    parser.add_argument("--limit", type=int, default=10, help="Maximum chunks to show.")
    parser.add_argument(
        "--show-content", action="store_true", default=True, help="Include raw_text content."
    )
    parser.add_argument("--no-content", dest="show_content", action="store_false", help="Hide content.")
    parser.add_argument("--config", default="llm_config.yaml", help="Path to llm config.")
    return parser.parse_args()


def load_retrieval_json(path: Path) -> dict[str, Any]:
    """Load and parse retrieval.json file."""
    if not path.exists():
        raise FileNotFoundError(f"retrieval.json not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fetch_chunk_content(
    store: ChromaStore, chunk_id: str
) -> dict[str, Any] | None:
    """Fetch full chunk content from ChromaDB."""
    payload = store.get(ids=[chunk_id], limit=1)
    ids = payload.get("ids", [])
    metadatas = payload.get("metadatas", [])
    documents = payload.get("documents", [])
    if not ids or not metadatas:
        return None
    return {
        "id": ids[0],
        "metadata": metadatas[0] if metadatas else {},
        "document": documents[0] if documents else "",
    }


def format_chunk_display(
    chunk_data: dict[str, Any],
    retrieval_chunk: dict[str, Any],
    show_content: bool = True,
) -> str:
    """Format chunk information for display."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"Chunk ID: {retrieval_chunk['chunk_id']}")
    lines.append(f"City: {retrieval_chunk['city_name']}")
    if retrieval_chunk.get("city_key"):
        lines.append(f"City Key: {retrieval_chunk['city_key']}")
    lines.append(f"Source: {retrieval_chunk['source_path']}")
    lines.append(f"Heading: {retrieval_chunk['heading_path']}")
    lines.append(f"Block Type: {retrieval_chunk['block_type']}")
    lines.append(f"Distance: {retrieval_chunk['distance']:.6f}")
    
    metadata = chunk_data.get("metadata", {})
    if metadata:
        lines.append(f"Chunk Index: {metadata.get('chunk_index', 'N/A')}")
        lines.append(f"Token Count: {metadata.get('token_count', 'N/A')}")
        if show_content:
            raw_text = metadata.get("raw_text", "")
            if raw_text:
                lines.append("-" * 80)
                lines.append("Content:")
                lines.append("-" * 80)
                lines.append(raw_text)
                lines.append("-" * 80)
    
    return "\n".join(lines)


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    config = load_config(Path(args.config))
    
    retrieval_path = Path(args.retrieval_json)
    retrieval_data = load_retrieval_json(retrieval_path)
    
    store = ChromaStore(
        persist_path=config.vector_store.chroma_persist_path,
        collection_name=config.vector_store.chroma_collection_name,
    )
    
    chunks = retrieval_data.get("chunks", [])
    if not chunks:
        logger.warning("No chunks found in retrieval.json")
        return
    
    # Filter chunks
    filtered_chunks = chunks
    if args.chunk_id:
        filtered_chunks = [c for c in chunks if c.get("chunk_id") == args.chunk_id]
        if not filtered_chunks:
            logger.warning("Chunk not found: %s", args.chunk_id)
            return
    if args.city:
        requested_city_key = normalize_city_key(args.city)
        filtered_chunks = [
            c
            for c in filtered_chunks
            if normalize_city_key(str(c.get("city_key", "")).strip()) == requested_city_key
        ]
    if args.min_distance is not None:
        filtered_chunks = [
            c
            for c in filtered_chunks
            if c.get("distance", float("inf")) >= args.min_distance
        ]
    if args.max_distance is not None:
        filtered_chunks = [
            c
            for c in filtered_chunks
            if c.get("distance", float("inf")) <= args.max_distance
        ]
    
    # Sort by distance ascending (closest first)
    filtered_chunks.sort(key=lambda c: c.get("distance", float("inf")))
    
    # Limit results
    filtered_chunks = filtered_chunks[: args.limit]
    
    logger.info("Showing %d chunks (from %d total)", len(filtered_chunks), len(chunks))
    logger.info("Queries used: %s", ", ".join(retrieval_data.get("queries", [])))
    logger.info("")
    
    for retrieval_chunk in filtered_chunks:
        chunk_id = retrieval_chunk["chunk_id"]
        chunk_data = fetch_chunk_content(store, chunk_id)
        if not chunk_data:
            logger.warning("Could not fetch chunk content for: %s", chunk_id)
            continue
        
        output = format_chunk_display(chunk_data, retrieval_chunk, args.show_content)
        logger.info(output)
        logger.info("")


if __name__ == "__main__":
    main()
