"""
Brief: Inspect accepted/rejected markdown decision chunk content for one run.

Inputs:
- CLI args:
  - --run-dir: Run artifact directory containing `markdown/` files (for example `output/20260306_1034`).
  - --decision: Which decision set to inspect: `accepted`, `rejected`, or `both` (default: `both`).
  - --limit: Optional maximum rows per decision set after filtering. Omit to dump all rows.
  - --city: Optional city filter (case-insensitive city key or name).
  - --show-content / --no-content: Include or hide chunk content text (default: show content).
  - --max-content-chars: Maximum characters shown for content preview per chunk (default: 800).
  - --output-file: Optional report file path. Defaults to `<run-dir>/markdown/decision_chunks_report.md`.
  - --stdout: Also print the report text to stdout.
  - --config: Path to llm config used to open Chroma store (default: `llm_config.yaml`).
- Files/paths:
  - `<run-dir>/markdown/accepted_excerpts.json`
  - `<run-dir>/markdown/rejected_excerpts.json`
  - `<run-dir>/markdown/retrieval.json`
  - Chroma collection configured in `llm_config.yaml` (`vector_store.*`).
- Env vars:
  - Optional `.env` values consumed by `load_config` (for example `CHROMA_PERSIST_PATH`).

Outputs:
- Writes a compact, human-readable report file (`.md`) with retrieval metadata and optional chunk content.
- Optionally prints the same report to stdout (`--stdout`).

Usage (from project root):
- python -m backend.scripts.inspect_decision_chunks --run-dir output/20260306_1034 --decision both
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Literal

from backend.modules.vector_store.chroma_store import ChromaStore
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import load_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)

DecisionKind = Literal["accepted", "rejected"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect accepted/rejected chunk decisions for a run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory (example: output/20260306_1034).",
    )
    parser.add_argument(
        "--decision",
        choices=["accepted", "rejected", "both"],
        default="both",
        help="Decision set to inspect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum rows per decision set after filtering.",
    )
    parser.add_argument(
        "--city",
        help="Optional city filter (case-insensitive key or display name).",
    )
    parser.add_argument(
        "--show-content",
        action="store_true",
        default=True,
        help="Include chunk content preview.",
    )
    parser.add_argument(
        "--no-content",
        dest="show_content",
        action="store_false",
        help="Hide chunk content preview.",
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=800,
        help="Maximum content preview characters per chunk.",
    )
    parser.add_argument(
        "--output-file",
        help=(
            "Optional output report file path. "
            "Defaults to <run-dir>/markdown/decision_chunks_report.md."
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the generated report to stdout.",
    )
    parser.add_argument(
        "--config",
        default="llm_config.yaml",
        help="Path to llm config.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    """Load JSON file and validate top-level object shape."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.as_posix()}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path.as_posix()}")
    return payload


def _decision_ids(payload: dict[str, Any], decision: DecisionKind) -> list[str]:
    """Extract ordered decision ids from accepted/rejected artifact payload."""
    key = "accepted_chunk_ids" if decision == "accepted" else "rejected_chunk_ids"
    raw = payload.get(key, [])
    if not isinstance(raw, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in raw:
        chunk_id = str(value).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        output.append(chunk_id)
    return output


def _normalize_retrieval_index(
    retrieval_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build chunk-id keyed retrieval metadata lookup map."""
    chunks = retrieval_payload.get("chunks", [])
    if not isinstance(chunks, list):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for item in chunks:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        if chunk_id:
            index[chunk_id] = item
    return index


def _passes_city_filter(item: dict[str, Any], city_filter: str | None) -> bool:
    """Return true when retrieval item matches optional city filter."""
    if not city_filter:
        return True
    requested_key = normalize_city_key(city_filter)
    item_city_key = normalize_city_key(str(item.get("city_key", "")).strip())
    if item_city_key:
        return item_city_key == requested_key
    return normalize_city_key(str(item.get("city_name", "")).strip()) == requested_key


def _truncate(value: str, max_chars: int) -> str:
    """Trim long text content for compact terminal output."""
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return f"{value[:max_chars].rstrip()} ... [truncated]"


def _default_output_path(run_dir: Path) -> Path:
    """Return default report path for a run directory."""
    return run_dir / "markdown" / "decision_chunks_report.md"


def _chunk_content(store: ChromaStore, chunk_id: str) -> str:
    """Fetch chunk raw text from Chroma metadata/document fields."""
    payload = store.get(ids=[chunk_id], limit=1)
    metadatas = payload.get("metadatas", [])
    documents = payload.get("documents", [])
    metadata = metadatas[0] if isinstance(metadatas, list) and metadatas else {}
    if isinstance(metadata, dict):
        raw_text = str(metadata.get("raw_text", "")).strip()
        if raw_text:
            return raw_text
    if isinstance(documents, list) and documents:
        return str(documents[0]).strip()
    return ""


def _collect_rows(
    decision: DecisionKind,
    chunk_ids: list[str],
    retrieval_index: dict[str, dict[str, Any]],
    store: ChromaStore,
    city_filter: str | None,
    show_content: bool,
    max_content_chars: int,
    limit: int | None,
) -> tuple[list[dict[str, str]], int]:
    """Collect decision rows with retrieval metadata and optional content."""
    rows: list[dict[str, str]] = []
    shown = 0
    missing_in_retrieval = 0
    for chunk_id in chunk_ids:
        retrieval_item = retrieval_index.get(chunk_id)
        if retrieval_item is None:
            missing_in_retrieval += 1
            continue
        if not _passes_city_filter(retrieval_item, city_filter):
            continue
        if limit is not None and shown >= max(limit, 0):
            break
        shown += 1
        content = ""
        if show_content:
            raw_content = _chunk_content(store, chunk_id)
            content = _truncate(raw_content, max_content_chars) if raw_content else ""
        rows.append(
            {
                "index": str(shown),
                "decision": decision,
                "city_name": str(retrieval_item.get("city_name", "")),
                "city_key": str(retrieval_item.get("city_key", "")),
                "distance": str(retrieval_item.get("distance", "")),
                "block_type": str(retrieval_item.get("block_type", "")),
                "chunk_id": chunk_id,
                "source_path": str(retrieval_item.get("source_path", "")),
                "heading_path": str(retrieval_item.get("heading_path", "")),
                "content": content,
            }
        )
    return rows, missing_in_retrieval


def _render_section(
    decision: DecisionKind,
    rows: list[dict[str, str]],
    total_ids: int,
    missing_in_retrieval: int,
    include_content: bool,
) -> list[str]:
    """Render one decision section as markdown lines."""
    lines = [f"## {decision.capitalize()} Chunks", ""]
    if not rows:
        lines.append("- No rows to show after filtering.")
        lines.append("")
    for row in rows:
        lines.append(
            (
                f"- [{row['index']}] city={row['city_name']} ({row['city_key']}) "
                f"distance={row['distance']} block={row['block_type']} id={row['chunk_id']}"
            )
        )
        lines.append(f"  - source: {row['source_path']}")
        lines.append(f"  - heading: {row['heading_path']}")
        if include_content:
            lines.append(
                "  - content: "
                + (row["content"] if row["content"] else "<not found in vector store>")
            )
    lines.append("")
    lines.append(
        f"- Summary: total_ids={total_ids}, shown={len(rows)}, missing_in_retrieval={missing_in_retrieval}"
    )
    lines.append("")
    return lines


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    run_dir = Path(args.run_dir)
    markdown_dir = run_dir / "markdown"
    retrieval_path = markdown_dir / "retrieval.json"
    accepted_path = markdown_dir / "accepted_excerpts.json"
    rejected_path = markdown_dir / "rejected_excerpts.json"

    retrieval_payload = _read_json(retrieval_path)
    accepted_payload = _read_json(accepted_path)
    rejected_payload = _read_json(rejected_path)
    retrieval_index = _normalize_retrieval_index(retrieval_payload)

    config = load_config(Path(args.config))
    store = ChromaStore(
        persist_path=config.vector_store.chroma_persist_path,
        collection_name=config.vector_store.chroma_collection_name,
    )
    output_path = (
        Path(args.output_file)
        if isinstance(args.output_file, str) and args.output_file.strip()
        else _default_output_path(run_dir)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Decision Chunk Inspection Report",
        "",
        f"- run_dir: {run_dir.as_posix()}",
        f"- decision: {args.decision}",
        f"- city_filter: {args.city or '<none>'}",
        f"- limit: {args.limit if args.limit is not None else 'all'}",
        f"- show_content: {args.show_content}",
        "",
    ]

    if args.decision in {"accepted", "both"}:
        accepted_ids = _decision_ids(accepted_payload, "accepted")
        accepted_rows, accepted_missing = _collect_rows(
            decision="accepted",
            chunk_ids=accepted_ids,
            retrieval_index=retrieval_index,
            store=store,
            city_filter=args.city,
            show_content=args.show_content,
            max_content_chars=args.max_content_chars,
            limit=args.limit,
        )
        report_lines.extend(
            _render_section(
                decision="accepted",
                rows=accepted_rows,
                total_ids=len(accepted_ids),
                missing_in_retrieval=accepted_missing,
                include_content=args.show_content,
            )
        )
    if args.decision in {"rejected", "both"}:
        rejected_ids = _decision_ids(rejected_payload, "rejected")
        rejected_rows, rejected_missing = _collect_rows(
            decision="rejected",
            chunk_ids=rejected_ids,
            retrieval_index=retrieval_index,
            store=store,
            city_filter=args.city,
            show_content=args.show_content,
            max_content_chars=args.max_content_chars,
            limit=args.limit,
        )
        report_lines.extend(
            _render_section(
                decision="rejected",
                rows=rejected_rows,
                total_ids=len(rejected_ids),
                missing_in_retrieval=rejected_missing,
                include_content=args.show_content,
            )
        )

    output_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    logger.info("Wrote decision chunk report: %s", output_path.as_posix())
    if args.stdout:
        print("\n".join(report_lines))


if __name__ == "__main__":
    main()
