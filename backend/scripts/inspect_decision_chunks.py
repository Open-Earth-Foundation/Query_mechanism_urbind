"""
Brief: Inspect accepted and rejected markdown decision chunks for one run.

Inputs:
- CLI args:
  - `--run-dir`: Run artifact directory containing `markdown/` files (for example `output/20260306_1034`).
  - `--decision`: Which decision set to inspect: `accepted`, `rejected`, or `both` (default: `both`).
  - `--limit`: Optional maximum rows per decision set after filtering. Omit to dump all rows.
  - `--city`: Optional city filter (case-insensitive city key or display name).
  - `--show-content` / `--no-content`: Include or hide chunk content text (default: show content).
  - `--max-content-chars`: Maximum characters shown for content preview per chunk (default: `800`).
  - `--output-file`: Optional report file path. Defaults to `<run-dir>/markdown/decision_chunks_report.md`.
  - `--stdout`: Also print the report text to stdout.
  - `--config`: Path to `llm_config.yaml` used to resolve markdown chunks (default: `llm_config.yaml`).
- Files/paths:
  - `<run-dir>/markdown/accepted_excerpts.json`
  - `<run-dir>/markdown/rejected_excerpts.json`
  - `<run-dir>/markdown/batches.json`
  - Source markdown files referenced by those batch artifacts.
- Env vars:
  - Optional `.env` values consumed by `load_config` (for example `MARKDOWN_DIR`).

Outputs:
- Writes a compact human-readable report file (`.md`) with per-chunk metadata and optional chunk content.
- Optionally prints the same report to stdout (`--stdout`).

Usage (from project root):
- python -m backend.scripts.inspect_decision_chunks --run-dir output/20260306_1034 --decision both
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from backend.api.services.source_chunks import load_source_chunks
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import AppConfig, load_config
from backend.utils.json_io import read_json_object
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)

DecisionKind = Literal["accepted", "rejected"]


class ChunkContentStore(Protocol):
    """Minimal content lookup interface used by row collection."""

    def get(self, ids: list[str], limit: int) -> dict[str, object]:
        """Return metadata and documents for the requested chunk ids."""


@dataclass(frozen=True)
class MarkdownChunkStore:
    """Resolve persisted chunk ids back into markdown content."""

    run_dir: Path
    markdown_dir: Path
    config: AppConfig

    def get(self, ids: list[str], limit: int) -> dict[str, object]:
        """Return markdown chunk content in the legacy metadata/documents shape."""
        requested_ids = list(ids[: max(limit, 0)]) if limit else list(ids)
        if not requested_ids:
            return {"metadatas": [], "documents": []}
        try:
            chunks = load_source_chunks(
                run_dir=self.run_dir,
                markdown_dir=self.markdown_dir,
                config=self.config,
                chunk_ids=requested_ids,
            )
        except FileNotFoundError:
            return {"metadatas": [], "documents": []}

        return {
            "metadatas": [
                {
                    "raw_text": chunk.content,
                    "city_name": chunk.city_name,
                    "source_path": chunk.source_path,
                }
                for chunk in chunks
            ],
            "documents": [chunk.content for chunk in chunks],
        }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect accepted and rejected markdown chunk decisions for a run."
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
    payload = read_json_object(path, logger=logger, error_prefix="Failed to read JSON")
    if payload is None:
        raise FileNotFoundError(f"File not found or invalid JSON: {path.as_posix()}")
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
    """Build a chunk-id keyed lookup map from batch or legacy chunk payloads."""
    chunks: list[dict[str, Any]] = []
    raw_chunks = retrieval_payload.get("chunks")
    if isinstance(raw_chunks, list):
        chunks.extend(item for item in raw_chunks if isinstance(item, dict))

    raw_batches = retrieval_payload.get("batches")
    if isinstance(raw_batches, list):
        for batch in raw_batches:
            if not isinstance(batch, dict):
                continue
            batch_chunks = batch.get("chunks")
            if not isinstance(batch_chunks, list):
                continue
            chunks.extend(item for item in batch_chunks if isinstance(item, dict))

    index: dict[str, dict[str, Any]] = {}
    for item in chunks:
        chunk_id = str(item.get("chunk_id", "")).strip()
        if chunk_id:
            index[chunk_id] = item
    return index


def _passes_city_filter(item: dict[str, Any], city_filter: str | None) -> bool:
    """Return true when an item matches the optional city filter."""
    if not city_filter:
        return True
    requested_key = normalize_city_key(city_filter)
    item_city_key = normalize_city_key(str(item.get("city_key", "")).strip())
    if item_city_key:
        return item_city_key == requested_key
    return normalize_city_key(str(item.get("city_name", "")).strip()) == requested_key


def _truncate(value: str, max_chars: int) -> str:
    """Trim long text content for compact report output."""
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return f"{value[:max_chars].rstrip()} ... [truncated]"


def _default_output_path(run_dir: Path) -> Path:
    """Return the default report path for a run directory."""
    return run_dir / "markdown" / "decision_chunks_report.md"


def _chunk_content(store: ChunkContentStore, chunk_id: str) -> str:
    """Fetch chunk raw text from the content store."""
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
    store: ChunkContentStore,
    city_filter: str | None,
    show_content: bool,
    max_content_chars: int,
    limit: int | None,
) -> tuple[list[dict[str, str]], int]:
    """Collect decision rows with metadata and optional content."""
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
                "chunk_id": chunk_id,
                "chunk_index": str(retrieval_item.get("chunk_index", "")),
                "block_type": str(retrieval_item.get("block_type", "")),
                "source_path": str(
                    retrieval_item.get("source_path", retrieval_item.get("path", ""))
                ),
                "heading_path": str(retrieval_item.get("heading_path", "")),
                "content": content,
            }
        )
    return rows, missing_in_retrieval


def _render_section(
    decision: DecisionKind,
    rows: list[dict[str, str]],
    total_ids: int,
    missing_in_index: int,
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
                f"chunk_index={row['chunk_index']} block={row['block_type']} id={row['chunk_id']}"
            )
        )
        lines.append(f"  - source: {row['source_path']}")
        lines.append(f"  - heading: {row['heading_path']}")
        if include_content:
            lines.append(
                "  - content: "
                + (row["content"] if row["content"] else "<chunk content not found>")
            )
    lines.append("")
    lines.append(
        f"- Summary: total_ids={total_ids}, shown={len(rows)}, missing_in_index={missing_in_index}"
    )
    lines.append("")
    return lines


def main() -> None:
    """Script entry point."""
    args = parse_args()

    run_dir = Path(args.run_dir)
    markdown_dir = run_dir / "markdown"
    batches_path = markdown_dir / "batches.json"
    accepted_path = markdown_dir / "accepted_excerpts.json"
    rejected_path = markdown_dir / "rejected_excerpts.json"

    batches_payload = _read_json(batches_path)
    accepted_payload = _read_json(accepted_path)
    rejected_payload = _read_json(rejected_path)
    retrieval_index = _normalize_retrieval_index(batches_payload)

    config = load_config(Path(args.config))
    store = MarkdownChunkStore(
        run_dir=run_dir,
        markdown_dir=config.markdown_dir,
        config=config,
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
                missing_in_index=accepted_missing,
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
                missing_in_index=rejected_missing,
                include_content=args.show_content,
            )
        )

    output_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    logger.info("Wrote decision chunk report: %s", output_path.as_posix())
    if args.stdout:
        print("\n".join(report_lines))


if __name__ == "__main__":
    setup_logger()
    main()
