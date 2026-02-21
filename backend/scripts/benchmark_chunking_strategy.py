"""
Brief: Run chunking strategy benchmark on a random markdown subset.

Inputs:
- CLI args:
  - --docs-dir: Markdown corpus root (default: documents).
  - --sample-size: Number of markdown files sampled (default: 20; 0 means all files).
  - --seed: Random seed for deterministic sampling (default: 42).
  - --chunk-tokens: Chunk token budget override (default: from config).
  - --overlap-tokens: Overlap token budget override (default: from config).
  - --table-row-group-max-rows: Table split row-group size override (default: from config).
  - --output-dir: Directory for benchmark artifacts (default: output/chunk_benchmarks).
  - --compare-to: Optional baseline benchmark JSON path.
  - --max-score-drop: Max allowed drop vs baseline before failure (default: 0.03).
  - --fail-on-regression: Exit 1 when score regression exceeds threshold.
  - --config: Path to llm_config.yaml (default: llm_config.yaml).

Outputs:
- Benchmark JSON with final score and individual metric values.
- Human-readable markdown report with metric breakdown.
- Optional non-zero exit code when regression threshold is violated.

Usage (from project root):
- python -m backend.scripts.benchmark_chunking_strategy --sample-size 25 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.modules.vector_store.benchmarking import run_chunking_benchmark
from backend.utils.config import load_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Benchmark markdown chunking strategy.")
    parser.add_argument("--docs-dir", default="documents", help="Markdown documents directory.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of markdown files sampled. Use 0 for all files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sample seed.")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        help="Override chunk token budget for benchmark run.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        help="Override chunk overlap token budget for benchmark run.",
    )
    parser.add_argument(
        "--table-row-group-max-rows",
        type=int,
        help="Override max rows per split table group.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/chunk_benchmarks",
        help="Directory where benchmark artifacts are stored.",
    )
    parser.add_argument(
        "--compare-to",
        help="Optional baseline benchmark JSON path for score comparison.",
    )
    parser.add_argument(
        "--max-score-drop",
        type=float,
        default=0.03,
        help="Allowed final score drop vs baseline before failing.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if regression exceeds threshold.",
    )
    parser.add_argument("--config", default="llm_config.yaml", help="Path to llm config.")
    return parser.parse_args()


def _timestamp_slug() -> str:
    """Create timestamp slug for benchmark artifact naming."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_markdown_report(path: Path, payload: dict) -> None:
    """Write markdown summary with final score and metric breakdown."""
    metrics = payload["metrics"]
    counts = payload["counts"]
    sample = payload["sample"]
    lines = [
        "# Chunking Benchmark Report",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Sample size: {sample['sample_size']} / {sample['total_docs_available']}",
        f"- Seed: {sample['seed']}",
        "",
        "## Sampled documents",
        "",
        *[f"- {doc}" for doc in sample.get("sampled_docs", [])],
        "",
        "## Final score",
        "",
        f"- Final accuracy (final_accuracy_score): {metrics['final_accuracy_score']:.4f}  ",
        "  Combined scalar score (0–1) computed from the individual metrics below.",
        "",
        "## Metric breakdown",
        "",
        f"- Caption linkage (caption_linkage_rate): {metrics['caption_linkage_rate']:.4f}  ",
        "  Extraction success: (# source tables with `Table ...` captions that ended up with a `table_title` on at least one table chunk) / (# captioned source tables).",
        f"- Table header validity (table_header_valid_rate): {metrics['table_header_valid_rate']:.4f}  ",
        "  Structural integrity: (# table chunks whose `raw_text` parses as a valid markdown header + separator row) / (# table chunks).",
        f"- Table detection recall (table_detection_rate): {metrics['table_detection_rate']:.4f}  ",
        "  Table recall proxy: (# table chunks produced by the packer) / (# source tables detected in markdown), capped at 1.0 (can exceed 1.0 before capping when large tables are split).",
        f"- Heading alignment (heading_alignment_rate): {metrics['heading_alignment_rate']:.4f}  ",
        "  Section alignment: (# chunks where `heading_path` equals the heading stack implied by the source at `start_line`) / (# chunks where a heading context exists).",
        f"- Token budget compliance (token_budget_compliance_rate): {metrics['token_budget_compliance_rate']:.4f}  ",
        "  Budget respect: (# chunks with `token_count` <= configured chunk token budget) / (# total chunks).",
        "",
        "## Counts",
        "",
        f"- Source tables (source_tables): {counts['source_tables']}  ",
        "  Count of markdown tables detected directly from the source text across sampled files.",
        f"- Captioned source tables (source_captioned_tables): {counts['source_captioned_tables']}  ",
        "  Count of source tables that have a preceding one-line `Table ...` caption.",
        f"- Table chunks (detected_table_chunks): {counts['detected_table_chunks']}  ",
        "  Count of chunks whose `block_type` is `table` after parsing and packing.",
        f"- Table chunks with caption (table_chunks_with_caption): {counts['table_chunks_with_caption']}  ",
        "  Count of table chunks that carry a non-empty `table_title` (caption attached).",
        f"- Table chunks with valid header (table_chunks_with_valid_header): {counts['table_chunks_with_valid_header']}  ",
        "  Count of table chunks whose `raw_text` parses to a valid markdown header + separator row.",
        f"- Total chunks (total_chunks): {counts['total_chunks']}  ",
        "  Count of all chunks (paragraph, list, table, code) produced by the packer across sampled files.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    config = load_config(Path(args.config))
    chunk_tokens = (
        args.chunk_tokens
        if args.chunk_tokens is not None
        else config.vector_store.embedding_chunk_tokens
    )
    overlap_tokens = (
        args.overlap_tokens
        if args.overlap_tokens is not None
        else config.vector_store.embedding_chunk_overlap_tokens
    )
    table_rows = (
        args.table_row_group_max_rows
        if args.table_row_group_max_rows is not None
        else config.vector_store.table_row_group_max_rows
    )

    result = run_chunking_benchmark(
        docs_dir=Path(args.docs_dir),
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        table_row_group_max_rows=table_rows,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample": {
            "seed": result.seed,
            "sample_size": result.sample_size,
            "total_docs_available": result.total_docs_available,
            "sampled_docs": result.sampled_docs,
        },
        "config": {
            "chunk_tokens": chunk_tokens,
            "overlap_tokens": overlap_tokens,
            "table_row_group_max_rows": table_rows,
        },
        "counts": result.counts,
        "metrics": result.metrics,
        "per_file": result.per_file,
    }

    output_dir = Path(args.output_dir) / _timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    report_path = output_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown_report(report_path, payload)

    logger.info("Benchmark complete final_score=%.4f", result.metrics["final_accuracy_score"])
    logger.info("Benchmark JSON: %s", json_path.as_posix())
    logger.info("Benchmark report: %s", report_path.as_posix())

    if args.compare_to:
        baseline_path = Path(args.compare_to)
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_score = float(baseline_payload["metrics"]["final_accuracy_score"])
        current_score = float(result.metrics["final_accuracy_score"])
        score_delta = current_score - baseline_score
        logger.info(
            "Baseline comparison baseline=%.4f current=%.4f delta=%.4f",
            baseline_score,
            current_score,
            score_delta,
        )
        if args.fail_on_regression and score_delta < -abs(args.max_score_drop):
            raise SystemExit(
                f"Benchmark regression detected. Score delta {score_delta:.4f} is below "
                f"allowed drop {-abs(args.max_score_drop):.4f}."
            )


if __name__ == "__main__":
    main()
