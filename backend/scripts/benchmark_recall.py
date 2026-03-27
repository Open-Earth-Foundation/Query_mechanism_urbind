"""
Brief: Run the rigorous recall/precision benchmark against a gold-standard dataset.

Inputs:
- CLI args:
  - `--gold-file`: Path to the versioned gold benchmark JSON file.
  - `--benchmark-id`: Optional benchmark id. Defaults to a UTC timestamp when omitted.
  - `--output-dir`: Root directory for benchmark artifacts (default: `output/benchmarks/recall`).
  - `--config`: Path to `llm_config.yaml` used for live pipeline runs and fact judging.
  - `--case-id`: Optional repeatable gold case filter.
  - `--log-llm-payload`: Enable or disable full LLM payload logs for live runs and fact judging.
- Files/paths:
  - The gold file must match the schema `{"version": 1, "cases": [...]}` and contain
    `case_id`, `question`, `gold_chunk_ids`, `gold_facts`, `gold_city`, and optional
    `selected_cities`.
- Env vars:
  - `OPENROUTER_API_KEY` is required because Stage B and Stage C fact recall are judged with an LLM.

Outputs:
- `output/benchmarks/recall/<benchmark_id>/benchmark_report.json`
- `output/benchmarks/recall/<benchmark_id>/benchmark_report.md`
- Optional live pipeline artifacts under `output/benchmarks/recall/<benchmark_id>/runs/`

Usage (from project root):
- python -m backend.scripts.benchmark_recall --gold-file tests/fixtures/benchmark_gold.json
- python -m backend.scripts.benchmark_recall --gold-file tests/fixtures/benchmark_gold.json --case-id sample_case
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmarks.gold_recall.runner import run_recall_benchmark
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Run the rigorous recall benchmark against a gold dataset."
    )
    parser.add_argument("--gold-file", required=True, help="Path to gold benchmark JSON.")
    parser.add_argument(
        "--benchmark-id",
        help="Optional benchmark id. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/benchmarks/recall",
        help="Root directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--config",
        default="llm_config.yaml",
        help="Path to llm_config.yaml.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Optional repeatable gold case filter.",
    )
    parser.add_argument(
        "--log-llm-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable full LLM payload logging.",
    )
    return parser.parse_args()


def _default_benchmark_id() -> str:
    """Build a default benchmark id from the current UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    benchmark_id = args.benchmark_id if args.benchmark_id else _default_benchmark_id()
    report = run_recall_benchmark(
        benchmark_id=benchmark_id,
        gold_file=Path(args.gold_file),
        output_dir=Path(args.output_dir),
        config_path=Path(args.config),
        selected_case_ids=args.case_id or [],
        log_llm_payload=args.log_llm_payload,
    )
    logger.info("Recall benchmark completed: %s", report.benchmark_id)
    logger.info("Artifacts: %s", report.output_dir)


if __name__ == "__main__":
    setup_logger()
    main()
