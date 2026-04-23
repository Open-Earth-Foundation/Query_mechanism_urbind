"""
Brief: Run the Krakow TEF mapping benchmark against curated CCC source truth.

Inputs:
- CLI args:
  - `--source-truth`: Mapped Krakow source-truth JSON file with an `initiatives` list;
    default `assets/tef_mapping/all_correct_initiatives_mapped_to_tef.json`.
  - `--tef-catalog-dir`: Local TEF catalog directory; default `tef_mapping`.
  - `--output-dir`: Benchmark output root; default
    `output/tef_benchmarks/krakow_tef_mapping`.
  - `--benchmark-id`: Optional benchmark run directory name; defaults to a UTC timestamp.
  - `--config`: Path to `llm_config.yaml`; default `llm_config.yaml`.
  - `--max-workers`: Optional worker override for initiative-level TEF mapper LLM calls.
  - `--limit`: Optional first-N initiative limit for smoke checks; omit for the full Krakow run.
  - `--log-llm-payload`: Enable logging of full LLM request/response payloads.
- Files/paths: source truth is converted to mapper-ready initiative JSONL before mapping.
- Env vars: `OPENROUTER_API_KEY` is required for live TEF mapping.

Outputs:
- `output-dir/<benchmark-id>/00_inputs/initiatives.jsonl`: mapper-ready source-truth input.
- `output-dir/<benchmark-id>/01_tef_mapping/`: standard TEF mapping artifacts.
- `output-dir/<benchmark-id>/02_comparison/tef_benchmark_issues.json`: P1-P3 issue JSON.
- `output-dir/<benchmark-id>/02_comparison/tef_benchmark_report.md`: human-readable report.
- `output-dir/<benchmark-id>/benchmark_summary.json`: benchmark paths and counts.

Usage (from project root):
- python -m backend.scripts.benchmark_krakow_tef_mapping --max-workers 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from backend.modules.tef_mapper.benchmark import (
    DEFAULT_KRAKOW_BENCHMARK_OUTPUT_ROOT,
    DEFAULT_KRAKOW_TEF_SOURCE_TRUTH,
    run_krakow_tef_benchmark,
)
from backend.utils.config import load_config, resolve_openrouter_api_key
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Run the Krakow TEF mapping benchmark against source truth."
    )
    parser.add_argument(
        "--source-truth",
        default=str(DEFAULT_KRAKOW_TEF_SOURCE_TRUTH),
        help="Mapped Krakow source-truth JSON file.",
    )
    parser.add_argument(
        "--tef-catalog-dir",
        default="tef_mapping",
        help="Local TEF catalog directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_KRAKOW_BENCHMARK_OUTPUT_ROOT),
        help="Root directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--benchmark-id",
        help="Optional benchmark run directory name.",
    )
    parser.add_argument(
        "--config",
        default="llm_config.yaml",
        help="Path to llm_config.yaml.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Override initiative-level TEF mapper worker count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to the first N source-truth initiatives for smoke checks.",
    )
    parser.add_argument(
        "--log-llm-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="log_llm_payload",
        help="Enable or disable full LLM payload logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    load_dotenv()

    result = run_krakow_tef_benchmark(
        config=load_config(Path(args.config)),
        api_key=resolve_openrouter_api_key(),
        source_truth_path=Path(args.source_truth),
        tef_catalog_dir=Path(args.tef_catalog_dir),
        output_root=Path(args.output_dir),
        benchmark_id=args.benchmark_id,
        max_workers=args.max_workers,
        limit=args.limit,
        log_llm_payload=args.log_llm_payload,
    )
    logger.info("Krakow TEF benchmark summary: %s", result)


if __name__ == "__main__":
    main()
