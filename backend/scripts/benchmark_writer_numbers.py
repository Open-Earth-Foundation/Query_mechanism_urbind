"""
Brief: Run the writer numeric benchmark against frozen Krakow, Poland, and optional all-cities cases.

Inputs:
- CLI args:
  - `--benchmark-file`: JSON fixture with `version`, `default_mode`, and `cases[]`.
    Defaults to `backend/benchmarks/writer_numeric/writer_numeric_benchmark.json`.
  - `--config`: Path to `llm_config.yaml`.
  - `--output-dir`: Directory where benchmark artifacts are written.
    Defaults to `output/benchmarks/writer_numeric`.
  - `--run-id`: Optional stable benchmark ID. Defaults to a UTC timestamp slug.
  - `--mode`: `ccc_only`, `full_pipeline`, or `both`. Defaults to the fixture's
    `default_mode`, which is `ccc_only`.
  - `--include-optional-cases`: Include fixture cases marked as optional. The
    frozen 102-city all-cities case is optional by default because it can take
    a long time and consume a large number of LLM tokens.
- Files/paths: expects the live markdown corpus under `documents/` and a valid LLM config.
- Env vars: `OPENROUTER_API_KEY` is required for the live pipeline and numeric extractor.

Outputs:
- `benchmark_summary.json`: full persisted benchmark payload with case-level results.
- `benchmark_report.md`: human-readable per-metric diff report.
- `runs/<case_id>__<mode>/final.md`: final writer output from the live pipeline.
- `runs/<case_id>__<mode>/context_bundle.json`: live writer context used for the run.
- `runs/<case_id>__<mode>/extracted_numbers.json`: structured extractor output.

Usage (from project root):
- python -m backend.scripts.benchmark_writer_numbers
- python -m backend.scripts.benchmark_writer_numbers --include-optional-cases
- python -m backend.scripts.benchmark_writer_numbers --mode both --run-id writer_numeric_smoke
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmarks.writer_numeric.runner import (
    DEFAULT_BENCHMARK_FILE,
    DEFAULT_OUTPUT_DIR,
    load_writer_numeric_benchmark_dataset,
    run_writer_numeric_benchmark,
    select_benchmark_cases,
)
from backend.utils.config import get_openrouter_api_key, load_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the writer numeric benchmark against frozen benchmark cases."
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=DEFAULT_BENCHMARK_FILE,
        help="JSON fixture describing the frozen benchmark cases.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("llm_config.yaml"),
        help="LLM config YAML path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional benchmark ID. Defaults to a UTC timestamp slug.",
    )
    parser.add_argument(
        "--mode",
        choices=("ccc_only", "full_pipeline", "both"),
        default=None,
        help="Pipeline mode to benchmark. Defaults to the fixture default_mode.",
    )
    parser.add_argument(
        "--include-optional-cases",
        action="store_true",
        help=(
            "Include fixture cases marked as optional, such as the expensive "
            "all-cities run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    dataset = load_writer_numeric_benchmark_dataset(args.benchmark_file)
    requested_mode = args.mode or dataset.default_mode
    selected_cases = select_benchmark_cases(
        dataset,
        include_optional_cases=args.include_optional_cases,
    )
    benchmark_id = args.run_id or datetime.now(timezone.utc).strftime(
        "writer_numeric_%Y%m%dT%H%M%SZ"
    )
    config = load_config(args.config)
    api_key = get_openrouter_api_key()
    logger.info(
        "Starting writer numeric benchmark benchmark_id=%s mode=%s cases=%d",
        benchmark_id,
        requested_mode,
        len(selected_cases),
    )
    report = run_writer_numeric_benchmark(
        benchmark_file=args.benchmark_file,
        output_dir=args.output_dir,
        benchmark_id=benchmark_id,
        requested_mode=requested_mode,
        config=config,
        api_key=api_key,
        include_optional_cases=args.include_optional_cases,
    )
    logger.info(
        "Writer numeric benchmark completed benchmark_id=%s outputs=%d mismatches=%d missing=%d",
        report.benchmark_id,
        report.summary.output_count,
        report.summary.mismatch_count,
        report.summary.missing_count,
    )


if __name__ == "__main__":
    setup_logger()
    main()
