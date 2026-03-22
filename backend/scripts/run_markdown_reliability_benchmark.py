"""
Brief: Run a markdown-only reliability benchmark across a fixed OpenRouter model matrix.

Warning:
- This benchmark can consume a lot of tokens and become expensive quickly because it
  replays the markdown stage across every enabled model and all markdown batches.
- Start with a small city subset and a short `--model` list before running the full matrix.

Inputs:
- CLI args:
  - `--matrix-config`: Path to the reliability matrix YAML file that defines the fixed
    question, retrieval queries, markdown defaults, payload capture mode, and model list.
  - `--config`: Path to `llm_config.yaml` used for vector-store retrieval and base agent settings.
  - `--benchmark-id`: Optional benchmark id. Defaults to a UTC timestamp when omitted.
  - `--output-dir`: Root directory for benchmark artifacts. The script writes one subdirectory
    per run at `output/reliability_testing/<benchmark_id>/` by default.
  - `--model`: Optional repeatable model-id filter. When provided, only matching entries from
    the matrix are executed, including entries that are disabled by default in the YAML.
  - `--city`: Optional repeatable city filter. When provided, it overrides `selected_cities`
    from the matrix for this run only.
- Files/paths:
  - The matrix YAML must match the schema in
    `backend/benchmarks/reliability_testing/config/markdown_model_matrix.yml`.
  - The vector index and manifest referenced by `llm_config.yaml` must already exist.
- Env vars:
  - `OPENROUTER_API_KEY`: Required unless the caller injects an API key by other means.
  - `LOG_LEVEL`: Optional logging verbosity used by `setup_logger`.

Outputs:
- `output/reliability_testing/<benchmark_id>/progress.json`
- `output/reliability_testing/<benchmark_id>/retrieval.json`
- `output/reliability_testing/<benchmark_id>/batches.json`
- `output/reliability_testing/<benchmark_id>/benchmark_report.json`
- `output/reliability_testing/<benchmark_id>/benchmark_report.md`
- `output/reliability_testing/<benchmark_id>/<model_slug>/run.log`
- `output/reliability_testing/<benchmark_id>/<model_slug>/error_log.txt`
- `output/reliability_testing/<benchmark_id>/<model_slug>/model_result.json`
- `output/reliability_testing/<benchmark_id>/<model_slug>/markdown/*.json`

Artifacts are flushed incrementally: the root progress/report snapshots are written at
startup and refreshed after each model completes, and each finished model gets its own
`model_result.json`.

Usage (from project root):
- python -m backend.scripts.run_markdown_reliability_benchmark
- python -m backend.scripts.run_markdown_reliability_benchmark --benchmark-id ev_markdown_reliability
- python -m backend.scripts.run_markdown_reliability_benchmark --model x-ai/grok-4.1-fast --model openai/gpt-5-nano
- python -m backend.scripts.run_markdown_reliability_benchmark --city Aachen --city Leipzig --city Mannheim --city Munich --city Oslo --city Lisbon --city Porto --city Turin --city Zaragoza --city "Vitoria Gasteiz"
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmarks.reliability_testing.runner import (
    DEFAULT_MATRIX_CONFIG_PATH,
    run_markdown_reliability_benchmark,
)
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Run the markdown reliability benchmark across a model matrix."
    )
    parser.add_argument(
        "--matrix-config",
        default=str(DEFAULT_MATRIX_CONFIG_PATH),
        help="Path to the reliability benchmark matrix YAML.",
    )
    parser.add_argument(
        "--config",
        default="llm_config.yaml",
        help="Path to llm_config.yaml.",
    )
    parser.add_argument(
        "--benchmark-id",
        help="Optional benchmark id. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/reliability_testing",
        help="Root directory for benchmark artifacts.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Optional repeatable model-id filter.",
    )
    parser.add_argument(
        "--city",
        action="append",
        help=(
            "Optional repeatable city filter. Overrides selected_cities from the "
            "matrix for this run only."
        ),
    )
    return parser.parse_args(argv)


def _default_benchmark_id() -> str:
    """Build a UTC timestamp slug for the default benchmark id."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main(argv: list[str] | None = None) -> None:
    """Script entry point."""
    args = parse_args(argv)
    setup_logger()
    benchmark_id = args.benchmark_id if args.benchmark_id else _default_benchmark_id()
    report = run_markdown_reliability_benchmark(
        benchmark_id=benchmark_id,
        output_dir=Path(args.output_dir),
        config_path=Path(args.config),
        matrix_config_path=Path(args.matrix_config),
        selected_models=args.model or [],
        selected_cities=args.city or None,
    )
    logger.info("Reliability benchmark completed: %s", report.benchmark_id)
    logger.info("Artifacts: %s", report.output_dir)


if __name__ == "__main__":
    main()
