"""
Brief: Backward-compatible run-log token analyzer that includes reasoning tokens.

Inputs:
- Files/paths:
  - `--run-log`: Path to a pipeline `run.log` file.
- CLI args:
  - `--run-log`: Required path to a run log file.
  - `--input-price-per-1m`: USD price per 1M input tokens for cost estimation.
  - `--output-price-per-1m`: USD price per 1M output tokens for cost estimation.
- Env vars:
  - `LOG_LEVEL`: Logging verbosity level used by `setup_logger`.

Outputs:
- Logs to stdout with token totals and estimated cost.

Usage (from project root):
- python -m app.scripts.temp_analyze --run-log output/20260209_1056/run.log
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.scripts.analyze_run_tokens import summarize_run_log
from app.utils.logging_config import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Analyze token usage from a run log (includes reasoning tokens)."
    )
    parser.add_argument(
        "--run-log",
        required=True,
        help="Path to a pipeline run.log file.",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=0.16,
        help="USD price per 1M input tokens.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=0.64,
        help="USD price per 1M output tokens.",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    summarize_run_log(
        run_log_path=Path(args.run_log),
        include_reasoning=True,
        input_price_per_1m=args.input_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
    )


if __name__ == "__main__":
    main()

