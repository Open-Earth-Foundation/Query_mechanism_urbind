"""
Brief: Summarize token usage from a pipeline run log and estimate API cost.

Inputs:
- Files/paths:
  - `--run-log`: Path to a pipeline `run.log` file containing `LLM_USAGE` entries.
- CLI args:
  - `--run-log`: Required path to a run log file.
  - `--include-reasoning`: Include `reasoning_tokens` totals in the summary.
  - `--input-price-per-1m`: USD price per 1M input tokens for cost estimation.
  - `--output-price-per-1m`: USD price per 1M output tokens for cost estimation.
- Env vars:
  - `LOG_LEVEL`: Logging verbosity level used by `setup_logger`.

Outputs:
- Logs to stdout with token totals, call count, and estimated cost.

Usage (from project root):
- python -m app.scripts.analyze_run_tokens --run-log output/20260209_1056/run.log
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from app.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)

_INPUT_TOKEN_PATTERN = re.compile(r'"input_tokens"\s*:\s*(\d+)')
_OUTPUT_TOKEN_PATTERN = re.compile(r'"output_tokens"\s*:\s*(\d+)')
_REASONING_TOKEN_PATTERN = re.compile(r'"reasoning_tokens"\s*:\s*(\d+)')


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Summarize token usage from a pipeline run log."
    )
    parser.add_argument(
        "--run-log",
        required=True,
        help="Path to a pipeline run.log file.",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include reasoning token totals when available.",
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


def _extract_token_values(pattern: re.Pattern[str], content: str) -> list[int]:
    return [int(value) for value in pattern.findall(content)]


def summarize_run_log(
    run_log_path: Path,
    include_reasoning: bool,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> None:
    """Summarize token usage from one run log path."""
    if not run_log_path.exists():
        raise FileNotFoundError(f"Run log file not found: {run_log_path}")

    content = run_log_path.read_text(encoding="utf-8")
    input_tokens = _extract_token_values(_INPUT_TOKEN_PATTERN, content)
    output_tokens = _extract_token_values(_OUTPUT_TOKEN_PATTERN, content)
    reasoning_tokens = _extract_token_values(_REASONING_TOKEN_PATTERN, content)

    call_count = max(len(input_tokens), len(output_tokens))
    total_input = sum(input_tokens)
    total_output = sum(output_tokens)
    total_reasoning = sum(reasoning_tokens)

    input_cost = total_input / 1_000_000 * input_price_per_1m
    output_cost = total_output / 1_000_000 * output_price_per_1m

    logger.info("Run log: %s", run_log_path)
    logger.info("Total API calls: %d", call_count)
    logger.info("Total input tokens: %s", f"{total_input:,}")
    logger.info("Total output tokens: %s", f"{total_output:,}")
    if include_reasoning:
        logger.info("Total reasoning tokens: %s", f"{total_reasoning:,}")
    if call_count > 0:
        logger.info("Average input tokens/call: %s", f"{(total_input // call_count):,}")
        logger.info("Average output tokens/call: %s", f"{(total_output // call_count):,}")
    logger.info("Estimated input cost: $%.4f", input_cost)
    logger.info("Estimated output cost: $%.4f", output_cost)
    logger.info("Estimated total cost: $%.4f", input_cost + output_cost)


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    summarize_run_log(
        run_log_path=Path(args.run_log),
        include_reasoning=args.include_reasoning,
        input_price_per_1m=args.input_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
    )


if __name__ == "__main__":
    main()

