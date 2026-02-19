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
- python -m backend.scripts.analyze_run_tokens --run-log output/20260209_1056/run.log
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path

from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


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


def _extract_token_value(usage: Mapping[str, object], keys: list[str]) -> int:
    """Return the first numeric token value found for the provided key aliases."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _extract_reasoning_tokens(usage: Mapping[str, object]) -> int:
    """Extract reasoning tokens from usage payloads when available."""
    details = usage.get("output_tokens_details")
    if isinstance(details, Mapping):
        reasoning_tokens = details.get("reasoning_tokens")
        if isinstance(reasoning_tokens, (int, float)):
            return int(reasoning_tokens)
    return _extract_token_value(usage, ["reasoning_tokens"])


def _parse_llm_usage_lines(run_log_path: Path) -> tuple[int, int, int, int]:
    """Parse only LLM_USAGE lines from run.log and return aggregate token totals."""
    call_count = 0
    total_input = 0
    total_output = 0
    total_reasoning = 0

    with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "LLM_USAGE " not in line:
                continue
            payload = line.split("LLM_USAGE ", 1)[1].strip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = data.get("usage")
            if not isinstance(usage, Mapping):
                continue

            call_count += 1
            total_input += _extract_token_value(
                usage, ["input_tokens", "prompt_tokens"]
            )
            total_output += _extract_token_value(
                usage, ["output_tokens", "completion_tokens"]
            )
            total_reasoning += _extract_reasoning_tokens(usage)

    return call_count, total_input, total_output, total_reasoning


def summarize_run_log(
    run_log_path: Path,
    include_reasoning: bool,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> None:
    """Summarize token usage from one run log path."""
    if not run_log_path.exists():
        raise FileNotFoundError(f"Run log file not found: {run_log_path}")

    call_count, total_input, total_output, total_reasoning = _parse_llm_usage_lines(
        run_log_path
    )

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
