"""
Brief: Calculate token usage across markdown documents and estimate batch run cost.

Inputs:
- Files/paths:
  - `--documents-dir`: Directory containing markdown documents to analyze.
  - `--prompt-path`: Optional prompt file included in per-request token estimates.
- CLI args:
  - `--documents-dir`: Directory of `.md` files to scan.
  - `--recursive`: Scan markdown files recursively.
  - `--prompt-path`: Prompt file path for system prompt token counting.
  - `--num-batches`: Number of markdown request batches to model.
  - `--avg-output-tokens`: Average output tokens per batch for cost estimation.
  - `--input-price-per-1m`: USD price per 1M input tokens.
  - `--output-price-per-1m`: USD price per 1M output tokens.
  - `--top`: Number of largest markdown files (by token count) to print.
- Env vars:
  - `LOG_LEVEL`: Logging verbosity level used by `setup_logger`.

Outputs:
- Logs to stdout with per-file token counts, totals, and estimated run cost.

Usage (from project root):
- python -m app.scripts.calculate_tokens --documents-dir documents --recursive
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.utils.logging_config import setup_logger
from app.utils.tokenization import count_tokens

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Calculate markdown token usage and estimate processing cost."
    )
    parser.add_argument(
        "--documents-dir",
        default="documents",
        help="Directory containing markdown documents.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan markdown files under --documents-dir.",
    )
    parser.add_argument(
        "--prompt-path",
        default="app/prompts/markdown_researcher_system.md",
        help="Prompt file path for system prompt token counting.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        help="Number of markdown batches to model. Defaults to markdown file count.",
    )
    parser.add_argument(
        "--avg-output-tokens",
        type=int,
        default=1500,
        help="Average output tokens per batch used for cost estimation.",
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
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of largest markdown files to print.",
    )
    return parser.parse_args()


def _collect_markdown_files(documents_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(path for path in documents_dir.glob(pattern) if path.is_file())


def _read_token_count(path: Path) -> int:
    return count_tokens(path.read_text(encoding="utf-8"))


def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    input_cost = input_tokens / 1_000_000 * input_price_per_1m
    output_cost = output_tokens / 1_000_000 * output_price_per_1m
    logger.info("Estimated input cost: $%.4f", input_cost)
    logger.info("Estimated output cost: $%.4f", output_cost)
    logger.info("Estimated total cost: $%.4f", input_cost + output_cost)
    return input_cost + output_cost


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    documents_dir = Path(args.documents_dir)
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    markdown_files = _collect_markdown_files(documents_dir, args.recursive)
    if not markdown_files:
        logger.warning("No markdown files found under %s", documents_dir)
        return

    file_totals: list[tuple[Path, int, float]] = []
    total_tokens = 0
    for file_path in markdown_files:
        token_count = _read_token_count(file_path)
        size_kb = file_path.stat().st_size / 1024
        file_totals.append((file_path, token_count, size_kb))
        total_tokens += token_count

    logger.info("Markdown files analyzed: %d", len(file_totals))
    logger.info("Total markdown tokens: %s", f"{total_tokens:,}")

    prompt_tokens = 0
    prompt_path = Path(args.prompt_path)
    if prompt_path.exists():
        prompt_tokens = _read_token_count(prompt_path)
        logger.info("Prompt tokens (%s): %s", prompt_path, f"{prompt_tokens:,}")
    else:
        logger.warning("Prompt file not found, using 0 prompt tokens: %s", prompt_path)

    num_batches = args.num_batches if args.num_batches is not None else len(file_totals)
    num_batches = max(num_batches, 1)
    avg_output_tokens = max(args.avg_output_tokens, 0)

    avg_tokens_per_batch = total_tokens // num_batches
    input_tokens_per_batch = prompt_tokens + avg_tokens_per_batch
    estimated_input_tokens = num_batches * input_tokens_per_batch
    estimated_output_tokens = num_batches * avg_output_tokens

    logger.info("Modeled batch count: %d", num_batches)
    logger.info("Average markdown tokens per batch: %s", f"{avg_tokens_per_batch:,}")
    logger.info("Estimated total input tokens per run: %s", f"{estimated_input_tokens:,}")
    logger.info(
        "Estimated total output tokens per run: %s", f"{estimated_output_tokens:,}"
    )

    _estimate_cost(
        input_tokens=estimated_input_tokens,
        output_tokens=estimated_output_tokens,
        input_price_per_1m=args.input_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
    )

    top_n = max(args.top, 0)
    if top_n == 0:
        return

    logger.info("Top %d markdown files by token count:", top_n)
    for rank, (file_path, token_count, size_kb) in enumerate(
        sorted(file_totals, key=lambda item: item[1], reverse=True)[:top_n],
        start=1,
    ):
        share = token_count / total_tokens * 100 if total_tokens else 0.0
        logger.info(
            "%d. %s | tokens=%s | size_kb=%.1f | share=%.1f%%",
            rank,
            file_path,
            f"{token_count:,}",
            size_kb,
            share,
        )


if __name__ == "__main__":
    main()

