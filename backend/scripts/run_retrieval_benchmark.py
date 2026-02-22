"""
Brief: Benchmark standard chunking versus vector-store retrieval.

Behavior: The benchmark runs every question from the questions file (not a single
query repeated N times). For each question it runs that question `repetitions`
times per mode. So total runs = len(questions) * repetitions * len(modes). Run IDs
use rNN = repetition index and qNN = question index (e.g. vector_store_r01_q02 =
repetition 1, second question). To get identical queries across all runs, use a
questions file with only one question.

Inputs:
- CLI args:
  - --benchmark-id: Optional benchmark id. Defaults to UTC timestamp when omitted.
  - --output-dir: Root output directory for benchmark artifacts (default: output/benchmarks).
  - --config: Path to llm_config.yaml (default: llm_config.yaml).
  - --docs-dir: Markdown directory used by both benchmark modes (default: documents).
  - --questions-file: Newline-delimited benchmark questions file (default: backend/benchmarks/prompts/retrieval_questions.txt).
  - --city: Optional city filter; repeatable.
  - --repetitions: Number of repetitions per question per mode (default: 1).
  - --mode: Run only these modes; repeatable; choices: standard_chunking, vector_store. Default: both.
  - --use-query-overrides/--no-use-query-overrides: Enable/disable fixed retrieval queries for benchmark stability (default: enabled).
  - --query-overrides: JSON file mapping benchmark question -> canonical query + retrieval queries (default: backend/benchmarks/prompts/retrieval_query_overrides.json).
  - --log-llm-payload/--no-log-llm-payload: Enable/disable full LLM payload logging (default: off).
- Files/paths:
  - Env files use dotenv KEY=VALUE format.
  - Question file supports comments via lines starting with '#'.
- Env vars:
  - OPENROUTER_API_KEY is required for LLM calls.

Outputs:
- output/benchmarks/<benchmark_id>/benchmark_report.json
- output/benchmarks/<benchmark_id>/benchmark_report.md
- output/benchmarks/<benchmark_id>/runs/<mode>/<run_id>/... pipeline artifacts

Usage (from project root):
- python -m backend.scripts.run_retrieval_benchmark --city Munster --city Leipzig --city Mannheim
- python -m backend.scripts.run_retrieval_benchmark --mode vector_store --repetitions 5 --questions-file my_questions.txt --query-overrides my_overrides.json --city Aachen
- python -m backend.scripts.run_retrieval_benchmark --mode vector_store --repetitions 2
  (uses default backend/benchmarks/prompts/retrieval_questions.txt and retrieval_query_overrides.json)
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmarks.runner import (
    BenchmarkModeConfig,
    run_retrieval_strategy_benchmark,
)
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)
DEFAULT_BASE_ENV_FILE = Path("backend/benchmarks/config/base.env")
DEFAULT_STANDARD_ENV_FILE = Path("backend/benchmarks/config/mode_standard.env")
DEFAULT_VECTOR_ENV_FILE = Path("backend/benchmarks/config/mode_vector.env")
DEFAULT_QUERY_OVERRIDES_FILE = Path(
    "backend/benchmarks/prompts/retrieval_query_overrides.json"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Benchmark standard chunking vs vector-store retrieval."
    )
    parser.add_argument(
        "--benchmark-id",
        help="Optional benchmark id. Defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/benchmarks",
        help="Benchmark output directory root.",
    )
    parser.add_argument(
        "--config",
        default="llm_config.yaml",
        help="Path to llm config.",
    )
    parser.add_argument(
        "--docs-dir",
        default="documents",
        help="Markdown docs directory used in both modes.",
    )
    parser.add_argument(
        "--questions-file",
        default="backend/benchmarks/prompts/retrieval_questions.txt",
        help="Path to newline-delimited benchmark questions file.",
    )
    parser.add_argument(
        "--city",
        action="append",
        help="Optional city filter (repeatable).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repetitions per question per mode.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=["standard_chunking", "vector_store"],
        help="Benchmark mode(s) to run. Default: both. Repeat to run multiple modes.",
    )
    parser.add_argument(
        "--use-query-overrides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fixed benchmark retrieval queries from --query-overrides.",
    )
    parser.add_argument(
        "--query-overrides",
        default=str(DEFAULT_QUERY_OVERRIDES_FILE),
        help="JSON file mapping question -> canonical query + retrieval queries.",
    )
    parser.add_argument(
        "--log-llm-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable full LLM payload logs for benchmark runs.",
    )
    return parser.parse_args()


def _default_benchmark_id() -> str:
    """Build default benchmark id from UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    benchmark_id = args.benchmark_id if args.benchmark_id else _default_benchmark_id()

    all_mode_configs = [
        BenchmarkModeConfig(
            name="standard_chunking",
            env_files=[DEFAULT_BASE_ENV_FILE, DEFAULT_STANDARD_ENV_FILE],
        ),
        BenchmarkModeConfig(
            name="vector_store",
            env_files=[DEFAULT_BASE_ENV_FILE, DEFAULT_VECTOR_ENV_FILE],
        ),
    ]
    modes = args.mode or []
    mode_configs = (
        [m for m in all_mode_configs if m.name in modes]
        if modes
        else all_mode_configs
    )

    report = run_retrieval_strategy_benchmark(
        benchmark_id=benchmark_id,
        output_dir=Path(args.output_dir),
        config_path=Path(args.config),
        docs_dir=Path(args.docs_dir),
        questions_file=Path(args.questions_file),
        selected_cities=args.city or [],
        repetitions=args.repetitions,
        mode_configs=mode_configs,
        use_query_overrides=bool(args.use_query_overrides),
        query_overrides_path=Path(args.query_overrides) if args.query_overrides else None,
        log_llm_payload=args.log_llm_payload,
    )

    logger.info("Benchmark completed: %s", report.benchmark_id)
    logger.info("Artifacts: %s", report.output_dir)
    for mode_name, summary in report.mode_summary.items():
        logger.info(
            (
                "Mode=%s runtime_median=%.2fs tokens_median=%.0f "
                "chunks_mean=%.1f excerpts_mean=%.1f"
            ),
            mode_name,
            summary["runtime_seconds_median"],
            summary["total_tokens_median"],
            summary["markdown_chunk_count_mean"],
            summary["markdown_excerpt_count_mean"],
        )
    for mode_name, summary in report.judge_summary.items():
        logger.info(
            ("Judge mode=%s score_median=%.2f score_mean=%.2f " "wins=%.0f ties=%.0f"),
            mode_name,
            summary["median_judge_score"],
            summary["mean_judge_score"],
            summary["wins"],
            summary["ties"],
        )


if __name__ == "__main__":
    main()
