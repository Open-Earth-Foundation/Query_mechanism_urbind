"""
Brief: Run a batch of end-to-end questions through the pipeline.

Inputs:
- --questions-file: path to a newline-delimited questions file
- --question: optional single question (can be passed multiple times); overrides --questions-file when provided
- --config: path to llm_config.yaml
- --markdown-path: override documents folder
- --city: limit markdown loading to selected city names (repeatable)
- --log-llm-payload: log full LLM request/response payloads (default: on)
- --no-log-llm-payload: disable LLM payload logging
- OPENROUTER_API_KEY (env var)

Outputs:
- output/<run_id>/... artifacts for each question

Usage (from project root):

- python -m backend.scripts.run_e2e_queries
- python -m backend.scripts.run_e2e_queries --questions-file assets/e2e_questions.txt
- python -m backend.scripts.run_e2e_queries --question "What initiatives exist for Munich?"
- python -m backend.scripts.run_e2e_queries --question "What initiatives exist for Munich and Leipzig?" --city Munich --city Leipzig
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from backend.modules.orchestrator.module import run_pipeline
from backend.utils.config import load_config, resolve_path_relative_to_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run end-to-end query batch.")
    parser.add_argument(
        "--questions-file",
        default="assets/e2e_questions.txt",
        help="Path to newline-delimited questions file.",
    )
    parser.add_argument(
        "--question",
        action="append",
        help="Single question to run (can be repeated). Overrides --questions-file when provided.",
    )
    parser.add_argument(
        "--config", default="llm_config.yaml", help="Path to llm_config.yaml"
    )
    parser.add_argument("--markdown-path", help="Override markdown documents path.")
    parser.add_argument(
        "--city",
        action="append",
        help="Limit markdown loading to selected city names (repeatable).",
    )
    parser.add_argument(
        "--log-llm-payload",
        action="store_true",
        default=True,
        help="Log full LLM request/response payloads (default: on).",
    )
    parser.add_argument(
        "--no-log-llm-payload",
        action="store_false",
        dest="log_llm_payload",
        help="Disable LLM payload logging.",
    )
    return parser.parse_args()


def load_questions(path: Path, overrides: list[str] | None) -> list[str]:
    """Load questions from file unless CLI overrides are provided."""
    if overrides:
        return [q.strip() for q in overrides if q.strip()]

    questions: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            questions.append(cleaned)

    return questions


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.markdown_path:
        config.markdown_dir = resolve_path_relative_to_config(
            config_path,
            Path(args.markdown_path),
        )

    questions = load_questions(Path(args.questions_file), args.question)
    if not questions:
        logger.warning("No questions provided. Provide --question or a questions file.")
        return

    total_start = time.perf_counter()
    logger.info("Starting batch of %d questions", len(questions))
    for question in questions:
        logger.info("Running question: %s", question)
        start = time.perf_counter()
        run_pipeline(
            question=question,
            config=config,
            log_llm_payload=args.log_llm_payload,
            selected_cities=args.city,
            config_path=config_path,
        )
        elapsed = time.perf_counter() - start
        logger.info("Completed question in %.2f seconds", elapsed)
    total_elapsed = time.perf_counter() - total_start
    logger.info("Completed %d questions in %.2f seconds", len(questions), total_elapsed)


if __name__ == "__main__":
    main()
