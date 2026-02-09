"""
Brief: Run the multi-agent document builder pipeline.

Inputs:
- --question: user question to answer
- --run-id: optional run identifier
- --config: path to llm_config.yaml
- --enable-sql: enable SQL lookups (disabled by default)
- --db-path: override source DB path
- --db-url: override source DB URL
- --markdown-path: override documents folder
- --log-llm-payload: log full LLM request/response payloads (default: on)
- --no-log-llm-payload: disable LLM payload logging
- OPENROUTER_API_KEY (env var)

Outputs:
- output/<run_id>/run.json and artifact files
- output/<run_id>/final.md

Usage (from project root):
- python -m app.scripts.run_pipeline --question "..."
- python -m app.scripts.run_pipeline --enable-sql --question "..." --db-path path/to/source.db
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# Disable OpenAI Agents tracing before importing to prevent 401 errors with OpenRouter
# (OpenRouter keys are not recognized by OpenAI's tracing endpoint)
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

from app.modules.orchestrator.module import run_pipeline
from app.utils.config import load_config
from app.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run the document builder pipeline.")
    parser.add_argument("--question", required=True, help="User question to answer.")
    parser.add_argument("--run-id", help="Optional run id.")
    parser.add_argument(
        "--config", default="llm_config.yaml", help="Path to llm_config.yaml"
    )
    parser.add_argument(
        "--enable-sql",
        action="store_true",
        help="Enable SQL lookups (disabled by default).",
    )
    parser.add_argument("--db-path", help="Override source DB path.")
    parser.add_argument("--db-url", help="Override source DB URL.")
    parser.add_argument("--markdown-path", help="Override markdown documents path.")
    parser.add_argument(
        "--log-llm-payload",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="log_llm_payload",
        help="Enable or disable logging of full LLM request/response payloads (default: on).",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    config = load_config(Path(args.config))
    if args.db_path:
        config.source_db_path = Path(args.db_path)
    if args.db_url:
        config.source_db_url = args.db_url
    if args.markdown_path:
        config.markdown_dir = Path(args.markdown_path)
    if args.enable_sql:
        config.enable_sql = True

    logger.info("Starting pipeline")
    run_pipeline(
        question=args.question,
        config=config,
        run_id=args.run_id,
        log_llm_payload=args.log_llm_payload,
    )


if __name__ == "__main__":
    main()
