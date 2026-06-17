"""
Brief: Run the multi-agent document builder pipeline.

Inputs:
- --question: user question to answer
- --run-id: optional run identifier
- --config: path to llm_config.yaml
- --markdown-path: override documents folder
- --city: limit markdown loading to selected city names (repeatable)
- --log-llm-payload: log full LLM request/response payloads (default: off)
- --no-log-llm-payload: disable LLM payload logging
- OPENROUTER_API_KEY (env var)

Outputs:
- output/<run_id>/api_state.json and artifact files
- output/<run_id>/final.md

Usage (from project root):

- python -m backend.scripts.run_pipeline --question "..."
- python -m backend.scripts.run_pipeline --question "..." --city Munich --city Leipzig
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from backend.modules.orchestrator.module import run_pipeline
from backend.utils.config import load_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def _build_invocation_command(args: argparse.Namespace) -> str:
    """Render the direct CLI invocation in a stable rerunnable form."""
    command = [
        "python",
        "-m",
        "backend.scripts.run_pipeline",
        "--question",
        args.question,
    ]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if args.config != "llm_config.yaml":
        command.extend(["--config", args.config])
    if args.markdown_path:
        command.extend(["--markdown-path", args.markdown_path])
    for city in args.city or []:
        command.extend(["--city", city])
    if args.log_llm_payload:
        command.append("--log-llm-payload")
    return subprocess.list2cmdline(command)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run the document builder pipeline.")
    parser.add_argument("--question", required=True, help="User question to answer.")
    parser.add_argument("--run-id", help="Optional run id.")
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
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="log_llm_payload",
        help="Enable or disable logging of full LLM request/response payloads (default: off).",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    config = load_config(Path(args.config))
    if args.markdown_path:
        config.markdown_dir = Path(args.markdown_path)

    logger.info("Starting pipeline")
    run_pipeline(
        question=args.question,
        config=config,
        run_id=args.run_id,
        log_llm_payload=args.log_llm_payload,
        selected_cities=args.city,
        config_path=Path(args.config),
        invocation_command=_build_invocation_command(args),
    )


if __name__ == "__main__":
    main()
