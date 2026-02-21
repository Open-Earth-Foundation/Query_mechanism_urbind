"""
Brief: Compare two final markdown outputs with GPT-5.2 judge rubric scoring.

Inputs:
- CLI args:
  - --left-final: Path to left candidate markdown file.
  - --right-final: Path to right candidate markdown file.
  - --question: Optional original question (recommended for best judging context).
  - --left-label: Label for left candidate (default: left).
  - --right-label: Label for right candidate (default: right).
  - --config: Path to llm_config.yaml (default: llm_config.yaml).
  - --output-json: Optional path to persist judge result JSON.
  - --log-llm-payload/--no-log-llm-payload: Enable/disable full payload logs (default: off).
- Files/paths:
  - Left/right inputs should be markdown outputs such as `output/<run_id>/final.md`.
- Env vars:
  - OPENROUTER_API_KEY is required.

Outputs:
- JSON judge result written to stdout.
- Optional JSON file at --output-json.

Usage (from project root):
- python -m backend.scripts.judge_final_outputs --left-final output/run_a/final.md --right-final output/run_b/final.md --question "..."
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from backend.benchmarks.judge import judge_final_outputs
from backend.utils.config import get_openrouter_api_key, load_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Judge two final markdown outputs.")
    parser.add_argument("--left-final", required=True, help="Path to left final markdown file.")
    parser.add_argument(
        "--right-final",
        required=True,
        help="Path to right final markdown file.",
    )
    parser.add_argument(
        "--question",
        default="",
        help="Optional original question for judging context.",
    )
    parser.add_argument(
        "--left-label",
        default="left",
        help="Label for left candidate output.",
    )
    parser.add_argument(
        "--right-label",
        default="right",
        help="Label for right candidate output.",
    )
    parser.add_argument("--config", default="llm_config.yaml", help="Path to llm config.")
    parser.add_argument(
        "--output-json",
        help="Optional path to save judge result JSON.",
    )
    parser.add_argument(
        "--log-llm-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable full LLM payload logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    left_path = Path(args.left_final)
    right_path = Path(args.right_final)
    left_text = left_path.read_text(encoding="utf-8")
    right_text = right_path.read_text(encoding="utf-8")

    config = load_config(Path(args.config))
    api_key = get_openrouter_api_key()
    evaluation = judge_final_outputs(
        question=args.question,
        left_label=args.left_label,
        left_text=left_text,
        right_label=args.right_label,
        right_text=right_text,
        config=config,
        api_key=api_key,
        log_llm_payload=args.log_llm_payload,
    )
    payload = evaluation.model_dump()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)  # intentional CLI output

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        logger.info("Judge result written to: %s", output_path.as_posix())


if __name__ == "__main__":
    main()
