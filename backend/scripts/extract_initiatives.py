"""
Brief: Extract city climate initiatives from Markdown documents into JSONL artifacts.

Inputs:
- CLI args:
  - `--markdown-path`: Markdown file or directory containing source `*.md` files.
  - `--city`: Optional city name filter matched against markdown file stems; repeatable.
  - `--run-id`: Optional run identifier used as the output subdirectory name.
  - `--output-dir`: Root output directory for extraction artifacts.
  - `--config`: Path to `llm_config.yaml`.
  - `--max-workers`: Optional worker override for segment-level LLM calls.
  - `--log-llm-payload`: Enable logging of full LLM request/response payloads.
- Files/paths: source Markdown is read from `--markdown-path`; output is written under
  `--output-dir/<run-id>/`.
- Env vars: `OPENROUTER_API_KEY` is required for LLM extraction.

Outputs:
- `00_source/source_manifest.json`
- `01_segments/segments.jsonl`
- `02_raw_extractions/raw_segment_extractions.jsonl`
- `03_deduped/initiatives.jsonl` with canonical v1 initiative objects only
- `03_deduped/initiative_records.jsonl` with generated ids and source_quote audit citations
- `04_review/review_items.jsonl`
- `summary.json`
- `README.md`

Usage (from project root):
- python -m backend.scripts.extract_initiatives --markdown-path documents --city Krakow
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from backend.modules.initiative_extractor import extract_initiatives
from backend.utils.config import load_config, resolve_openrouter_api_key
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Extract climate initiatives from Markdown documents."
    )
    parser.add_argument(
        "--markdown-path",
        default="documents",
        help="Markdown file or directory containing source .md files.",
    )
    parser.add_argument(
        "--city",
        action="append",
        help="Limit extraction to selected city file stems; repeatable.",
    )
    parser.add_argument("--run-id", help="Optional output run identifier.")
    parser.add_argument(
        "--output-dir",
        default="output/initiative_extraction",
        help="Root directory for initiative extraction artifacts.",
    )
    parser.add_argument(
        "--config",
        default="llm_config.yaml",
        help="Path to llm_config.yaml.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Override segment-level worker count.",
    )
    parser.add_argument(
        "--log-llm-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="log_llm_payload",
        help="Enable or disable full LLM payload logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    load_dotenv()

    config = load_config(Path(args.config))
    api_key = resolve_openrouter_api_key()
    result = extract_initiatives(
        markdown_path=Path(args.markdown_path),
        config=config,
        api_key=api_key,
        output_root=Path(args.output_dir),
        run_id=args.run_id,
        selected_cities=args.city,
        max_workers=args.max_workers,
        log_llm_payload=args.log_llm_payload,
    )
    logger.info("Initiative extraction finished: %s", result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
