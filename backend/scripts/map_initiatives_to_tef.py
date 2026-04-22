"""
Brief: Map extracted city climate initiatives to TEF targets as JSON artifacts.

Inputs:
- CLI args:
  - `--extraction-run-dir`: Initiative extraction run directory containing
    `03_deduped/initiative_records.jsonl`, for example `output/initiative_extraction/<run-id>`.
    Older runs with only `03_deduped/initiatives.jsonl` are also accepted.
  - `--initiatives-jsonl`: Direct path to a deduplicated initiative records JSONL file; overrides
    `--extraction-run-dir` input discovery when provided. Direct input rows must include generated
    record metadata, not only canonical v1 initiative objects.
  - `--tef-catalog-dir`: Local TEF catalog directory, default `tef_mapping`.
  - `--city`: Optional city filter matched against extracted initiative city names; repeatable.
  - `--limit`: Optional maximum number of initiatives to map after filtering.
  - `--run-id`: Optional output run identifier used as the output subdirectory name.
  - `--output-dir`: Root output directory for TEF mapping artifacts.
  - `--config`: Path to `llm_config.yaml`.
  - `--max-workers`: Optional worker override for initiative-level LLM calls.
  - `--log-llm-payload`: Enable logging of full LLM request/response payloads.
- Files/paths: extracted initiatives are read from JSONL; TEF catalog JSON is read from
  `--tef-catalog-dir`; output is written under `--output-dir/<run-id>/`.
- Env vars: `OPENROUTER_API_KEY` is required for LLM mapping.

Outputs:
- `00_source/source_manifest.json`
- `01_inputs/initiatives.jsonl`
- `02_sector_routes/sector_routes.jsonl`
- `03_subsector_routes/subsector_routes.jsonl`
- `04_transition_mappings/transition_mappings.jsonl`
- `05_final_mappings/final_mappings.jsonl`
- `06_review/review_items.jsonl`
- `summary.json`
- `README.md`

Usage (from project root):
- python -m backend.scripts.map_initiatives_to_tef --extraction-run-dir output/initiative_extraction/five_cities_20260421_002 --city Krakow
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from backend.modules.tef_mapper import map_initiatives_to_tef
from backend.utils.config import load_config, resolve_openrouter_api_key
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Map extracted initiatives to TEF targets as JSON artifacts."
    )
    parser.add_argument(
        "--extraction-run-dir",
        help="Initiative extraction run directory containing 03_deduped/initiative_records.jsonl.",
    )
    parser.add_argument(
        "--initiatives-jsonl",
        help="Direct path to deduplicated initiative records JSONL; overrides extraction-run discovery.",
    )
    parser.add_argument(
        "--tef-catalog-dir",
        default="tef_mapping",
        help="Local TEF mapping catalog directory.",
    )
    parser.add_argument(
        "--city",
        action="append",
        help="Limit mapping to selected extracted initiative city names; repeatable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of initiatives to map after filtering.",
    )
    parser.add_argument("--run-id", help="Optional output run identifier.")
    parser.add_argument(
        "--output-dir",
        default="output/tef_mapping",
        help="Root directory for TEF mapping artifacts.",
    )
    parser.add_argument(
        "--config",
        default="llm_config.yaml",
        help="Path to llm_config.yaml.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Override initiative-level worker count.",
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
    extraction_run_dir = Path(args.extraction_run_dir) if args.extraction_run_dir else None
    initiatives_jsonl = Path(args.initiatives_jsonl) if args.initiatives_jsonl else None
    result = map_initiatives_to_tef(
        config=config,
        api_key=api_key,
        tef_catalog_dir=Path(args.tef_catalog_dir),
        output_root=Path(args.output_dir),
        extraction_run_dir=extraction_run_dir,
        initiatives_jsonl=initiatives_jsonl,
        run_id=args.run_id,
        selected_cities=args.city,
        limit=args.limit,
        max_workers=args.max_workers,
        log_llm_payload=args.log_llm_payload,
    )
    logger.info("TEF mapping finished: %s", result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
