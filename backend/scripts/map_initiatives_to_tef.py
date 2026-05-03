"""
Brief: Extract city climate initiatives and map them to TEF targets as JSON artifacts.

Inputs:
- CLI args:
  - `--mapping-only`: Skip extraction and map an existing extraction run or JSONL input.
  - `--markdown-path`: Markdown file or directory containing source `*.md` files for the
    default extraction-plus-mapping run.
  - `--extraction-output-dir`: Root output directory for extraction artifacts in the default
    extraction-plus-mapping run.
  - `--extraction-run-id`: Optional extraction run identifier; defaults to `--run-id` when
    provided, otherwise the extractor generates a timestamp.
  - `--extraction-run-dir`: Mapping-only initiative extraction run directory containing
    `03_deduped/initiative_records.jsonl`, for example `output/initiative_extraction/<run-id>`.
  - `--initiatives-jsonl`: Mapping-only direct path to a deduplicated initiative records JSONL
    file; overrides `--extraction-run-dir` input discovery when provided. Direct input rows must
    include generated record metadata, not only canonical v1 initiative objects.
  - `--tef-catalog-dir`: Local TEF catalog directory, default `tef_mapping`.
  - `--city`: Optional city filter; repeat for multiple cities. Omit to process all discovered
    Markdown cities in the default run or all input initiatives in mapping-only mode.
  - `--limit`: Optional maximum number of initiatives to map after filtering.
  - `--run-id`: Optional output run identifier used as the output subdirectory name.
  - `--output-dir`: Root output directory for TEF mapping artifacts.
  - `--config`: Path to `llm_config.yaml`.
  - `--max-workers`: Optional worker override for initiative-level LLM calls.
  - `--extraction-max-workers`: Optional worker override for extraction segment-level LLM calls.
  - `--log-llm-payload`: Enable logging of full LLM request/response payloads.
- Files/paths: default runs read Markdown from `--markdown-path`, write extraction artifacts under
  `--extraction-output-dir/<run-id>/`, read TEF catalog JSON from `--tef-catalog-dir`, and write
  mapping artifacts under `--output-dir/<run-id>/`. Mapping-only runs read extracted initiatives
  from JSONL.
- Env vars: `OPENROUTER_API_KEY` is required for LLM extraction and mapping.

Outputs:
- Extraction artifacts under `--extraction-output-dir/<run-id>/` unless `--mapping-only`.
- TEF mapping artifacts under `--output-dir/<run-id>/`, including:
  - `00_source/source_manifest.json`
  - `01_inputs/initiatives.jsonl`
  - `02_sector_routes/sector_routes.jsonl`
  - `03_subsector_routes/subsector_routes.jsonl`
  - `04_transition_mappings/transition_mappings.jsonl`
  - `05_final_mappings/final_mappings.jsonl` with copied `source_quote` values
  - `06_review/review_items.jsonl`
  - `07_numeric_facts/initiative_numeric_facts.jsonl` with copied `source_quote`
    values and metric/unit classification metadata
  - `08_tef_groups/tef_grouped_initiatives.jsonl` with copied `source_quote` values
  - `08_tef_groups/tef_metric_rollups.json`
  - `summary.json`
  - `README.md`

Usage (from project root):
- python -m backend.scripts.map_initiatives_to_tef --markdown-path documents --city Krakow
- python -m backend.scripts.map_initiatives_to_tef --mapping-only --extraction-run-dir output/initiative_extraction/five_cities_20260421_002 --city Krakow
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from backend.modules.initiative_extractor import extract_initiatives
from backend.modules.tef_mapper import map_initiatives_to_tef
from backend.utils.config import load_config, resolve_openrouter_api_key
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Extract initiatives and map them to TEF targets as JSON artifacts."
    )
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="Skip extraction and map an existing extraction run or initiatives JSONL.",
    )
    parser.add_argument(
        "--markdown-path",
        default="documents",
        help="Markdown file or directory containing source .md files for extraction.",
    )
    parser.add_argument(
        "--extraction-output-dir",
        default="output/initiative_extraction",
        help="Root directory for extraction artifacts in default full-run mode.",
    )
    parser.add_argument(
        "--extraction-run-id",
        help="Optional extraction run identifier; defaults to --run-id when provided.",
    )
    parser.add_argument(
        "--extraction-run-dir",
        help=(
            "Mapping-only extraction run directory containing "
            "03_deduped/initiative_records.jsonl."
        ),
    )
    parser.add_argument(
        "--initiatives-jsonl",
        help="Mapping-only direct path to deduplicated initiative records JSONL.",
    )
    parser.add_argument(
        "--tef-catalog-dir",
        default="tef_mapping",
        help="Local TEF mapping catalog directory.",
    )
    parser.add_argument(
        "--city",
        action="append",
        help=(
            "Limit extraction and mapping to a selected city; repeat for multiple cities. "
            "Omit for all cities."
        ),
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
        "--extraction-max-workers",
        type=int,
        help="Override segment-level extraction worker count in default full-run mode.",
    )
    parser.add_argument(
        "--log-llm-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="log_llm_payload",
        help="Enable or disable full LLM payload logging.",
    )
    args = parser.parse_args()
    has_mapping_input = bool(args.extraction_run_dir or args.initiatives_jsonl)
    if args.mapping_only and not has_mapping_input:
        parser.error("--mapping-only requires --extraction-run-dir or --initiatives-jsonl.")
    if not args.mapping_only and has_mapping_input:
        parser.error("Use --mapping-only with --extraction-run-dir or --initiatives-jsonl.")
    return args


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    load_dotenv()

    config = load_config(Path(args.config))
    api_key = resolve_openrouter_api_key()
    extraction_run_dir = None
    initiatives_jsonl = None
    mapping_run_id = args.run_id

    if args.mapping_only:
        extraction_run_dir = Path(args.extraction_run_dir) if args.extraction_run_dir else None
        initiatives_jsonl = Path(args.initiatives_jsonl) if args.initiatives_jsonl else None
    else:
        extraction_run_id = args.extraction_run_id or args.run_id
        logger.info(
            "Starting initiative extraction before TEF mapping: markdown_path=%s cities=%s",
            args.markdown_path,
            args.city or "all",
        )
        extraction_result = extract_initiatives(
            markdown_path=Path(args.markdown_path),
            config=config,
            api_key=api_key,
            output_root=Path(args.extraction_output_dir),
            run_id=extraction_run_id,
            selected_cities=args.city,
            max_workers=args.extraction_max_workers,
            log_llm_payload=args.log_llm_payload,
        )
        logger.info(
            "Initiative extraction finished: %s",
            extraction_result.model_dump(mode="json"),
        )
        extraction_run_dir = Path(extraction_result.output_dir)
        mapping_run_id = args.run_id or extraction_result.run_id

    result = map_initiatives_to_tef(
        config=config,
        api_key=api_key,
        tef_catalog_dir=Path(args.tef_catalog_dir),
        output_root=Path(args.output_dir),
        extraction_run_dir=extraction_run_dir,
        initiatives_jsonl=initiatives_jsonl,
        run_id=mapping_run_id,
        selected_cities=args.city,
        limit=args.limit,
        max_workers=args.max_workers,
        log_llm_payload=args.log_llm_payload,
    )
    logger.info("TEF mapping finished: %s", result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
