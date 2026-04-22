"""
Brief: Build numeric rollup artifacts for an existing TEF mapping run.

Inputs:
- CLI args:
  - `--tef-run-dir`: TEF mapping run directory containing
    `05_final_mappings/final_mappings.jsonl`.
  - `--extraction-run-dir`: Optional initiative extraction run directory containing
    `03_deduped/initiative_records.jsonl`.
  - `--initiative-records-jsonl`: Optional direct path to pipeline initiative records JSONL;
    overrides `--extraction-run-dir` and TEF run input discovery.
- Files/paths: reads TEF final mappings and pipeline initiative records. Initiative records
  must wrap the clean canonical v1 initiative object in the `initiative` field.
- Env vars: none.

Outputs:
- `07_numeric_facts/initiative_numeric_facts.jsonl`
- `08_tef_groups/tef_grouped_initiatives.jsonl`
- `08_tef_groups/tef_metric_rollups.json`
- Logs a short count summary.

Usage (from project root):
- python -m backend.scripts.rollup_tef_numeric_facts --tef-run-dir output/tef_mapping/<run_id> --extraction-run-dir output/initiative_extraction/<run_id>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.modules.tef_mapper.numeric_rollup import rollup_existing_tef_run
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Build numeric rollup artifacts for an existing TEF mapping run."
    )
    parser.add_argument(
        "--tef-run-dir",
        required=True,
        help="TEF mapping run directory containing 05_final_mappings/final_mappings.jsonl.",
    )
    parser.add_argument(
        "--extraction-run-dir",
        help="Initiative extraction run directory containing 03_deduped/initiative_records.jsonl.",
    )
    parser.add_argument(
        "--initiative-records-jsonl",
        help="Direct path to pipeline initiative records JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()
    summary = rollup_existing_tef_run(
        tef_run_dir=Path(args.tef_run_dir),
        extraction_run_dir=Path(args.extraction_run_dir) if args.extraction_run_dir else None,
        initiative_records_jsonl=(
            Path(args.initiative_records_jsonl) if args.initiative_records_jsonl else None
        ),
    )
    logger.info("TEF numeric rollup finished: %s", summary)


if __name__ == "__main__":
    main()
