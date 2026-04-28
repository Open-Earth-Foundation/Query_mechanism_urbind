"""
Brief: Run one or more declared ingestions from the data sources manifest.

Resolves the upstream source (clones GitHub repos to temp), dispatches
the registered handler, and persists per-ingestion state to
`.state/sources/{ingestion_id}.json`.

Inputs:
- CLI args:
  - ingestion_ids: positional list of ingestion ids to run; if empty, runs all.
  - --manifest: path to manifest yaml (default: backend/data/sources_manifest.yaml).
  - --state-root: directory holding state files (default: .state/sources).
  - --project-root: project root for output resolution (default: cwd).

Outputs:
- Output artefacts produced by handlers (markdown, parquet, vector
  collections, allowlist yaml).
- Per-ingestion state files.
- Stdout summary.
- Non-zero exit on any handler failure.

Usage (from project root):
- uv run python -m backend.scripts.sources_ingest urbind_tier1_cities
- uv run python -m backend.scripts.sources_ingest          # run every ingestion
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.modules.sources.manifest import (
    DEFAULT_MANIFEST_PATH,
    Manifest,
    load_manifest,
)
from backend.modules.sources.runner import run_ingestion
from backend.modules.sources.state import DEFAULT_STATE_ROOT
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manifest-declared ingestions.")
    parser.add_argument(
        "ingestion_ids",
        nargs="*",
        help="Ingestion ids to run; runs all declared ingestions if omitted.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sources manifest yaml.",
    )
    parser.add_argument(
        "--state-root",
        default=str(DEFAULT_STATE_ROOT),
        help="Directory to write per-ingestion state files.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Project root for output path resolution.",
    )
    return parser.parse_args()


def _select_ingestions(manifest: Manifest, ids: list[str]) -> list[tuple]:
    if not ids:
        return manifest.iter_ingestions()
    selected: list[tuple] = []
    for ingestion_id in ids:
        found = manifest.get_ingestion(ingestion_id)
        if not found:
            raise KeyError(f"ingestion {ingestion_id!r} not found in manifest")
        selected.append(found)
    return selected


def main() -> int:
    setup_logger()
    args = parse_args()

    manifest_path = Path(args.manifest)
    state_root = Path(args.state_root)
    project_root = Path(args.project_root).resolve()

    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError:
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    try:
        targets = _select_ingestions(manifest, args.ingestion_ids)
    except KeyError as exc:
        logger.error("%s", exc)
        return 1

    if not targets:
        logger.info("Nothing to do — no ingestions declared.")
        return 0

    failures = 0
    for source, ingestion in targets:
        try:
            state = run_ingestion(
                source,
                ingestion,
                project_root=project_root,
                state_root=state_root,
            )
        except Exception:
            logger.exception("ingestion %s failed", ingestion.id)
            failures += 1
            continue

        extras = state.model_dump(
            exclude={"ingestion_id", "source_id", "last_ingested_at", "source_commit"}
        )
        summary_bits = [f"last={state.last_ingested_at}"]
        if state.source_commit:
            summary_bits.append(f"commit={state.source_commit[:8]}")
        for key in ("file_count", "row_count", "chunk_count", "extracted_entries"):
            if key in extras and extras[key] is not None:
                summary_bits.append(f"{key}={extras[key]}")
        print(f"✓ {ingestion.id}: {', '.join(summary_bits)}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
