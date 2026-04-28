"""
Brief: Print the status of every declared data source and its ingestions.

Reads `backend/data/sources_manifest.yaml` plus the per-ingestion state
files under `.state/sources/` and prints a tree of sources -> ingestions
with last-ingested timestamps and counts.

Inputs:
- CLI args:
  - --manifest: path to manifest yaml (default: backend/data/sources_manifest.yaml).
  - --state-root: directory holding state files (default: .state/sources).

Outputs:
- Stdout report.
- Non-zero exit if the manifest fails to load.

Usage (from project root):
- uv run python -m backend.scripts.sources_status
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.modules.sources.manifest import (
    DEFAULT_MANIFEST_PATH,
    Manifest,
    SourceConfig,
    load_manifest,
)
from backend.modules.sources.state import (
    DEFAULT_STATE_ROOT,
    IngestionState,
    load_state,
)
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print data source manifest status.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sources manifest yaml.",
    )
    parser.add_argument(
        "--state-root",
        default=str(DEFAULT_STATE_ROOT),
        help="Directory holding per-ingestion state files.",
    )
    return parser.parse_args()


def _format_state_summary(state: IngestionState | None) -> str:
    if state is None:
        return "never ingested"
    bits: list[str] = []
    if state.last_ingested_at:
        bits.append(f"last {state.last_ingested_at}")
    if state.source_commit:
        bits.append(f"commit {state.source_commit[:8]}")
    extras = state.model_dump(exclude={"ingestion_id", "source_id", "last_ingested_at", "source_commit"})
    for key in ("file_count", "row_count", "chunk_count", "extracted_entries"):
        if key in extras and extras[key] is not None:
            bits.append(f"{key}={extras[key]}")
    return ", ".join(bits) if bits else "ingested (no metadata)"


def _print_source(source: SourceConfig, state_root: Path) -> None:
    header = f"{source.id} [{source.provider}]"
    if source.provider == "github" and source.repo:
        header += f" {source.repo}"
        if source.pinned_commit:
            header += f" @ {source.pinned_commit[:8]}"
    print(header)

    if not source.ingestions:
        print("  (no ingestions)")
        return

    last_index = len(source.ingestions) - 1
    for index, ingestion in enumerate(source.ingestions):
        prefix = "  └─" if index == last_index else "  ├─"
        state = load_state(ingestion.id, state_root)
        summary = _format_state_summary(state)
        print(f"{prefix} {ingestion.id} [{ingestion.kind}] — {summary}")


def print_status(manifest: Manifest, state_root: Path) -> None:
    print(f"Manifest version: {manifest.version}")
    print(f"Sources: {len(manifest.sources)}\n")

    if not manifest.sources:
        print("(no sources declared)")
        return

    for index, source in enumerate(manifest.sources):
        _print_source(source, state_root)
        if index < len(manifest.sources) - 1:
            print()


def main() -> int:
    setup_logger()
    args = parse_args()

    manifest_path = Path(args.manifest)
    state_root = Path(args.state_root)

    try:
        manifest = load_manifest(manifest_path)
    except FileNotFoundError:
        logger.error("Manifest not found: %s", manifest_path)
        return 1
    except Exception:
        logger.exception("Failed to load manifest %s", manifest_path)
        return 1

    print_status(manifest, state_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
