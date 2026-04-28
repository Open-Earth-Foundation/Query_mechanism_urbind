"""Run a manifest ingestion: resolve upstream, dispatch handler, persist state."""

from __future__ import annotations

import logging
from pathlib import Path

from backend.modules.sources.handlers import IngestionContext, get_handler
from backend.modules.sources.manifest import (
    IngestionConfig,
    Manifest,
    SourceConfig,
)
from backend.modules.sources.state import IngestionState, save_state
from backend.modules.sources.upstream import resolve_upstream, resolved_commit

logger = logging.getLogger(__name__)


def run_ingestion(
    source: SourceConfig,
    ingestion: IngestionConfig,
    *,
    project_root: Path,
    state_root: Path | None = None,
) -> IngestionState:
    """Resolve upstream, run the registered handler, persist state, return it."""
    if not ingestion.handler:
        raise ValueError(
            f"ingestion {ingestion.id!r}: missing `handler` (skipping is not yet supported)"
        )
    handler = get_handler(ingestion.handler)

    with resolve_upstream(source) as upstream_root:
        commit = resolved_commit(upstream_root, source)
        ctx = IngestionContext(
            source=source,
            ingestion=ingestion,
            upstream_root=upstream_root,
            project_root=project_root,
            resolved_commit=commit,
        )
        logger.info(
            "running ingestion %s (kind=%s, handler=%s)",
            ingestion.id,
            ingestion.kind,
            ingestion.handler,
        )
        state = handler(ctx)

    save_state(state, state_root)
    return state


def run_ingestion_by_id(
    manifest: Manifest,
    ingestion_id: str,
    *,
    project_root: Path,
    state_root: Path | None = None,
) -> IngestionState:
    found = manifest.get_ingestion(ingestion_id)
    if not found:
        raise KeyError(f"ingestion {ingestion_id!r} not found in manifest")
    source, ingestion = found
    return run_ingestion(
        source,
        ingestion,
        project_root=project_root,
        state_root=state_root,
    )


__all__ = ["run_ingestion", "run_ingestion_by_id"]
