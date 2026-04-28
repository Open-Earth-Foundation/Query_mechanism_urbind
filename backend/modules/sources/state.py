"""Per-ingestion state files: read/write helpers.

State lives at `.state/sources/{ingestion_id}.json` by default and is
gitignored.  Each ingestion handler is free to attach extra fields
(file lists, hashes, counts) — the model allows them.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

DEFAULT_STATE_ROOT = Path(".state/sources")


class IngestionState(BaseModel):
    """State emitted by an ingestion run.

    Handlers may attach kind-specific fields (e.g. file lists, row counts,
    chunk counts).  Required fields are kept minimal so consumers can
    always read identity + freshness without parsing handler-specific shape.
    """

    model_config = ConfigDict(extra="allow")

    ingestion_id: str
    source_id: str
    last_ingested_at: str | None = None
    source_commit: str | None = None


def state_path(ingestion_id: str, root: Path | None = None) -> Path:
    """Return the conventional state file path for an ingestion."""
    return (root or DEFAULT_STATE_ROOT) / f"{ingestion_id}.json"


def load_state(ingestion_id: str, root: Path | None = None) -> IngestionState | None:
    """Load state for an ingestion, or return None if no state file exists."""
    path = state_path(ingestion_id, root)
    if not path.exists():
        return None
    return IngestionState.model_validate_json(path.read_text(encoding="utf-8"))


def save_state(state: IngestionState, root: Path | None = None) -> Path:
    """Persist state and return the path written."""
    path = state_path(state.ingestion_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_STATE_ROOT",
    "IngestionState",
    "load_state",
    "save_state",
    "state_path",
]
