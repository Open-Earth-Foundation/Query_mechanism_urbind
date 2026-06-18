"""Shared vector-store update status file helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from backend.utils.config import AppConfig

VectorStoreUpdateStatus = Literal[
    "pending",
    "skipped",
    "checking",
    "stale",
    "running",
    "completed",
    "failed",
]


def now_iso() -> str:
    """Return the current UTC timestamp as an ISO string."""
    return datetime.now(timezone.utc).isoformat()


def get_update_status_path(config: AppConfig) -> Path:
    """Return the status file path colocated with the Chroma index."""
    return config.vector_store.chroma_persist_path / "update_status.json"


def read_update_status(path: Path) -> dict[str, object] | None:
    """Read a vector-store update status file if it exists and is valid JSON."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_update_status(
    path: Path,
    *,
    status: VectorStoreUpdateStatus,
    trigger: str,
    update_mode: str,
    message: str,
    started_at: str | None = None,
    completed_at: str | None = None,
    error: str | None = None,
    stats: dict[str, object] | None = None,
    job_name: str | None = None,
) -> dict[str, object]:
    """Write and return one JSON-safe vector-store update status payload."""
    timestamp = now_iso()
    payload: dict[str, object] = {
        "status": status,
        "trigger": trigger,
        "update_mode": update_mode,
        "message": message,
        "started_at": started_at,
        "completed_at": completed_at,
        "updated_at": timestamp,
        "error": error,
        "stats": stats,
        "job_name": job_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    return payload


__all__ = [
    "VectorStoreUpdateStatus",
    "get_update_status_path",
    "now_iso",
    "read_update_status",
    "write_update_status",
]
