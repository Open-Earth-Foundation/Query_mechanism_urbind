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


def compact_update_stats(
    stats: dict[str, object] | None,
    *,
    sample_limit: int = 5,
) -> dict[str, object] | None:
    """Return a compact status-friendly summary of update stats."""
    if not isinstance(stats, dict):
        return None
    changed_files = stats.get("changed_files")
    changed_list = changed_files if isinstance(changed_files, list) else []
    deleted_files = stats.get("deleted_files")
    deleted_list = deleted_files if isinstance(deleted_files, list) else []
    return {
        "files_indexed": stats.get("files_indexed", 0),
        "files_changed": stats.get("files_changed", 0),
        "files_unchanged": stats.get("files_unchanged", 0),
        "files_deleted": stats.get("files_deleted", 0),
        "chunks_created": stats.get("chunks_created", 0),
        "table_chunks": stats.get("table_chunks", 0),
        "min_tokens": stats.get("min_tokens", 0),
        "avg_tokens": stats.get("avg_tokens", 0.0),
        "max_tokens": stats.get("max_tokens", 0),
        "dry_run": stats.get("dry_run", False),
        "update_mode": stats.get("update_mode"),
        "changed_file_entries": len(changed_list),
        "deleted_file_entries": len(deleted_list),
        "changed_files_sample": changed_list[:sample_limit],
        "deleted_files_sample": deleted_list[:sample_limit],
    }


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
        "stats": compact_update_stats(stats),
        "job_name": job_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    return payload


__all__ = [
    "compact_update_stats",
    "VectorStoreUpdateStatus",
    "get_update_status_path",
    "now_iso",
    "read_update_status",
    "write_update_status",
]
