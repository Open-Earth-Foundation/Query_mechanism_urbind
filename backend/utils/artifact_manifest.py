"""Helpers for reading and resolving run artifact manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.utils.json_io import read_json_object


def read_artifact_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Return the run-local manifest payload when present and valid."""
    return read_json_object(run_dir / "manifest.json")


def resolve_manifest_alias(run_dir: Path, alias: str) -> Path | None:
    """Resolve one artifact alias from ``manifest.json`` to a run-local path."""
    manifest = read_artifact_manifest(run_dir)
    if manifest is None:
        return None
    aliases = manifest.get("aliases")
    if not isinstance(aliases, dict):
        return None
    alias_payload = aliases.get(alias)
    if not isinstance(alias_payload, dict):
        return None
    raw_path: Any = alias_payload.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    resolved_run_dir = run_dir.resolve()
    candidate = Path(raw_path)
    resolved = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (resolved_run_dir / candidate).resolve(strict=False)
    )
    if resolved_run_dir not in resolved.parents and resolved != resolved_run_dir:
        return None
    return resolved if resolved.exists() else None

__all__ = ["read_artifact_manifest", "resolve_manifest_alias"]
