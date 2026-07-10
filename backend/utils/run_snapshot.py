from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from backend.modules.vector_store.manifest import load_manifest
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import AppConfig


def _json_hash(payload: object) -> str:
    """Return a stable SHA256 hash for one JSON-serializable payload."""
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str | None:
    """Return a SHA256 file hash when the file exists."""
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_git(args: list[str], cwd: Path) -> str | None:
    """Run one git command and return trimmed stdout on success."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def build_execution_snapshot(
    *,
    argv: list[str],
    cwd: Path,
    config_path: Path,
    requested_run_id: str,
    resolved_run_id: str,
    invocation_command: str | None,
) -> dict[str, object]:
    """Capture how this run was invoked."""
    snapshot = {
        "argv": list(argv),
        "cwd": str(cwd),
        "config_path": str(config_path),
        "requested_run_id": requested_run_id,
        "resolved_run_id": resolved_run_id,
        "invocation_command": invocation_command,
    }
    snapshot["snapshot_hash"] = _json_hash(snapshot)
    return snapshot


def build_code_snapshot(repo_root: Path) -> dict[str, object]:
    """Capture git state so local code changes are visible in run artifacts."""
    top_level = _run_git(["rev-parse", "--show-toplevel"], repo_root)
    if not top_level:
        snapshot = {
            "repo_root": str(repo_root),
            "git_available": False,
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
            "changed_files": [],
        }
        snapshot["snapshot_hash"] = _json_hash(snapshot)
        return snapshot

    git_root = Path(top_level)
    status_output = _run_git(["status", "--short"], git_root) or ""
    changed_files: list[dict[str, str]] = []
    for line in status_output.splitlines():
        if len(line) < 4:
            continue
        changed_files.append(
            {
                "status": line[:2].strip() or "??",
                "path": line[3:].strip(),
            }
        )

    snapshot = {
        "repo_root": str(git_root),
        "git_available": True,
        "git_commit": _run_git(["rev-parse", "HEAD"], git_root),
        "git_branch": _run_git(["branch", "--show-current"], git_root),
        "git_dirty": bool(changed_files),
        "changed_files": changed_files,
    }
    snapshot["snapshot_hash"] = _json_hash(snapshot)
    return snapshot


def build_config_snapshot(config: AppConfig, config_path: Path) -> dict[str, object]:
    """Capture the resolved application config used for the run."""
    resolved_config = config.model_dump(mode="json")
    snapshot = {
        "config_path": str(config_path),
        "config_file_exists": config_path.exists(),
        "config_file_hash": _file_hash(config_path),
        "resolved_config": resolved_config,
    }
    snapshot["snapshot_hash"] = _json_hash(snapshot)
    return snapshot


def _serialize_update_stats(update_stats: object | None) -> dict[str, object] | None:
    """Return JSON-safe vector-store update stats when an update ran."""
    if update_stats is None:
        return None
    if is_dataclass(update_stats) and not isinstance(update_stats, type):
        payload = asdict(update_stats)
    elif isinstance(update_stats, dict):
        payload = dict(update_stats)
    elif hasattr(update_stats, "__dict__"):
        payload = dict(vars(update_stats))
    else:
        return None
    return json.loads(json.dumps(payload, default=str))


def build_vector_store_snapshot(
    config: AppConfig,
    *,
    update_stats: object | None = None,
    selected_cities: list[str] | None = None,
) -> dict[str, object]:
    """Capture vector-store state that can affect retrieval reproducibility."""
    manifest_path = config.vector_store.index_manifest_path
    manifest: dict[str, Any] = {}
    files: dict[str, object] = {}
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        files_payload = manifest.get("files", {})
        files = files_payload if isinstance(files_payload, dict) else {}
    chunk_count = 0
    for payload in files.values():
        if isinstance(payload, dict):
            chunk_ids = payload.get("chunk_ids")
            if isinstance(chunk_ids, list):
                chunk_count += len(chunk_ids)

    manifest_summary = {
        "index_version": manifest.get("index_version") if manifest else None,
        "created_at": manifest.get("created_at") if manifest else None,
        "updated_at": manifest.get("updated_at") if manifest else None,
        "embedding_model": manifest.get("embedding_model") if manifest else None,
        "embedding_base_url": (
            manifest.get("index_settings", {}).get("embedding_base_url")
            if isinstance(manifest.get("index_settings"), dict)
            else None
        ),
        "embedding_api_key_env": (
            manifest.get("index_settings", {}).get("embedding_api_key_env")
            if isinstance(manifest.get("index_settings"), dict)
            else None
        ),
        "distance_metric": (
            manifest.get("index_settings", {}).get("distance_metric")
            if isinstance(manifest.get("index_settings"), dict)
            else None
        ),
        "embedding_chunk_tokens": manifest.get("embedding_chunk_tokens") if manifest else None,
        "embedding_chunk_overlap_tokens": (
            manifest.get("embedding_chunk_overlap_tokens") if manifest else None
        ),
        "file_count": len(files),
        "chunk_count": chunk_count,
        "index_settings_signature": manifest.get("index_settings_signature") if manifest else None,
    }
    snapshot = {
        "enabled": config.vector_store.enabled,
        "persist_path": str(config.vector_store.chroma_persist_path),
        "collection_name": config.vector_store.chroma_collection_name,
        "distance_metric": config.vector_store.distance_metric,
        "index_manifest_path": str(manifest_path),
        "index_manifest_exists": manifest_path.exists(),
        "index_manifest_hash": _file_hash(manifest_path),
        "resolved_settings": config.vector_store.model_dump(mode="json"),
        "index_settings": manifest.get("index_settings") if manifest else {},
        "manifest_summary": manifest_summary,
    }
    update_payload = _serialize_update_stats(update_stats)
    if update_payload is not None:
        was_dry_run = bool(update_payload.get("dry_run"))
        snapshot["auto_update"] = {
            "checked": True,
            "ran": not was_dry_run,
            "applied": not was_dry_run,
            "dry_run": was_dry_run,
            "update_mode": update_payload.get("update_mode"),
            "trigger": "auto_update_on_run",
            "selected_cities": selected_cities or [],
            "stats": update_payload,
        }
    snapshot["snapshot_hash"] = _json_hash(snapshot)
    return snapshot


def build_documents_snapshot(
    markdown_dir: Path,
    selected_cities: list[str] | None = None,
) -> dict[str, object]:
    """Capture the markdown corpus state used as pipeline input."""
    files: list[dict[str, object]] = []
    selected_city_keys = {
        normalize_city_key(city)
        for city in (selected_cities or [])
        if isinstance(city, str) and city.strip()
    }
    selected_city_files: list[str] = []
    used_markdown_files: list[str] = []
    source_library_file_count = 0
    if markdown_dir.exists():
        for path in sorted(markdown_dir.rglob("*.md")):
            relative_path = path.relative_to(markdown_dir).as_posix()
            stat = path.stat()
            files.append(
                {
                    "path": relative_path,
                    "size_bytes": stat.st_size,
                    "modified_at_ns": stat.st_mtime_ns,
                }
            )
            if relative_path.startswith("source_library/"):
                source_library_file_count += 1
                continue
            city_key = normalize_city_key(path.stem)
            if city_key in selected_city_keys:
                selected_city_files.append(relative_path)
            if not selected_city_keys or city_key in selected_city_keys:
                used_markdown_files.append(relative_path)

    snapshot = {
        "markdown_dir": str(markdown_dir),
        "markdown_dir_exists": markdown_dir.exists(),
        "file_count": len(files),
        "summary": {
            "total_document_files": len(files),
            "selected_city_files": selected_city_files,
            "source_library_files": source_library_file_count,
            "used_markdown_files": used_markdown_files,
        },
        "files": files,
    }
    snapshot["snapshot_hash"] = _json_hash(snapshot)
    return snapshot


__all__ = [
    "build_code_snapshot",
    "build_config_snapshot",
    "build_documents_snapshot",
    "build_execution_snapshot",
    "build_vector_store_snapshot",
]
