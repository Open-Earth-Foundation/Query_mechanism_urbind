from __future__ import annotations

import hashlib
import inspect
import logging
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def compute_file_hash(content: str) -> str:
    """Compute stable file hash for manifest comparison."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_content_hash(content: str) -> str:
    """Compute stable content hash for chunk identity."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_chunk_id(source_path: str, chunk_index: int, content_hash: str) -> str:
    """Build deterministic chunk identifier."""
    payload = f"{source_path}:{chunk_index}:{content_hash}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"chunk_{digest[:24]}"


def default_manifest() -> dict[str, Any]:
    """Return default manifest payload for a new index."""
    timestamp = now_iso()
    return {
        "index_version": 1,
        "created_at": timestamp,
        "updated_at": timestamp,
        "embedding_model": "",
        "embedding_chunk_tokens": 0,
        "embedding_chunk_overlap_tokens": 0,
        "index_settings": {},
        "index_settings_signature": "",
        "files": {},
    }


def load_manifest(path: Path) -> dict[str, Any]:
    """Load manifest from disk or return default when absent."""
    if not path.exists():
        return default_manifest()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return default_manifest()
    files_payload = payload.setdefault("files", {})
    if isinstance(files_payload, dict) and not files_payload:
        metadata_keys = {
            key
            for key in (
                "created_at",
                "updated_at",
                "embedding_model",
                "embedding_chunk_tokens",
                "embedding_chunk_overlap_tokens",
                "index_settings",
                "index_settings_signature",
            )
            if payload.get(key)
        }
        if metadata_keys:
            logger.warning(
                "Loaded vector-store manifest with empty files payload path=%s metadata_keys=%s",
                path,
                sorted(metadata_keys),
            )
    return payload


def _caller_metadata() -> dict[str, object]:
    """Return the first external caller frame outside this module."""
    module_path = Path(__file__).resolve()
    cwd = Path.cwd().resolve()
    for frame in inspect.stack()[1:]:
        frame_path = Path(frame.filename).resolve()
        if frame_path == module_path:
            continue
        try:
            caller_path = frame_path.relative_to(cwd).as_posix()
        except ValueError:
            caller_path = str(frame_path)
        return {
            "caller_file": caller_path,
            "caller_function": frame.function,
            "caller_line": frame.lineno,
        }
    return {
        "caller_file": str(module_path),
        "caller_function": "unknown",
        "caller_line": 0,
    }


def _write_manifest_audit(
    *,
    path: Path,
    manifest: dict[str, Any],
    file_count: int,
    chunk_count: int,
    reason: str,
    audit_dir: Path | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Append one structured manifest-write audit record under output/system."""
    cwd = Path.cwd()
    resolved_audit_dir = audit_dir or (cwd / "output" / "system" / "vector_store_manifest_writes")
    resolved_audit_dir.mkdir(parents=True, exist_ok=True)
    caller = _caller_metadata()
    try:
        manifest_resolved_path = str(path.resolve(strict=False))
    except OSError:
        manifest_resolved_path = str(path)
    payload = {
        "timestamp": now_iso(),
        "manifest_path": str(path),
        "manifest_resolved_path": manifest_resolved_path,
        "file_count": file_count,
        "chunk_count": chunk_count,
        "updated_at": manifest.get("updated_at"),
        "reason": reason,
        "cwd": str(cwd),
        "pid": os.getpid(),
        "audit_dir": str(resolved_audit_dir),
        **caller,
        "metadata": metadata or {},
    }
    history_path = resolved_audit_dir / "history.jsonl"
    latest_path = resolved_audit_dir / "latest.json"
    with history_path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
    latest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    return payload


def save_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    reason: str | None = None,
    audit_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write manifest JSON to disk and audit who triggered the write."""
    files_payload = manifest.get("files", {})
    files = files_payload if isinstance(files_payload, dict) else {}
    chunk_count = 0
    for payload in files.values():
        if isinstance(payload, dict):
            chunk_ids = payload.get("chunk_ids")
            if isinstance(chunk_ids, list):
                chunk_count += len(chunk_ids)
    caller = _caller_metadata()
    write_reason = reason or str(caller.get("caller_function") or "unknown")

    log_fn = logger.warning if not files else logger.info
    log_fn(
        "Saving vector-store manifest path=%s file_count=%d chunk_count=%d "
        "updated_at=%s reason=%s caller=%s:%s",
        path,
        len(files),
        chunk_count,
        manifest.get("updated_at"),
        write_reason,
        caller.get("caller_file"),
        caller.get("caller_line"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_manifest_audit(
        path=path,
        manifest=manifest,
        file_count=len(files),
        chunk_count=chunk_count,
        reason=write_reason,
        audit_dir=audit_dir,
        metadata=metadata,
    )


def mark_manifest_updated(
    manifest: dict[str, Any],
    embedding_model: str,
    embedding_chunk_tokens: int,
    embedding_chunk_overlap_tokens: int,
) -> dict[str, Any]:
    """Refresh manifest-level timestamps and embedding config fields."""
    if "created_at" not in manifest:
        manifest["created_at"] = now_iso()
    manifest["updated_at"] = now_iso()
    manifest["embedding_model"] = embedding_model
    manifest["embedding_chunk_tokens"] = embedding_chunk_tokens
    manifest["embedding_chunk_overlap_tokens"] = embedding_chunk_overlap_tokens
    return manifest


__all__ = [
    "build_chunk_id",
    "compute_content_hash",
    "compute_file_hash",
    "default_manifest",
    "load_manifest",
    "mark_manifest_updated",
    "save_manifest",
]
