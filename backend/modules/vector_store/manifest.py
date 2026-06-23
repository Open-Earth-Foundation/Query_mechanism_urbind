from __future__ import annotations

import hashlib
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
    payload.setdefault("files", {})
    return payload


def _manifest_write_audit_enabled() -> bool:
    """Return true when manifest-write audit artifacts should be persisted."""
    raw_value = os.getenv("VECTOR_STORE_MANIFEST_WRITE_AUDIT_ENABLED")
    if raw_value is not None:
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return "PYTEST_CURRENT_TEST" not in os.environ


def _append_manifest_write_audit(
    *,
    path: Path,
    manifest: dict[str, Any],
    file_count: int,
    chunk_count: int,
    reason: str | None,
    docs_dir: Path | None,
    metadata: dict[str, Any] | None,
) -> None:
    """Append one manifest-write audit record under output/system artifacts."""
    if not _manifest_write_audit_enabled():
        return
    runs_dir = Path(os.getenv("RUNS_DIR", "output"))
    audit_dir = runs_dir / "system" / "vector_store_manifest_writes"
    audit_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_iso()
    payload = {
        "timestamp": timestamp,
        "manifest_path": str(path),
        "manifest_resolved_path": str(path.resolve()),
        "file_count": file_count,
        "chunk_count": chunk_count,
        "updated_at": manifest.get("updated_at"),
        "reason": reason,
        "docs_dir": str(docs_dir) if docs_dir is not None else None,
        "docs_dir_resolved": str(docs_dir.resolve()) if docs_dir is not None else None,
        "cwd": str(Path.cwd()),
        "metadata": metadata or {},
    }
    latest_path = audit_dir / "latest.json"
    history_path = audit_dir / "history.jsonl"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def save_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    reason: str | None = None,
    docs_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write manifest JSON to disk."""
    files_payload = manifest.get("files", {})
    files = files_payload if isinstance(files_payload, dict) else {}
    chunk_count = 0
    for payload in files.values():
        if isinstance(payload, dict):
            chunk_ids = payload.get("chunk_ids")
            if isinstance(chunk_ids, list):
                chunk_count += len(chunk_ids)

    log_fn = logger.warning if not files else logger.info
    log_fn(
        "Saving vector-store manifest path=%s file_count=%d chunk_count=%d updated_at=%s "
        "reason=%s docs_dir=%s metadata=%s",
        path,
        len(files),
        chunk_count,
        manifest.get("updated_at"),
        reason,
        docs_dir,
        metadata or {},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _append_manifest_write_audit(
        path=path,
        manifest=manifest,
        file_count=len(files),
        chunk_count=chunk_count,
        reason=reason,
        docs_dir=docs_dir,
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
