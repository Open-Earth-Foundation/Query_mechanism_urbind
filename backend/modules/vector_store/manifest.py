from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write manifest JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
