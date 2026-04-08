from __future__ import annotations

import hashlib


def compute_content_hash(content: str) -> str:
    """Compute a stable content hash for deterministic chunk ids."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_chunk_id(source_path: str, chunk_index: int, content_hash: str) -> str:
    """Build a deterministic chunk identifier from source path, index, and content."""
    payload = f"{source_path}:{chunk_index}:{content_hash}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"chunk_{digest[:24]}"


__all__ = ["build_chunk_id", "compute_content_hash"]
