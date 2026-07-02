"""Cross-process lock helpers for vector-store update/build writes."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)
DEFAULT_LOCK_WAIT_SECONDS = 1.0
DEFAULT_LOCK_POLL_SECONDS = 0.1
LOCK_FILENAME = "vector_store_update.lock"


def _now_iso() -> str:
    """Return the current UTC timestamp as an ISO string."""
    return datetime.now(timezone.utc).isoformat()


def vector_store_update_lock_path(persist_path: Path) -> Path:
    """Return the canonical lock file path for one vector-store persist root."""
    return persist_path / LOCK_FILENAME


def _read_lock_payload(lock_path: Path) -> dict[str, Any] | None:
    """Read lock metadata if the file currently exists and contains valid JSON."""
    if not lock_path.exists():
        return None
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


@dataclass(frozen=True)
class VectorStoreUpdateLockError(RuntimeError):
    """Raised when another process already owns the vector-store write lock."""

    lock_path: Path
    holder: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        holder_bits: list[str] = []
        if self.holder:
            operation = self.holder.get("operation")
            pid = self.holder.get("pid")
            hostname = self.holder.get("hostname")
            started_at = self.holder.get("started_at")
            if operation:
                holder_bits.append(f"operation={operation}")
            if pid:
                holder_bits.append(f"pid={pid}")
            if hostname:
                holder_bits.append(f"host={hostname}")
            if started_at:
                holder_bits.append(f"started_at={started_at}")
        detail = " ".join(holder_bits) if holder_bits else "holder metadata unavailable"
        RuntimeError.__init__(
            self,
            f"Vector store update already in progress lock_path={self.lock_path} {detail}",
        )


@dataclass(frozen=True)
class VectorStoreUpdateLockHandle:
    """Metadata describing one successful vector-store write-lock acquisition."""

    path: Path
    operation: str
    acquired_after_seconds: float
    waited_for_holder: bool


@contextmanager
def acquire_vector_store_update_lock(
    persist_path: Path,
    *,
    operation: str,
    wait_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_LOCK_POLL_SECONDS,
) -> Iterator[VectorStoreUpdateLockHandle]:
    """Create an exclusive lock file so only one writer updates a persist root."""
    lock_path = vector_store_update_lock_path(persist_path)
    lock_token = uuid.uuid4().hex
    lock_payload = {
        "token": lock_token,
        "operation": operation,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "started_at": _now_iso(),
    }
    deadline = time.monotonic() + max(wait_seconds, 0.0)
    poll_interval = max(poll_seconds, 0.01)
    started_wait = time.monotonic()
    waited_for_holder = False
    persist_path.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            waited_for_holder = True
            holder = _read_lock_payload(lock_path)
            if time.monotonic() >= deadline:
                raise VectorStoreUpdateLockError(lock_path=lock_path, holder=holder) from exc
            time.sleep(poll_interval)
    with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
        json.dump(lock_payload, lock_file, indent=2, ensure_ascii=True)
    logger.info(
        "Acquired vector-store update lock path=%s operation=%s",
        lock_path,
        operation,
    )
    try:
        yield VectorStoreUpdateLockHandle(
            path=lock_path,
            operation=operation,
            acquired_after_seconds=round(time.monotonic() - started_wait, 6),
            waited_for_holder=waited_for_holder,
        )
    finally:
        current_payload = _read_lock_payload(lock_path)
        if current_payload and current_payload.get("token") != lock_token:
            logger.warning(
                "Skipping vector-store update lock cleanup because ownership changed path=%s",
                lock_path,
            )
            return
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return
        logger.info(
            "Released vector-store update lock path=%s operation=%s",
            lock_path,
            operation,
        )


__all__ = [
    "VectorStoreUpdateLockError",
    "VectorStoreUpdateLockHandle",
    "acquire_vector_store_update_lock",
    "vector_store_update_lock_path",
]
