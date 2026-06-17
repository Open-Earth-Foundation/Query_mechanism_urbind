"""Vector-store startup warm-up state and execution."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Literal

from backend.modules.vector_store.indexer import update_markdown_index
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)

VectorStoreWarmupStatus = Literal["pending", "skipped", "running", "completed", "failed"]


class VectorStoreWarmup:
    """Run one non-blocking vector-store refresh and expose compact status."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._thread: Thread | None = None
        self._status: VectorStoreWarmupStatus = "pending"
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None
        self._message = "Vector store warm-up has not started."
        self._error: str | None = None
        self._stats: dict[str, int] | None = None
        self._enabled = False
        self._auto_update_on_run = False

    def start(self, *, config: AppConfig, docs_dir: Path) -> None:
        """Start the warm-up once when vector auto-update is enabled."""
        with self._lock:
            self._enabled = config.vector_store.enabled
            self._auto_update_on_run = config.vector_store.auto_update_on_run
            if not config.vector_store.enabled:
                self._skip_locked("Vector store is disabled.")
                return
            if not config.vector_store.auto_update_on_run:
                self._skip_locked("Vector auto-update is disabled.")
                return
            if self._thread is not None:
                return
            self._status = "running"
            self._started_at = datetime.now(timezone.utc)
            self._completed_at = None
            self._message = "Refreshing vector store before accepting new runs."
            self._error = None
            self._stats = None
            self._thread = Thread(
                target=self._run,
                kwargs={"config": config, "docs_dir": docs_dir},
                name="vector-warmup",
                daemon=True,
            )
            self._thread.start()

    def _skip_locked(self, message: str) -> None:
        """Mark the warm-up as skipped while the caller holds the lock."""
        self._status = "skipped"
        self._started_at = None
        self._completed_at = datetime.now(timezone.utc)
        self._message = message
        self._error = None
        self._stats = None

    def _run(self, *, config: AppConfig, docs_dir: Path) -> None:
        """Refresh the vector store and update status for API consumers."""
        logger.info("Vector store startup warm-up started docs_dir=%s", docs_dir)
        try:
            stats = update_markdown_index(
                config=config,
                docs_dir=docs_dir,
                selected_cities=None,
                dry_run=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vector store startup warm-up failed")
            with self._lock:
                self._status = "failed"
                self._completed_at = datetime.now(timezone.utc)
                self._message = "Vector store startup warm-up failed."
                self._error = str(exc)
            return

        stats_payload = {
            "files_changed": stats.files_changed,
            "files_unchanged": stats.files_unchanged,
            "files_deleted": stats.files_deleted,
            "chunks_created": stats.chunks_created,
        }
        logger.info(
            "Vector store startup warm-up completed changed=%d unchanged=%d deleted=%d chunks=%d",
            stats.files_changed,
            stats.files_unchanged,
            stats.files_deleted,
            stats.chunks_created,
        )
        with self._lock:
            self._status = "completed"
            self._completed_at = datetime.now(timezone.utc)
            self._message = "Vector store is up to date."
            self._error = None
            self._stats = stats_payload

    def snapshot(self) -> dict[str, object]:
        """Return a thread-safe status payload for API responses."""
        with self._lock:
            return {
                "status": self._status,
                "enabled": self._enabled,
                "auto_update_on_run": self._auto_update_on_run,
                "started_at": self._started_at,
                "completed_at": self._completed_at,
                "message": self._message,
                "error": self._error,
                "stats": dict(self._stats) if self._stats is not None else None,
            }

    def is_blocking_runs(self) -> bool:
        """Return True while startup warm-up should block new run submissions."""
        with self._lock:
            return self._status == "running"

    def shutdown(self, *, wait: bool = False) -> None:
        """Optionally wait for warm-up completion during API shutdown."""
        thread = self._thread
        if wait and thread is not None:
            thread.join()


__all__ = ["VectorStoreWarmup", "VectorStoreWarmupStatus"]
