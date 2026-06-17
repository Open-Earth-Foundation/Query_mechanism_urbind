"""Vector-store startup warm-up state and execution."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Literal

from backend.modules.vector_store.indexer import update_markdown_index
from backend.utils.config import AppConfig
from backend.utils.run_snapshot import build_vector_store_snapshot

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
        self._stats: dict[str, object] | None = None
        self._latest_artifact: str | None = None
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
            self._latest_artifact = None
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
        self._latest_artifact = None

    def _run(self, *, config: AppConfig, docs_dir: Path) -> None:
        """Refresh the vector store and update status for API consumers."""
        logger.info("Vector store startup warm-up started docs_dir=%s", docs_dir)
        started_at = datetime.now(timezone.utc)
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
                self._latest_artifact = self._write_artifact(
                    config=config,
                    docs_dir=docs_dir,
                    status="failed",
                    started_at=started_at,
                    completed_at=self._completed_at,
                    message=self._message,
                    error=self._error,
                    stats=None,
                )
            return

        stats_payload = self._stats_payload(stats)
        logger.info(
            "Vector store startup warm-up completed changed=%d unchanged=%d deleted=%d chunks=%d",
            stats.files_changed,
            stats.files_unchanged,
            stats.files_deleted,
            stats.chunks_created,
        )
        completed_at = datetime.now(timezone.utc)
        latest_artifact = self._write_artifact(
            config=config,
            docs_dir=docs_dir,
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            message="Vector store is up to date.",
            error=None,
            stats=stats,
        )
        with self._lock:
            self._status = "completed"
            self._completed_at = completed_at
            self._message = "Vector store is up to date."
            self._error = None
            self._stats = stats_payload
            self._latest_artifact = latest_artifact

    def _stats_payload(self, stats: object) -> dict[str, object]:
        """Return JSON-safe warm-up stats for status responses and artifacts."""
        return {
            "files_indexed": getattr(stats, "files_indexed", 0),
            "files_changed": getattr(stats, "files_changed", 0),
            "files_unchanged": getattr(stats, "files_unchanged", 0),
            "files_deleted": getattr(stats, "files_deleted", 0),
            "chunks_created": getattr(stats, "chunks_created", 0),
            "table_chunks": getattr(stats, "table_chunks", 0),
            "min_tokens": getattr(stats, "min_tokens", 0),
            "avg_tokens": getattr(stats, "avg_tokens", 0.0),
            "max_tokens": getattr(stats, "max_tokens", 0),
            "dry_run": getattr(stats, "dry_run", False),
            "update_mode": getattr(stats, "update_mode", None),
            "changed_files": list(getattr(stats, "changed_files", [])),
            "deleted_files": list(getattr(stats, "deleted_files", [])),
        }

    def _write_artifact(
        self,
        *,
        config: AppConfig,
        docs_dir: Path,
        status: VectorStoreWarmupStatus,
        started_at: datetime,
        completed_at: datetime,
        message: str,
        error: str | None,
        stats: object | None,
    ) -> str:
        """Persist startup warm-up diagnostics outside any user run directory."""
        artifact_dir = config.runs_dir / "system" / "vector_store_warmup"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        timestamp = completed_at.strftime("%Y%m%dT%H%M%SZ")
        timestamped_path = artifact_dir / f"{timestamp}.json"
        latest_path = artifact_dir / "latest.json"
        latest_label = Path("system") / "vector_store_warmup" / "latest.json"
        payload = {
            "event_type": "vector_store_startup_warmup",
            "trigger": "api_startup",
            "status": status,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "message": message,
            "error": error,
            "docs_dir": str(docs_dir),
            "stats": self._stats_payload(stats) if stats is not None else None,
            "vector_store_snapshot": build_vector_store_snapshot(
                config,
                update_stats=stats,
                selected_cities=None,
            ),
        }
        timestamped_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, default=str),
            encoding="utf-8",
        )
        latest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, default=str),
            encoding="utf-8",
        )
        return latest_label.as_posix()

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
                "latest_artifact": self._latest_artifact,
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
