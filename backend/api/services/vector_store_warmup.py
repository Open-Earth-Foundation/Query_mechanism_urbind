"""Vector-store startup and run-time update coordination."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from backend.api.services.kubernetes_vector_job import (
    KubernetesJobError,
    create_vector_store_update_job,
)
from backend.modules.vector_store.indexer import update_markdown_index
from backend.modules.vector_store.update_status import (
    VectorStoreUpdateStatus,
    get_update_status_path,
    read_update_status,
    write_update_status,
)
from backend.utils.config import AppConfig
from backend.utils.run_snapshot import build_vector_store_snapshot

logger = logging.getLogger(__name__)
DEFAULT_UPDATE_TIMEOUT_SECONDS = 7200


class VectorStoreWarmup:
    """Coordinate vector-store freshness without forcing production indexing in API pods."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._thread: Thread | None = None
        self._status: VectorStoreUpdateStatus = "pending"
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None
        self._message = "Vector store warm-up has not started."
        self._error: str | None = None
        self._stats: dict[str, object] | None = None
        self._latest_artifact: str | None = None
        self._job_name: str | None = None
        self._enabled = False
        self._auto_update_on_run = False
        self._update_mode = "local_process"
        self._status_path: Path | None = None

    def start(self, *, config: AppConfig, docs_dir: Path) -> None:
        """Start one startup freshness check when vector retrieval is enabled."""
        self._configure(config)
        with self._lock:
            if not config.vector_store.enabled:
                self._skip_locked("Vector store is disabled.")
                self._write_skipped_status(config=config, trigger="startup")
                return
            if self._thread is not None:
                return
            self._mark_checking_locked("api_startup")
            self._thread = Thread(
                target=self._run_for_trigger,
                kwargs={
                    "config": config,
                    "docs_dir": docs_dir,
                    "trigger": "startup",
                    "allow_updates": config.vector_store.auto_update_on_run,
                },
                name="vector-store-update-coordinator",
                daemon=True,
            )
            self._thread.start()

    def ensure_ready_for_run(self, *, config: AppConfig, docs_dir: Path) -> str | None:
        """Return a blocking reason after starting an update if the index is stale."""
        self._configure(config)
        if not config.vector_store.enabled:
            return None

        with self._lock:
            self._sync_from_status_file_locked()
            if self._status in {"checking", "running"}:
                return self._message
            if not config.vector_store.auto_update_on_run and self._status in {"stale", "failed"}:
                return self._message
            self._mark_checking_locked("api_run")

        self._run_for_trigger(
            config=config,
            docs_dir=docs_dir,
            trigger="run",
            allow_updates=config.vector_store.auto_update_on_run,
        )
        with self._lock:
            if self._status in {"checking", "running", "stale", "failed"}:
                return self._message
        return None

    def _configure(self, config: AppConfig) -> None:
        """Refresh config-derived flags used in status responses."""
        with self._lock:
            self._enabled = config.vector_store.enabled
            self._auto_update_on_run = config.vector_store.auto_update_on_run
            self._update_mode = config.vector_store.update_mode
            self._status_path = get_update_status_path(config)

    def _skip_locked(self, message: str) -> None:
        """Mark the warm-up as skipped while the caller holds the lock."""
        self._status = "skipped"
        self._started_at = None
        self._completed_at = datetime.now(timezone.utc)
        self._message = message
        self._error = None
        self._stats = None
        self._job_name = None
        self._latest_artifact = None

    def _write_skipped_status(self, *, config: AppConfig, trigger: str) -> None:
        """Persist the canonical skipped status for disabled startup flows."""
        write_update_status(
            get_update_status_path(config),
            status="skipped",
            trigger=trigger,
            update_mode=config.vector_store.update_mode,
            message=self._message,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _mark_checking_locked(self, trigger: str) -> None:
        """Mark this coordinator as checking freshness."""
        self._status = "checking"
        self._started_at = datetime.now(timezone.utc)
        self._completed_at = None
        self._message = "Checking vector store freshness."
        self._error = None
        self._stats = None
        self._job_name = None
        self._latest_artifact = None
        if self._status_path is not None:
            write_update_status(
                self._status_path,
                status="checking",
                trigger=trigger,
                update_mode=self._update_mode,
                message=self._message,
                started_at=self._started_at.isoformat(),
            )

    def _run_for_trigger(
        self,
        *,
        config: AppConfig,
        docs_dir: Path,
        trigger: str,
        allow_updates: bool,
    ) -> None:
        """Check freshness and either update locally or start a Kubernetes Job."""
        logger.info(
            "Vector store refresh check started trigger=%s mode=%s docs_dir=%s",
            trigger,
            config.vector_store.update_mode,
            docs_dir,
        )
        started_at = datetime.now(timezone.utc)
        try:
            dry_run_stats = update_markdown_index(
                config=config,
                docs_dir=docs_dir,
                selected_cities=None,
                dry_run=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(
                config=config,
                docs_dir=docs_dir,
                trigger=trigger,
                started_at=started_at,
                message="Vector store freshness check failed.",
                error=str(exc),
            )
            logger.exception("Vector store freshness check failed")
            return

        dry_run_payload = self._stats_payload(dry_run_stats)
        added_files = self._changed_files_by_status(dry_run_payload, "added")
        modified_files = self._changed_files_by_status(dry_run_payload, "modified")
        deleted_files = self._deleted_file_sources(dry_run_payload)
        if not self._needs_update(dry_run_stats):
            logger.info(
                "Vector store refresh check result trigger=%s stale=false added=%d "
                "changed=%d deleted=%d unchanged=%d",
                trigger,
                len(added_files),
                len(modified_files),
                len(deleted_files),
                int(dry_run_payload.get("files_unchanged", 0)),
            )
            completed_at = datetime.now(timezone.utc)
            self._complete(
                config=config,
                docs_dir=docs_dir,
                trigger=trigger,
                started_at=started_at,
                completed_at=completed_at,
                message="Vector store is up to date.",
                stats=dry_run_stats,
            )
            return

        logger.info(
            "Vector store refresh check result trigger=%s stale=true added=%d "
            "changed=%d deleted=%d unchanged=%d update_mode=%s auto_update_on_run=%s",
            trigger,
            len(added_files),
            len(modified_files),
            len(deleted_files),
            int(dry_run_payload.get("files_unchanged", 0)),
            str(dry_run_payload.get("update_mode")),
            allow_updates,
        )
        logger.info(
            "Vector store affected files trigger=%s added=%s changed=%s deleted=%s",
            trigger,
            added_files,
            modified_files,
            deleted_files,
        )
        if not allow_updates:
            self._mark_manual_maintenance_required(
                config=config,
                docs_dir=docs_dir,
                trigger=trigger,
                started_at=started_at,
                stats_payload=dry_run_payload,
            )
            return
        if config.vector_store.update_mode == "kubernetes_job":
            self._start_kubernetes_update(
                config=config,
                docs_dir=docs_dir,
                trigger=trigger,
                started_at=started_at,
                stats_payload=dry_run_payload,
            )
            return

        self._run_local_update(
            config=config,
            docs_dir=docs_dir,
            trigger=trigger,
            started_at=started_at,
        )

    def _run_local_update(
        self,
        *,
        config: AppConfig,
        docs_dir: Path,
        trigger: str,
        started_at: datetime,
    ) -> None:
        """Run the actual vector update inside this process for local development."""
        status_path = get_update_status_path(config)
        write_update_status(
            status_path,
            status="running",
            trigger=trigger,
            update_mode=config.vector_store.update_mode,
            message="Refreshing vector store in the API process.",
            started_at=started_at.isoformat(),
        )
        with self._lock:
            self._status = "running"
            self._message = "Refreshing vector store in the API process."
        logger.info(
            "Vector store local refresh started trigger=%s docs_dir=%s",
            trigger,
            docs_dir,
        )
        try:
            stats = update_markdown_index(
                config=config,
                docs_dir=docs_dir,
                selected_cities=None,
                dry_run=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vector store local update failed")
            self._fail(
                config=config,
                docs_dir=docs_dir,
                trigger=trigger,
                started_at=started_at,
                message="Vector store update failed.",
                error=str(exc),
            )
            return
        logger.info(
            "Vector store refresh completed trigger=%s status=completed indexed=%d "
            "added=%d changed=%d deleted=%d unchanged=%d chunks=%d",
            trigger,
            self._count_changed_files_by_status(stats, "indexed"),
            self._count_changed_files_by_status(stats, "added"),
            self._count_changed_files_by_status(stats, "modified"),
            stats.files_deleted,
            stats.files_unchanged,
            stats.chunks_created,
        )
        self._complete(
            config=config,
            docs_dir=docs_dir,
            trigger=trigger,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            message="Vector store is up to date.",
            stats=stats,
            )

    def _start_kubernetes_update(
        self,
        *,
        config: AppConfig,
        docs_dir: Path,
        trigger: str,
        started_at: datetime,
        stats_payload: dict[str, object],
    ) -> None:
        """Create a Kubernetes updater Job and keep runs blocked until it completes."""
        status_path = get_update_status_path(config)
        try:
            job_name = create_vector_store_update_job(trigger=trigger)
        except KubernetesJobError as exc:
            logger.exception("Vector store updater Job creation failed")
            self._fail(
                config=config,
                docs_dir=docs_dir,
                trigger=trigger,
                started_at=started_at,
                message="Vector store is stale and updater Job creation failed.",
                error=str(exc),
            )
            return
        logger.info(
            "Vector store updater Job started trigger=%s job_name=%s added=%d changed=%d deleted=%d",
            trigger,
            job_name,
            len(self._changed_files_by_status(stats_payload, "added")),
            len(self._changed_files_by_status(stats_payload, "modified")),
            len(self._deleted_file_sources(stats_payload)),
        )

        write_update_status(
            status_path,
            status="running",
            trigger=trigger,
            update_mode=config.vector_store.update_mode,
            message="Vector store is stale; updater Job is running.",
            started_at=started_at.isoformat(),
            stats=stats_payload,
            job_name=job_name,
        )
        with self._lock:
            self._status = "running"
            self._message = "Vector store is stale; updater Job is running."
            self._stats = stats_payload
            self._job_name = job_name
            self._error = None
            self._latest_artifact = self._write_artifact(
                config=config,
                docs_dir=docs_dir,
                status="running",
                trigger=trigger,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                message=self._message,
                error=None,
                stats_payload=stats_payload,
                job_name=job_name,
            )

    def _mark_manual_maintenance_required(
        self,
        *,
        config: AppConfig,
        docs_dir: Path,
        trigger: str,
        started_at: datetime,
        stats_payload: dict[str, object],
    ) -> None:
        """Persist a stale status that instructs operators to run the maintenance workflow."""
        completed_at = datetime.now(timezone.utc)
        message = (
            "Vector store is stale. Run `bash scripts/update_vector_store_maintenance.sh` "
            "and retry."
        )
        write_update_status(
            get_update_status_path(config),
            status="stale",
            trigger=trigger,
            update_mode=config.vector_store.update_mode,
            message=message,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            stats=stats_payload,
        )
        latest_artifact = self._write_artifact(
            config=config,
            docs_dir=docs_dir,
            status="stale",
            trigger=trigger,
            started_at=started_at,
            completed_at=completed_at,
            message=message,
            error=None,
            stats_payload=stats_payload,
            job_name=None,
        )
        with self._lock:
            self._status = "stale"
            self._completed_at = completed_at
            self._message = message
            self._error = None
            self._stats = stats_payload
            self._job_name = None
            self._latest_artifact = latest_artifact

    def _complete(
        self,
        *,
        config: AppConfig,
        docs_dir: Path,
        trigger: str,
        started_at: datetime,
        completed_at: datetime,
        message: str,
        stats: object,
    ) -> None:
        """Mark the vector update coordinator as completed."""
        stats_payload = self._stats_payload(stats)
        write_update_status(
            get_update_status_path(config),
            status="completed",
            trigger=trigger,
            update_mode=config.vector_store.update_mode,
            message=message,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            stats=stats_payload,
        )
        latest_artifact = self._write_artifact(
            config=config,
            docs_dir=docs_dir,
            status="completed",
            trigger=trigger,
            started_at=started_at,
            completed_at=completed_at,
            message=message,
            error=None,
            stats_payload=stats_payload,
            update_stats=stats,
            job_name=None,
        )
        with self._lock:
            self._status = "completed"
            self._completed_at = completed_at
            self._message = message
            self._error = None
            self._stats = stats_payload
            self._job_name = None
            self._latest_artifact = latest_artifact

    def _fail(
        self,
        *,
        config: AppConfig,
        docs_dir: Path,
        trigger: str,
        started_at: datetime,
        message: str,
        error: str,
    ) -> None:
        """Mark the vector update coordinator as failed and persist diagnostics."""
        completed_at = datetime.now(timezone.utc)
        write_update_status(
            get_update_status_path(config),
            status="failed",
            trigger=trigger,
            update_mode=config.vector_store.update_mode,
            message=message,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            error=error,
        )
        latest_artifact = self._write_artifact(
            config=config,
            docs_dir=docs_dir,
            status="failed",
            trigger=trigger,
            started_at=started_at,
            completed_at=completed_at,
            message=message,
            error=error,
            stats_payload=None,
            job_name=None,
        )
        with self._lock:
            self._status = "failed"
            self._completed_at = completed_at
            self._message = message
            self._error = error
            self._stats = None
            self._job_name = None
            self._latest_artifact = latest_artifact

    def _needs_update(self, stats: object) -> bool:
        """Return true when dry-run stats indicate the persisted index is stale."""
        return (
            int(getattr(stats, "files_changed", 0)) > 0
            or int(getattr(stats, "files_deleted", 0)) > 0
            or str(getattr(stats, "update_mode", "")) != "incremental_update"
        )

    def _stats_payload(self, stats: object) -> dict[str, object]:
        """Return JSON-safe update stats for status responses and artifacts."""
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

    def _changed_files_by_status(
        self,
        stats_payload: dict[str, object],
        status: str,
    ) -> list[str]:
        """Return changed-file source paths for one status label."""
        changed_files = stats_payload.get("changed_files")
        if not isinstance(changed_files, list):
            return []
        return [
            str(item.get("source_path"))
            for item in changed_files
            if isinstance(item, dict)
            and str(item.get("status")) == status
            and str(item.get("source_path", "")).strip()
        ]

    def _deleted_file_sources(self, stats_payload: dict[str, object]) -> list[str]:
        """Return deleted-file source paths from one stats payload."""
        deleted_files = stats_payload.get("deleted_files")
        if not isinstance(deleted_files, list):
            return []
        return [
            str(item.get("source_path"))
            for item in deleted_files
            if isinstance(item, dict) and str(item.get("source_path", "")).strip()
        ]

    def _count_changed_files_by_status(self, stats: object, status: str) -> int:
        """Return the number of changed files with one status on index stats."""
        changed_files = getattr(stats, "changed_files", [])
        if not isinstance(changed_files, list):
            return 0
        return sum(
            1
            for item in changed_files
            if isinstance(item, dict) and str(item.get("status")) == status
        )

    def _write_artifact(
        self,
        *,
        config: AppConfig,
        docs_dir: Path,
        status: VectorStoreUpdateStatus,
        trigger: str,
        started_at: datetime,
        completed_at: datetime,
        message: str,
        error: str | None,
        stats_payload: dict[str, object] | None,
        update_stats: object | None = None,
        job_name: str | None = None,
    ) -> str:
        """Persist vector update diagnostics outside any user run directory."""
        artifact_dir = config.runs_dir / "system" / "vector_store_warmup"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        timestamp = completed_at.strftime("%Y%m%dT%H%M%SZ")
        timestamped_path = artifact_dir / f"{timestamp}.json"
        latest_path = artifact_dir / "latest.json"
        latest_label = Path("system") / "vector_store_warmup" / "latest.json"
        payload = {
            "event_type": "vector_store_update",
            "trigger": trigger,
            "status": status,
            "update_mode": config.vector_store.update_mode,
            "job_name": job_name,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "message": message,
            "error": error,
            "docs_dir": str(docs_dir),
            "stats": stats_payload,
            "vector_store_snapshot": build_vector_store_snapshot(
                config,
                update_stats=update_stats,
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

    def _status_from_file(self) -> dict[str, object] | None:
        """Return current shared status-file payload when available."""
        path = self._status_path
        if path is None:
            return None
        return read_update_status(path)

    def _has_active_status_timed_out(self, payload: dict[str, object]) -> bool:
        """Return true when a checking/running status has exceeded its timeout window."""
        if payload.get("status") not in {"checking", "running"}:
            return False
        raw_started_at = payload.get("started_at") or payload.get("updated_at")
        if not isinstance(raw_started_at, str) or not raw_started_at.strip():
            return False
        try:
            started_at = datetime.fromisoformat(raw_started_at)
        except ValueError:
            return False
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        raw_timeout = os.getenv("VECTOR_STORE_UPDATE_JOB_TIMEOUT_SECONDS", "").strip()
        try:
            timeout_seconds = int(raw_timeout) if raw_timeout else DEFAULT_UPDATE_TIMEOUT_SECONDS
        except ValueError:
            timeout_seconds = DEFAULT_UPDATE_TIMEOUT_SECONDS
        elapsed_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
        return elapsed_seconds > max(timeout_seconds, 1)

    def _is_compatible_file_status(self, payload: dict[str, object]) -> bool:
        """Return true when the persisted status file matches the active update mode."""
        file_update_mode = str(payload.get("update_mode", "")).strip()
        return not file_update_mode or file_update_mode == self._update_mode

    def _timed_out_active_status_message(
        self,
        payload: dict[str, object],
    ) -> tuple[str, str]:
        """Return user-facing message and error for one timed-out active status."""
        if payload.get("status") == "checking":
            return (
                "Vector store freshness check timed out before completing.",
                "Vector store freshness check timed out.",
            )
        if self._update_mode == "local_process":
            return (
                "Vector store local update timed out before writing a completion status.",
                "Vector store local update timed out.",
            )
        return (
            "Vector store updater Job timed out before writing a completion status.",
            "Vector store updater Job timed out.",
        )

    def _reconcile_timed_out_file_status(
        self,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Persist a failed status when a stale active status has timed out."""
        message, error = self._timed_out_active_status_message(payload)
        status_path = self._status_path
        if status_path is None:
            updated_payload = dict(payload)
            updated_payload.update(
                {
                    "status": "failed",
                    "message": message,
                    "error": error,
                }
            )
            return updated_payload
        return write_update_status(
            status_path,
            status="failed",
            trigger=str(payload.get("trigger") or "unknown"),
            update_mode=self._update_mode,
            message=message,
            started_at=str(payload.get("started_at")) if payload.get("started_at") else None,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=error,
            stats=payload.get("stats") if isinstance(payload.get("stats"), dict) else None,
            job_name=str(payload.get("job_name")) if payload.get("job_name") else None,
        )

    def _sync_from_status_file_locked(self) -> None:
        """Refresh in-memory status from the shared status file while holding the lock."""
        file_status = self._status_from_file()
        if not isinstance(file_status, dict) or not self._is_compatible_file_status(file_status):
            return
        file_state = str(file_status.get("status", "")).strip()
        if file_state not in {
            "checking",
            "stale",
            "running",
            "completed",
            "failed",
        }:
            return
        if self._has_active_status_timed_out(file_status):
            file_status = self._reconcile_timed_out_file_status(file_status)
            message, error = self._timed_out_active_status_message(file_status)
            self._status = "failed"
            self._message = message
            self._error = error
            self._stats = (
                file_status.get("stats")
                if isinstance(file_status.get("stats"), dict)
                else None
            )
            self._job_name = str(file_status.get("job_name")) if file_status.get("job_name") else None
            return
        self._status = file_state
        self._message = str(file_status.get("message") or self._message)
        self._error = str(file_status.get("error")) if file_status.get("error") else None
        self._stats = file_status.get("stats") if isinstance(file_status.get("stats"), dict) else None
        self._job_name = str(file_status.get("job_name")) if file_status.get("job_name") else None

    def snapshot(self) -> dict[str, object]:
        """Return a thread-safe status payload for API responses."""
        file_status = self._status_from_file()
        if (
            isinstance(file_status, dict)
            and self._is_compatible_file_status(file_status)
            and self._has_active_status_timed_out(file_status)
        ):
            file_status = self._reconcile_timed_out_file_status(file_status)
        with self._lock:
            payload = {
                "status": self._status,
                "enabled": self._enabled,
                "auto_update_on_run": self._auto_update_on_run,
                "update_mode": self._update_mode,
                "started_at": self._started_at,
                "completed_at": self._completed_at,
                "message": self._message,
                "error": self._error,
                "stats": dict(self._stats) if self._stats is not None else None,
                "job_name": self._job_name,
                "latest_artifact": self._latest_artifact,
            }
        if (
            self._enabled
            and self._auto_update_on_run
            and isinstance(file_status, dict)
            and self._is_compatible_file_status(file_status)
        ):
            file_state = str(file_status.get("status", "")).strip()
            if file_state in {
                "checking",
                "stale",
                "running",
                "completed",
                "failed",
            }:
                payload.update(
                    {
                        "status": file_state,
                        "message": str(file_status.get("message") or payload["message"]),
                        "error": file_status.get("error"),
                        "stats": file_status.get("stats"),
                        "job_name": file_status.get("job_name"),
                    }
                )
        return payload

    def is_blocking_runs(self) -> bool:
        """Return True while vector state should block new run submissions."""
        status_value = str(self.snapshot().get("status", ""))
        return status_value in {"checking", "stale", "running", "failed"}

    def shutdown(self, *, wait: bool = False) -> None:
        """Optionally wait for warm-up completion during API shutdown."""
        thread = self._thread
        if wait and thread is not None:
            thread.join()


__all__ = ["VectorStoreWarmup"]
