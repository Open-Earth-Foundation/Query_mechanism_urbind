"""Thread-safe run state store for async API execution."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.api.models import RunError, RunStatus
from app.utils.paths import build_run_id

logger = logging.getLogger(__name__)

TERMINAL_STATUSES: set[RunStatus] = {
    "completed",
    "completed_with_gaps",
    "failed",
    "stopped",
}
IN_PROGRESS_STATUSES: set[RunStatus] = {"queued", "running"}
SUCCESS_STATUSES: set[RunStatus] = {"completed", "completed_with_gaps"}
VALID_RUN_STATUSES: set[RunStatus] = TERMINAL_STATUSES | IN_PROGRESS_STATUSES

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class DuplicateRunIdError(ValueError):
    """Raised when a run id is already present in memory or on disk."""


@dataclass(frozen=True)
class RunRecord:
    """Single run state record."""

    run_id: str
    question: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    finish_reason: str | None = None
    error: RunError | None = None
    final_output_path: Path | None = None
    context_bundle_path: Path | None = None
    run_log_path: Path | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize record to JSON-friendly payload."""
        return {
            "run_id": self.run_id,
            "question": self.question,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "finish_reason": self.finish_reason,
            "error": self.error.model_dump() if self.error else None,
            "final_output_path": str(self.final_output_path) if self.final_output_path else None,
            "context_bundle_path": str(self.context_bundle_path)
            if self.context_bundle_path
            else None,
            "run_log_path": str(self.run_log_path) if self.run_log_path else None,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RunRecord | None:
        """Deserialize record from persisted payload."""
        run_id = payload.get("run_id")
        question = payload.get("question")
        status = payload.get("status")
        started_at_raw = payload.get("started_at")

        if not isinstance(run_id, str) or not run_id.strip():
            return None
        if not isinstance(question, str):
            return None
        if not isinstance(status, str) or status not in VALID_RUN_STATUSES:
            return None
        if not isinstance(started_at_raw, str):
            return None

        started_at = _parse_datetime(started_at_raw)
        if started_at is None:
            return None

        completed_at_raw = payload.get("completed_at")
        completed_at: datetime | None = None
        if isinstance(completed_at_raw, str):
            completed_at = _parse_datetime(completed_at_raw)

        error_raw = payload.get("error")
        error_payload: RunError | None = None
        if isinstance(error_raw, dict):
            code = error_raw.get("code")
            message = error_raw.get("message")
            if isinstance(code, str) and isinstance(message, str):
                error_payload = RunError(code=code, message=message)

        return cls(
            run_id=run_id,
            question=question,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            finish_reason=payload.get("finish_reason")
            if isinstance(payload.get("finish_reason"), str)
            else None,
            error=error_payload,
            final_output_path=Path(payload["final_output_path"])
            if isinstance(payload.get("final_output_path"), str)
            else None,
            context_bundle_path=Path(payload["context_bundle_path"])
            if isinstance(payload.get("context_bundle_path"), str)
            else None,
            run_log_path=Path(payload["run_log_path"])
            if isinstance(payload.get("run_log_path"), str)
            else None,
        )


def _parse_datetime(value: str) -> datetime | None:
    """Parse ISO datetime and coerce to UTC-aware when needed."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON object from file, returning None on parse errors."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(raw, dict):
        return raw
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload to file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def _coerce_status(value: str | None) -> RunStatus | None:
    """Return status only when value is a supported run status."""
    if value in VALID_RUN_STATUSES:
        return value
    return None


class RunStore:
    """Persisted in-memory store used by API routes and background workers."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir
        self._state_dir = runs_dir / "_api_state"
        self._lock = threading.Lock()
        self._records: dict[str, RunRecord] = {}
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._load_state_files()

    @property
    def runs_dir(self) -> Path:
        """Return run artifacts root directory."""
        return self._runs_dir

    def create_queued_run(self, question: str, requested_run_id: str | None) -> RunRecord:
        """Create queued run record with a unique run id."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question must be a non-empty string.")

        with self._lock:
            run_id = self._allocate_run_id_locked(requested_run_id)
            record = RunRecord(
                run_id=run_id,
                question=cleaned_question,
                status="queued",
                started_at=_utc_now(),
                run_log_path=self._runs_dir / run_id / "run.json",
            )
            self._records[run_id] = record
            self._persist_record_locked(record)
            logger.info(
                "Run queued run_id=%s question_chars=%d",
                run_id,
                len(cleaned_question),
            )
            return record

    def mark_running(self, run_id: str) -> RunRecord:
        """Transition a run from queued to running."""
        with self._lock:
            record = self._get_or_hydrate_locked(run_id)
            if record is None:
                raise KeyError(f"Run `{run_id}` not found.")
            updated = RunRecord(
                run_id=record.run_id,
                question=record.question,
                status="running",
                started_at=record.started_at,
                completed_at=None,
                finish_reason=None,
                error=None,
                final_output_path=record.final_output_path,
                context_bundle_path=record.context_bundle_path,
                run_log_path=record.run_log_path,
            )
            self._records[run_id] = updated
            self._persist_record_locked(updated)
            logger.info("Run status update run_id=%s status=running", run_id)
            return updated

    def mark_terminal(
        self,
        run_id: str,
        status: RunStatus,
        finish_reason: str | None = None,
        error: RunError | None = None,
        final_output_path: Path | None = None,
        context_bundle_path: Path | None = None,
        run_log_path: Path | None = None,
    ) -> RunRecord:
        """Set final run state and persist terminal metadata."""
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Invalid terminal status: {status}")

        with self._lock:
            record = self._get_or_hydrate_locked(run_id)
            if record is None:
                raise KeyError(f"Run `{run_id}` not found.")
            completed_at = _utc_now()
            updated = RunRecord(
                run_id=record.run_id,
                question=record.question,
                status=status,
                started_at=record.started_at,
                completed_at=completed_at,
                finish_reason=finish_reason,
                error=error,
                final_output_path=final_output_path or record.final_output_path,
                context_bundle_path=context_bundle_path or record.context_bundle_path,
                run_log_path=run_log_path or record.run_log_path,
            )
            self._records[run_id] = updated
            self._persist_record_locked(updated)
            elapsed_seconds = (completed_at - record.started_at).total_seconds()
            logger.info(
                "Run terminal run_id=%s status=%s finish_reason=%s elapsed_seconds=%.2f error_code=%s",
                run_id,
                status,
                finish_reason,
                elapsed_seconds,
                error.code if error else None,
            )
            return updated

    def get_run(self, run_id: str) -> RunRecord | None:
        """Return run record from memory or artifact files."""
        with self._lock:
            record = self._get_or_hydrate_locked(run_id)
            return record

    def list_runs(self) -> list[RunRecord]:
        """Return all known runs sorted by newest first."""
        with self._lock:
            for path in self._state_dir.glob("*.json"):
                run_id = path.stem
                if run_id not in self._records:
                    self._get_or_hydrate_locked(run_id)
            return sorted(
                self._records.values(),
                key=lambda record: record.started_at,
                reverse=True,
            )

    def _load_state_files(self) -> None:
        """Load persisted API state snapshots into memory."""
        for path in self._state_dir.glob("*.json"):
            payload = _read_json(path)
            if payload is None:
                continue
            record = RunRecord.from_payload(payload)
            if record is None:
                continue
            self._records[record.run_id] = record

    def _normalize_run_id(self, run_id: str) -> str:
        """Validate and normalize candidate run id."""
        candidate = run_id.strip()
        if not candidate:
            raise ValueError("run_id must be a non-empty string.")
        if not RUN_ID_PATTERN.fullmatch(candidate):
            raise ValueError(
                "run_id may contain only letters, numbers, underscore, and hyphen."
            )
        return candidate

    def _allocate_run_id_locked(self, requested_run_id: str | None) -> str:
        """Allocate unique run id while holding lock."""
        if requested_run_id:
            candidate = self._normalize_run_id(requested_run_id)
            if self._is_reserved_locked(candidate):
                raise DuplicateRunIdError(f"Run id `{candidate}` already exists.")
            return candidate

        base = build_run_id()
        if not self._is_reserved_locked(base):
            return base

        counter = 1
        while True:
            candidate = f"{base}_{counter:02d}"
            if not self._is_reserved_locked(candidate):
                return candidate
            counter += 1

    def _is_reserved_locked(self, run_id: str) -> bool:
        """Check if run id is already in store or existing artifacts."""
        if run_id in self._records:
            return True
        if (self._state_dir / f"{run_id}.json").exists():
            return True
        if (self._runs_dir / run_id).exists():
            return True
        return False

    def _get_or_hydrate_locked(self, run_id: str) -> RunRecord | None:
        """Return record, hydrating from persisted files when missing."""
        existing = self._records.get(run_id)
        if existing is not None:
            return existing

        state_path = self._state_dir / f"{run_id}.json"
        payload = _read_json(state_path)
        if payload is not None:
            record = RunRecord.from_payload(payload)
            if record is not None:
                self._records[run_id] = record
                return record

        run_log_path = self._runs_dir / run_id / "run.json"
        run_log = _read_json(run_log_path)
        if run_log is None:
            return None

        status = _coerce_status(run_log.get("status")) or "failed"
        started_at = _parse_datetime(str(run_log.get("started_at"))) or _utc_now()
        completed_at = None
        completed_raw = run_log.get("completed_at")
        if isinstance(completed_raw, str):
            completed_at = _parse_datetime(completed_raw)

        finish_reason = run_log.get("finish_reason")
        finish_reason_value = finish_reason if isinstance(finish_reason, str) else None

        artifacts = run_log.get("artifacts")
        final_output_path: Path | None = None
        context_bundle_path: Path | None = None
        if isinstance(artifacts, dict):
            final_raw = artifacts.get("final_output")
            context_raw = artifacts.get("context_bundle")
            if isinstance(final_raw, str):
                final_output_path = Path(final_raw)
            if isinstance(context_raw, str):
                context_bundle_path = Path(context_raw)
        if final_output_path is None:
            fallback_final = self._runs_dir / run_id / "final.md"
            if fallback_final.exists():
                final_output_path = fallback_final
        if context_bundle_path is None:
            fallback_context = self._runs_dir / run_id / "context_bundle.json"
            if fallback_context.exists():
                context_bundle_path = fallback_context

        record = RunRecord(
            run_id=run_id,
            question=str(run_log.get("question") or ""),
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            finish_reason=finish_reason_value,
            error=None,
            final_output_path=final_output_path,
            context_bundle_path=context_bundle_path,
            run_log_path=run_log_path,
        )
        self._records[run_id] = record
        self._persist_record_locked(record)
        return record

    def _persist_record_locked(self, record: RunRecord) -> None:
        """Persist run record to API state file (and run directory when available)."""
        payload = record.to_payload()
        state_path = self._state_dir / f"{record.run_id}.json"
        try:
            _write_json(state_path, payload)
        except OSError:
            logger.exception("Failed to persist API state for run_id=%s", record.run_id)

        run_dir = self._runs_dir / record.run_id
        if run_dir.exists():
            try:
                _write_json(run_dir / "api_state.json", payload)
            except OSError:
                logger.exception(
                    "Failed to persist run-local API state for run_id=%s", record.run_id
                )


__all__ = [
    "DuplicateRunIdError",
    "RunRecord",
    "RunStore",
    "TERMINAL_STATUSES",
    "IN_PROGRESS_STATUSES",
    "SUCCESS_STATUSES",
]
