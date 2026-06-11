"""Thread-safe run state store for async API execution."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.api.models import RunError, RunStatus
from backend.utils.artifact_manifest import resolve_manifest_alias
from backend.utils.json_io import read_json_object, write_json
from backend.utils.paths import build_run_id

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
    api_state_path: Path | None = None

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

        return cls(
            run_id=run_id,
            question=question,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            finish_reason=payload.get("finish_reason")
            if isinstance(payload.get("finish_reason"), str)
            else None,
            error=_parse_error_payload(payload.get("error")),
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


def _coerce_status(value: str | None) -> RunStatus | None:
    """Return status only when value is a supported run status."""
    if value in VALID_RUN_STATUSES:
        return value
    return None


def _parse_error_payload(raw_value: object) -> RunError | None:
    """Return a structured run error when persisted payload fields are valid."""
    if not isinstance(raw_value, dict):
        return None
    code = raw_value.get("code")
    message = raw_value.get("message")
    if not isinstance(code, str) or not isinstance(message, str):
        return None
    return RunError(code=code, message=message)


def _extract_question_from_api_state(api_state: dict[str, Any]) -> str:
    """Extract display question from API state using modern and legacy fields."""
    question_raw = api_state.get("question")
    if isinstance(question_raw, str) and question_raw.strip():
        return question_raw.strip()

    inputs_raw = api_state.get("inputs")
    if isinstance(inputs_raw, dict):
        original_question = inputs_raw.get("original_question")
        if isinstance(original_question, str) and original_question.strip():
            return original_question.strip()
        initial_question = inputs_raw.get("initial_question")
        if isinstance(initial_question, str) and initial_question.strip():
            return initial_question.strip()
        canonical_research_query = inputs_raw.get("canonical_research_query")
        if (
            isinstance(canonical_research_query, str)
            and canonical_research_query.strip()
        ):
            return canonical_research_query.strip()
    return ""


class RunStore:
    """Persisted in-memory store used by API routes and background workers."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir
        self._lock = threading.Lock()
        self._records: dict[str, RunRecord] = {}

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
                api_state_path=self._runs_dir / run_id / "api_state.json",
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
                api_state_path=record.api_state_path,
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
        api_state_path: Path | None = None,
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
                api_state_path=api_state_path or record.api_state_path,
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
            self._load_runs_from_artifact_dirs_locked()
            self._prune_stale_terminal_records_locked()
            records = self._exclude_legacy_alias_records_locked(
                list(self._records.values())
            )
            return sorted(
                records,
                key=lambda record: record.started_at,
                reverse=True,
            )

    def _prune_stale_terminal_records_locked(self) -> None:
        """Drop terminal records whose run artifacts are missing from runs directory."""
        stale_run_ids: list[str] = []
        for run_id, record in self._records.items():
            if record.status in IN_PROGRESS_STATUSES:
                continue
            if self._resolve_existing_api_state_path(record) is not None:
                continue
            stale_run_ids.append(run_id)

        for run_id in stale_run_ids:
            self._records.pop(run_id, None)

    def _resolve_existing_api_state_path(self, record: RunRecord) -> Path | None:
        """Resolve existing api_state.json path for a record, if present on disk."""
        if record.api_state_path is not None and record.api_state_path.exists():
            return record.api_state_path
        fallback = self._runs_dir / record.run_id / "api_state.json"
        if fallback.exists():
            return fallback
        return None

    def _exclude_legacy_alias_records_locked(
        self, records: list[RunRecord]
    ) -> list[RunRecord]:
        """Drop known alias ids from legacy `_01` artifact naming collisions."""
        folder_owner: dict[str, str] = {}
        for record in records:
            if record.api_state_path is None:
                continue
            folder_name = record.api_state_path.parent.name
            existing_owner = folder_owner.get(folder_name)
            if existing_owner is None:
                folder_owner[folder_name] = record.run_id
                continue
            if existing_owner == folder_name and record.run_id != folder_name:
                folder_owner[folder_name] = record.run_id

        alias_run_ids = {
            folder_name
            for folder_name, owner_run_id in folder_owner.items()
            if owner_run_id != folder_name
        }
        if not alias_run_ids:
            return records
        return [record for record in records if record.run_id not in alias_run_ids]

    def _load_runs_from_artifact_dirs_locked(self) -> None:
        """Hydrate runs directly from artifact folders when state snapshots are missing."""
        for api_state_path in self._runs_dir.glob("*/api_state.json"):
            self._hydrate_from_api_state_locked(api_state_path)

    def _hydrate_from_api_state_locked(self, api_state_path: Path) -> RunRecord | None:
        """Create a run record from an api_state.json artifact when possible."""
        if not api_state_path.exists():
            return None
        if self._has_record_for_run_folder_locked(api_state_path.parent.name):
            return None

        api_state = read_json_object(api_state_path)
        if api_state is None:
            return None

        run_id_raw = api_state.get("run_id")
        run_id = (
            run_id_raw.strip()
            if isinstance(run_id_raw, str) and run_id_raw.strip()
            else api_state_path.parent.name
        )
        if run_id in self._records:
            return self._records[run_id]

        status = _coerce_status(api_state.get("status")) or "failed"
        started_at = _parse_datetime(str(api_state.get("started_at"))) or _utc_now()
        completed_at: datetime | None = None
        completed_raw = api_state.get("completed_at")
        if isinstance(completed_raw, str):
            completed_at = _parse_datetime(completed_raw)

        finish_reason_raw = api_state.get("finish_reason")
        finish_reason = finish_reason_raw if isinstance(finish_reason_raw, str) else None

        final_output_path: Path | None = resolve_manifest_alias(
            api_state_path.parent, "final_output"
        )
        context_bundle_path: Path | None = resolve_manifest_alias(
            api_state_path.parent, "context_bundle"
        )
        if final_output_path is None:
            fallback_final = api_state_path.parent / "final.md"
            if fallback_final.exists():
                final_output_path = fallback_final
        if context_bundle_path is None:
            fallback_context = api_state_path.parent / "context_bundle.json"
            if fallback_context.exists():
                context_bundle_path = fallback_context

        question = _extract_question_from_api_state(api_state)

        record = RunRecord(
            run_id=run_id,
            question=question,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            finish_reason=finish_reason,
            error=_parse_error_payload(api_state.get("error")),
            final_output_path=final_output_path,
            context_bundle_path=context_bundle_path,
            api_state_path=api_state_path,
        )
        self._records[run_id] = record
        self._persist_record_locked(record)
        return record

    def _has_record_for_run_folder_locked(self, folder_name: str) -> bool:
        """Return True when a record already points to the same artifact folder."""
        for record in self._records.values():
            if record.api_state_path is None:
                continue
            if record.api_state_path.parent.name == folder_name:
                return True
        return False

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
        if (self._runs_dir / run_id).exists():
            return True
        return False

    def _get_or_hydrate_locked(self, run_id: str) -> RunRecord | None:
        """Return record, hydrating from persisted files when missing."""
        existing = self._records.get(run_id)
        if existing is not None:
            return existing

        api_state_path = self._runs_dir / run_id / "api_state.json"
        api_state = read_json_object(api_state_path)
        if api_state is None:
            return None

        status = _coerce_status(api_state.get("status")) or "failed"
        started_at = _parse_datetime(str(api_state.get("started_at"))) or _utc_now()
        completed_at = None
        completed_raw = api_state.get("completed_at")
        if isinstance(completed_raw, str):
            completed_at = _parse_datetime(completed_raw)

        finish_reason = api_state.get("finish_reason")
        finish_reason_value = finish_reason if isinstance(finish_reason, str) else None

        final_output_path: Path | None = resolve_manifest_alias(
            api_state_path.parent, "final_output"
        )
        context_bundle_path: Path | None = resolve_manifest_alias(
            api_state_path.parent, "context_bundle"
        )
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
            question=_extract_question_from_api_state(api_state),
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            finish_reason=finish_reason_value,
            error=_parse_error_payload(api_state.get("error")),
            final_output_path=final_output_path,
            context_bundle_path=context_bundle_path,
            api_state_path=api_state_path,
        )
        self._records[run_id] = record
        self._persist_record_locked(record)
        return record

    def _persist_record_locked(self, record: RunRecord) -> None:
        """Persist run record to run-local state file when run directory exists."""
        run_dir = self._runs_dir / record.run_id
        if run_dir.exists():
            try:
                api_state_path = run_dir / "api_state.json"
                existing_payload = read_json_object(api_state_path)
                payload = (
                    dict(existing_payload)
                    if isinstance(existing_payload, dict)
                    else {}
                )
                payload.update(record.to_payload())
                write_json(api_state_path, payload, default=str)
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
