"""Persisted background jobs for split-mode context chat replies.

Inputs:
- `StartChatJobCommand`: one queued split-mode chat request snapshot.
- `ChatMemoryStore`: persisted chat transcript store used to append final assistant replies.
- `processor`: callable that turns one command into the final assistant answer payload.

Outputs:
- JSON job records under `output/<run_id>/chat_jobs/<conversation_id>/<job_id>.json`
- appended assistant messages or deterministic failure messages in chat sessions
- polling metadata for the chat job status endpoint
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from backend.api.models import RunError
from backend.api.services.chat_memory import ChatMemoryStore
from backend.utils.json_io import read_json_object, write_json

logger = logging.getLogger(__name__)

ChatJobStatus = Literal["queued", "running", "completed", "failed"]
VALID_CHAT_JOB_STATUSES: set[ChatJobStatus] = {"queued", "running", "completed", "failed"}
IN_PROGRESS_CHAT_JOB_STATUSES: set[ChatJobStatus] = {"queued", "running"}
CHAT_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
CHAT_JOB_ERROR_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
CHAT_JOB_FAILURE_MESSAGE = "The long-context answer did not complete. Please retry."
CHAT_JOB_INTERRUPTED_MESSAGE = (
    "The long-context answer was interrupted before completion. Please retry."
)


def _utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _parse_datetime(value: object) -> datetime | None:
    """Parse one persisted ISO timestamp."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalize_error_message(message: str) -> str:
    """Mask obvious key-like fragments in stored error text."""
    cleaned = message.strip() or "Unknown split-mode chat job failure."
    return CHAT_JOB_ERROR_PATTERN.sub("sk-***", cleaned)


def build_chat_job_failure_message(*, interrupted: bool = False) -> str:
    """Return the deterministic assistant reply for failed split-mode jobs."""
    if interrupted:
        return CHAT_JOB_INTERRUPTED_MESSAGE
    return CHAT_JOB_FAILURE_MESSAGE


@dataclass(frozen=True)
class StartChatJobCommand:
    """Parameters required to finish one queued split-mode chat answer."""

    run_id: str
    conversation_id: str
    job_id: str
    job_number: int
    original_question: str
    user_content: str
    history: list[dict[str, str]]
    context_run_ids: list[str]
    followup_bundle_ids: list[str]
    token_cap: int
    assistant_routing: dict[str, object] | None = None
    api_key_override: str | None = None

    def to_request_payload(self) -> dict[str, Any]:
        """Return the persisted request snapshot without secret overrides."""
        payload: dict[str, Any] = {
            "original_question": self.original_question,
            "user_content": self.user_content,
            "history": self.history,
            "context_run_ids": self.context_run_ids,
            "followup_bundle_ids": self.followup_bundle_ids,
            "token_cap": self.token_cap,
        }
        if self.assistant_routing:
            payload["assistant_routing"] = self.assistant_routing
        return payload


@dataclass(frozen=True)
class ChatJobResult:
    """Final assistant answer payload produced by the job worker."""

    assistant_content: str
    assistant_citations: list[dict[str, object]] | None = None
    assistant_citation_warning: str | None = None


@dataclass(frozen=True)
class ChatJobRecord:
    """Persisted chat-job record used for polling and reconciliation."""

    job_id: str
    job_number: int
    run_id: str
    conversation_id: str
    status: ChatJobStatus
    created_at: datetime
    request_payload: dict[str, Any]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    finish_reason: str | None = None
    error: RunError | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize one chat-job record for JSON storage."""
        return {
            "job_id": self.job_id,
            "job_number": self.job_number,
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "finish_reason": self.finish_reason,
            "error": self.error.model_dump() if self.error else None,
            "request": self.request_payload,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChatJobRecord | None:
        """Deserialize one persisted chat-job record."""
        job_id = payload.get("job_id")
        job_number = payload.get("job_number")
        run_id = payload.get("run_id")
        conversation_id = payload.get("conversation_id")
        status = payload.get("status")
        created_at = _parse_datetime(payload.get("created_at"))
        request_payload = payload.get("request")
        if not isinstance(job_id, str) or not CHAT_JOB_ID_PATTERN.fullmatch(job_id):
            return None
        if not isinstance(job_number, int) or job_number <= 0:
            return None
        if not isinstance(run_id, str) or not run_id.strip():
            return None
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            return None
        if status not in VALID_CHAT_JOB_STATUSES:
            return None
        if created_at is None or not isinstance(request_payload, dict):
            return None

        error_payload: RunError | None = None
        raw_error = payload.get("error")
        if isinstance(raw_error, dict):
            code = raw_error.get("code")
            message = raw_error.get("message")
            if isinstance(code, str) and isinstance(message, str):
                error_payload = RunError(code=code, message=message)

        return cls(
            job_id=job_id,
            job_number=job_number,
            run_id=run_id.strip(),
            conversation_id=conversation_id.strip(),
            status=status,
            created_at=created_at,
            started_at=_parse_datetime(payload.get("started_at")),
            completed_at=_parse_datetime(payload.get("completed_at")),
            finish_reason=payload.get("finish_reason")
            if isinstance(payload.get("finish_reason"), str)
            else None,
            error=error_payload,
            request_payload=request_payload,
        )


class ChatJobStore:
    """Persist chat-job records under each run directory."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir
        self._lock = threading.Lock()

    def create_queued_job(self, command: StartChatJobCommand) -> ChatJobRecord:
        """Persist one newly accepted queued chat job."""
        with self._lock:
            path = self.job_path(command.run_id, command.conversation_id, command.job_id)
            if path.exists():
                raise ValueError(f"Chat job `{command.job_id}` already exists.")
            record = ChatJobRecord(
                job_id=command.job_id,
                job_number=command.job_number,
                run_id=command.run_id,
                conversation_id=command.conversation_id,
                status="queued",
                created_at=_utc_now(),
                request_payload=command.to_request_payload(),
            )
            write_json(path, record.to_payload(), default=str)
            return record

    def get_job(self, run_id: str, conversation_id: str, job_id: str) -> ChatJobRecord | None:
        """Load one persisted chat job by identifiers."""
        with self._lock:
            return self._read_record_locked(run_id, conversation_id, job_id)

    def mark_running(
        self,
        run_id: str,
        conversation_id: str,
        job_id: str,
    ) -> ChatJobRecord:
        """Transition a queued job to running."""
        with self._lock:
            record = self._require_record_locked(run_id, conversation_id, job_id)
            updated = ChatJobRecord(
                job_id=record.job_id,
                job_number=record.job_number,
                run_id=record.run_id,
                conversation_id=record.conversation_id,
                status="running",
                created_at=record.created_at,
                request_payload=record.request_payload,
                started_at=_utc_now(),
            )
            self._write_record_locked(updated)
            return updated

    def mark_terminal(
        self,
        run_id: str,
        conversation_id: str,
        job_id: str,
        *,
        status: Literal["completed", "failed"],
        finish_reason: str | None = None,
        error: RunError | None = None,
    ) -> ChatJobRecord:
        """Persist one terminal chat job state."""
        with self._lock:
            record = self._require_record_locked(run_id, conversation_id, job_id)
            updated = ChatJobRecord(
                job_id=record.job_id,
                job_number=record.job_number,
                run_id=record.run_id,
                conversation_id=record.conversation_id,
                status=status,
                created_at=record.created_at,
                request_payload=record.request_payload,
                started_at=record.started_at,
                completed_at=_utc_now(),
                finish_reason=finish_reason,
                error=error,
            )
            self._write_record_locked(updated)
            return updated

    def list_in_progress_jobs(self) -> list[ChatJobRecord]:
        """List all queued or running chat jobs across runs."""
        records: list[ChatJobRecord] = []
        with self._lock:
            for path in self._runs_dir.glob("*/chat_jobs/*/*.json"):
                payload = read_json_object(path)
                if not isinstance(payload, dict):
                    continue
                record = ChatJobRecord.from_payload(payload)
                if record is None or record.status not in IN_PROGRESS_CHAT_JOB_STATUSES:
                    continue
                records.append(record)
        return records

    def job_path(self, run_id: str, conversation_id: str, job_id: str) -> Path:
        """Return the persisted path for one chat job."""
        return self._runs_dir / run_id / "chat_jobs" / conversation_id / f"{job_id}.json"

    def _require_record_locked(
        self,
        run_id: str,
        conversation_id: str,
        job_id: str,
    ) -> ChatJobRecord:
        """Load one record or raise when missing or invalid."""
        record = self._read_record_locked(run_id, conversation_id, job_id)
        if record is None:
            raise KeyError(
                f"Chat job `{job_id}` not found for run `{run_id}` conversation `{conversation_id}`."
            )
        return record

    def _read_record_locked(
        self,
        run_id: str,
        conversation_id: str,
        job_id: str,
    ) -> ChatJobRecord | None:
        """Load one record from disk while holding the store lock."""
        payload = read_json_object(self.job_path(run_id, conversation_id, job_id))
        if not isinstance(payload, dict):
            return None
        return ChatJobRecord.from_payload(payload)

    def _write_record_locked(self, record: ChatJobRecord) -> None:
        """Persist one record while holding the store lock."""
        write_json(
            self.job_path(record.run_id, record.conversation_id, record.job_id),
            record.to_payload(),
            default=str,
        )


class ChatJobExecutor:
    """Run split-mode chat jobs in a dedicated background worker pool."""

    def __init__(
        self,
        *,
        job_store: ChatJobStore,
        chat_memory_store: ChatMemoryStore,
        processor: Callable[[StartChatJobCommand], ChatJobResult],
        max_workers: int = 1,
    ) -> None:
        self._job_store = job_store
        self._chat_memory_store = chat_memory_store
        self._processor = processor
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="chat-job-worker",
        )

    def submit(self, command: StartChatJobCommand) -> ChatJobRecord:
        """Persist and dispatch one queued split-mode chat job."""
        record = self._job_store.create_queued_job(command)
        logger.info(
            "Context chat split job queued run_id=%s conversation_id=%s job_id=%s job_number=%d",
            command.run_id,
            command.conversation_id,
            command.job_id,
            command.job_number,
        )
        self._executor.submit(self._execute, command)
        return record

    def reconcile_interrupted_jobs(self) -> None:
        """Mark leftover in-progress jobs as failed after an API restart."""
        interrupted_jobs = self._job_store.list_in_progress_jobs()
        for record in interrupted_jobs:
            logger.warning(
                "Context chat split job interrupted run_id=%s conversation_id=%s job_id=%s job_number=%d status=%s",
                record.run_id,
                record.conversation_id,
                record.job_id,
                record.job_number,
                record.status,
            )
            self._persist_failure(
                record,
                finish_reason="interrupted",
                error=RunError(
                    code="CHAT_JOB_INTERRUPTED",
                    message="Split-mode chat job was interrupted during API restart.",
                ),
                interrupted=True,
            )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown background chat-job workers."""
        self._executor.shutdown(wait=wait)

    def _execute(self, command: StartChatJobCommand) -> None:
        """Run one split-mode chat job and persist terminal state."""
        try:
            record = self._job_store.mark_running(
                command.run_id,
                command.conversation_id,
                command.job_id,
            )
            self._chat_memory_store.update_pending_job_status(
                command.run_id,
                command.conversation_id,
                command.job_id,
                "running",
            )
            logger.info(
                "Context chat split job running run_id=%s conversation_id=%s job_id=%s job_number=%d",
                record.run_id,
                record.conversation_id,
                record.job_id,
                record.job_number,
            )
            result = self._processor(command)
            self._chat_memory_store.append_assistant_message(
                command.run_id,
                command.conversation_id,
                result.assistant_content,
                citations=result.assistant_citations,
                citation_warning=result.assistant_citation_warning,
                routing=command.assistant_routing,
            )
            self._chat_memory_store.clear_pending_job(
                command.run_id,
                command.conversation_id,
                command.job_id,
            )
            completed = self._job_store.mark_terminal(
                command.run_id,
                command.conversation_id,
                command.job_id,
                status="completed",
                finish_reason="completed",
            )
            logger.info(
                "Context chat split job completed run_id=%s conversation_id=%s job_id=%s job_number=%d",
                completed.run_id,
                completed.conversation_id,
                completed.job_id,
                completed.job_number,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Context chat split job failed run_id=%s conversation_id=%s job_id=%s job_number=%d",
                command.run_id,
                command.conversation_id,
                command.job_id,
                command.job_number,
            )
            self._persist_failure(
                self._job_store.get_job(command.run_id, command.conversation_id, command.job_id)
                or ChatJobRecord(
                    job_id=command.job_id,
                    job_number=command.job_number,
                    run_id=command.run_id,
                    conversation_id=command.conversation_id,
                    status="running",
                    created_at=_utc_now(),
                    request_payload=command.to_request_payload(),
                ),
                finish_reason="chat_job_error",
                error=RunError(
                    code="CHAT_JOB_ERROR",
                    message=_normalize_error_message(str(exc)),
                ),
                interrupted=False,
            )

    def _persist_failure(
        self,
        record: ChatJobRecord,
        *,
        finish_reason: str,
        error: RunError,
        interrupted: bool,
    ) -> None:
        """Persist terminal failure state and a deterministic assistant reply."""
        try:
            self._chat_memory_store.append_assistant_message(
                record.run_id,
                record.conversation_id,
                build_chat_job_failure_message(interrupted=interrupted),
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to append split-job failure assistant message run_id=%s conversation_id=%s job_id=%s",
                record.run_id,
                record.conversation_id,
                record.job_id,
            )
        try:
            self._chat_memory_store.clear_pending_job(
                record.run_id,
                record.conversation_id,
                record.job_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to clear pending split-job pointer run_id=%s conversation_id=%s job_id=%s",
                record.run_id,
                record.conversation_id,
                record.job_id,
            )
        try:
            self._job_store.mark_terminal(
                record.run_id,
                record.conversation_id,
                record.job_id,
                status="failed",
                finish_reason=finish_reason,
                error=error,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to persist split-job terminal failure state run_id=%s conversation_id=%s job_id=%s",
                record.run_id,
                record.conversation_id,
                record.job_id,
            )
        logger.info(
            "Context chat split job failed run_id=%s conversation_id=%s job_id=%s job_number=%d finish_reason=%s",
            record.run_id,
            record.conversation_id,
            record.job_id,
            record.job_number,
            finish_reason,
        )


__all__ = [
    "ChatJobExecutor",
    "ChatJobRecord",
    "ChatJobResult",
    "ChatJobStatus",
    "ChatJobStore",
    "IN_PROGRESS_CHAT_JOB_STATUSES",
    "StartChatJobCommand",
    "build_chat_job_failure_message",
]
