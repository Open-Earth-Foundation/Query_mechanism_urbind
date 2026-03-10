"""Persistent run-scoped chat session memory."""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.utils.json_io import read_json_object, write_json

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _utc_now_iso() -> str:
    """Return ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


class ChatSessionExistsError(ValueError):
    """Raised when creating a chat session with existing id."""


class ChatSessionNotFoundError(FileNotFoundError):
    """Raised when chat session file is missing."""


class ChatSessionPendingJobError(RuntimeError):
    """Raised when a session already has an active pending chat job."""


class ChatMemoryStore:
    """Read/write chat transcripts stored under each run directory."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir
        self._lock = threading.Lock()

    def list_sessions(self, run_id: str) -> list[str]:
        """List conversation identifiers for a run."""
        chat_dir = self._chat_dir(run_id)
        if not chat_dir.exists():
            return []
        return sorted(path.stem for path in chat_dir.glob("*.json") if path.is_file())

    def create_session(self, run_id: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Create empty chat session payload and persist it."""
        with self._lock:
            resolved_id = self._resolve_conversation_id(conversation_id)
            path = self._session_path(run_id, resolved_id)
            if path.exists():
                raise ChatSessionExistsError(
                    f"Conversation `{resolved_id}` already exists for run `{run_id}`."
                )
            now = _utc_now_iso()
            payload = {
                "run_id": run_id,
                "conversation_id": resolved_id,
                "created_at": now,
                "updated_at": now,
                "context_run_ids": [run_id],
                "followup_bundles": [],
                "prompt_context_cache": None,
                "next_job_number": 1,
                "pending_job": None,
                "messages": [],
            }
            write_json(path, payload, default=str)
            return payload

    def get_session(self, run_id: str, conversation_id: str) -> dict[str, Any]:
        """Read session payload from disk."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = read_json_object(path)
            if payload is None:
                raise ChatSessionNotFoundError(
                    f"Conversation `{conversation_id}` not found for run `{run_id}`."
                )
            return payload

    def append_turn(
        self,
        run_id: str,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        assistant_citations: list[dict[str, object]] | None = None,
        assistant_citation_warning: str | None = None,
        assistant_routing: dict[str, object] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Append a user/assistant turn and persist session."""
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            user_message = self._build_message("user", user_content)
            assistant_message = self._build_message("assistant", assistant_content)
            if assistant_citations:
                assistant_message["citations"] = assistant_citations
            if assistant_citation_warning:
                assistant_message["citation_warning"] = assistant_citation_warning
            if assistant_routing:
                assistant_message["routing"] = assistant_routing
            messages = self._messages_list(payload)
            messages.append(user_message)
            messages.append(assistant_message)
            payload["updated_at"] = assistant_message["created_at"]
            write_json(self._session_path(run_id, conversation_id), payload, default=str)
            return payload, user_message, assistant_message

    def append_user_message(
        self,
        run_id: str,
        conversation_id: str,
        content: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Append one user message and persist session."""
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            user_message = self._build_message("user", content)
            self._messages_list(payload).append(user_message)
            payload["updated_at"] = user_message["created_at"]
            write_json(self._session_path(run_id, conversation_id), payload, default=str)
            return payload, user_message

    def append_assistant_message(
        self,
        run_id: str,
        conversation_id: str,
        content: str,
        citations: list[dict[str, object]] | None = None,
        citation_warning: str | None = None,
        routing: dict[str, object] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Append one assistant message and persist session."""
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            assistant_message = self._build_message("assistant", content)
            if citations:
                assistant_message["citations"] = citations
            if citation_warning:
                assistant_message["citation_warning"] = citation_warning
            if routing:
                assistant_message["routing"] = routing
            self._messages_list(payload).append(assistant_message)
            payload["updated_at"] = assistant_message["created_at"]
            write_json(self._session_path(run_id, conversation_id), payload, default=str)
            return payload, assistant_message

    def update_context_runs(
        self,
        run_id: str,
        conversation_id: str,
        context_run_ids: list[str],
    ) -> dict[str, Any]:
        """Persist selected context run ids for a chat session."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            normalized = self._normalize_context_run_ids(run_id, context_run_ids)
            payload["context_run_ids"] = normalized
            payload["prompt_context_cache"] = None
            payload["updated_at"] = _utc_now_iso()
            write_json(path, payload, default=str)
            return payload

    def attach_followup_bundle(
        self,
        run_id: str,
        conversation_id: str,
        bundle_id: str,
        city_key: str,
        target_city: str,
        created_at: str,
        max_followup_bundles: int,
    ) -> dict[str, Any]:
        """Attach a new follow-up bundle, replacing older bundles for the same city."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            existing = self._normalize_followup_bundles(payload.get("followup_bundles"))
            filtered = [entry for entry in existing if entry["city_key"] != city_key]
            filtered.insert(
                0,
                {
                    "bundle_id": self._resolve_session_id_value(bundle_id, "bundle_id"),
                    "city_key": city_key,
                    "target_city": target_city.strip(),
                    "created_at": created_at.strip(),
                },
            )
            payload["followup_bundles"] = self._trim_followup_bundles(
                filtered,
                max_followup_bundles,
            )
            payload["prompt_context_cache"] = None
            payload["updated_at"] = _utc_now_iso()
            write_json(path, payload, default=str)
            return payload

    def prune_followup_bundles(
        self,
        run_id: str,
        conversation_id: str,
        keep_bundle_ids: list[str],
    ) -> dict[str, Any]:
        """Persist only the selected follow-up bundles for a session."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            keep = {
                self._resolve_session_id_value(bundle_id, "bundle_id")
                for bundle_id in keep_bundle_ids
                if isinstance(bundle_id, str) and bundle_id.strip()
            }
            existing = self._normalize_followup_bundles(payload.get("followup_bundles"))
            payload["followup_bundles"] = [
                entry for entry in existing if entry["bundle_id"] in keep
            ]
            payload["prompt_context_cache"] = None
            payload["updated_at"] = _utc_now_iso()
            write_json(path, payload, default=str)
            return payload

    def update_prompt_context_cache(
        self,
        run_id: str,
        conversation_id: str,
        prompt_context_cache: dict[str, object] | None,
    ) -> dict[str, Any]:
        """Persist the combined prompt-context cache for a session selection."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            payload["prompt_context_cache"] = (
                dict(prompt_context_cache) if isinstance(prompt_context_cache, dict) else None
            )
            write_json(path, payload, default=str)
            return payload

    def create_pending_job(
        self,
        run_id: str,
        conversation_id: str,
        job_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Persist one pending chat job and reserve the next session-local number."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            existing = self._normalize_pending_job(payload.get("pending_job"))
            if existing is not None:
                raise ChatSessionPendingJobError(
                    "This chat session already has a pending long-context answer."
                )
            next_job_number = self._next_job_number(payload)
            now = _utc_now_iso()
            pending_job = {
                "job_id": self._resolve_session_id_value(job_id, "job_id"),
                "job_number": next_job_number,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
            }
            payload["pending_job"] = pending_job
            payload["next_job_number"] = next_job_number + 1
            payload["updated_at"] = now
            write_json(path, payload, default=str)
            return payload, pending_job

    def update_pending_job_status(
        self,
        run_id: str,
        conversation_id: str,
        job_id: str,
        status: str,
    ) -> dict[str, Any]:
        """Update the persisted status of the active pending job."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            pending_job = self._normalize_pending_job(payload.get("pending_job"))
            if pending_job is None or pending_job["job_id"] != job_id:
                raise ChatSessionPendingJobError(
                    "Pending chat job no longer matches the requested session job."
                )
            pending_job["status"] = status.strip()
            pending_job["updated_at"] = _utc_now_iso()
            payload["pending_job"] = pending_job
            payload["updated_at"] = pending_job["updated_at"]
            write_json(path, payload, default=str)
            return payload

    def clear_pending_job(
        self,
        run_id: str,
        conversation_id: str,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Clear the active pending chat job when it matches the expected id."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = self._load_session_payload_locked(run_id, conversation_id)
            pending_job = self._normalize_pending_job(payload.get("pending_job"))
            if pending_job is None:
                return payload
            if job_id is not None and pending_job["job_id"] != job_id:
                return payload
            payload["pending_job"] = None
            payload["updated_at"] = _utc_now_iso()
            write_json(path, payload, default=str)
            return payload

    def _chat_dir(self, run_id: str) -> Path:
        """Return chat directory path for a run."""
        return self._runs_dir / run_id / "chat"

    def _session_path(self, run_id: str, conversation_id: str) -> Path:
        """Resolve chat session file path."""
        cleaned = self._resolve_conversation_id(conversation_id)
        return self._chat_dir(run_id) / f"{cleaned}.json"

    def _resolve_conversation_id(self, conversation_id: str | None) -> str:
        """Validate provided conversation id or generate a new one."""
        if conversation_id is None:
            return uuid4().hex
        cleaned = conversation_id.strip()
        if not cleaned:
            raise ValueError("conversation_id must be non-empty when provided.")
        return self._resolve_session_id_value(cleaned, "conversation_id")

    def _resolve_session_id_value(self, value: str, field_name: str) -> str:
        """Validate a stored run/chat identifier against the session id pattern."""
        cleaned = value.strip()
        if not SESSION_ID_PATTERN.fullmatch(cleaned):
            raise ValueError(
                f"{field_name} may contain only letters, numbers, underscore, and hyphen."
            )
        return cleaned

    def _normalize_context_run_ids(self, run_id: str, context_run_ids: list[str]) -> list[str]:
        """Validate, de-duplicate, and normalize selected context run ids."""
        normalized: list[str] = []
        seen: set[str] = set()
        normalized.append(self._resolve_session_id_value(run_id, "run_id"))
        seen.add(normalized[0])
        for context_run_id in context_run_ids:
            if not isinstance(context_run_id, str):
                continue
            cleaned = context_run_id.strip()
            if not cleaned:
                continue
            cleaned = self._resolve_session_id_value(cleaned, "context_run_ids")
            if cleaned in seen:
                continue
            normalized.append(cleaned)
            seen.add(cleaned)
        return normalized

    def _normalize_followup_bundles(self, value: object) -> list[dict[str, str]]:
        """Validate and normalize persisted follow-up bundle metadata."""
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            raw_bundle_id = item.get("bundle_id")
            raw_city_key = item.get("city_key")
            raw_target_city = item.get("target_city")
            raw_created_at = item.get("created_at")
            if not isinstance(raw_bundle_id, str) or not raw_bundle_id.strip():
                continue
            if not isinstance(raw_city_key, str) or not raw_city_key.strip():
                continue
            if not isinstance(raw_target_city, str) or not raw_target_city.strip():
                continue
            if not isinstance(raw_created_at, str) or not raw_created_at.strip():
                continue
            bundle_id = self._resolve_session_id_value(raw_bundle_id, "bundle_id")
            if bundle_id in seen:
                continue
            seen.add(bundle_id)
            normalized.append(
                {
                    "bundle_id": bundle_id,
                    "city_key": raw_city_key.strip(),
                    "target_city": raw_target_city.strip(),
                    "created_at": raw_created_at.strip(),
                }
            )
        return normalized

    def _trim_followup_bundles(
        self,
        bundles: list[dict[str, str]],
        max_followup_bundles: int,
    ) -> list[dict[str, str]]:
        """Trim follow-up bundles to the configured maximum."""
        if max_followup_bundles <= 0:
            return []
        return bundles[:max_followup_bundles]

    def _load_session_payload_locked(self, run_id: str, conversation_id: str) -> dict[str, Any]:
        """Load one session payload while holding the store lock."""
        path = self._session_path(run_id, conversation_id)
        payload = read_json_object(path)
        if payload is None:
            raise ChatSessionNotFoundError(
                f"Conversation `{conversation_id}` not found for run `{run_id}`."
            )
        payload["pending_job"] = self._normalize_pending_job(payload.get("pending_job"))
        payload["next_job_number"] = self._next_job_number(payload)
        self._messages_list(payload)
        return payload

    def _messages_list(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the mutable message list stored in the session payload."""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = []
            payload["messages"] = messages
        return messages

    def _build_message(self, role: str, content: str) -> dict[str, Any]:
        """Build one persisted chat message payload."""
        return {
            "role": role,
            "content": content,
            "created_at": _utc_now_iso(),
        }

    def _next_job_number(self, payload: dict[str, Any]) -> int:
        """Return the next valid session-local chat job number."""
        raw_value = payload.get("next_job_number")
        if isinstance(raw_value, int) and raw_value > 0:
            return raw_value
        return 1

    def _normalize_pending_job(self, value: object) -> dict[str, Any] | None:
        """Validate the persisted pending-job payload."""
        if not isinstance(value, dict):
            return None
        raw_job_id = value.get("job_id")
        raw_job_number = value.get("job_number")
        raw_status = value.get("status")
        raw_created_at = value.get("created_at")
        raw_updated_at = value.get("updated_at")
        if not isinstance(raw_job_id, str) or not raw_job_id.strip():
            return None
        if not isinstance(raw_job_number, int) or raw_job_number <= 0:
            return None
        if not isinstance(raw_status, str) or not raw_status.strip():
            return None
        if not isinstance(raw_created_at, str) or not raw_created_at.strip():
            return None
        if not isinstance(raw_updated_at, str) or not raw_updated_at.strip():
            return None
        return {
            "job_id": self._resolve_session_id_value(raw_job_id, "job_id"),
            "job_number": raw_job_number,
            "status": raw_status.strip(),
            "created_at": raw_created_at.strip(),
            "updated_at": raw_updated_at.strip(),
        }


__all__ = [
    "ChatMemoryStore",
    "ChatSessionExistsError",
    "ChatSessionNotFoundError",
    "ChatSessionPendingJobError",
]
