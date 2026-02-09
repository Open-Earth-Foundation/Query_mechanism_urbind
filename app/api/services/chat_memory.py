"""Persistent run-scoped chat session memory."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _utc_now_iso() -> str:
    """Return ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from disk."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


class ChatSessionExistsError(ValueError):
    """Raised when creating a chat session with existing id."""


class ChatSessionNotFoundError(FileNotFoundError):
    """Raised when chat session file is missing."""


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
                "messages": [],
            }
            _write_json(path, payload)
            return payload

    def get_session(self, run_id: str, conversation_id: str) -> dict[str, Any]:
        """Read session payload from disk."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = _read_json(path)
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
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Append a user/assistant turn and persist session."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = _read_json(path)
            if payload is None:
                raise ChatSessionNotFoundError(
                    f"Conversation `{conversation_id}` not found for run `{run_id}`."
                )
            messages = payload.get("messages")
            if not isinstance(messages, list):
                messages = []
                payload["messages"] = messages

            user_message = {
                "role": "user",
                "content": user_content,
                "created_at": _utc_now_iso(),
            }
            assistant_message = {
                "role": "assistant",
                "content": assistant_content,
                "created_at": _utc_now_iso(),
            }
            messages.append(user_message)
            messages.append(assistant_message)
            payload["updated_at"] = assistant_message["created_at"]
            _write_json(path, payload)
            return payload, user_message, assistant_message

    def update_context_runs(
        self,
        run_id: str,
        conversation_id: str,
        context_run_ids: list[str],
    ) -> dict[str, Any]:
        """Persist selected context run ids for a chat session."""
        path = self._session_path(run_id, conversation_id)
        with self._lock:
            payload = _read_json(path)
            if payload is None:
                raise ChatSessionNotFoundError(
                    f"Conversation `{conversation_id}` not found for run `{run_id}`."
                )
            normalized = self._normalize_context_run_ids(context_run_ids)
            payload["context_run_ids"] = normalized
            payload["updated_at"] = _utc_now_iso()
            _write_json(path, payload)
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
        if not SESSION_ID_PATTERN.fullmatch(cleaned):
            raise ValueError(
                "conversation_id may contain only letters, numbers, underscore, and hyphen."
            )
        return cleaned

    def _normalize_context_run_ids(self, context_run_ids: list[str]) -> list[str]:
        """Validate, de-duplicate, and normalize selected context run ids."""
        normalized: list[str] = []
        seen: set[str] = set()
        for run_id in context_run_ids:
            if not isinstance(run_id, str):
                continue
            cleaned = run_id.strip()
            if not cleaned:
                continue
            if not SESSION_ID_PATTERN.fullmatch(cleaned):
                raise ValueError(
                    "context_run_ids may contain only letters, numbers, underscore, and hyphen."
                )
            if cleaned in seen:
                continue
            normalized.append(cleaned)
            seen.add(cleaned)
        if not normalized:
            raise ValueError("context_run_ids must include at least one run id.")
        return normalized


__all__ = [
    "ChatMemoryStore",
    "ChatSessionExistsError",
    "ChatSessionNotFoundError",
]
