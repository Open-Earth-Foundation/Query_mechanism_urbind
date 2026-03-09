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
                "followup_bundles": [],
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
        assistant_citations: list[dict[str, object]] | None = None,
        assistant_citation_warning: str | None = None,
        assistant_routing: dict[str, object] | None = None,
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
            if assistant_citations:
                assistant_message["citations"] = assistant_citations
            if assistant_citation_warning:
                assistant_message["citation_warning"] = assistant_citation_warning
            if assistant_routing:
                assistant_message["routing"] = assistant_routing
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
            normalized = self._normalize_context_run_ids(run_id, context_run_ids)
            payload["context_run_ids"] = normalized
            payload["updated_at"] = _utc_now_iso()
            _write_json(path, payload)
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
            payload = _read_json(path)
            if payload is None:
                raise ChatSessionNotFoundError(
                    f"Conversation `{conversation_id}` not found for run `{run_id}`."
                )
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
            payload["updated_at"] = _utc_now_iso()
            _write_json(path, payload)
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
            payload = _read_json(path)
            if payload is None:
                raise ChatSessionNotFoundError(
                    f"Conversation `{conversation_id}` not found for run `{run_id}`."
                )
            keep = {
                self._resolve_session_id_value(bundle_id, "bundle_id")
                for bundle_id in keep_bundle_ids
                if isinstance(bundle_id, str) and bundle_id.strip()
            }
            existing = self._normalize_followup_bundles(payload.get("followup_bundles"))
            payload["followup_bundles"] = [
                entry for entry in existing if entry["bundle_id"] in keep
            ]
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


__all__ = [
    "ChatMemoryStore",
    "ChatSessionExistsError",
    "ChatSessionNotFoundError",
]
