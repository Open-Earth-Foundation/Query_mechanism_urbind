"""Run-local LLM call recording for MLflow and artifact inspection."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.utils.artifact_writer import resolve_stage_number, stage_file_dir_name
from backend.utils.json_io import write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmCallContext:
    """Stable metadata attached to one LLM invocation."""

    stage_name: str
    stage_family: str
    agent: str
    call_kind: str
    provider: str = "openrouter"
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    """Return a stable filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip().lower())
    return slug.strip("._") or "llm_call"


def safe_serialize(value: Any) -> Any:
    """Serialize SDK and Pydantic objects into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): safe_serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_serialize(item) for item in value]
    if dataclasses.is_dataclass(value):
        try:
            return safe_serialize(dataclasses.asdict(value))
        except Exception:  # noqa: BLE001
            return str(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return safe_serialize(model_dump())
        except Exception:  # noqa: BLE001
            pass
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return safe_serialize(to_dict())
        except Exception:  # noqa: BLE001
            pass
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        try:
            filtered = {
                key: item for key, item in value_dict.items() if not key.startswith("_")
            }
            return safe_serialize(filtered)
        except Exception:  # noqa: BLE001
            pass
    return str(value)


class LlmCallRecorder:
    """Write full LLM request and response artifacts for one run."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.index_path = run_dir / "llm_calls" / "index.jsonl"
        self._lock = threading.Lock()
        self._next_index = self._load_next_index()

    def _load_next_index(self) -> int:
        """Return the next call index after any existing index rows."""
        if not self.index_path.exists():
            return 1
        try:
            with self.index_path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip()) + 1
        except OSError:
            return 1

    def _call_path(self, call_index: int, context: LlmCallContext) -> Path:
        """Return the stage-local artifact path for one LLM call."""
        stage_dir = self.run_dir / "stage_files" / stage_file_dir_name(context.stage_name)
        filename = (
            f"{call_index:04d}_{_safe_name(context.agent)}_"
            f"{_safe_name(context.call_kind)}.json"
        )
        return stage_dir / "llm_calls" / filename

    def record_call(
        self,
        context: LlmCallContext,
        *,
        request: object,
        response: object | None = None,
        error: BaseException | dict[str, object] | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> Path:
        """Persist one request/response pair and append it to the run index."""
        started = started_at or _utc_now_iso()
        ended = ended_at or _utc_now_iso()
        error_payload: dict[str, object] | None = None
        status = "success"
        if error is not None:
            status = "error"
            if isinstance(error, BaseException):
                error_payload = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            else:
                error_payload = dict(error)

        with self._lock:
            call_index = self._next_index
            self._next_index += 1
            path = self._call_path(call_index, context)
            relative_path = path.relative_to(self.run_dir).as_posix()
            payload = {
                "call_index": call_index,
                "run_id": self.run_id,
                "stage_number": resolve_stage_number(context.stage_name),
                "stage_name": context.stage_name,
                "stage_family": context.stage_family,
                "agent": context.agent,
                "call_kind": context.call_kind,
                "provider": context.provider,
                "model": context.model,
                "status": status,
                "started_at": started,
                "ended_at": ended,
                "metadata": safe_serialize(context.metadata),
                "request": safe_serialize(request),
                "response": safe_serialize(response),
                "error": safe_serialize(error_payload),
            }
            write_json(path, payload, ensure_ascii=False, default=str)
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            index_row = {
                "call_index": call_index,
                "run_id": self.run_id,
                "stage_number": payload["stage_number"],
                "stage_name": context.stage_name,
                "stage_family": context.stage_family,
                "agent": context.agent,
                "call_kind": context.call_kind,
                "provider": context.provider,
                "model": context.model,
                "status": status,
                "started_at": started,
                "ended_at": ended,
                "path": relative_path,
            }
            with self.index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(index_row, ensure_ascii=False, default=str) + "\n")
            return path

    def read_call_records(self) -> list[dict[str, Any]]:
        """Load recorded call payloads in index order."""
        if not self.index_path.exists():
            return []

        records: list[dict[str, Any]] = []
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_path = row.get("path")
                if not isinstance(raw_path, str):
                    continue
                path = self.run_dir / raw_path
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    logger.warning("Could not read LLM call artifact: %s", path)
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records


def record_openai_chat_completion(
    client: Any,
    request_kwargs: dict[str, object],
    *,
    context: LlmCallContext,
    recorder: LlmCallRecorder | None,
) -> Any:
    """Call ``chat.completions.create`` and record the raw payload when enabled."""
    if recorder is None:
        return client.chat.completions.create(**request_kwargs)

    started_at = _utc_now_iso()
    try:
        response = client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        recorder.record_call(
            context,
            request=request_kwargs,
            error=exc,
            started_at=started_at,
            ended_at=_utc_now_iso(),
        )
        raise

    recorder.record_call(
        context,
        request=request_kwargs,
        response=response,
        started_at=started_at,
        ended_at=_utc_now_iso(),
    )
    return response


__all__ = [
    "LlmCallContext",
    "LlmCallRecorder",
    "record_openai_chat_completion",
    "safe_serialize",
]
