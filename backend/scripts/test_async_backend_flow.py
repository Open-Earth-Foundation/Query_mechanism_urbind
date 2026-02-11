"""
Brief: Smoke-test async backend flow by triggering a run, polling status, and fetching output + context.

Inputs:
- CLI args:
  - `--base-url`: Backend base URL (for example `http://127.0.0.1:8000`).
  - `--question`: Question sent to `POST /api/v1/runs`.
  - `--run-id`: Optional run id to request.
  - `--cities`: Optional comma-separated city names to limit processed markdown files.
  - `--markdown-path`: Optional markdown directory override sent to backend.
  - `--config-path`: Optional config path override sent to backend.
  - `--log-llm-payload`: Request backend LLM payload logging.
  - `--start-path`: Start endpoint path.
  - `--status-path-template`: Status endpoint template with `{run_id}`.
  - `--output-path-template`: Output endpoint template with `{run_id}`.
  - `--context-path-template`: Context endpoint template with `{run_id}`.
  - `--chat-sessions-path-template`: Chat sessions endpoint template with `{run_id}`.
  - `--chat-message-path-template`: Chat message endpoint template with `{run_id}` and `{conversation_id}`.
  - `--exercise-chat`: Validate chat session + message endpoints after document generation.
  - `--chat-message`: Chat prompt used for endpoint validation when `--exercise-chat` is enabled.
  - `--poll-interval-seconds`: Poll interval for status checks.
  - `--max-wait-seconds`: Max time to wait for terminal status.
  - `--request-timeout-seconds`: HTTP timeout per request.
  - `--accepted-terminal-statuses`: Comma-separated successful terminal statuses.
  - `--artifacts-dir`: Directory where smoke-test API responses are saved.
- Files/paths:
  - Writes JSON artifacts under `--artifacts-dir/<run_id>/`.
- Env vars:
  - `LOG_LEVEL`: Logging verbosity.

Outputs:
- Logs API lifecycle checks and failures.
- Writes:
  - `smoke_start.json`
  - `smoke_status_history.json`
  - `smoke_output.json`
  - `smoke_context.json`
  - `smoke_chat_session.json` (when `--exercise-chat`)
  - `smoke_chat_message.json` (when `--exercise-chat`)
  - `smoke_summary.json`

Usage (from project root):
- python -m backend.scripts.test_async_backend_flow --question "What are main climate initiatives?"
- python -m backend.scripts.test_async_backend_flow --base-url http://127.0.0.1:8000 --question "What initiatives exist for Munich?" --poll-interval-seconds 3 --max-wait-seconds 1200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {
    "completed",
    "completed_with_gaps",
    "failed",
    "stopped",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the backend flow smoke test."""
    parser = argparse.ArgumentParser(
        description="Smoke-test async backend run lifecycle endpoints."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Backend base URL.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to send to the run trigger endpoint.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run id to request.",
    )
    parser.add_argument(
        "--cities",
        help="Optional comma-separated city names for backend filtering.",
    )
    parser.add_argument(
        "--markdown-path",
        help="Optional markdown path override sent to backend.",
    )
    parser.add_argument(
        "--config-path",
        help="Optional config path override sent to backend.",
    )
    parser.add_argument(
        "--log-llm-payload",
        action="store_true",
        default=False,
        help="Request backend LLM payload logging.",
    )
    parser.add_argument(
        "--start-path",
        default="/api/v1/runs",
        help="Run trigger endpoint path.",
    )
    parser.add_argument(
        "--status-path-template",
        default="/api/v1/runs/{run_id}/status",
        help="Status endpoint path template containing {run_id}.",
    )
    parser.add_argument(
        "--output-path-template",
        default="/api/v1/runs/{run_id}/output",
        help="Output endpoint path template containing {run_id}.",
    )
    parser.add_argument(
        "--context-path-template",
        default="/api/v1/runs/{run_id}/context",
        help="Context endpoint path template containing {run_id}.",
    )
    parser.add_argument(
        "--chat-sessions-path-template",
        default="/api/v1/runs/{run_id}/chat/sessions",
        help="Chat sessions endpoint path template containing {run_id}.",
    )
    parser.add_argument(
        "--chat-message-path-template",
        default="/api/v1/runs/{run_id}/chat/sessions/{conversation_id}/messages",
        help="Chat message endpoint template containing {run_id} and {conversation_id}.",
    )
    parser.add_argument(
        "--exercise-chat",
        action="store_true",
        default=False,
        help="Validate backend chat endpoints after run completion.",
    )
    parser.add_argument(
        "--chat-message",
        default="Summarize the context bundle in three bullets.",
        help="Prompt sent to chat endpoint when --exercise-chat is enabled.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=3.0,
        help="Seconds between status polls.",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=1200.0,
        help="Max seconds to wait for terminal status.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--accepted-terminal-statuses",
        default="completed,completed_with_gaps",
        help="Comma-separated terminal statuses considered successful.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("output") / "api_smoke_tests",
        help="Directory where smoke-test response artifacts are written.",
    )
    return parser.parse_args()


def _normalize_statuses(raw: str) -> set[str]:
    """Normalize comma-separated status names."""
    statuses = {item.strip() for item in raw.split(",") if item.strip()}
    if not statuses:
        raise ValueError("At least one accepted terminal status is required.")
    return statuses


def _join_url(base_url: str, path_or_url: str) -> str:
    """Join base URL with a path unless an absolute URL is already provided."""
    candidate = path_or_url.strip()
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    return f"{base_url.rstrip('/')}/{candidate.lstrip('/')}"


def _request_json(
    method: str,
    url: str,
    timeout_seconds: float,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    """Send an HTTP request and parse JSON response."""
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=data, method=method, headers=headers)

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {url} failed with status {exc.code}: {raw_error}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    if not raw_body.strip():
        return status_code, {}

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{method} {url} returned non-JSON response: {raw_body[:500]}"
        ) from exc

    if isinstance(parsed, dict):
        return status_code, parsed
    return status_code, {"data": parsed}


def _extract_run_id(payload: dict[str, object]) -> str:
    """Extract run id from API payload."""
    run_id = payload.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id.strip()
    raise RuntimeError("Start response is missing a non-empty `run_id`.")


def _resolve_endpoint_url(
    payload: dict[str, object],
    field_name: str,
    base_url: str,
    fallback_path: str,
) -> str:
    """Resolve endpoint URL from payload field or fallback path."""
    value = payload.get(field_name)
    if isinstance(value, str) and value.strip():
        return _join_url(base_url, value.strip())
    return _join_url(base_url, fallback_path)


def _extract_status(payload: dict[str, object]) -> str:
    """Extract status value from status endpoint payload."""
    candidates: list[object] = [payload.get("status"), payload.get("run_status")]
    nested_run = payload.get("run")
    if isinstance(nested_run, dict):
        candidates.append(nested_run.get("status"))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise RuntimeError("Status response does not include a recognized status field.")


def _write_json(path: Path, payload: object) -> None:
    """Write JSON payload to file with stable formatting."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def _build_start_payload(args: argparse.Namespace) -> dict[str, object]:
    """Build trigger endpoint payload from CLI args."""
    payload: dict[str, object] = {
        "question": args.question,
        "log_llm_payload": args.log_llm_payload,
    }
    if args.run_id:
        payload["run_id"] = args.run_id
    if args.cities:
        payload["cities"] = [item.strip() for item in args.cities.split(",") if item.strip()]
    if args.markdown_path:
        payload["markdown_path"] = args.markdown_path
    if args.config_path:
        payload["config_path"] = args.config_path
    return payload


def _poll_status_until_terminal(
    run_id: str,
    status_url: str,
    poll_interval_seconds: float,
    max_wait_seconds: float,
    request_timeout_seconds: float,
) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    """Poll run status endpoint until a terminal state is observed."""
    deadline = time.monotonic() + max_wait_seconds
    history: list[dict[str, object]] = []

    while True:
        status_code, payload = _request_json(
            "GET", status_url, timeout_seconds=request_timeout_seconds
        )
        status_value = _extract_status(payload)
        history.append(
            {
                "received_at": datetime.now(timezone.utc).isoformat(),
                "http_status": status_code,
                "status": status_value,
                "payload": payload,
            }
        )
        logger.info("Run %s status=%s", run_id, status_value)

        if status_value in TERMINAL_STATUSES:
            return status_value, payload, history

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Run {run_id} did not reach terminal state in {max_wait_seconds} seconds."
            )

        time.sleep(poll_interval_seconds)


def _validate_output_payload(payload: dict[str, object]) -> None:
    """Validate that output endpoint payload contains final content."""
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        return
    raise RuntimeError("Output response must include a non-empty `content` string.")


def _validate_context_payload(payload: dict[str, object]) -> None:
    """Validate that context endpoint payload contains context bundle."""
    context_bundle = payload.get("context_bundle")
    if isinstance(context_bundle, dict):
        return
    raise RuntimeError("Context response must include `context_bundle` as an object.")


def _validate_chat_session_payload(payload: dict[str, object]) -> str:
    """Validate chat session creation payload and return conversation id."""
    conversation_id = payload.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise RuntimeError("Chat session response must include `conversation_id`.")
    return conversation_id


def _validate_chat_message_payload(payload: dict[str, object]) -> None:
    """Validate chat message response payload."""
    assistant_message = payload.get("assistant_message")
    if not isinstance(assistant_message, dict):
        raise RuntimeError("Chat message response must include `assistant_message`.")
    content = assistant_message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "Chat message response must include non-empty assistant content."
        )


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    accepted_statuses = _normalize_statuses(args.accepted_terminal_statuses)
    base_url = args.base_url.rstrip("/")
    start_url = _join_url(base_url, args.start_path)
    start_payload = _build_start_payload(args)

    try:
        start_http_status, start_response = _request_json(
            "POST",
            start_url,
            timeout_seconds=args.request_timeout_seconds,
            payload=start_payload,
        )
        if start_http_status not in {200, 202}:
            raise RuntimeError(
                f"Unexpected start endpoint status: {start_http_status}."
            )

        run_id = _extract_run_id(start_response)
        status_url = _resolve_endpoint_url(
            start_response,
            "status_url",
            base_url,
            args.status_path_template.format(run_id=run_id),
        )
        output_url = _resolve_endpoint_url(
            start_response,
            "output_url",
            base_url,
            args.output_path_template.format(run_id=run_id),
        )
        context_url = _resolve_endpoint_url(
            start_response,
            "context_url",
            base_url,
            args.context_path_template.format(run_id=run_id),
        )
        chat_sessions_url = _join_url(
            base_url, args.chat_sessions_path_template.format(run_id=run_id)
        )

        logger.info("Started run_id=%s", run_id)
        logger.info("Polling status endpoint: %s", status_url)

        terminal_status, final_status_payload, status_history = _poll_status_until_terminal(
            run_id=run_id,
            status_url=status_url,
            poll_interval_seconds=args.poll_interval_seconds,
            max_wait_seconds=args.max_wait_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
        )

        if terminal_status not in accepted_statuses:
            raise RuntimeError(
                f"Run {run_id} finished with unexpected status `{terminal_status}`. "
                f"Accepted statuses: {sorted(accepted_statuses)}."
            )

        _, output_response = _request_json(
            "GET", output_url, timeout_seconds=args.request_timeout_seconds
        )
        _, context_response = _request_json(
            "GET", context_url, timeout_seconds=args.request_timeout_seconds
        )
        _validate_output_payload(output_response)
        _validate_context_payload(context_response)

        run_artifacts_dir = args.artifacts_dir / run_id
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        _write_json(run_artifacts_dir / "smoke_start.json", start_response)
        _write_json(run_artifacts_dir / "smoke_status_history.json", status_history)
        _write_json(run_artifacts_dir / "smoke_output.json", output_response)
        _write_json(run_artifacts_dir / "smoke_context.json", context_response)

        chat_session_response: dict[str, object] | None = None
        chat_message_response: dict[str, object] | None = None
        if args.exercise_chat:
            _, chat_session_response = _request_json(
                "POST",
                chat_sessions_url,
                timeout_seconds=args.request_timeout_seconds,
                payload={},
            )
            conversation_id = _validate_chat_session_payload(chat_session_response)
            chat_message_url = _join_url(
                base_url,
                args.chat_message_path_template.format(
                    run_id=run_id, conversation_id=conversation_id
                ),
            )
            _, chat_message_response = _request_json(
                "POST",
                chat_message_url,
                timeout_seconds=args.request_timeout_seconds,
                payload={"content": args.chat_message},
            )
            _validate_chat_message_payload(chat_message_response)
            _write_json(
                run_artifacts_dir / "smoke_chat_session.json", chat_session_response
            )
            _write_json(
                run_artifacts_dir / "smoke_chat_message.json", chat_message_response
            )

        _write_json(
            run_artifacts_dir / "smoke_summary.json",
            {
                "run_id": run_id,
                "start_http_status": start_http_status,
                "terminal_status": terminal_status,
                "status_url": status_url,
                "output_url": output_url,
                "context_url": context_url,
                "chat_sessions_url": chat_sessions_url,
                "chat_exercised": args.exercise_chat,
                "accepted_terminal_statuses": sorted(accepted_statuses),
                "artifacts_dir": str(run_artifacts_dir),
                "final_status_payload": final_status_payload,
                "chat_session_response": chat_session_response,
                "chat_message_response": chat_message_response,
            },
        )

        logger.info(
            "Smoke test passed for run_id=%s. Artifacts saved to %s",
            run_id,
            run_artifacts_dir,
        )
    except (RuntimeError, TimeoutError, ValueError) as exc:
        logger.exception("Smoke test failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
