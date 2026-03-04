from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic type variable for call_with_retries; allows the function to work with any return type while preserving type safety


@dataclass(frozen=True)
class RetrySettings:
    """Shared retry settings used by backend components."""

    max_attempts: int
    backoff_base_seconds: float
    backoff_max_seconds: float

    @classmethod
    def bounded(
        cls,
        *,
        max_attempts: int,
        backoff_base_seconds: float,
        backoff_max_seconds: float,
    ) -> RetrySettings:
        resolved_attempts = max(int(max_attempts), 1)
        resolved_base = max(float(backoff_base_seconds), 0.0)
        resolved_max = max(float(backoff_max_seconds), resolved_base)
        return cls(
            max_attempts=resolved_attempts,
            backoff_base_seconds=resolved_base,
            backoff_max_seconds=resolved_max,
        )


_STATUS_FROM_MESSAGE_PATTERNS = (
    re.compile(r"\bHTTP(?:/[0-9.]+)?\s+(\d{3})\b", flags=re.IGNORECASE),
    re.compile(r"\bstatus(?:_code)?\s*[:=]\s*(\d{3})\b", flags=re.IGNORECASE),
    re.compile(r"\berror code\s*[:=]\s*(\d{3})\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(\d{3})\s+("
        r"Too Many Requests|Not Found|Bad Request|Unauthorized|Forbidden|"
        r"Service Unavailable|Gateway Timeout|Internal Server Error"
        r")\b",
        flags=re.IGNORECASE,
    ),
)


def _sanitize_log_text(value: str, *, max_chars: int = 500) -> str:
    """Normalize text for single-line log output."""
    collapsed = " ".join(value.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."


def _quote(value: str) -> str:
    """Quote free-text values for key=value logging."""
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _extract_http_status_from_context(context: dict[str, object] | None) -> int | None:
    """Try to read an HTTP status code from retry context payload."""
    if not isinstance(context, dict):
        return None
    for key in ("http_status", "status_code", "http_status_code", "provider_status_code"):
        value = context.get(key)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.isdigit():
                numeric = int(candidate)
                if 100 <= numeric <= 599:
                    return numeric
    return None


def _extract_http_status_from_message(error_message: str) -> int | None:
    """Best-effort extraction of HTTP status from an error message."""
    for pattern in _STATUS_FROM_MESSAGE_PATTERNS:
        match = pattern.search(error_message)
        if not match:
            continue
        candidate = match.group(1)
        if candidate.isdigit():
            numeric = int(candidate)
            if 100 <= numeric <= 599:
                return numeric
    return None


def _is_rate_limited(
    *,
    error_type: str,
    error_message: str,
    http_status: int | None,
) -> bool:
    """Return True when retry signals provider-side rate limiting."""
    if http_status == 429:
        return True
    lowered_type = error_type.casefold()
    lowered_message = error_message.casefold()
    return (
        "ratelimit" in lowered_type
        or "rate_limit" in lowered_type
        or "rate limit" in lowered_type
        or "too many requests" in lowered_message
        or "rate limit" in lowered_message
        or "ratelimit" in lowered_message
    )


def _derive_retry_reason(
    *,
    error_type: str,
    error_message: str,
    http_status: int | None,
) -> str:
    """Create a concise human-readable retry reason."""
    lowered_message = error_message.casefold()
    if _is_rate_limited(
        error_type=error_type,
        error_message=error_message,
        http_status=http_status,
    ):
        return "provider rate limit"
    if http_status is not None:
        return f"provider HTTP {http_status}"
    if "max turns" in lowered_message:
        return "agent max turns exceeded"
    if error_type == "RetryableBadOutput":
        return "retryable model output validation failure"
    cleaned = _sanitize_log_text(error_message)
    if cleaned:
        return cleaned
    return f"retry triggered with error type {error_type}"


def _format_retry_context(context: dict[str, object] | None) -> str:
    """Format context map as stable key=value pairs."""
    if not isinstance(context, dict) or not context:
        return "none"
    parts: list[str] = []
    for key in sorted(context.keys()):
        value = context[key]
        if isinstance(value, list):
            rendered = ",".join(_sanitize_log_text(str(item), max_chars=60) for item in value[:10])
            if len(value) > 10:
                rendered = f"{rendered},...(+{len(value)-10})"
        elif isinstance(value, dict):
            rendered = _sanitize_log_text(str(value), max_chars=120)
        else:
            rendered = _sanitize_log_text(str(value), max_chars=120)
        parts.append(f"{key}={rendered}")
    return "; ".join(parts)


def _format_retry_log_line(payload: dict[str, object]) -> str:
    """Render retry payload as single-line key=value text (no JSON)."""
    operation = str(payload.get("operation", "unknown"))
    run_id = str(payload.get("run_id", "unknown"))
    attempt = int(payload.get("attempt", 0))
    max_attempts = int(payload.get("max_attempts", 0))
    error_type = _sanitize_log_text(str(payload.get("error_type", "unknown")), max_chars=120)
    error_message = str(payload.get("error_message", ""))
    next_backoff_seconds = payload.get("next_backoff_seconds")
    context = payload.get("context")
    typed_context = context if isinstance(context, dict) else None
    http_status = _extract_http_status_from_context(
        typed_context
    ) or _extract_http_status_from_message(error_message)
    rate_limited = _is_rate_limited(
        error_type=error_type,
        error_message=error_message,
        http_status=http_status,
    )
    reason = _derive_retry_reason(
        error_type=error_type,
        error_message=error_message,
        http_status=http_status,
    )
    backoff_text = (
        f"{float(next_backoff_seconds):.3f}"
        if isinstance(next_backoff_seconds, (int, float))
        else "none"
    )
    context_text = _format_retry_context(typed_context)
    message_text = _sanitize_log_text(error_message)
    status_text = str(http_status) if http_status is not None else "n/a"
    return (
        f"operation={operation} run_id={run_id} attempt={attempt}/{max_attempts} "
        f"error=true error_type={error_type} reason={_quote(reason)} "
        f"http_status={status_text} rate_limited={'true' if rate_limited else 'false'} "
        f"next_backoff_seconds={backoff_text} "
        f"error_message={_quote(message_text)} context={_quote(context_text)}"
    )


def compute_retry_delay_seconds(attempt: int, settings: RetrySettings) -> float:
    """Compute exponential backoff with light jitter for a given failed attempt."""
    exponent = max(attempt - 1, 0)
    base_delay = min(
        settings.backoff_max_seconds,
        settings.backoff_base_seconds * (2**exponent),
    )
    jitter = random.uniform(0.0, base_delay * 0.1) if base_delay > 0 else 0.0
    return base_delay + jitter


def _build_retry_payload(
    *,
    operation: str,
    run_id: str | None,
    attempt: int,
    max_attempts: int,
    error_type: str,
    error_message: str,
    next_backoff_seconds: float | None,
    context: dict[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "operation": operation,
        "run_id": run_id or "unknown",
        "attempt": attempt,
        "max_attempts": max_attempts,
        "error_type": error_type,
        "error_message": error_message,
        "next_backoff_seconds": next_backoff_seconds,
    }
    if context:
        payload["context"] = context
    return payload


def log_retry_event(
    *,
    operation: str,
    run_id: str | None,
    attempt: int,
    max_attempts: int,
    error_type: str,
    error_message: str,
    next_backoff_seconds: float | None,
    context: dict[str, object] | None = None,
) -> None:
    """Log one retry event in a structured single-line text format."""
    payload = _build_retry_payload(
        operation=operation,
        run_id=run_id,
        attempt=attempt,
        max_attempts=max_attempts,
        error_type=error_type,
        error_message=error_message,
        next_backoff_seconds=next_backoff_seconds,
        context=context,
    )
    logger.warning("RETRY_EVENT %s", _format_retry_log_line(payload))


def log_retry_exhausted(
    *,
    operation: str,
    run_id: str | None,
    attempt: int,
    max_attempts: int,
    error_type: str,
    error_message: str,
    context: dict[str, object] | None = None,
) -> None:
    """Log when retry attempts are exhausted."""
    payload = _build_retry_payload(
        operation=operation,
        run_id=run_id,
        attempt=attempt,
        max_attempts=max_attempts,
        error_type=error_type,
        error_message=error_message,
        next_backoff_seconds=None,
        context=context,
    )
    logger.error("RETRY_EXHAUSTED %s", _format_retry_log_line(payload))


def call_with_retries(
    func: Callable[[], T],
    *,
    operation: str,
    retry_settings: RetrySettings,
    should_retry: Callable[[Exception], bool] | None = None,
    run_id: str | None = None,
    context: dict[str, object] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Execute callable with retry/backoff and structured retry logs."""
    attempts = retry_settings.max_attempts
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if should_retry is not None and not should_retry(exc):
                raise
            if attempt >= attempts:
                log_retry_exhausted(
                    operation=operation,
                    run_id=run_id,
                    attempt=attempt,
                    max_attempts=attempts,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    context=context,
                )
                raise
            delay_seconds = compute_retry_delay_seconds(attempt, retry_settings)
            log_retry_event(
                operation=operation,
                run_id=run_id,
                attempt=attempt,
                max_attempts=attempts,
                error_type=type(exc).__name__,
                error_message=str(exc),
                next_backoff_seconds=delay_seconds,
                context=context,
            )
            if delay_seconds > 0:
                sleep(delay_seconds)
    raise RuntimeError("Retry loop unexpectedly completed without returning.")


__all__ = [
    "RetrySettings",
    "compute_retry_delay_seconds",
    "log_retry_event",
    "log_retry_exhausted",
    "call_with_retries",
]
