from __future__ import annotations

import json
import logging
import random
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
    """Log one retry event in a structured single-line format."""
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
    logger.warning("RETRY_EVENT %s", json.dumps(payload, ensure_ascii=False, default=str))


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
    logger.error("RETRY_EXHAUSTED %s", json.dumps(payload, ensure_ascii=False, default=str))


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
