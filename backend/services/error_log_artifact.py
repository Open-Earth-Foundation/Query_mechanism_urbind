"""Helpers for extracting and writing error-focused run log artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_NO_ERRORS_MESSAGE = "No ERROR/CRITICAL or RETRY_EXHAUSTED entries found in run.log."
_ERROR_MARKERS = (" - ERROR - ", " - CRITICAL - ", "RETRY_EXHAUSTED ")


def _is_log_entry_start(line: str) -> bool:
    """Return True if the line opens a new log record (starts with a timestamp year)."""
    return len(line) > 4 and line[:4].isdigit() and line[4] == "-"


def write_error_log_artifact(run_log_path: Path, output_path: Path) -> Path | None:
    """Write filtered error lines from ``run.log`` to ``error_log.txt``.

    Captures each ERROR/CRITICAL/RETRY_EXHAUSTED log line together with any
    continuation lines that follow it (e.g. the full Python traceback), stopping
    when the next timestamped log entry begins.
    """
    if not run_log_path.exists():
        return None

    selected_lines: list[str] = []
    in_error_block = False
    try:
        with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                if _is_log_entry_start(stripped):
                    in_error_block = any(marker in stripped for marker in _ERROR_MARKERS)
                    if in_error_block:
                        selected_lines.append(stripped)
                elif in_error_block:
                    selected_lines.append(stripped)
    except OSError:
        logger.exception("Failed to read run.log while extracting errors path=%s", run_log_path)
        return None

    if not selected_lines:
        selected_lines.append(_NO_ERRORS_MESSAGE)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(selected_lines), encoding="utf-8")
    except OSError:
        logger.exception("Failed to write error log artifact path=%s", output_path)
        return None
    return output_path


__all__ = ["write_error_log_artifact"]
