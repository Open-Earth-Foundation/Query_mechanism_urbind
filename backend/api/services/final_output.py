"""Helpers for normalizing persisted final markdown documents."""

from __future__ import annotations


def strip_legacy_finish_reason_footer(content: str) -> str:
    """Remove the old trailing ``Finish reason: ...`` footer when present."""
    lines = content.splitlines()
    if not lines:
        return content

    end_index = len(lines) - 1
    while end_index >= 0 and not lines[end_index].strip():
        end_index -= 1
    if end_index < 0:
        return ""
    if not lines[end_index].startswith("Finish reason:"):
        return content

    start_index = end_index
    while start_index > 0 and not lines[start_index - 1].strip():
        start_index -= 1
    if start_index > 0 and lines[start_index - 1].strip() == "---":
        start_index -= 1
        while start_index > 0 and not lines[start_index - 1].strip():
            start_index -= 1

    return "\n".join(lines[:start_index]).rstrip()


__all__ = ["strip_legacy_finish_reason_footer"]
