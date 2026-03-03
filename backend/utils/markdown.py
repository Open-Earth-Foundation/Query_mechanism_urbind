"""Helpers for consistent markdown rendering."""

from __future__ import annotations


def render_question_section(question: str) -> str:
    """
    Render the top-level question section as a markdown blockquote.

    The returned markdown keeps multiline questions inside one adjacent block,
    allowing the UI to style the whole question in a single container.
    """
    normalized = question.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = normalized.split("\n")
    if not lines:
        lines = [""]
    quoted_lines = [f"> {line.rstrip()}" if line.strip() else ">" for line in lines]
    quoted_block = "\n".join(quoted_lines)
    return f"# Question\n{quoted_block}\n\n"


__all__ = ["render_question_section"]
