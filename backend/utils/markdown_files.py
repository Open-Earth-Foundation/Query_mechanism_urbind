"""Markdown file discovery helpers."""

from __future__ import annotations

from pathlib import Path


def list_markdown_files(markdown_dir: Path) -> list[Path]:
    """Return all Markdown files under ``markdown_dir``, recursively.

    City identity is derived from ``Path.stem`` (see
    ``markdown_researcher.split_documents_by_city``), so files in
    subdirectories such as ``additional/<city>_<slug>/<city>.md`` bucket
    alongside their top-level counterparts.
    """
    return sorted(path for path in markdown_dir.rglob("*.md") if path.is_file())


__all__ = ["list_markdown_files"]
