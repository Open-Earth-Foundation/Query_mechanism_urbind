"""Markdown file discovery helpers."""

from __future__ import annotations

from pathlib import Path


def list_markdown_files(markdown_dir: Path) -> list[Path]:
    """Return top-level Markdown files without scanning document subfolders."""
    return sorted(path for path in markdown_dir.glob("*.md") if path.is_file())


__all__ = ["list_markdown_files"]
