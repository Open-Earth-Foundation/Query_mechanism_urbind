"""Markdown file discovery helpers."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def list_markdown_files(markdown_dir: Path) -> list[Path]:
    """Return top-level Markdown files without scanning document subfolders."""
    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown directory does not exist: {markdown_dir}")
    if not markdown_dir.is_dir():
        raise NotADirectoryError(f"Markdown directory is not a folder: {markdown_dir}")
    logger.info(
        "Listing markdown files markdown_dir=%s resolved=%s",
        markdown_dir,
        markdown_dir.resolve(),
    )
    return sorted(path for path in markdown_dir.glob("*.md") if path.is_file())


__all__ = ["list_markdown_files"]
