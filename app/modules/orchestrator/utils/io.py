"""I/O utilities for orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.run_logger import RunLogger
    from app.utils.paths import RunPaths

logger = logging.getLogger(__name__)


def write_json(path: Path, payload: object) -> None:
    """
    Write payload as formatted JSON to file.

    Args:
        path: Target file path
        payload: Object to serialize as JSON
    """
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def load_context_bundle(paths: RunPaths) -> dict:
    """
    Load context bundle from file if it exists.

    Args:
        paths: Run paths containing context bundle location

    Returns:
        Context bundle dictionary, empty if file doesn't exist
    """
    if not paths.context_bundle.exists():
        return {}
    return json.loads(paths.context_bundle.read_text(encoding="utf-8"))


def write_draft_and_final(
    question: str,
    content: str,
    paths: RunPaths,
    run_logger: RunLogger,
) -> None:
    """
    Write draft and final output files.

    Creates both a numbered draft and final output file with a question header.

    Args:
        question: Original user question
        content: Generated content
        paths: Run paths for output
        run_logger: Logger for recording artifacts
    """
    question_header = f"# Question\n{question}\n\n"
    rendered_content = f"{question_header}{content}"

    draft_index = len(run_logger.run_log.get("drafts", [])) + 1
    draft_path = paths.drafts_dir / f"draft_{draft_index:02d}.md"
    draft_path.write_text(rendered_content, encoding="utf-8")
    run_logger.record_draft(draft_path)

    final_path = paths.final_output
    final_path.write_text(rendered_content, encoding="utf-8")
    run_logger.record_artifact("final_output", final_path)


__all__ = [
    "write_json",
    "load_context_bundle",
    "write_draft_and_final",
]
