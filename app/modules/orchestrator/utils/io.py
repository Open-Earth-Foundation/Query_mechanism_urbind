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
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
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


def write_final_output(
    question: str,
    content: str,
    paths: RunPaths,
    run_logger: RunLogger,
    finish_reason: str = "completed",
) -> None:
    """
    Write the final output file.

    Args:
        question: Original user question
        content: Generated content
        paths: Run paths for output
        run_logger: Logger for recording artifacts
        finish_reason: Reason for finishing (included in output)
    """
    question_header = f"# Question\n{question}\n\n"
    finish_note = f"\n\n---\nFinish reason: {finish_reason}\n"
    rendered_content = f"{question_header}{content}{finish_note}"
    final_path = paths.final_output
    final_path.write_text(rendered_content, encoding="utf-8")
    run_logger.record_artifact("final_output", final_path)


__all__ = [
    "write_json",
    "load_context_bundle",
    "write_final_output",
]
