"""I/O utilities for orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

from backend.api.services.context_prompt_cache import compute_prompt_context_cache, write_prompt_context_cache
from backend.services.run_logger import RunLogger
from backend.utils.config import AppConfig
from backend.utils.paths import RunPaths

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
    config: AppConfig,
    finish_reason: str = "completed",
) -> None:
    """
    Write the final output file.

    Args:
        question: Original user question
        content: Generated content
        paths: Run paths for output
        run_logger: Logger for recording artifacts
        finish_reason: Finish reason for the output
    """
    question_header = f"# Question\n{question}\n\n"
    finish_note = f"\n\n---\nFinish reason: {finish_reason}\n"
    rendered_content = f"{question_header}{content}{finish_note}"
    final_path = paths.final_output
    final_path.write_text(rendered_content, encoding="utf-8")
    run_logger.record_artifact("final_output", final_path)
    prompt_context_tokens, prompt_context_kind = compute_prompt_context_cache(
        question=question,
        final_document=rendered_content,
        context_bundle=run_logger.context_bundle,
        config=config,
    )
    run_logger.context_bundle = write_prompt_context_cache(
        context_bundle_path=paths.context_bundle,
        markdown_excerpts_path=paths.markdown_excerpts,
        context_bundle=run_logger.context_bundle,
        prompt_context_tokens=prompt_context_tokens,
        prompt_context_kind=prompt_context_kind,
    )


__all__ = [
    "write_json",
    "load_context_bundle",
    "write_final_output",
]
