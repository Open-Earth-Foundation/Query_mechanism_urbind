"""Decision action handlers for orchestration iterations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.modules.writer.models import WriterOutput
    from app.services.run_logger import RunLogger
    from app.utils.config import AppConfig
    from app.utils.paths import RunPaths

from app.modules.orchestrator.utils.error_handlers import (
    detach_run_file_logger,
    handle_orchestration_error,
)
from app.modules.orchestrator.utils.io import write_final_output


def handle_write_decision(
    question: str,
    context_bundle: dict,
    paths: RunPaths,
    run_logger: RunLogger,
    run_log_handler: logging.FileHandler,
    writer_func: Callable[..., WriterOutput],
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> RunPaths | None:
    """
    Execute write decision to generate final output.

    Args:
        question: Original user question
        context_bundle: Accumulated context for writing
        paths: Run paths for output
        run_logger: Logger for recording run artifacts
        run_log_handler: File handler for run logs
        writer_func: Function to call for writing
        config: Application configuration
        api_key: API key for external services
        log_llm_payload: Whether to log full LLM request/response payloads

    Returns:
        Run paths if successful, None to continue iteration
    """
    try:
        writer_output = writer_func(
            question,
            context_bundle,
            config,
            api_key,
            log_llm_payload=log_llm_payload,
        )
        write_final_output(
            question,
            writer_output.content,
            paths,
            run_logger,
            finish_reason="completed (write)",
        )
        run_logger.finalize(
            "completed",
            final_output_path=paths.final_output,
            finish_reason="completed (write)",
        )
        detach_run_file_logger(run_log_handler)
        return paths
    except (ValueError, RuntimeError, OSError) as exc:
        return handle_orchestration_error(
            run_logger,
            run_log_handler,
            paths,
            error_code="WRITER_ERROR",
            message="Writer failed",
            reason="writer_failed",
            exc=exc,
        )


__all__ = [
    "handle_write_decision",
]
