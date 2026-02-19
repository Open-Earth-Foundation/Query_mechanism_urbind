"""Error handling utilities for orchestrator."""

import logging

from backend.services.run_logger import RunLogger
from backend.utils.paths import RunPaths

logger = logging.getLogger(__name__)


def detach_run_file_logger(run_log_handler: logging.FileHandler) -> None:
    """Safely remove and close a run-specific file handler from the root logger."""
    root_logger = logging.getLogger()
    root_logger.removeHandler(run_log_handler)
    run_log_handler.close()


def handle_orchestration_error(
    run_logger: RunLogger,
    run_log_handler: logging.FileHandler,
    paths: RunPaths,
    error_code: str,
    message: str,
    reason: str,
    exc: Exception,
) -> RunPaths:
    """
    Standardized error handling and finalization for orchestration failures.

    Logs the error, records decision, and finalizes the run with failure status.

    Args:
        run_logger: Logger for recording run artifacts
        run_log_handler: File handler for run logs
        paths: Run paths for output
        error_code: Error code for classification (e.g., "SQL_EXECUTION_ERROR")
        message: Human-readable error message
        reason: Reason for finalization (e.g., "sql_execution_failed")
        exc: The exception that occurred

    Returns:
        The run paths for the failed run
    """
    logger.exception(message)
    run_logger.record_decision(
        {
            "status": "error",
            "run_id": paths.base_dir.name,
            "reason": message,
            "error": {"code": error_code, "message": str(exc)},
        }
    )
    run_logger.finalize("failed", finish_reason=reason)
    detach_run_file_logger(run_log_handler)
    return paths


def handle_task_error(
    task_name: str,
    exc: Exception,
    run_logger: RunLogger,
    run_log_handler: logging.FileHandler,
    paths: RunPaths,
) -> RunPaths:
    """
    Handle errors from parallel task execution (SQL and Markdown).

    Args:
        task_name: Name of the task that failed ("sql" or "markdown")
        exc: The exception that occurred
        run_logger: Logger for recording run artifacts
        run_log_handler: File handler for run logs
        paths: Run paths for output

    Returns:
        The run paths for the failed run
    """
    if task_name == "sql":
        return handle_orchestration_error(
            run_logger,
            run_log_handler,
            paths,
            error_code="SQL_EXECUTION_ERROR",
            message="SQL execution failed",
            reason="sql_execution_failed",
            exc=exc,
        )
    else:  # markdown
        return handle_orchestration_error(
            run_logger,
            run_log_handler,
            paths,
            error_code="MARKDOWN_ERROR",
            message="Markdown extraction failed",
            reason="markdown_extraction_failed",
            exc=exc,
        )


__all__ = ["detach_run_file_logger", "handle_orchestration_error", "handle_task_error"]
