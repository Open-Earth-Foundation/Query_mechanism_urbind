"""Decision action handlers for orchestration iterations."""

from __future__ import annotations

import inspect
import logging
from typing import Callable

from backend.modules.orchestrator.utils.error_handlers import (
    detach_run_file_logger,
    handle_orchestration_error,
)
from backend.modules.orchestrator.utils.io import write_final_output
from backend.modules.writer.models import WriterOutput
from backend.services.llm_observability import LlmCallRecorder
from backend.services.progress_tracker import ProgressTracker
from backend.services.run_logger import RunLogger
from backend.utils.config import AppConfig
from backend.utils.paths import RunPaths


def _resolve_writer_completion_state(
    writer_output: WriterOutput,
) -> tuple[str, str]:
    """Return terminal status and finish reason for one writer result."""
    coverage = writer_output.citation_coverage
    if coverage is None or coverage.status == "confirmed":
        return "completed", "completed (write)"
    return (
        "completed_with_gaps",
        f"completed_with_gaps (writer partial citation coverage {coverage.coverage_ratio})",
    )


def _record_writer_diagnostics(
    run_logger: RunLogger,
    paths: RunPaths,
    writer_output: WriterOutput,
) -> None:
    """Persist writer coverage metadata when the final draft is partial."""
    coverage = writer_output.citation_coverage
    if coverage is None:
        return
    coverage_payload = coverage.model_dump()
    run_logger.record_writer_citation_coverage(coverage_payload)
    if coverage.status == "confirmed":
        return
    run_logger.record_decision(
        {
            "status": "success",
            "run_id": paths.base_dir.name,
            "reason": "Writer returned a partial draft because citation coverage remained incomplete.",
            "writer_citation_coverage": coverage_payload,
        }
    )


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
    progress: ProgressTracker | None = None,
    llm_recorder: LlmCallRecorder | None = None,
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
        llm_recorder: Optional recorder for run-local LLM call artifacts

    Returns:
        Run paths if successful, None to continue iteration
    """
    try:
        writer_kwargs: dict[str, object] = {"log_llm_payload": log_llm_payload}
        writer_signature = inspect.signature(writer_func)
        writer_accepts_extra_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in writer_signature.parameters.values()
        )
        if "run_id" in writer_signature.parameters or writer_accepts_extra_kwargs:
            writer_kwargs["run_id"] = paths.base_dir.name
        if "run_logger" in writer_signature.parameters or writer_accepts_extra_kwargs:
            writer_kwargs["run_logger"] = run_logger
        if "paths" in writer_signature.parameters or writer_accepts_extra_kwargs:
            writer_kwargs["paths"] = paths
        if "llm_recorder" in writer_signature.parameters or writer_accepts_extra_kwargs:
            writer_kwargs["llm_recorder"] = llm_recorder
        writer_output = writer_func(
            question,
            context_bundle,
            config,
            api_key,
            **writer_kwargs,
        )
        terminal_status, finish_reason = _resolve_writer_completion_state(writer_output)
        _record_writer_diagnostics(run_logger, paths, writer_output)
        coverage = writer_output.citation_coverage
        if coverage is not None and progress is not None:
            progress.start_step(
                "writer_citation_coverage",
                "Recording writer citation coverage",
            )
            progress.add_item(
                "writer_citation_coverage",
                f"Citation coverage: {coverage.coverage_ratio}",
                metadata={
                    "status": coverage.status,
                    "confirmed_city_count": coverage.coverage_confirmed,
                    "required_city_count": coverage.coverage_required,
                },
            )
            progress.complete_step("writer_citation_coverage")
        write_final_output(
            question,
            writer_output.content,
            paths,
            run_logger,
            config,
        )
        run_logger.write_stage_detail(
            "writer",
            {
                "inputs": {
                    "question": question,
                    "analysis_mode": context_bundle.get("analysis_mode"),
                },
                "outputs": {
                    "final_output": run_logger.artifact_label(paths.final_output),
                    "citation_coverage": (
                        coverage.model_dump() if coverage is not None else None
                    ),
                },
                "metrics": {
                    "final_output_chars": len(writer_output.content),
                    "citation_coverage_ratio": (
                        coverage.coverage_ratio if coverage is not None else None
                    ),
                    "confirmed_city_count": (
                        coverage.coverage_confirmed if coverage is not None else None
                    ),
                    "required_city_count": (
                        coverage.coverage_required if coverage is not None else None
                    ),
                },
            },
        )
        run_logger.finalize(
            terminal_status,
            final_output_path=paths.final_output,
            finish_reason=finish_reason,
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
