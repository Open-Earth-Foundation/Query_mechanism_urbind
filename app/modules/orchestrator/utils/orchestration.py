"""Orchestration loop helpers for orchestrator."""

import logging
from typing import Callable

from app.modules.orchestrator.agent import decide_next_action
from app.modules.orchestrator.models import OrchestratorDecision
from app.modules.orchestrator.utils.error_handlers import detach_run_file_logger
from app.modules.orchestrator.utils.handlers import handle_write_decision
from app.modules.orchestrator.utils.io import (
    load_context_bundle,
    write_draft_and_final,
)
from app.modules.writer.models import WriterOutput
from app.services.run_logger import RunLogger
from app.utils.config import AppConfig
from app.utils.paths import RunPaths

logger = logging.getLogger(__name__)


def run_orchestration_loop(
    question: str,
    max_iterations: int,
    paths: RunPaths,
    run_logger: RunLogger,
    run_log_handler: logging.FileHandler,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool,
    decide_func: Callable[..., OrchestratorDecision] = decide_next_action,
    writer_func: Callable[..., WriterOutput] | None = None,
) -> RunPaths:
    """
    Execute the orchestrator decision loop.

    Handles decision-making and final writing/stop behavior using the prepared context bundle.

    Args:
        question: Original user question
        max_iterations: Maximum iterations before fallback
        paths: Run paths
        run_logger: Run logger
        run_log_handler: File handler for run logs
        config: App configuration
        api_key: API key for agents
        log_llm_payload: Whether to log full LLM request/response payloads
        decide_func: Function to make orchestration decisions
        writer_func: Function to write final output

    Returns:
        Run paths for the completed run
    """
    for iteration_num in range(max_iterations):
        logger.info(
            "Starting orchestration iteration %d/%d", iteration_num + 1, max_iterations
        )

        context_bundle = load_context_bundle(paths)
        decision = decide_func(
            question,
            context_bundle,
            config,
            api_key,
            log_llm_payload=log_llm_payload,
        )
        run_logger.record_decision(decision.model_dump())
        logger.debug("Orchestrator decision: action=%s", decision.action)

        if decision.status == "error":
            logger.error("Orchestrator decision failed")
            run_logger.finalize("failed", finish_reason="decision_error")
            detach_run_file_logger(run_log_handler)
            return paths

        # Handle write decision
        if decision.action == "write":
            if writer_func is None:
                logger.error("Writer function not provided for write action")
                run_logger.finalize("failed", finish_reason="writer_not_provided")
                detach_run_file_logger(run_log_handler)
                return paths

            result = handle_write_decision(
                question,
                context_bundle,
                paths,
                run_logger,
                run_log_handler,
                writer_func,
                config,
                api_key,
                log_llm_payload=log_llm_payload,
            )
            return result

        # Handle stop decision
        if decision.action == "stop":
            logger.info("Orchestrator decided to stop")
            run_logger.finalize("stopped", finish_reason="stopped_by_orchestrator")
            detach_run_file_logger(run_log_handler)
            return paths

    # Fallback writer after max iterations
    logger.warning(
        "Reached max iterations (%d), running fallback writer", max_iterations
    )
    if writer_func is None:
        logger.error("Writer function not provided for fallback writer")
        run_logger.finalize("failed", finish_reason="writer_not_provided")
        detach_run_file_logger(run_log_handler)
        return paths

    context_bundle = load_context_bundle(paths)
    try:
        writer_output = writer_func(
            question,
            context_bundle,
            config,
            api_key,
            log_llm_payload=log_llm_payload,
        )
        write_draft_and_final(
            question,
            writer_output.content,
            paths,
            run_logger,
            finish_reason="completed_with_gaps (max iterations)",
        )
        run_logger.finalize(
            "completed_with_gaps",
            final_output_path=paths.final_output,
            finish_reason="completed_with_gaps (max iterations)",
        )
        detach_run_file_logger(run_log_handler)
        return paths
    except (ValueError, RuntimeError, OSError) as exc:
        logger.exception("Fallback writer failed")
        run_logger.record_decision(
            {
                "status": "error",
                "run_id": paths.base_dir.name,
                "reason": "Fallback writer failed",
                "error": {"code": "WRITER_FALLBACK_ERROR", "message": str(exc)},
            }
        )
        run_logger.finalize("failed", finish_reason="max_iterations_exceeded")
        detach_run_file_logger(run_log_handler)
        return paths


__all__ = ["run_orchestration_loop"]
