"""Decision action handlers for orchestration iterations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pathlib import Path
    from app.modules.writer.models import WriterOutput
    from app.modules.markdown_researcher.models import MarkdownResearchResult
    from app.services.run_logger import RunLogger
    from app.utils.config import AppConfig
    from app.utils.paths import RunPaths
    from app.modules.sql_researcher.models import SqlQueryPlan

from app.modules.orchestrator.utils.error_handlers import handle_orchestration_error
from app.modules.orchestrator.utils.io import write_draft_and_final

logger = logging.getLogger(__name__)


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
        write_draft_and_final(
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
        run_log_handler.close()
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


def handle_sql_decision(
    follow_up_question: str,
    schema_summary: dict,
    city_names: list[str],
    paths: RunPaths,
    run_logger: RunLogger,
    run_log_handler: logging.FileHandler,
    config: AppConfig,
    api_key: str,
    sql_plan_func: Callable[..., SqlQueryPlan],
    sql_rounds_func: Callable,
    cap_results_func: Callable,
    run_logger_update_sql_bundle: Callable,
    write_json_func: Callable,
    log_llm_payload: bool = False,
) -> RunPaths | None:
    """
    Execute SQL decision to run additional queries.

    Args:
        follow_up_question: Question for the new SQL queries
        schema_summary: Database schema information
        city_names: List of city names for context
        paths: Run paths for output
        run_logger: Logger for recording run artifacts
        run_log_handler: File handler for run logs
        config: Application configuration
        api_key: API key for external services
        sql_plan_func: Function to generate SQL plans
        sql_rounds_func: Function to execute SQL rounds
        cap_results_func: Function to cap results to token limit
        run_logger_update_sql_bundle: Function to update SQL bundle in logger
        write_json_func: Function to write JSON files
        log_llm_payload: Whether to log full LLM request/response payloads

    Returns:
        Run paths if error occurred, None to continue iteration
    """
    try:
        # Plans SQL queries
        sql_plan = sql_plan_func(
            follow_up_question,
            schema_summary,
            city_names,
            config,
            api_key,
            per_city_focus=True,
            log_llm_payload=log_llm_payload,
        )
        write_json_func(paths.sql_queries, sql_plan.model_dump())
        run_logger.record_artifact("sql_queries", paths.sql_queries)

        if sql_plan.status == "error":
            run_logger.record_decision(sql_plan.model_dump())
            run_logger.finalize("failed", finish_reason="sql_plan_error")
            run_log_handler.close()
            return paths

        # Executes planned queries
        sql_plan, full_results, sql_rounds = sql_rounds_func(
            follow_up_question, sql_plan
        )
        sql_rounds_path = paths.sql_dir / "rounds.json"
        write_json_func(sql_rounds_path, sql_rounds)
        run_logger.record_artifact("sql_rounds", sql_rounds_path)
        write_json_func(paths.sql_queries, sql_plan.model_dump())
        run_logger.record_artifact("sql_queries", paths.sql_queries)

        capped_results, total_tokens, truncated = cap_results_func(
            full_results, config.sql_researcher.max_result_tokens
        )
        write_json_func(
            paths.sql_results_full, [res.model_dump() for res in full_results]
        )
        write_json_func(paths.sql_results, [res.model_dump() for res in capped_results])
        run_logger.record_artifact("sql_results_full", paths.sql_results_full)
        run_logger.record_artifact("sql_results", paths.sql_results)

        run_logger_update_sql_bundle(sql_plan, capped_results, total_tokens, truncated)
        return None  # Continue iteration

    except (ValueError, RuntimeError, OSError) as exc:
        return handle_orchestration_error(
            run_logger,
            run_log_handler,
            paths,
            error_code="SQL_EXECUTION_ERROR",
            message="SQL execution failed",
            reason="sql_execution_failed",
            exc=exc,
        )


def handle_markdown_decision(
    follow_up_question: str,
    documents: list[dict[str, str]],
    paths: RunPaths,
    run_logger: RunLogger,
    run_log_handler: logging.FileHandler,
    config: AppConfig,
    api_key: str,
    markdown_func: Callable[..., MarkdownResearchResult],
    write_json_func: Callable,
    log_llm_payload: bool = False,
) -> RunPaths | None:
    """
    Execute markdown decision to extract additional excerpts.

    Args:
        follow_up_question: Question for markdown extraction
        documents: Loaded markdown documents
        paths: Run paths for output
        run_logger: Logger for recording run artifacts
        run_log_handler: File handler for run logs
        config: Application configuration
        api_key: API key for external services
        markdown_func: Function to extract markdown excerpts
        write_json_func: Function to write JSON files
        log_llm_payload: Whether to log full LLM request/response payloads

    Returns:
        Run paths if error occurred, None to continue iteration
    """
    try:
        markdown_result = markdown_func(
            follow_up_question,
            documents,
            config,
            api_key,
            log_llm_payload=log_llm_payload,
        )
        write_json_func(paths.markdown_excerpts, markdown_result.model_dump())
        run_logger.record_artifact("markdown_excerpts", paths.markdown_excerpts)
        run_logger.update_markdown_bundle(markdown_result.model_dump())

        if markdown_result.status == "error":
            run_logger.record_decision(markdown_result.model_dump())
            run_logger.finalize("failed", finish_reason="markdown_result_error")
            run_log_handler.close()
            return paths

        return None  # Continue iteration

    except (ValueError, RuntimeError, OSError) as exc:
        return handle_orchestration_error(
            run_logger,
            run_log_handler,
            paths,
            error_code="MARKDOWN_ERROR",
            message="Markdown extraction failed",
            reason="markdown_extraction_failed",
            exc=exc,
        )


__all__ = [
    "handle_write_decision",
    "handle_sql_decision",
    "handle_markdown_decision",
]
