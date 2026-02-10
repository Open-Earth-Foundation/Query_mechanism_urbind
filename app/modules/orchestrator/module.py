from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg

from app.modules.markdown_researcher.agent import extract_markdown_excerpts
from app.modules.markdown_researcher.models import MarkdownResearchResult
from app.modules.markdown_researcher.services import load_markdown_documents
from app.modules.orchestrator.agent import decide_next_action, refine_research_question
from app.modules.orchestrator.models import (
    OrchestratorDecision,
    ResearchQuestionRefinement,
)
from app.modules.orchestrator.utils import (
    attach_run_file_logger,
    collect_identifiers,
    fetch_city_list,
    handle_task_error,
    run_orchestration_loop,
    run_sql_rounds,
)
from app.modules.orchestrator.utils.error_handlers import (
    detach_run_file_logger,
)
from app.modules.orchestrator.utils.io import (
    load_context_bundle,
    write_draft_and_final,
    write_json,
)
from app.modules.sql_researcher.agent import plan_sql_queries
from app.modules.sql_researcher.models import SqlQueryPlan
from app.modules.sql_researcher.services import (
    build_sql_research_result,
    cap_results,
    execute_queries,
)
from app.modules.writer.agent import write_markdown
from app.modules.writer.models import WriterOutput
from app.services.db_client import DbClient, get_db_client
from app.services.run_logger import RunLogger
from app.services.schema_registry import load_schema
from app.utils.config import AppConfig, get_openrouter_api_key
from app.utils.paths import RunPaths, build_run_id, create_run_paths

logger = logging.getLogger(__name__)


def run_pipeline(
    question: str,
    config: AppConfig,
    run_id: str | None = None,
    log_llm_payload: bool = True,
    selected_cities: list[str] | None = None,
    sql_plan_func: Callable[..., SqlQueryPlan] = plan_sql_queries,
    markdown_func: Callable[..., MarkdownResearchResult] = extract_markdown_excerpts,
    refine_question_func: Callable[
        ..., ResearchQuestionRefinement
    ] = refine_research_question,
    decide_func: Callable[..., OrchestratorDecision] = decide_next_action,
    writer_func: Callable[..., WriterOutput] = write_markdown,
) -> RunPaths:
    """
    Run the multi-agent document builder pipeline.

    Orchestrates SQL research (when enabled), markdown extraction, and document writing in an agentic loop.

    Args:
        question: User question to answer
        config: Application configuration
        run_id: Optional run identifier
        log_llm_payload: Whether to log full LLM request/response payloads
        selected_cities: Optional list of city names to limit markdown document loading
        sql_plan_func: SQL planning function (default: plan_sql_queries)
        markdown_func: Markdown extraction function (default: extract_markdown_excerpts)
        refine_question_func: Question refinement function (default: refine_research_question)
        decide_func: Orchestration decision function (default: decide_next_action)
        writer_func: Document writing function (default: write_markdown)

    Returns:
        Run paths containing output artifacts
    """
    api_key = get_openrouter_api_key()
    run_id_value = run_id or build_run_id()
    paths = create_run_paths(
        config.runs_dir, run_id_value, config.orchestrator.context_bundle_name
    )
    run_logger = RunLogger(paths, question)
    run_logger.record_artifact("context_bundle", paths.context_bundle)
    run_log_handler = attach_run_file_logger(paths.base_dir)

    # First step in the chain: rewrite the user question into a research-ready form.
    research_question = question
    try:
        refinement = refine_question_func(
            question,
            config,
            api_key,
            log_llm_payload=log_llm_payload,
        )
        candidate = refinement.research_question.strip()
        if candidate:
            research_question = candidate
        else:
            logger.warning(
                "Research question refinement returned empty output; using original question."
            )
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        logger.warning(
            "Research question refinement failed; using original question. error=%s",
            exc,
        )

    run_logger.update_research_question(research_question)
    write_json(
        paths.research_question,
        {
            "original_question": question,
            "research_question": research_question,
        },
    )
    run_logger.record_artifact("research_question", paths.research_question)

    sql_enabled = config.enable_sql
    schema_summary: dict = {}
    identifiers: list[str] = []
    city_names: list[str] = []
    db_client: DbClient | None = None

    if sql_enabled:
        # Load schema and extract identifiers
        models_dir = Path(__file__).resolve().parents[2] / "db_models"
        schema_summary = load_schema(models_dir).model_dump()
        write_json(paths.schema_summary, schema_summary)
        run_logger.record_artifact("schema_summary", paths.schema_summary)
        identifiers = collect_identifiers(schema_summary)

        # Get database client and city list
        db_client = get_db_client(config.source_db_path, config.source_db_url)
        try:
            city_names = fetch_city_list(db_client)
        except (sqlite3.OperationalError, psycopg.DatabaseError, OSError) as e:
            logger.warning("Failed to fetch city list: %s", e)
        write_json(paths.city_list, city_names)
        run_logger.record_artifact("city_list", paths.city_list)
    else:
        logger.info("SQL disabled; running without database lookups")

    # Run initial SQL and markdown in parallel
    def _run_initial_sql() -> dict[str, object]:
        sql_plan = sql_plan_func(
            research_question,
            schema_summary,
            city_names,
            config,
            api_key,
            per_city_focus=True,
            log_llm_payload=log_llm_payload,
        )
        if sql_plan.status == "error":
            return {"status": "error", "plan": sql_plan}
        sql_plan, full_results, sql_rounds = run_sql_rounds(
            research_question,
            sql_plan,
            db_client,
            config,
            api_key,
            log_llm_payload,
            schema_summary,
            city_names,
            identifiers,
            sql_plan_func,
            write_json,
            run_logger,
            paths,
        )
        capped_results, total_tokens, truncated = cap_results(
            full_results, config.sql_researcher.max_result_tokens
        )
        return {
            "status": "ok",
            "plan": sql_plan,
            "full_results": full_results,
            "sql_rounds": sql_rounds,
            "capped_results": capped_results,
            "total_tokens": total_tokens,
            "truncated": truncated,
        }

    def _run_initial_markdown() -> dict[str, object]:
        documents = load_markdown_documents(
            config.markdown_dir,
            config.markdown_researcher,
            selected_cities=selected_cities,
        )
        markdown_result = markdown_func(
            research_question,
            documents,
            config,
            api_key,
            log_llm_payload=log_llm_payload,
        )
        return {"documents": documents, "result": markdown_result}

    sql_payload: dict[str, object] | None = None
    markdown_payload: dict[str, object] | None = None

    with ThreadPoolExecutor(max_workers=2 if sql_enabled else 1) as executor:
        futures = {}
        if sql_enabled:
            futures[executor.submit(_run_initial_sql)] = "sql"
        futures[executor.submit(_run_initial_markdown)] = "markdown"
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                payload = future.result()
                if task_name == "sql":
                    sql_payload = payload
                else:
                    markdown_payload = payload
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                return handle_task_error(
                    task_name, exc, run_logger, run_log_handler, paths
                )

    if sql_enabled and not sql_payload:
        run_logger.finalize("failed", finish_reason="sql_execution_failed")
        detach_run_file_logger(run_log_handler)
        return paths
    if not markdown_payload:
        run_logger.finalize("failed", finish_reason="markdown_extraction_failed")
        detach_run_file_logger(run_log_handler)
        return paths

    if sql_enabled and sql_payload and sql_payload.get("status") == "error":
        sql_plan_obj = sql_payload.get("plan")
        if isinstance(sql_plan_obj, SqlQueryPlan):
            run_logger.record_decision(sql_plan_obj.model_dump())
        run_logger.finalize("failed", finish_reason="sql_plan_error")
        detach_run_file_logger(run_log_handler)
        return paths

    if sql_enabled and sql_payload:
        # Process and log initial SQL results
        sql_plan: SqlQueryPlan = sql_payload["plan"]  # type: ignore
        full_results: list = sql_payload["full_results"]  # type: ignore
        sql_rounds: list[dict[str, object]] = sql_payload["sql_rounds"]  # type: ignore
        capped_results: list = sql_payload["capped_results"]  # type: ignore
        total_tokens: int = sql_payload["total_tokens"]  # type: ignore
        truncated: bool = sql_payload["truncated"]  # type: ignore

        sql_rounds_path = paths.sql_dir / "rounds.json"
        write_json(sql_rounds_path, sql_rounds)
        run_logger.record_artifact("sql_rounds", sql_rounds_path)
        write_json(paths.sql_queries, sql_plan.model_dump())
        run_logger.record_artifact("sql_queries", paths.sql_queries)

        full_payload = [result.model_dump() for result in full_results]
        capped_payload = [result.model_dump() for result in capped_results]
        write_json(paths.sql_results_full, full_payload)
        write_json(paths.sql_results, capped_payload)
        run_logger.record_artifact("sql_results_full", paths.sql_results_full)
        run_logger.record_artifact("sql_results", paths.sql_results)

        sql_result = build_sql_research_result(
            queries=sql_plan.queries,
            results=capped_results,
            total_tokens=total_tokens,
            truncated=truncated,
        )
        run_logger.update_sql_bundle(sql_result.model_dump())

    # Process and log initial markdown results
    documents = markdown_payload["documents"]
    markdown_result = markdown_payload["result"]
    if isinstance(markdown_result, MarkdownResearchResult):
        write_json(paths.markdown_excerpts, markdown_result.model_dump())
        run_logger.record_artifact("markdown_excerpts", paths.markdown_excerpts)
        run_logger.update_markdown_bundle(markdown_result.model_dump())
        if markdown_result.status == "error":
            run_logger.record_decision(markdown_result.model_dump())
            run_logger.finalize("failed", finish_reason="markdown_result_error")
            detach_run_file_logger(run_log_handler)
            return paths
    else:
        run_logger.finalize("failed", finish_reason="markdown_extraction_failed")
        detach_run_file_logger(run_log_handler)
        return paths

    # Run orchestration loop
    return run_orchestration_loop(
        question,
        max_iterations=4,
        paths=paths,
        run_logger=run_logger,
        run_log_handler=run_log_handler,
        config=config,
        api_key=api_key,
        log_llm_payload=log_llm_payload,
        decide_func=decide_func,
        writer_func=writer_func,
    )


__all__ = ["run_pipeline"]
