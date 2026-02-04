from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg

from app.modules.markdown_researcher.agent import extract_markdown_excerpts
from app.modules.markdown_researcher.models import MarkdownResearchResult
from app.modules.markdown_researcher.services import load_markdown_documents
from app.modules.orchestrator.agent import decide_next_action
from app.modules.orchestrator.models import OrchestratorDecision
from app.modules.orchestrator.utils.error_handlers import handle_task_error
from app.modules.orchestrator.utils.handlers import (
    handle_markdown_decision,
    handle_sql_decision,
    handle_write_decision,
)
from app.modules.orchestrator.utils.io import (
    load_context_bundle,
    write_draft_and_final,
    write_json,
)
from app.modules.sql_researcher.agent import plan_sql_queries
from app.modules.sql_researcher.models import SqlQueryPlan, SqlResearchResult
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


def _attach_run_file_logger(run_dir: Path) -> logging.FileHandler:
    """Attach file handler to root logger for run-specific logging."""
    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.getLogger().level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def _detach_run_file_logger(handler: logging.FileHandler) -> None:
    """Remove and close file handler from root logger."""
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def _fetch_city_list(db_client: DbClient, max_rows: int = 500) -> list[str]:
    """
    Fetch list of city names from database.

    Attempts multiple query variations to account for different schema configurations.

    Args:
        db_client: Database client
        max_rows: Maximum cities to return

    Returns:
        List of city names, empty if fetch fails
    """
    queries = [
        'SELECT "cityName" FROM "City"',
        "SELECT cityName FROM City",
        'SELECT "city_name" FROM "city"',
        "SELECT city_name FROM city",
    ]
    for sql in queries:
        try:
            columns, rows = db_client.query(sql)
        except (sqlite3.OperationalError, psycopg.DatabaseError, ValueError) as e:
            logger.debug("City list query attempt failed: %s", e)
            continue
        if not columns or not rows:
            continue
        names = [str(row[0]) for row in rows if row and row[0] is not None]
        return names[:max_rows]
    logger.warning("Could not fetch city list from database")
    return []


def _collect_sql_execution_errors(results: list[object]) -> list[dict[str, object]]:
    """Extract error results from SQL execution results."""
    errors: list[dict[str, object]] = []
    for result in results:
        if isinstance(result, dict):
            columns = result.get("columns", [])
            rows = result.get("rows", [])
            query_id = result.get("query_id")
        else:
            columns = getattr(result, "columns", [])
            rows = getattr(result, "rows", [])
            query_id = getattr(result, "query_id", None)
        if columns == ["error"] and rows:
            errors.append({"query_id": query_id, "error": rows[0][0]})
    return errors


def _summarize_sql_results(
    results: list[object], max_rows: int = 3
) -> list[dict[str, object]]:
    """Summarize SQL results for agent context."""
    summary: list[dict[str, object]] = []
    for result in results:
        if isinstance(result, dict):
            query_id = result.get("query_id")
            columns = result.get("columns", [])
            rows = result.get("rows", [])
            row_count = result.get("row_count", len(rows))
        else:
            query_id = getattr(result, "query_id", None)
            columns = getattr(result, "columns", [])
            rows = getattr(result, "rows", [])
            row_count = getattr(result, "row_count", len(rows))
        sample_rows = rows[:max_rows] if isinstance(rows, list) else []
        summary.append(
            {
                "query_id": query_id,
                "columns": columns,
                "row_count": row_count,
                "sample_rows": sample_rows,
            }
        )
    return summary


def _collect_identifiers(schema_summary: dict) -> list[str]:
    """Extract all identifiers (tables, columns) from schema for SQL sanitization."""
    identifiers: set[str] = set()
    for table in schema_summary.get("tables", []):
        table_name = table.get("name")
        if table_name:
            identifiers.add(str(table_name))
        for column in table.get("columns", []):
            if column:
                identifiers.add(str(column))
        for fk in table.get("foreign_keys", []):
            for key in ("column", "ref_table", "ref_column"):
                value = fk.get(key)
                if value:
                    identifiers.add(str(value))
    return sorted(identifiers)


def _execute_sql_plan(
    question: str,
    sql_plan: SqlQueryPlan,
    db_client: DbClient,
    config: AppConfig,
    api_key: str,
    schema_summary: dict,
    city_names: list[str],
    identifiers: list[str],
    sql_plan_func: Callable[..., SqlQueryPlan],
    write_json_func: Callable,
    run_logger: RunLogger,
    paths: RunPaths,
    allow_retry: bool = True,
) -> tuple[SqlQueryPlan, list]:
    """
    Execute a SQL plan and handle errors with retry.

    Args:
        question: Question being answered
        sql_plan: Plan containing queries to execute
        db_client: Database client
        config: App configuration
        api_key: API key for agents
        schema_summary: Database schema for reference
        city_names: Cities for context
        identifiers: Database identifiers for SQL sanitization
        sql_plan_func: Function to generate new plans
        write_json_func: Function to write JSON artifacts
        run_logger: Run logger
        paths: Run paths
        allow_retry: Whether to retry on errors

    Returns:
        Tuple of updated SQL plan and results
    """
    full_results = execute_queries(
        db_client,
        sql_plan.queries,
        config.sql_researcher.max_rows,
        identifiers=identifiers,
    )

    if allow_retry:
        sql_errors = _collect_sql_execution_errors(full_results)
        if sql_errors:
            retry_plan = sql_plan_func(
                question,
                schema_summary,
                city_names,
                paths.base_dir.name,
                config,
                api_key,
                sql_execution_errors=sql_errors,
                previous_queries=[query.model_dump() for query in sql_plan.queries],
                per_city_focus=True,
            )
            if retry_plan.status != "error":
                sql_plan = retry_plan
                write_json_func(paths.sql_queries, sql_plan.model_dump())
                run_logger.record_artifact("sql_queries", paths.sql_queries)
                full_results = execute_queries(
                    db_client,
                    sql_plan.queries,
                    config.sql_researcher.max_rows,
                    identifiers=identifiers,
                )

    return sql_plan, full_results


def _run_sql_rounds(
    question: str,
    initial_plan: SqlQueryPlan,
    db_client: DbClient,
    config: AppConfig,
    api_key: str,
    schema_summary: dict,
    city_names: list[str],
    identifiers: list[str],
    sql_plan_func: Callable[..., SqlQueryPlan],
    write_json_func: Callable,
    run_logger: RunLogger,
    paths: RunPaths,
) -> tuple[SqlQueryPlan, list, list[dict[str, object]]]:
    """
    Execute multiple rounds of SQL planning and execution.

    Args:
        question: Question being answered
        initial_plan: Initial SQL plan
        db_client: Database client
        config: App configuration
        api_key: API key for agents
        schema_summary: Database schema for reference
        city_names: Cities for context
        identifiers: Database identifiers for SQL sanitization
        sql_plan_func: Function to generate new plans
        write_json_func: Function to write JSON artifacts
        run_logger: Run logger
        paths: Run paths

    Returns:
        Tuple of final plan, all results, and rounds history
    """
    rounds_to_run = max(1, config.sql_researcher.pre_orchestrator_rounds)
    current_plan = initial_plan
    all_results: list = []
    all_queries = []
    rounds_payload: list[dict[str, object]] = []

    for round_index in range(rounds_to_run):
        current_plan, round_results = _execute_sql_plan(
            question,
            current_plan,
            db_client,
            config,
            api_key,
            schema_summary,
            city_names,
            identifiers,
            sql_plan_func,
            write_json_func,
            run_logger,
            paths,
            allow_retry=True,
        )
        all_results.extend(round_results)
        all_queries.extend(current_plan.queries)
        rounds_payload.append(
            {
                "round": round_index + 1,
                "plan": current_plan.model_dump(),
                "results": [result.model_dump() for result in round_results],
            }
        )

        if round_index + 1 >= rounds_to_run:
            break

        summary = _summarize_sql_results(round_results)
        next_plan = sql_plan_func(
            question,
            schema_summary,
            city_names,
            paths.base_dir.name,
            config,
            api_key,
            sql_results_summary=summary,
            previous_queries=[query.model_dump() for query in current_plan.queries],
            per_city_focus=True,
        )

        if next_plan.status == "error" or not next_plan.queries:
            break
        current_plan = next_plan

    combined_plan = current_plan
    if all_queries:
        combined_plan = current_plan.model_copy(update={"queries": all_queries})

    return combined_plan, all_results, rounds_payload


def _run_orchestration_loop(
    question: str,
    max_iterations: int,
    sql_payload: dict[str, object],
    markdown_payload: dict[str, object],
    paths: RunPaths,
    run_logger: RunLogger,
    run_log_handler: logging.FileHandler,
    config: AppConfig,
    api_key: str,
    schema_summary: dict,
    city_names: list[str],
    identifiers: list[str],
    db_client: DbClient,
    documents: list[dict[str, str]],
    sql_plan_func: Callable[..., SqlQueryPlan],
    sql_rounds_func: Callable,
    cap_results_func: Callable,
    markdown_func: Callable[..., MarkdownResearchResult],
    decide_func: Callable[..., OrchestratorDecision],
    writer_func: Callable[..., WriterOutput],
) -> RunPaths:
    """
    Execute orchestration loop to iteratively refine results.

    Handles decision-making, query execution, markdown extraction, and writing.

    Args:
        question: Original user question
        max_iterations: Maximum iterations before fallback
        sql_payload: Initial SQL execution results
        markdown_payload: Initial markdown extraction results
        paths: Run paths
        run_logger: Run logger
        run_log_handler: File handler for run logs
        config: App configuration
        api_key: API key for agents
        schema_summary: Database schema
        city_names: Cities for context
        identifiers: Database identifiers
        db_client: Database client
        documents: Loaded markdown documents
        sql_plan_func: Function to generate SQL plans
        sql_rounds_func: Function to execute SQL rounds
        cap_results_func: Function to cap results
        markdown_func: Function to extract markdown
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
            question, context_bundle, paths.base_dir.name, config, api_key
        )
        run_logger.record_decision(decision.model_dump())
        logger.debug("Orchestrator decision: action=%s", decision.action)

        if decision.status == "error":
            logger.error("Orchestrator decision failed")
            run_logger.finalize("failed", finish_reason="decision_error")
            _detach_run_file_logger(run_log_handler)
            return paths

        # Handle write decision
        if decision.action == "write":
            result = handle_write_decision(
                question,
                context_bundle,
                paths,
                run_logger,
                run_log_handler,
                writer_func,
                config,
                api_key,
            )
            return result

        # Handle SQL decision
        if decision.action == "run_sql":
            follow_up = decision.follow_up_question or question
            result = handle_sql_decision(
                follow_up,
                schema_summary,
                city_names,
                paths,
                run_logger,
                run_log_handler,
                config,
                api_key,
                sql_plan_func,
                lambda q, p: sql_rounds_func(
                    q,
                    p,
                    db_client,
                    config,
                    api_key,
                    schema_summary,
                    city_names,
                    identifiers,
                    sql_plan_func,
                    write_json,
                    run_logger,
                    paths,
                ),
                cap_results_func,
                lambda plan, results, tokens, truncated: run_logger.update_sql_bundle(
                    build_sql_research_result(
                        paths.base_dir.name, plan.queries, results, tokens, truncated
                    ).model_dump()
                ),
                write_json,
            )
            if result:
                return result
            continue

        # Handle markdown decision
        if decision.action == "run_markdown":
            follow_up = decision.follow_up_question or question
            result = handle_markdown_decision(
                follow_up,
                documents,
                paths,
                run_logger,
                run_log_handler,
                config,
                api_key,
                markdown_func,
                write_json,
            )
            if result:
                return result
            continue

        # Handle stop decision
        if decision.action == "stop":
            logger.info("Orchestrator decided to stop")
            run_logger.finalize("stopped", finish_reason="stopped_by_orchestrator")
            _detach_run_file_logger(run_log_handler)
            return paths

    # Fallback writer after max iterations
    logger.warning(
        "Reached max iterations (%d), running fallback writer", max_iterations
    )
    context_bundle = load_context_bundle(paths)
    try:
        writer_output = writer_func(
            question, context_bundle, paths.base_dir.name, config, api_key
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
        _detach_run_file_logger(run_log_handler)
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
        _detach_run_file_logger(run_log_handler)
        return paths


def run_pipeline(
    question: str,
    config: AppConfig,
    run_id: str | None = None,
    sql_plan_func: Callable[..., SqlQueryPlan] = plan_sql_queries,
    markdown_func: Callable[..., MarkdownResearchResult] = extract_markdown_excerpts,
    decide_func: Callable[..., OrchestratorDecision] = decide_next_action,
    writer_func: Callable[..., WriterOutput] = write_markdown,
) -> RunPaths:
    """
    Run the multi-agent document builder pipeline.

    Orchestrates SQL research, markdown extraction, and document writing in an agentic loop.

    Args:
        question: User question to answer
        config: Application configuration
        run_id: Optional run identifier
        sql_plan_func: SQL planning function (default: plan_sql_queries)
        markdown_func: Markdown extraction function (default: extract_markdown_excerpts)
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
    run_log_handler = _attach_run_file_logger(paths.base_dir)

    # Load schema and extract identifiers
    models_dir = Path(__file__).resolve().parents[2] / "db_models"
    schema_summary = load_schema(models_dir).model_dump()
    write_json(paths.schema_summary, schema_summary)
    run_logger.record_artifact("schema_summary", paths.schema_summary)
    identifiers = _collect_identifiers(schema_summary)

    # Get database client and city list
    db_client = get_db_client(config.source_db_path, config.source_db_url)
    city_names: list[str] = []
    try:
        city_names = _fetch_city_list(db_client)
    except (sqlite3.OperationalError, psycopg.DatabaseError, OSError) as e:
        logger.warning("Failed to fetch city list: %s", e)
    write_json(paths.city_list, city_names)
    run_logger.record_artifact("city_list", paths.city_list)

    # Run initial SQL and markdown in parallel
    def _run_initial_sql() -> dict[str, object]:
        sql_plan = sql_plan_func(
            question,
            schema_summary,
            city_names,
            paths.base_dir.name,
            config,
            api_key,
            per_city_focus=True,
        )
        if sql_plan.status == "error":
            return {"status": "error", "plan": sql_plan}
        sql_plan, full_results, sql_rounds = _run_sql_rounds(
            question,
            sql_plan,
            db_client,
            config,
            api_key,
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
            config.markdown_dir, config.markdown_researcher
        )
        markdown_result = markdown_func(
            question, documents, paths.base_dir.name, config, api_key
        )
        return {"documents": documents, "result": markdown_result}

    sql_payload: dict[str, object] | None = None
    markdown_payload: dict[str, object] | None = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_initial_sql): "sql",
            executor.submit(_run_initial_markdown): "markdown",
        }
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

    if not sql_payload:
        run_logger.finalize("failed", finish_reason="sql_execution_failed")
        _detach_run_file_logger(run_log_handler)
        return paths
    if not markdown_payload:
        run_logger.finalize("failed", finish_reason="markdown_extraction_failed")
        _detach_run_file_logger(run_log_handler)
        return paths

    if sql_payload.get("status") == "error":
        sql_plan = sql_payload.get("plan")
        if isinstance(sql_plan, SqlQueryPlan):
            run_logger.record_decision(sql_plan.model_dump())
        run_logger.finalize("failed", finish_reason="sql_plan_error")
        _detach_run_file_logger(run_log_handler)
        return paths

    # Process and log initial SQL results
    sql_plan = sql_payload["plan"]
    full_results = sql_payload["full_results"]
    sql_rounds = sql_payload["sql_rounds"]
    capped_results = sql_payload["capped_results"]
    total_tokens = sql_payload["total_tokens"]
    truncated = sql_payload["truncated"]

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
        run_id=paths.base_dir.name,
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
            _detach_run_file_logger(run_log_handler)
            return paths
    else:
        run_logger.finalize("failed", finish_reason="markdown_extraction_failed")
        _detach_run_file_logger(run_log_handler)
        return paths

    # Run orchestration loop
    return _run_orchestration_loop(
        question,
        max_iterations=4,
        sql_payload=sql_payload,
        markdown_payload=markdown_payload,
        paths=paths,
        run_logger=run_logger,
        run_log_handler=run_log_handler,
        config=config,
        api_key=api_key,
        schema_summary=schema_summary,
        city_names=city_names,
        identifiers=identifiers,
        db_client=db_client,
        documents=documents,
        sql_plan_func=sql_plan_func,
        sql_rounds_func=_run_sql_rounds,
        cap_results_func=cap_results,
        markdown_func=markdown_func,
        decide_func=decide_func,
        writer_func=writer_func,
    )


__all__ = ["run_pipeline"]


def _attach_run_file_logger(run_dir: Path) -> logging.FileHandler:
    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.getLogger().level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def _detach_run_file_logger(handler: logging.FileHandler) -> None:
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def _load_context_bundle(paths: RunPaths) -> dict:
    if not paths.context_bundle.exists():
        return {}
    return json.loads(paths.context_bundle.read_text(encoding="utf-8"))


def _fetch_city_list(db_client: DbClient, max_rows: int = 500) -> list[str]:
    queries = [
        'SELECT "cityName" FROM "City"',
        "SELECT cityName FROM City",
        'SELECT "city_name" FROM "city"',
        "SELECT city_name FROM city",
    ]
    for sql in queries:
        try:
            columns, rows = db_client.query(sql)
        except (sqlite3.OperationalError, psycopg.DatabaseError, ValueError) as e:
            logger.debug("City list query attempt failed: %s", e)
            continue
        if not columns or not rows:
            continue
        names = [str(row[0]) for row in rows if row and row[0] is not None]
        return names[:max_rows]
    logger.warning("Could not fetch city list from database")
    return []


def _collect_sql_execution_errors(results: list[object]) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    for result in results:
        if isinstance(result, dict):
            columns = result.get("columns", [])
            rows = result.get("rows", [])
            query_id = result.get("query_id")
        else:
            columns = getattr(result, "columns", [])
            rows = getattr(result, "rows", [])
            query_id = getattr(result, "query_id", None)
        if columns == ["error"] and rows:
            errors.append({"query_id": query_id, "error": rows[0][0]})
    return errors


def _summarize_sql_results(
    results: list[object], max_rows: int = 3
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for result in results:
        if isinstance(result, dict):
            query_id = result.get("query_id")
            columns = result.get("columns", [])
            rows = result.get("rows", [])
            row_count = result.get("row_count", len(rows))
        else:
            query_id = getattr(result, "query_id", None)
            columns = getattr(result, "columns", [])
            rows = getattr(result, "rows", [])
            row_count = getattr(result, "row_count", len(rows))
        sample_rows = rows[:max_rows] if isinstance(rows, list) else []
        summary.append(
            {
                "query_id": query_id,
                "columns": columns,
                "row_count": row_count,
                "sample_rows": sample_rows,
            }
        )
    return summary


def _collect_identifiers(schema_summary: dict) -> list[str]:
    identifiers: set[str] = set()
    for table in schema_summary.get("tables", []):
        table_name = table.get("name")
        if table_name:
            identifiers.add(str(table_name))
        for column in table.get("columns", []):
            if column:
                identifiers.add(str(column))
        for fk in table.get("foreign_keys", []):
            for key in ("column", "ref_table", "ref_column"):
                value = fk.get(key)
                if value:
                    identifiers.add(str(value))
    return sorted(identifiers)


def run_pipeline(
    question: str,
    config: AppConfig,
    run_id: str | None = None,
    sql_plan_func: Callable[..., SqlQueryPlan] = plan_sql_queries,
    markdown_func: Callable[..., MarkdownResearchResult] = extract_markdown_excerpts,
    decide_func: Callable[..., OrchestratorDecision] = decide_next_action,
    writer_func: Callable[..., WriterOutput] = write_markdown,
) -> RunPaths:
    api_key = get_openrouter_api_key()
    run_id_value = run_id or build_run_id()
    paths = create_run_paths(
        config.runs_dir, run_id_value, config.orchestrator.context_bundle_name
    )
    run_logger = RunLogger(paths, question)
    run_logger.record_artifact("context_bundle", paths.context_bundle)
    run_log_handler = _attach_run_file_logger(paths.base_dir)

    models_dir = Path(__file__).resolve().parents[2] / "db_models"
    schema_summary = load_schema(models_dir).model_dump()
    _write_json(paths.schema_summary, schema_summary)
    run_logger.record_artifact("schema_summary", paths.schema_summary)
    identifiers = _collect_identifiers(schema_summary)

    def _execute_sql_plan(
        question_for_plan: str, sql_plan: SqlQueryPlan, allow_retry: bool = True
    ) -> tuple[SqlQueryPlan, list]:
        full_results = execute_queries(
            db_client,
            sql_plan.queries,
            config.sql_researcher.max_rows,
            identifiers=identifiers,
        )
        if allow_retry:
            sql_errors = _collect_sql_execution_errors(full_results)
            if sql_errors:
                retry_plan = sql_plan_func(
                    question_for_plan,
                    schema_summary,
                    city_names,
                    paths.base_dir.name,
                    config,
                    api_key,
                    sql_execution_errors=sql_errors,
                    previous_queries=[query.model_dump() for query in sql_plan.queries],
                    per_city_focus=True,
                )
                if retry_plan.status != "error":
                    sql_plan = retry_plan
                    _write_json(paths.sql_queries, sql_plan.model_dump())
                    run_logger.record_artifact("sql_queries", paths.sql_queries)
                    full_results = execute_queries(
                        db_client,
                        sql_plan.queries,
                        config.sql_researcher.max_rows,
                        identifiers=identifiers,
                    )
        return sql_plan, full_results

    def _run_sql_rounds(
        question_for_plan: str, initial_plan: SqlQueryPlan
    ) -> tuple[SqlQueryPlan, list, list[dict[str, object]]]:
        rounds_to_run = max(1, config.sql_researcher.pre_orchestrator_rounds)
        current_plan = initial_plan
        all_results: list = []
        all_queries = []
        rounds_payload: list[dict[str, object]] = []

        for round_index in range(rounds_to_run):
            current_plan, round_results = _execute_sql_plan(
                question_for_plan, current_plan, allow_retry=True
            )
            all_results.extend(round_results)
            all_queries.extend(current_plan.queries)
            rounds_payload.append(
                {
                    "round": round_index + 1,
                    "plan": current_plan.model_dump(),
                    "results": [result.model_dump() for result in round_results],
                }
            )
            if round_index + 1 >= rounds_to_run:
                break
            summary = _summarize_sql_results(round_results)
            next_plan = sql_plan_func(
                question_for_plan,
                schema_summary,
                city_names,
                paths.base_dir.name,
                config,
                api_key,
                sql_results_summary=summary,
                previous_queries=[query.model_dump() for query in current_plan.queries],
                per_city_focus=True,
            )
            if next_plan.status == "error" or not next_plan.queries:
                break
            current_plan = next_plan

        combined_plan = current_plan
        if all_queries:
            combined_plan = current_plan.model_copy(update={"queries": all_queries})
        return combined_plan, all_results, rounds_payload

    db_client = get_db_client(config.source_db_path, config.source_db_url)
    city_names: list[str] = []
    try:
        city_names = _fetch_city_list(db_client)
    except (sqlite3.OperationalError, psycopg.DatabaseError, OSError) as e:
        logger.warning("Failed to fetch city list: %s", e)
    _write_json(paths.city_list, city_names)
    run_logger.record_artifact("city_list", paths.city_list)

    def _run_initial_sql() -> dict[str, object]:
        sql_plan = sql_plan_func(
            question,
            schema_summary,
            city_names,
            paths.base_dir.name,
            config,
            api_key,
            per_city_focus=True,
        )
        if sql_plan.status == "error":
            return {"status": "error", "plan": sql_plan}
        sql_plan, full_results, sql_rounds = _run_sql_rounds(question, sql_plan)
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
            config.markdown_dir, config.markdown_researcher
        )
        markdown_result = markdown_func(
            question, documents, paths.base_dir.name, config, api_key
        )
        return {"documents": documents, "result": markdown_result}

    documents: list[dict[str, str]] = []
    sql_payload: dict[str, object] | None = None
    markdown_payload: dict[str, object] | None = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_initial_sql): "sql",
            executor.submit(_run_initial_markdown): "markdown",
        }
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                payload = future.result()
                if task_name == "sql":
                    sql_payload = payload
                else:
                    markdown_payload = payload
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                # These are known, handled exceptions from agent execution and file operations that cause the run to fail
                if task_name == "sql":
                    logger.exception("SQL execution failed")
                    run_logger.record_decision(
                        {
                            "status": "error",
                            "run_id": paths.base_dir.name,
                            "reason": "SQL execution failed",
                            "error": {
                                "code": "SQL_EXECUTION_ERROR",
                                "message": str(exc),
                            },
                        }
                    )
                    run_logger.finalize("failed", finish_reason="sql_execution_failed")
                    _detach_run_file_logger(run_log_handler)
                    return paths
                logger.exception("Markdown extraction failed")
                run_logger.record_decision(
                    {
                        "status": "error",
                        "run_id": paths.base_dir.name,
                        "reason": "Markdown extraction failed",
                        "error": {"code": "MARKDOWN_ERROR", "message": str(exc)},
                    }
                )
                run_logger.finalize(
                    "failed", finish_reason="markdown_extraction_failed"
                )
                _detach_run_file_logger(run_log_handler)
                return paths

    if not sql_payload:
        run_logger.finalize("failed", finish_reason="sql_execution_failed")
        _detach_run_file_logger(run_log_handler)
        return paths
    if not markdown_payload:
        run_logger.finalize("failed", finish_reason="markdown_extraction_failed")
        _detach_run_file_logger(run_log_handler)
        return paths

    if sql_payload.get("status") == "error":
        sql_plan = sql_payload.get("plan")
        if isinstance(sql_plan, SqlQueryPlan):
            run_logger.record_decision(sql_plan.model_dump())
        run_logger.finalize("failed", finish_reason="sql_plan_error")
        _detach_run_file_logger(run_log_handler)
        return paths

    sql_plan = sql_payload["plan"]
    full_results = sql_payload["full_results"]
    sql_rounds = sql_payload["sql_rounds"]
    capped_results = sql_payload["capped_results"]
    total_tokens = sql_payload["total_tokens"]
    truncated = sql_payload["truncated"]

    sql_rounds_path = paths.sql_dir / "rounds.json"
    _write_json(sql_rounds_path, sql_rounds)
    run_logger.record_artifact("sql_rounds", sql_rounds_path)
    _write_json(paths.sql_queries, sql_plan.model_dump())
    run_logger.record_artifact("sql_queries", paths.sql_queries)

    full_payload = [result.model_dump() for result in full_results]
    capped_payload = [result.model_dump() for result in capped_results]
    _write_json(paths.sql_results_full, full_payload)
    _write_json(paths.sql_results, capped_payload)
    run_logger.record_artifact("sql_results_full", paths.sql_results_full)
    run_logger.record_artifact("sql_results", paths.sql_results)

    sql_result = build_sql_research_result(
        run_id=paths.base_dir.name,
        queries=sql_plan.queries,
        results=capped_results,
        total_tokens=total_tokens,
        truncated=truncated,
    )
    run_logger.update_sql_bundle(sql_result.model_dump())

    documents = markdown_payload["documents"]
    markdown_result = markdown_payload["result"]
    if isinstance(markdown_result, MarkdownResearchResult):
        _write_json(paths.markdown_excerpts, markdown_result.model_dump())
        run_logger.record_artifact("markdown_excerpts", paths.markdown_excerpts)
        run_logger.update_markdown_bundle(markdown_result.model_dump())
        if markdown_result.status == "error":
            run_logger.record_decision(markdown_result.model_dump())
            run_logger.finalize("failed", finish_reason="markdown_result_error")
            _detach_run_file_logger(run_log_handler)
            return paths
    else:
        run_logger.finalize("failed", finish_reason="markdown_extraction_failed")
        _detach_run_file_logger(run_log_handler)
        return paths

    max_iterations = 4
    for _ in range(max_iterations):
        context_bundle = _load_context_bundle(paths)
        decision = decide_func(
            question, context_bundle, paths.base_dir.name, config, api_key
        )
        run_logger.record_decision(decision.model_dump())
        if decision.status == "error":
            run_logger.finalize("failed", finish_reason="decision_error")
            _detach_run_file_logger(run_log_handler)
            return paths

        if decision.action == "write":
            try:
                writer_output = writer_func(
                    question, context_bundle, paths.base_dir.name, config, api_key
                )
                question_header = f"# Question\n{question}\n\n"
                finish_note = "\n\n---\nFinish reason: completed (write)\n"
                rendered_content = (
                    f"{question_header}{writer_output.content}{finish_note}"
                )
                draft_index = len(run_logger.run_log.get("drafts", [])) + 1
                draft_path = paths.drafts_dir / f"draft_{draft_index:02d}.md"
                draft_path.write_text(rendered_content, encoding="utf-8")
                run_logger.record_draft(draft_path)

                final_path = paths.final_output
                final_path.write_text(rendered_content, encoding="utf-8")
                run_logger.record_artifact("final_output", final_path)
                run_logger.finalize(
                    "completed",
                    final_output_path=final_path,
                    finish_reason="completed (write)",
                )
                _detach_run_file_logger(run_log_handler)
                return paths
            except (ValueError, RuntimeError, OSError) as exc:
                logger.exception("Writer failed")
                run_logger.record_decision(
                    {
                        "status": "error",
                        "run_id": paths.base_dir.name,
                        "reason": "Writer failed",
                        "error": {"code": "WRITER_ERROR", "message": str(exc)},
                    }
                )
                run_logger.finalize("failed", finish_reason="writer_failed")
                _detach_run_file_logger(run_log_handler)
                return paths

        if decision.action == "run_sql":
            follow_up = decision.follow_up_question or question
            sql_plan = sql_plan_func(
                follow_up,
                schema_summary,
                city_names,
                paths.base_dir.name,
                config,
                api_key,
                per_city_focus=True,
            )
            _write_json(paths.sql_queries, sql_plan.model_dump())
            run_logger.record_artifact("sql_queries", paths.sql_queries)
            if sql_plan.status == "error":
                run_logger.record_decision(sql_plan.model_dump())
                run_logger.finalize("failed", finish_reason="sql_plan_error")
                _detach_run_file_logger(run_log_handler)
                return paths
            try:
                sql_plan, full_results, sql_rounds = _run_sql_rounds(
                    follow_up, sql_plan
                )
                sql_rounds_path = paths.sql_dir / "rounds.json"
                _write_json(sql_rounds_path, sql_rounds)
                run_logger.record_artifact("sql_rounds", sql_rounds_path)
                _write_json(paths.sql_queries, sql_plan.model_dump())
                run_logger.record_artifact("sql_queries", paths.sql_queries)
                capped_results, total_tokens, truncated = cap_results(
                    full_results, config.sql_researcher.max_result_tokens
                )
            except (ValueError, RuntimeError, OSError) as exc:
                logger.exception("SQL execution failed")
                run_logger.record_decision(
                    {
                        "status": "error",
                        "run_id": paths.base_dir.name,
                        "reason": "SQL execution failed",
                        "error": {"code": "SQL_EXECUTION_ERROR", "message": str(exc)},
                    }
                )
                run_logger.finalize("failed", finish_reason="sql_execution_failed")
                _detach_run_file_logger(run_log_handler)
                return paths
            _write_json(
                paths.sql_results_full, [res.model_dump() for res in full_results]
            )
            _write_json(paths.sql_results, [res.model_dump() for res in capped_results])
            run_logger.record_artifact("sql_results_full", paths.sql_results_full)
            run_logger.record_artifact("sql_results", paths.sql_results)
            sql_result = build_sql_research_result(
                run_id=paths.base_dir.name,
                queries=sql_plan.queries,
                results=capped_results,
                total_tokens=total_tokens,
                truncated=truncated,
            )
            run_logger.update_sql_bundle(sql_result.model_dump())
            continue

        if decision.action == "run_markdown":
            follow_up = decision.follow_up_question or question
            try:
                markdown_result = markdown_func(
                    follow_up, documents, paths.base_dir.name, config, api_key
                )
                _write_json(paths.markdown_excerpts, markdown_result.model_dump())
                run_logger.record_artifact("markdown_excerpts", paths.markdown_excerpts)
                run_logger.update_markdown_bundle(markdown_result.model_dump())
                if markdown_result.status == "error":
                    run_logger.record_decision(markdown_result.model_dump())
                    run_logger.finalize("failed", finish_reason="markdown_result_error")
                    _detach_run_file_logger(run_log_handler)
                    return paths
            except (ValueError, RuntimeError, OSError) as exc:
                logger.exception("Markdown extraction failed")
                run_logger.record_decision(
                    {
                        "status": "error",
                        "run_id": paths.base_dir.name,
                        "reason": "Markdown extraction failed",
                        "error": {"code": "MARKDOWN_ERROR", "message": str(exc)},
                    }
                )
                run_logger.finalize(
                    "failed", finish_reason="markdown_extraction_failed"
                )
                _detach_run_file_logger(run_log_handler)
                return paths
            continue

        if decision.action == "stop":
            run_logger.finalize("stopped", finish_reason="stopped_by_orchestrator")
            _detach_run_file_logger(run_log_handler)
            return paths

    context_bundle = _load_context_bundle(paths)
    try:
        writer_output = writer_func(
            question, context_bundle, paths.base_dir.name, config, api_key
        )
        question_header = f"# Question\n{question}\n\n"
        finish_note = "\n\n---\nFinish reason: completed_with_gaps (max iterations)\n"
        rendered_content = f"{question_header}{writer_output.content}{finish_note}"
        draft_index = len(run_logger.run_log.get("drafts", [])) + 1
        draft_path = paths.drafts_dir / f"draft_{draft_index:02d}.md"
        draft_path.write_text(rendered_content, encoding="utf-8")
        run_logger.record_draft(draft_path)

        final_path = paths.final_output
        final_path.write_text(rendered_content, encoding="utf-8")
        run_logger.record_artifact("final_output", final_path)
        run_logger.finalize(
            "completed_with_gaps",
            final_output_path=final_path,
            finish_reason="completed_with_gaps (max iterations)",
        )
        _detach_run_file_logger(run_log_handler)
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
        _detach_run_file_logger(run_log_handler)
        return paths


__all__ = ["run_pipeline"]
