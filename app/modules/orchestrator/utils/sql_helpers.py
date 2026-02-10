"""SQL research helpers for orchestrator."""

import logging
import sqlite3
from typing import Callable

import psycopg

from app.modules.sql_researcher.models import SqlQueryPlan
from app.services.db_client import DbClient
from app.utils.config import AppConfig
from app.services.run_logger import RunLogger
from app.utils.paths import RunPaths

logger = logging.getLogger(__name__)


def fetch_city_list(db_client: DbClient, max_rows: int = 500) -> list[str]:
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


def collect_sql_execution_errors(results: list[object]) -> list[dict[str, object]]:
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


def summarize_sql_results(
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


def collect_identifiers(schema_summary: dict) -> list[str]:
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


def execute_sql_plan(
    question: str,
    sql_plan: SqlQueryPlan,
    db_client: DbClient | None,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool,
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
        log_llm_payload: Whether to log full LLM request/response payloads
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
    from app.modules.sql_researcher.services import execute_queries

    full_results = execute_queries(
        db_client,
        sql_plan.queries,
        config.sql_researcher.max_rows,
        identifiers=identifiers,
    )

    if allow_retry:
        sql_errors = collect_sql_execution_errors(full_results)
        if sql_errors:
            retry_plan = sql_plan_func(
                question,
                schema_summary,
                city_names,
                config,
                api_key,
                sql_execution_errors=sql_errors,
                previous_queries=[query.model_dump() for query in sql_plan.queries],
                per_city_focus=True,
                log_llm_payload=log_llm_payload,
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


def run_sql_rounds(
    question: str,
    initial_plan: SqlQueryPlan,
    db_client: DbClient | None,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool,
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
        log_llm_payload: Whether to log full LLM request/response payloads
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
        current_plan, round_results = execute_sql_plan(
            question,
            current_plan,
            db_client,
            config,
            api_key,
            log_llm_payload,
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

        summary = summarize_sql_results(round_results)
        next_plan = sql_plan_func(
            question,
            schema_summary,
            city_names,
            config,
            api_key,
            sql_results_summary=summary,
            previous_queries=[query.model_dump() for query in current_plan.queries],
            per_city_focus=True,
            log_llm_payload=log_llm_payload,
        )

        if next_plan.status == "error" or not next_plan.queries:
            break
        current_plan = next_plan

    combined_plan = current_plan
    if all_queries:
        combined_plan = current_plan.model_copy(update={"queries": all_queries})

    return combined_plan, all_results, rounds_payload


__all__ = [
    "fetch_city_list",
    "collect_sql_execution_errors",
    "summarize_sql_results",
    "collect_identifiers",
    "execute_sql_plan",
    "run_sql_rounds",
]
