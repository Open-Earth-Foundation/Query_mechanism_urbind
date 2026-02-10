from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, function_tool

from app.modules.sql_researcher.models import SqlQueryPlan
from app.modules.sql_researcher.services import build_table_catalog, sanitize_queries, validate_queries
from app.services.agents import build_model_settings, build_openrouter_model, run_agent_sync
from app.utils.config import AppConfig
from app.utils.prompts import load_prompt
from app.utils.tokenization import get_max_input_tokens


def build_sql_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the SQL researcher agent."""
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "sql_researcher_system.md"
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(config.sql_researcher.model, api_key, config.openrouter_base_url)
    settings = build_model_settings(
        config.sql_researcher.temperature,
        config.sql_researcher.max_output_tokens,
    )

    @function_tool
    def submit_sql_queries(plan: SqlQueryPlan) -> SqlQueryPlan:
        return plan

    return Agent(
        name="SQL Researcher",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_sql_queries],
        output_type=SqlQueryPlan,
        tool_use_behavior="stop_on_first_tool",
    )


def plan_sql_queries(
    question: str,
    schema_summary: dict,
    city_names: list[str],
    config: AppConfig,
    api_key: str,
    sql_execution_errors: list[dict[str, object]] | None = None,
    previous_queries: list[dict[str, object]] | None = None,
    sql_results_summary: list[dict[str, object]] | None = None,
    per_city_focus: bool | None = None,
    log_llm_payload: bool = False,
) -> SqlQueryPlan:
    """Generate a SQL query plan for the current question."""
    agent = build_sql_agent(config, api_key)
    table_catalog = build_table_catalog(schema_summary)
    payload: dict[str, object] = {
        "question": question,
        "schema_summary": schema_summary,
        "table_catalog": table_catalog,
        "city_names": city_names,
        "per_city_focus": per_city_focus if per_city_focus is not None else bool(city_names),
        "context_window_tokens": config.sql_researcher.context_window_tokens,
        "max_input_tokens": get_max_input_tokens(
            config.sql_researcher.context_window_tokens,
            config.sql_researcher.max_output_tokens,
            config.sql_researcher.input_token_reserve,
            config.sql_researcher.max_input_tokens,
        ),
    }
    if sql_execution_errors:
        payload["sql_execution_errors"] = sql_execution_errors
    if previous_queries:
        payload["previous_queries"] = previous_queries
    if sql_results_summary:
        payload["sql_results_summary"] = sql_results_summary
    result = run_agent_sync(
        agent,
        json.dumps(payload, ensure_ascii=True),
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if not isinstance(output, SqlQueryPlan):
        raise ValueError("SQL researcher did not return a structured query plan.")

    validation_errors = validate_queries(output.queries, schema_summary)
    if validation_errors:
        retry_payload = {
            **payload,
            "previous_queries": [query.model_dump() for query in output.queries],
            "validation_errors": validation_errors,
        }
        retry_result = run_agent_sync(
            agent,
            json.dumps(retry_payload, ensure_ascii=True),
            log_llm_payload=log_llm_payload,
        )
        retry_output = retry_result.final_output
        if isinstance(retry_output, SqlQueryPlan):
            output = retry_output

    validation_errors = validate_queries(output.queries, schema_summary)
    if validation_errors:
        sanitized = sanitize_queries(output.queries, schema_summary)
        output = output.model_copy(update={"queries": sanitized})

    return output


__all__ = ["build_sql_agent", "plan_sql_queries"]
