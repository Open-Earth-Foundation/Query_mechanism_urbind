from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from agents import Agent, function_tool

from backend.modules.calculation_researcher.agent import (
    run_calculation_subagent as run_calculation_subagent_agent,
)
from backend.modules.calculation_researcher.models import (
    CalculationRequest,
    CalculationSubagentOutput,
)
from backend.modules.writer.models import WriterOutput
from backend.modules.writer.utils.markdown_helpers import (
    append_sections,
    extract_city_coverage_sets,
    extract_markdown_bundle,
    extract_missing_city_excerpts,
    extract_ref_city_mapping,
    extract_selected_city_names,
    render_cities_considered_section,
    render_no_evidence_section,
    resolve_analysis_mode,
    validate_writer_citations,
)
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.utils.city_normalization import format_city_display_name
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt
from backend.utils.retry import (
    RetrySettings,
    compute_retry_delay_seconds,
    log_retry_event,
    log_retry_exhausted,
)

logger = logging.getLogger(__name__)


def _resolve_writer_prompt_path(analysis_mode: str) -> Path:
    """Resolve writer prompt path for the selected analysis mode."""
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    if analysis_mode == "city_by_city":
        return prompts_dir / "writer_system_city_by_city.md"
    return prompts_dir / "writer_system_aggregate.md"


def _format_ref_tokens(ref_ids: list[str]) -> str:
    return "".join(f"[{ref_id}]" for ref_id in ref_ids)


def _render_calculated_values_section(
    records: list[dict[str, Any]],
) -> str:
    if not records:
        return ""

    lines: list[str] = ["## Calculated values"]
    for record in records:
        output = record.get("output")
        if not isinstance(output, dict):
            continue
        metric_name = str(output.get("metric_name", "")).strip() or "metric"
        total_value = output.get("total_value")
        coverage_observed = int(output.get("coverage_observed", 0) or 0)
        coverage_total = int(output.get("coverage_total", 0) or 0)
        final_ref_ids = [
            str(ref_id)
            for ref_id in output.get("final_ref_ids", [])
            if isinstance(ref_id, str)
        ]
        ref_suffix = f" {_format_ref_tokens(final_ref_ids)}" if final_ref_ids else ""

        if isinstance(total_value, (int, float)):
            rendered_total = f"{total_value:,.3f}".rstrip("0").rstrip(".")
        else:
            rendered_total = "unavailable"

        lines.append("")
        lines.append(
            f"Calculated saving command for {metric_name}: {rendered_total} "
            f"(observed coverage: {coverage_observed}/{coverage_total}){ref_suffix}"
        )
        lines.append("")
        lines.append("### Cities included in numeric sum")
        included_cities = output.get("included_cities", [])
        if isinstance(included_cities, list) and included_cities:
            for city in included_cities:
                if not isinstance(city, dict):
                    continue
                city_name = str(city.get("city_name", "")).strip() or "(unknown city)"
                year = city.get("year")
                year_text = str(year) if isinstance(year, int) else "n/a"
                city_ref_ids = [
                    str(ref_id)
                    for ref_id in city.get("ref_ids", [])
                    if isinstance(ref_id, str)
                ]
                city_refs = _format_ref_tokens(city_ref_ids)
                lines.append(f"- {city_name} ({year_text}) {city_refs}".rstrip())
        else:
            lines.append("- (none)")

        lines.append("")
        lines.append("### Cities excluded from numeric sum (policy-only)")
        excluded_cities = output.get("excluded_policy_cities", [])
        if isinstance(excluded_cities, list) and excluded_cities:
            for city in excluded_cities:
                if not isinstance(city, dict):
                    continue
                city_name = str(city.get("city_name", "")).strip() or "(unknown city)"
                reason = str(city.get("reason_no_numeric", "")).strip() or "no numeric value"
                summary = str(city.get("policy_summary", "")).strip()
                city_ref_ids = [
                    str(ref_id)
                    for ref_id in city.get("ref_ids", [])
                    if isinstance(ref_id, str)
                ]
                city_refs = _format_ref_tokens(city_ref_ids)
                details = f"{reason}; {summary}".strip("; ").strip()
                lines.append(f"- {city_name}: {details} {city_refs}".rstrip())
        else:
            lines.append("- (none)")

        lines.append("")
        lines.append("### Assumptions used for calculation")
        assumptions = output.get("assumptions", [])
        if isinstance(assumptions, list) and assumptions:
            for assumption in assumptions:
                if not isinstance(assumption, dict):
                    continue
                statement = str(assumption.get("statement", "")).strip()
                if not statement:
                    continue
                assumption_ref_ids = [
                    str(ref_id)
                    for ref_id in assumption.get("ref_ids", [])
                    if isinstance(ref_id, str)
                ]
                assumption_refs = _format_ref_tokens(assumption_ref_ids)
                lines.append(f"- {statement} {assumption_refs}".rstrip())
        else:
            lines.append("- (none)")

    return "\n".join(lines).strip()


def _write_calculation_artifact(
    *,
    run_id: str | None,
    config: AppConfig,
    records: list[dict[str, Any]],
) -> None:
    if not run_id or not records:
        return
    artifact_path = config.runs_dir / run_id / "markdown" / "calculations.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps({"calculations": records}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def build_writer_agent(
    config: AppConfig,
    api_key: str,
    analysis_mode: str,
    *,
    question: str,
    context_bundle: dict[str, object],
    log_llm_payload: bool,
    calculation_records: list[dict[str, Any]],
) -> Agent:
    """Build the writer agent."""
    prompt_path = _resolve_writer_prompt_path(analysis_mode)
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(
        config.writer.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.writer.temperature,
        config.writer.max_output_tokens,
        reasoning_effort=config.writer.reasoning_effort,
    )

    @function_tool(name_override="run_calculation_subagent", strict_mode=True)
    def run_calculation_tool(request: CalculationRequest) -> CalculationSubagentOutput:
        """Delegate structured numeric aggregation to calculation subagent."""
        logger.info(
            "WRITER_CALCULATION_SUBAGENT_REQUEST %s",
            json.dumps(
                {
                    "metric_name": request.metric_name,
                    "operation": request.operation,
                    "city_scope_count": len(request.city_scope),
                    "city_scope": request.city_scope,
                },
                ensure_ascii=False,
            ),
        )
        output = run_calculation_subagent_agent(
            request=request,
            question=question,
            context_bundle=context_bundle,
            config=config,
            api_key=api_key,
            log_llm_payload=log_llm_payload,
        )
        calculation_records.append(
            {
                "request": request.model_dump(),
                "output": output.model_dump(),
            }
        )
        logger.info(
            "WRITER_CALCULATION_SUBAGENT_RESULT %s",
            json.dumps(
                {
                    "metric_name": output.metric_name,
                    "operation": output.operation,
                    "status": output.status,
                    "coverage_observed": output.coverage_observed,
                    "coverage_total": output.coverage_total,
                    "error": output.error.model_dump() if output.error is not None else None,
                },
                ensure_ascii=False,
            ),
        )
        return output

    @function_tool
    def submit_writer_output(output: WriterOutput) -> WriterOutput:
        """Return structured writer output unchanged."""
        return output

    return Agent(
        name="Writer",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[
            run_calculation_tool,
            submit_writer_output,
        ],
        output_type=WriterOutput,
        tool_use_behavior="run_llm_again",
    )


def _run_writer_once(
    *,
    agent: Agent,
    payload: dict[str, object],
    max_turns: int,
    log_llm_payload: bool,
) -> WriterOutput:
    """Run writer once and return structured output."""
    result = run_agent_sync(
        agent,
        json.dumps(payload, ensure_ascii=False),
        max_turns=max_turns,
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, WriterOutput):
        return output
    raise ValueError("Writer did not return structured output.")


def write_markdown(
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
    run_id: str | None = None,
) -> WriterOutput:
    """Generate the final markdown answer with city-coverage guardrails."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    selected_city_names = extract_selected_city_names(context_bundle, markdown_bundle)
    analysis_mode = resolve_analysis_mode(context_bundle)
    calculation_records: list[dict[str, Any]] = []
    agent = build_writer_agent(
        config,
        api_key,
        analysis_mode=analysis_mode,
        question=question,
        context_bundle=context_bundle,
        log_llm_payload=log_llm_payload,
        calculation_records=calculation_records,
    )
    retry_settings = RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )
    max_attempts = retry_settings.max_attempts

    previous_answer = ""
    missing_city_keys: list[str] = []

    for attempt in range(1, max_attempts + 1):
        start_record_index = len(calculation_records)
        payload: dict[str, object] = {
            "question": question,
            "context_bundle": context_bundle,
            "analysis_mode": analysis_mode,
            "selected_cities": selected_city_names,
        }
        if attempt > 1 and previous_answer:
            ref_city_map = extract_ref_city_mapping(markdown_bundle)[1]
            reconsideration_payload: dict[str, object] = {
                "previous_answer": previous_answer,
            }
            if missing_city_keys:
                missing_city_names = [
                    ref_city_map.get(city_key, format_city_display_name(city_key))
                    for city_key in missing_city_keys
                ]
                reconsideration_payload["missing_cities"] = missing_city_names
                reconsideration_payload["missing_city_excerpts"] = extract_missing_city_excerpts(
                    markdown_bundle,
                    missing_city_keys,
                )
            payload["reconsideration"] = reconsideration_payload

        output = _run_writer_once(
            agent=agent,
            payload=payload,
            max_turns=config.writer.max_turns,
            log_llm_payload=log_llm_payload,
        )
        attempt_records = calculation_records[start_record_index:]
        records_for_render = attempt_records or calculation_records

        (
            required_city_keys,
            missing_coverage_keys,
            no_evidence_keys,
            city_display_by_key,
        ) = extract_city_coverage_sets(
            content=output.content,
            markdown_bundle=markdown_bundle,
            selected_city_names=selected_city_names,
        )
        confirmed_city_count = len(required_city_keys) - len(missing_coverage_keys)
        required_city_count = len(required_city_keys)
        coverage_ratio = f"{confirmed_city_count}/{required_city_count}"
        no_evidence_names = [
            city_display_by_key.get(city_key, format_city_display_name(city_key))
            for city_key in no_evidence_keys
        ]
        cities_considered = selected_city_names or sorted(city_display_by_key.values())
        content = append_sections(
            output.content,
            [
                _render_calculated_values_section(records_for_render),
                render_no_evidence_section(no_evidence_names),
                render_cities_considered_section(cities_considered),
            ],
        )
        validate_writer_citations(content, context_bundle)

        if not missing_coverage_keys:
            _write_calculation_artifact(
                run_id=run_id,
                config=config,
                records=calculation_records,
            )
            logger.info(
                "WRITER_CITATION_COVERAGE %s",
                json.dumps(
                    {
                        "run_id": run_id or "unknown",
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "status": "confirmed",
                        "coverage_confirmed": confirmed_city_count,
                        "coverage_required": required_city_count,
                        "coverage_ratio": coverage_ratio,
                        "analysis_mode": analysis_mode,
                    },
                    ensure_ascii=False,
                ),
            )
            return WriterOutput(content=content)

        previous_answer = output.content
        missing_city_keys = missing_coverage_keys
        missing_city_names = [
            city_display_by_key.get(city_key, format_city_display_name(city_key))
            for city_key in missing_city_keys
        ]
        coverage_status = "retrying" if attempt < max_attempts else "exhausted"
        logger.warning(
            "WRITER_CITATION_COVERAGE %s",
            json.dumps(
                {
                    "run_id": run_id or "unknown",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "status": coverage_status,
                    "coverage_confirmed": confirmed_city_count,
                    "coverage_required": required_city_count,
                    "coverage_ratio": coverage_ratio,
                    "missing_cities": missing_city_names,
                    "analysis_mode": analysis_mode,
                },
                ensure_ascii=False,
            ),
        )
        if attempt < max_attempts:
            delay_seconds = compute_retry_delay_seconds(attempt, retry_settings)
            retry_error_type = "MissingCityCitationCoverage"
            retry_error_message = (
                f"Writer city citation coverage is {coverage_ratio}; retrying missing cities: "
                + ", ".join(missing_city_names)
            )
            log_retry_event(
                operation="writer.output_reconsideration",
                run_id=run_id,
                attempt=attempt,
                max_attempts=max_attempts,
                error_type=retry_error_type,
                error_message=retry_error_message,
                next_backoff_seconds=delay_seconds,
                context={
                    "missing_cities": missing_city_names,
                    "coverage_confirmed": confirmed_city_count,
                    "coverage_required": required_city_count,
                    "coverage_ratio": coverage_ratio,
                    "analysis_mode": analysis_mode,
                },
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            continue

        _write_calculation_artifact(
            run_id=run_id,
            config=config,
            records=calculation_records,
        )
        exhausted_error_type = "MissingCityCitationCoverage"
        exhausted_error_message = (
            f"Writer city citation coverage remains {coverage_ratio}; missing cities: "
            + ", ".join(missing_city_names)
        )
        log_retry_exhausted(
            operation="writer.output_reconsideration",
            run_id=run_id,
            attempt=attempt,
            max_attempts=max_attempts,
            error_type=exhausted_error_type,
            error_message=exhausted_error_message,
            context={
                "missing_cities": missing_city_names,
                "coverage_confirmed": confirmed_city_count,
                "coverage_required": required_city_count,
                "coverage_ratio": coverage_ratio,
                "analysis_mode": analysis_mode,
            },
        )
        return WriterOutput(content=content)

    raise RuntimeError("Writer retry loop ended unexpectedly.")


__all__ = ["build_writer_agent", "write_markdown"]
