from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, function_tool

from backend.modules.calculation_researcher.models import (
    CalculationError,
    CalculationRequest,
    CalculationSubagentOutput,
)
from backend.modules.orchestrator.utils.references import is_valid_ref_id
from backend.services.agents import (
    MaxToolCallsExceeded,
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.tools.calculator import (
    divide_numbers,
    multiply_numbers,
    subtract_numbers,
    sum_numbers,
)
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt


def _extract_expected_ref_ids(context_bundle: dict[str, object]) -> set[str]:
    markdown_payload = context_bundle.get("markdown")
    if not isinstance(markdown_payload, dict):
        return set()
    excerpts = markdown_payload.get("excerpts")
    if not isinstance(excerpts, list):
        return set()

    expected_ids: set[str] = set()
    for excerpt in excerpts:
        if not isinstance(excerpt, dict):
            continue
        ref_id = excerpt.get("ref_id")
        if not isinstance(ref_id, str):
            continue
        candidate = ref_id.strip()
        if is_valid_ref_id(candidate):
            expected_ids.add(candidate)
    return expected_ids


def _filter_valid_refs(ref_ids: list[str], expected_ref_ids: set[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in ref_ids:
        candidate = raw.strip()
        if not is_valid_ref_id(candidate):
            continue
        if candidate not in expected_ref_ids:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _normalize_output(
    output: CalculationSubagentOutput,
    request: CalculationRequest,
    expected_ref_ids: set[str],
) -> CalculationSubagentOutput:
    included_cities = []
    for city in output.included_cities:
        valid_refs = _filter_valid_refs(city.ref_ids, expected_ref_ids)
        if not valid_refs:
            continue
        included_cities.append(city.model_copy(update={"ref_ids": valid_refs}))

    excluded_policy_cities = []
    for city in output.excluded_policy_cities:
        valid_refs = _filter_valid_refs(city.ref_ids, expected_ref_ids)
        if not valid_refs:
            continue
        excluded_policy_cities.append(city.model_copy(update={"ref_ids": valid_refs}))

    assumptions = []
    for assumption in output.assumptions:
        valid_refs = _filter_valid_refs(assumption.ref_ids, expected_ref_ids)
        if not valid_refs:
            continue
        assumptions.append(assumption.model_copy(update={"ref_ids": valid_refs}))

    if request.year_rule == "same_year_only":
        available_years = {
            city.year for city in included_cities if isinstance(city.year, int)
        }
        if len(available_years) > 1:
            first_year = min(available_years)
            included_cities = [city for city in included_cities if city.year == first_year]
    elif request.year_rule == "user_specified_year" and request.target_year is not None:
        included_cities = [
            city for city in included_cities if city.year == request.target_year
        ]

    if output.unit is not None:
        included_cities = [city for city in included_cities if city.unit == output.unit]

    coverage_total = len({city.strip().casefold() for city in request.city_scope if city.strip()})
    coverage_observed = len(
        {
            city.city_name.strip().casefold()
            for city in included_cities
            if city.city_name.strip()
        }
    )

    collected_final_refs: list[str] = []
    for city in included_cities:
        for ref_id in city.ref_ids:
            if ref_id not in collected_final_refs:
                collected_final_refs.append(ref_id)
    for city in excluded_policy_cities:
        for ref_id in city.ref_ids:
            if ref_id not in collected_final_refs:
                collected_final_refs.append(ref_id)
    for assumption in assumptions:
        for ref_id in assumption.ref_ids:
            if ref_id not in collected_final_refs:
                collected_final_refs.append(ref_id)

    status = output.status
    total_value = output.total_value
    error = output.error
    if not included_cities:
        total_value = None
        status = "partial" if status != "error" else "error"
    if status == "error" and error is None:
        error = CalculationError(code="CALCULATION_FAILED", message="Calculation failed.")

    return output.model_copy(
        update={
            "metric_name": request.metric_name,
            "operation": request.operation,
            "status": status,
            "total_value": total_value,
            "coverage_observed": coverage_observed,
            "coverage_total": coverage_total,
            "included_cities": included_cities,
            "excluded_policy_cities": excluded_policy_cities,
            "assumptions": assumptions,
            "final_ref_ids": collected_final_refs,
            "error": error,
        }
    )


def build_calculation_agent(config: AppConfig, api_key: str) -> Agent:
    """Build calculation subagent used by writer for aggregate numeric operations."""
    prompt_path = (
        Path(__file__).resolve().parents[2] / "prompts" / "calculation_researcher_system.md"
    )
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(
        config.calculation_researcher.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.calculation_researcher.temperature,
        config.calculation_researcher.max_output_tokens,
        reasoning_effort=config.calculation_researcher.reasoning_effort,
    )

    @function_tool(name_override="sum_numbers", strict_mode=True)
    def sum_numbers_tool(numbers: list[float]) -> float:
        return sum_numbers(numbers, source="calculation_subagent")

    @function_tool(name_override="subtract_numbers", strict_mode=True)
    def subtract_numbers_tool(minuend: float, subtrahend: float) -> float:
        return subtract_numbers(minuend, subtrahend, source="calculation_subagent")

    @function_tool(name_override="multiply_numbers", strict_mode=True)
    def multiply_numbers_tool(numbers: list[float]) -> float:
        return multiply_numbers(numbers, source="calculation_subagent")

    @function_tool(name_override="divide_numbers", strict_mode=True)
    def divide_numbers_tool(dividend: float, divisor: float) -> float:
        return divide_numbers(dividend, divisor, source="calculation_subagent")

    @function_tool(name_override="submit_calculation_output", strict_mode=True)
    def submit_calculation_output(
        output: CalculationSubagentOutput,
    ) -> CalculationSubagentOutput:
        return output

    return Agent(
        name="Calculation Researcher",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[
            sum_numbers_tool,
            subtract_numbers_tool,
            multiply_numbers_tool,
            divide_numbers_tool,
            submit_calculation_output,
        ],
        output_type=CalculationSubagentOutput,
        tool_use_behavior="run_llm_again",
    )


def run_calculation_subagent(
    *,
    request: CalculationRequest,
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> CalculationSubagentOutput:
    """Execute calculation subagent and normalize output to strict invariants."""
    expected_ref_ids = _extract_expected_ref_ids(context_bundle)
    try:
        agent = build_calculation_agent(config, api_key)
        payload = {
            "question": question,
            "request": request.model_dump(),
            "context_bundle": context_bundle,
        }
        result = run_agent_sync(
            agent,
            json.dumps(payload, ensure_ascii=False),
            max_turns=config.calculation_researcher.max_turns,
            max_tool_calls=config.calculation_researcher.max_tool_calls,
            log_llm_payload=log_llm_payload,
        )
        output = result.final_output
        if not isinstance(output, CalculationSubagentOutput):
            raise ValueError("Calculation researcher did not return structured output.")
        return _normalize_output(output, request, expected_ref_ids)
    except (ValueError, RuntimeError, MaxToolCallsExceeded) as exc:
        return CalculationSubagentOutput(
            status="error",
            metric_name=request.metric_name,
            operation=request.operation,
            total_value=None,
            unit=None,
            coverage_observed=0,
            coverage_total=len(request.city_scope),
            included_cities=[],
            excluded_policy_cities=[],
            assumptions=[],
            final_ref_ids=[],
            error=CalculationError(
                code="CALCULATION_SUBAGENT_FAILED",
                message=str(exc),
            ),
        )


__all__ = ["build_calculation_agent", "run_calculation_subagent"]
