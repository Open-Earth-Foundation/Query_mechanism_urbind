from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from agents import Agent, function_tool

from backend.modules.writer.models import WriterOutput
from backend.modules.writer.utils.markdown_helpers import (
    append_sections,
    extract_city_coverage_sets,
    extract_markdown_bundle,
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


def build_writer_agent(config: AppConfig, api_key: str, analysis_mode: str) -> Agent:
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
            submit_writer_output,
        ],
        output_type=WriterOutput,
        tool_use_behavior="stop_on_first_tool",
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
    """Generate the final markdown answer with city-coverage guardrails.

    Runs the writer once, then retries at most once more if citation coverage
    is incomplete. Always returns the best available output.
    """
    markdown_bundle = extract_markdown_bundle(context_bundle)
    selected_city_names = extract_selected_city_names(context_bundle, markdown_bundle)
    analysis_mode = resolve_analysis_mode(context_bundle)
    agent = build_writer_agent(config, api_key, analysis_mode=analysis_mode)
    max_attempts = config.writer.max_coverage_attempts
    retry_settings = RetrySettings.bounded(
        max_attempts=max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )

    previous_answer = ""
    missing_city_keys: list[str] = []

    for attempt in range(1, max_attempts + 1):
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
            payload["reconsideration"] = reconsideration_payload

        output = _run_writer_once(
            agent=agent,
            payload=payload,
            max_turns=config.writer.max_turns,
            log_llm_payload=log_llm_payload,
        )

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
                render_no_evidence_section(no_evidence_names),
                render_cities_considered_section(cities_considered),
            ],
        )
        validate_writer_citations(content, context_bundle)

        if not missing_coverage_keys:
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

        previous_answer = content
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
            log_retry_event(
                operation="writer.output_reconsideration",
                run_id=run_id,
                attempt=attempt,
                max_attempts=max_attempts,
                error_type="MissingCityCitationCoverage",
                error_message=(
                    f"Writer city citation coverage is {coverage_ratio}; retrying missing cities: "
                    + ", ".join(missing_city_names)
                ),
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

        log_retry_exhausted(
            operation="writer.output_reconsideration",
            run_id=run_id,
            attempt=attempt,
            max_attempts=max_attempts,
            error_type="MissingCityCitationCoverage",
            error_message=(
                f"Writer city citation coverage remains {coverage_ratio}; missing cities: "
                + ", ".join(missing_city_names)
            ),
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
