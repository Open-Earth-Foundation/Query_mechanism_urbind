from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import time
from pathlib import Path
from typing import Any

from agents import Agent, function_tool
from agents.exceptions import MaxTurnsExceeded

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
from backend.tools.calculator import (
    sum_numbers,
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

NO_TOOL_FALLBACK_PROMPT_APPENDIX = """
<fallback_mode>
These fallback instructions override any conflicting tool-call requirements above.
Tools are unavailable in this fallback pass.
- Do not call any tools.
- Return the final answer as structured `WriterOutput` output directly.
- Do not mention turns, max turns, retries, tool status, finish reasons, or any runtime/debug details.
</fallback_mode>
"""


def _resolve_writer_prompt_path(analysis_mode: str) -> Path:
    """Resolve writer prompt path for the selected analysis mode."""
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    if analysis_mode == "city_by_city":
        return prompts_dir / "writer_system_city_by_city.md"
    return prompts_dir / "writer_system_aggregate.md"


def _build_no_tool_fallback_instructions(base_instructions: str) -> str:
    """Return no-tool fallback instructions while preserving the base writer prompt."""
    return (
        f"{base_instructions.strip()}\n\n"
        f"{NO_TOOL_FALLBACK_PROMPT_APPENDIX.strip()}\n"
    )


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

    @function_tool(name_override="sum_numbers", strict_mode=True)
    def sum_numbers_tool(numbers: list[float]) -> float:
        """Return arithmetic sum of provided numbers."""
        return sum_numbers(numbers, source="writer")

    @function_tool
    def submit_writer_output(output: WriterOutput) -> WriterOutput:
        """Return structured writer output unchanged."""
        return output

    return Agent(
        name="Writer",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[sum_numbers_tool, submit_writer_output],
        output_type=WriterOutput,
        tool_use_behavior="run_llm_again",
    )


def build_writer_no_tool_agent(
    config: AppConfig,
    api_key: str,
    analysis_mode: str,
) -> Agent:
    """Build a write-only fallback writer agent with no tools."""
    prompt_path = _resolve_writer_prompt_path(analysis_mode)
    instructions = _build_no_tool_fallback_instructions(load_prompt(prompt_path))
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
    return Agent(
        name="Writer (No-Tool Fallback)",
        instructions=instructions,
        model=model,
        model_settings=settings,
        output_type=WriterOutput,
    )


def _get_field(target: object, key: str) -> Any:
    """Return key-like field from dicts/objects."""
    if isinstance(target, Mapping):
        return target.get(key)
    return getattr(target, key, None)


def _extract_text_from_output_item(item: object) -> list[str]:
    """Extract text fragments from one output item payload."""
    texts: list[str] = []
    item_type = _get_field(item, "type")
    if item_type == "message":
        content = _get_field(item, "content")
        if isinstance(content, list):
            for part in content:
                part_type = _get_field(part, "type")
                if part_type in {"output_text", "text"}:
                    text = _get_field(part, "text")
                    if text:
                        texts.append(str(text))
                elif part_type is None:
                    text = _get_field(part, "text")
                    if text:
                        texts.append(str(text))
        elif content:
            texts.append(str(content))
    elif item_type in {"output_text", "text"}:
        text = _get_field(item, "text")
        if text:
            texts.append(str(text))
    else:
        text = _get_field(item, "text")
        if text:
            texts.append(str(text))
    return texts


def _extract_writer_draft_from_max_turns(exc: MaxTurnsExceeded) -> str:
    """Extract any best-effort textual draft from max-turn diagnostics."""
    run_data = getattr(exc, "run_data", None)
    if run_data is None:
        return ""

    raw_responses = getattr(run_data, "raw_responses", None)
    if not isinstance(raw_responses, list):
        return ""

    fragments: list[str] = []
    for response in raw_responses:
        output_items = _get_field(response, "output")
        if not isinstance(output_items, list):
            continue
        for output_item in output_items:
            fragments.extend(_extract_text_from_output_item(output_item))

    cleaned_fragments = [part.strip() for part in fragments if part and part.strip()]
    return "\n".join(cleaned_fragments).strip()


def _inject_fallback_draft(
    payload: dict[str, object],
    fallback_draft: str,
) -> dict[str, object]:
    """Attach fallback draft to reconsideration payload for write-only pass."""
    if not fallback_draft:
        return payload

    next_payload = dict(payload)
    reconsideration_payload: dict[str, object] = {}
    existing_reconsideration = next_payload.get("reconsideration")
    if isinstance(existing_reconsideration, dict):
        reconsideration_payload.update(existing_reconsideration)
    reconsideration_payload.setdefault("previous_answer", fallback_draft)
    reconsideration_payload["fallback_mode"] = "no_tool_writer"
    next_payload["reconsideration"] = reconsideration_payload
    return next_payload


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
    primary_agent = build_writer_agent(config, api_key, analysis_mode=analysis_mode)
    fallback_agent: Agent | None = None
    use_fallback_writer = False
    fallback_draft = ""
    retry_settings = RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )
    max_attempts = retry_settings.max_attempts

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
                reconsideration_payload["missing_city_excerpts"] = extract_missing_city_excerpts(
                    markdown_bundle,
                    missing_city_keys,
                )
            payload["reconsideration"] = reconsideration_payload

        active_agent = fallback_agent if use_fallback_writer and fallback_agent else primary_agent
        active_payload = (
            _inject_fallback_draft(payload, fallback_draft)
            if use_fallback_writer and fallback_draft
            else payload
        )
        try:
            output = _run_writer_once(
                agent=active_agent,
                payload=active_payload,
                max_turns=config.writer.max_turns,
                log_llm_payload=log_llm_payload,
            )
        except MaxTurnsExceeded as exc:
            if use_fallback_writer:
                raise
            fallback_draft = _extract_writer_draft_from_max_turns(exc)
            use_fallback_writer = True
            fallback_agent = build_writer_no_tool_agent(
                config,
                api_key,
                analysis_mode=analysis_mode,
            )
            fallback_payload = _inject_fallback_draft(payload, fallback_draft)
            logger.warning(
                "WRITER_MAX_TURNS_FALLBACK %s",
                json.dumps(
                    {
                        "run_id": run_id or "unknown",
                        "attempt": attempt,
                        "analysis_mode": analysis_mode,
                        "fallback_has_draft": bool(fallback_draft),
                    },
                    ensure_ascii=False,
                ),
            )
            output = _run_writer_once(
                agent=fallback_agent,
                payload=fallback_payload,
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


__all__ = [
    "build_writer_agent",
    "build_writer_no_tool_agent",
    "write_markdown",
]
