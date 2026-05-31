from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents import Agent, function_tool

from backend.modules.writer.models import (
    WriterCitationCoverage,
    WriterMultiPassPlan,
    WriterOutput,
    WriterSectionPlan,
    WriterSectionSpec,
)
from backend.modules.writer.utils.markdown_helpers import (
    append_sections,
    extract_city_coverage_sets,
    extract_markdown_bundle,
    extract_markdown_excerpts,
    extract_ref_city_mapping,
    extract_selected_city_names,
    normalize_reference_citations,
    render_cities_considered_section,
    render_no_evidence_section,
    resolve_analysis_mode,
    strip_existing_footer_sections,
    validate_writer_citations,
)
from backend.modules.writer.utils.multi_pass import (
    WriterBatch,
    build_writer_batch_drafts_payload,
    build_writer_context_bundle,
    build_writer_payload,
    plan_writer_multi_pass,
)
from backend.modules.writer.utils.section_first import (
    WriterSectionPlannerPayload,
    build_section_composer_payload,
    build_section_context_bundle,
    build_section_planner_payload,
    build_section_writer_payload,
    count_section_payload_tokens,
    sanitize_writer_section_plan,
)
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.services.run_logger import RunLogger
from backend.utils.city_normalization import format_city_display_name
from backend.utils.config import AppConfig
from backend.utils.json_io import write_json
from backend.utils.paths import RunPaths
from backend.utils.prompts import load_prompt
from openai import APIConnectionError, APIStatusError, APITimeoutError
from backend.utils.tokenization import get_max_input_tokens

from backend.utils.retry import (
    RetrySettings,
    call_with_retries,
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


def _resolve_writer_combine_prompt_path() -> Path:
    """Resolve the writer prompt used for draft-merging fallback."""
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    return prompts_dir / "writer_system_combine.md"


def _resolve_writer_section_planner_prompt_path() -> Path:
    """Resolve the aggregate section-planner prompt path."""
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    return prompts_dir / "writer_section_planner_system.md"


def _resolve_writer_section_prompt_path() -> Path:
    """Resolve the section writer prompt path."""
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    return prompts_dir / "writer_section_system.md"


def _resolve_writer_section_composer_prompt_path() -> Path:
    """Resolve the section composer prompt path."""
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    return prompts_dir / "writer_section_composer_system.md"


def _build_structured_writer_agent(
    *,
    config: AppConfig,
    api_key: str,
    prompt_path: Path,
    agent_name: str,
) -> Agent:
    """Build a writer-family agent that returns ``WriterOutput``."""
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
        name=agent_name,
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_writer_output],
        output_type=WriterOutput,
        tool_use_behavior="stop_on_first_tool",
    )


def build_writer_section_planner_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the section planner agent for aggregate writer runs."""
    instructions = load_prompt(_resolve_writer_section_planner_prompt_path())
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
        name="WriterSectionPlanner",
        instructions=instructions,
        model=model,
        model_settings=settings,
        output_type=WriterSectionPlan,
    )


def build_writer_agent(config: AppConfig, api_key: str, analysis_mode: str) -> Agent:
    """Build the primary writer agent."""
    return _build_structured_writer_agent(
        config=config,
        api_key=api_key,
        prompt_path=_resolve_writer_prompt_path(analysis_mode),
        agent_name="Writer",
    )


def build_writer_combine_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the draft-combine writer agent."""
    return _build_structured_writer_agent(
        config=config,
        api_key=api_key,
        prompt_path=_resolve_writer_combine_prompt_path(),
        agent_name="WriterCombine",
    )


def build_writer_section_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the section-scoped writer agent."""
    return _build_structured_writer_agent(
        config=config,
        api_key=api_key,
        prompt_path=_resolve_writer_section_prompt_path(),
        agent_name="WriterSection",
    )


def build_writer_section_composer_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the final section-draft composer agent."""
    return _build_structured_writer_agent(
        config=config,
        api_key=api_key,
        prompt_path=_resolve_writer_section_composer_prompt_path(),
        agent_name="WriterSectionComposer",
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


def _run_section_planner_once(
    *,
    agent: Agent,
    payload: dict[str, object],
    max_turns: int,
    log_llm_payload: bool,
) -> WriterSectionPlan:
    """Run the section planner once and return its structured plan."""
    result = run_agent_sync(
        agent,
        json.dumps(payload, ensure_ascii=False),
        max_turns=max_turns,
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, WriterSectionPlan):
        return output
    raise ValueError("Writer section planner did not return structured output.")


def _is_retryable_writer_error(exc: Exception) -> bool:
    """Return True for transient writer/provider errors worth retrying."""
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 429, 500, 502, 503, 504}
    return False


def _build_writer_api_retry_settings(config: AppConfig) -> RetrySettings:
    """Return shared retry settings for writer provider calls."""
    return RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )


def _build_citation_coverage(
    *,
    status: str,
    attempt: int,
    max_attempts: int,
    confirmed_city_count: int,
    required_city_count: int,
    coverage_ratio: str,
    missing_city_names: list[str],
    analysis_mode: str,
) -> WriterCitationCoverage:
    """Build structured citation-coverage diagnostics for the final writer draft."""
    normalized_status = "confirmed" if status == "confirmed" else "partial"
    return WriterCitationCoverage(
        status=normalized_status,
        attempt=attempt,
        max_attempts=max_attempts,
        coverage_confirmed=confirmed_city_count,
        coverage_required=required_city_count,
        coverage_ratio=coverage_ratio,
        missing_cities=missing_city_names,
        analysis_mode=analysis_mode,
    )


def _prepare_writer_content(
    *,
    content: str,
    context_bundle: dict[str, object],
    selected_city_names: list[str],
) -> tuple[str, list[str], list[str], dict[str, str], int, int, str]:
    """Append canonical footer sections and compute citation coverage for one draft."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    normalized_content = normalize_reference_citations(content)
    (
        required_city_keys,
        missing_coverage_keys,
        no_evidence_keys,
        city_display_by_key,
    ) = extract_city_coverage_sets(
        content=normalized_content,
        markdown_bundle=markdown_bundle,
        selected_city_names=selected_city_names,
        context_bundle=context_bundle,
    )
    confirmed_city_count = len(required_city_keys) - len(missing_coverage_keys)
    required_city_count = len(required_city_keys)
    coverage_ratio = f"{confirmed_city_count}/{required_city_count}"
    no_evidence_names = [
        city_display_by_key.get(city_key, format_city_display_name(city_key))
        for city_key in no_evidence_keys
    ]
    cities_considered = selected_city_names or sorted(city_display_by_key.values())
    prepared_content = append_sections(
        normalized_content,
        [
            render_no_evidence_section(no_evidence_names),
            render_cities_considered_section(cities_considered),
        ],
    )
    validate_writer_citations(prepared_content, context_bundle)
    return (
        prepared_content,
        missing_coverage_keys,
        no_evidence_names,
        city_display_by_key,
        confirmed_city_count,
        required_city_count,
        coverage_ratio,
    )


def _log_writer_citation_coverage(
    *,
    run_id: str | None,
    attempt: int,
    max_attempts: int,
    status: str,
    confirmed_city_count: int,
    required_city_count: int,
    coverage_ratio: str,
    analysis_mode: str,
    missing_city_names: list[str] | None = None,
) -> None:
    """Emit one structured writer citation-coverage log line."""
    payload: dict[str, object] = {
        "run_id": run_id or "unknown",
        "attempt": attempt,
        "max_attempts": max_attempts,
        "status": status,
        "coverage_confirmed": confirmed_city_count,
        "coverage_required": required_city_count,
        "coverage_ratio": coverage_ratio,
        "analysis_mode": analysis_mode,
    }
    if missing_city_names:
        payload["missing_cities"] = missing_city_names
    rendered = json.dumps(payload, ensure_ascii=False)
    if status == "confirmed":
        logger.info("WRITER_CITATION_COVERAGE %s", rendered)
        return
    logger.warning("WRITER_CITATION_COVERAGE %s", rendered)


def _write_markdown_single_bundle(
    *,
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    analysis_mode: str,
    selected_city_names: list[str],
    log_llm_payload: bool,
    run_id: str | None,
) -> WriterOutput:
    """Write one markdown bundle with citation-coverage reconsideration."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    agent = build_writer_agent(config, api_key, analysis_mode=analysis_mode)
    max_attempts = config.writer.max_coverage_attempts
    retry_settings = RetrySettings.bounded(
        max_attempts=max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )

    api_retry_settings = _build_writer_api_retry_settings(config)

    previous_answer = ""
    missing_city_keys: list[str] = []

    for attempt in range(1, max_attempts + 1):
        reconsideration_payload: dict[str, object] | None = None
        if attempt > 1 and previous_answer:
            ref_city_map = extract_ref_city_mapping(markdown_bundle)[1]
            reconsideration_payload = {"previous_answer": previous_answer}
            if missing_city_keys:
                reconsideration_payload["missing_cities"] = [
                    ref_city_map.get(city_key, format_city_display_name(city_key))
                    for city_key in missing_city_keys
                ]

        payload = build_writer_payload(
            question=question,
            context_bundle=context_bundle,
            analysis_mode=analysis_mode,
            selected_city_names=selected_city_names,
            reconsideration=reconsideration_payload,
        )
        output = call_with_retries(
            lambda: _run_writer_once(
                agent=agent,
                payload=payload,
                max_turns=config.writer.max_turns,
                log_llm_payload=log_llm_payload,
            ),
            operation="writer.llm_call",
            retry_settings=api_retry_settings,
            should_retry=_is_retryable_writer_error,
            run_id=run_id,
        )

        (
            content,
            missing_coverage_keys,
            _no_evidence_names,
            city_display_by_key,
            confirmed_city_count,
            required_city_count,
            coverage_ratio,
        ) = _prepare_writer_content(
            content=output.content,
            context_bundle=context_bundle,
            selected_city_names=selected_city_names,
        )
        if not missing_coverage_keys:
            _log_writer_citation_coverage(
                run_id=run_id,
                attempt=attempt,
                max_attempts=max_attempts,
                status="confirmed",
                confirmed_city_count=confirmed_city_count,
                required_city_count=required_city_count,
                coverage_ratio=coverage_ratio,
                analysis_mode=analysis_mode,
            )
            return WriterOutput(
                content=content,
                citation_coverage=_build_citation_coverage(
                    status="confirmed",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    confirmed_city_count=confirmed_city_count,
                    required_city_count=required_city_count,
                    coverage_ratio=coverage_ratio,
                    missing_city_names=[],
                    analysis_mode=analysis_mode,
                ),
            )

        previous_answer = content
        missing_city_keys = missing_coverage_keys
        missing_city_names = [
            city_display_by_key.get(city_key, format_city_display_name(city_key))
            for city_key in missing_city_keys
        ]
        coverage_status = "retrying" if attempt < max_attempts else "exhausted"
        _log_writer_citation_coverage(
            run_id=run_id,
            attempt=attempt,
            max_attempts=max_attempts,
            status=coverage_status,
            confirmed_city_count=confirmed_city_count,
            required_city_count=required_city_count,
            coverage_ratio=coverage_ratio,
            analysis_mode=analysis_mode,
            missing_city_names=missing_city_names,
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
        return WriterOutput(
            content=content,
            citation_coverage=_build_citation_coverage(
                status="partial",
                attempt=attempt,
                max_attempts=max_attempts,
                confirmed_city_count=confirmed_city_count,
                required_city_count=required_city_count,
                coverage_ratio=coverage_ratio,
                missing_city_names=missing_city_names,
                analysis_mode=analysis_mode,
            ),
        )

    raise RuntimeError("Writer retry loop ended unexpectedly.")


def _combine_writer_drafts(
    *,
    question: str,
    analysis_mode: str,
    selected_city_names: list[str],
    batch_outputs: list[WriterOutput],
    batches: list[WriterBatch],
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool,
    run_id: str | None,
) -> str:
    """Combine multiple batch drafts into one cited final answer."""
    combine_agent = build_writer_combine_agent(config, api_key)
    draft_answers: list[dict[str, object]] = []
    for batch, output in zip(batches, batch_outputs, strict=True):
        draft_answers.append(
            {
                "batch_index": batch.batch_index,
                "cities": batch.city_names,
                "content": strip_existing_footer_sections(output.content),
            }
        )

    payload: dict[str, object] = {
        "question": question,
        "analysis_mode": analysis_mode,
        "selected_cities": selected_city_names,
        "draft_answers": draft_answers,
    }
    combined_output = call_with_retries(
        lambda: _run_writer_once(
            agent=combine_agent,
            payload=payload,
            max_turns=config.writer.max_turns,
            log_llm_payload=log_llm_payload,
        ),
        operation="writer.combine_llm_call",
        retry_settings=_build_writer_api_retry_settings(config),
        should_retry=_is_retryable_writer_error,
        run_id=run_id,
    )
    return combined_output.content


def _persist_writer_multi_pass(
    *,
    plan: WriterMultiPassPlan,
    batches: list[WriterBatch],
    batch_outputs: list[WriterOutput],
    run_logger: RunLogger | None,
    paths: RunPaths | None,
) -> None:
    """Persist writer multi-pass diagnostics for developer tooling."""
    logger.info("WRITER_MULTI_PASS %s", json.dumps(plan.model_dump(), ensure_ascii=False))
    if run_logger is None:
        return

    run_logger.record_writer_multi_pass(plan.model_dump())
    run_logger.record_decision(
        {
            "status": "success",
            "reason": "Writer used multi-pass batching because the prompt exceeded the configured token threshold.",
            "writer_multi_pass": plan.model_dump(),
        }
    )
    if paths is None:
        return

    artifact_path = paths.base_dir / "writer" / "multi_pass.json"
    artifact_payload = {
        "plan": plan.model_dump(),
        "drafts": build_writer_batch_drafts_payload(
            batches=batches,
            drafts=[output.content for output in batch_outputs],
        ),
    }
    write_json(artifact_path, artifact_payload, ensure_ascii=False)
    run_logger.record_artifact("writer_multi_pass", artifact_path)


def _has_writer_visible_evidence(context_bundle: dict[str, object]) -> bool:
    """Return True when an aggregate run has evidence for section-first writing."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    if extract_markdown_excerpts(markdown_bundle):
        return True
    enrichment = context_bundle.get("enrichment")
    if not isinstance(enrichment, dict):
        return False
    for key in ("external_evidence", "web_findings", "assumptions", "enriched_fields"):
        value = enrichment.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _build_writer_section_plan(
    *,
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    analysis_mode: str,
    selected_city_names: list[str],
    log_llm_payload: bool,
    run_id: str | None,
) -> tuple[WriterSectionPlan, WriterSectionPlannerPayload]:
    """Plan question-specific aggregate sections from a compact evidence catalog."""
    planner_payload = build_section_planner_payload(
        question=question,
        context_bundle=context_bundle,
        analysis_mode=analysis_mode,
        selected_city_names=selected_city_names,
        max_input_tokens=config.writer.section_planner_max_input_tokens,
    )
    planner_agent = build_writer_section_planner_agent(config, api_key)
    raw_plan = call_with_retries(
        lambda: _run_section_planner_once(
            agent=planner_agent,
            payload=planner_payload.payload,
            max_turns=config.writer.max_turns,
            log_llm_payload=log_llm_payload,
        ),
        operation="writer.section_planner_llm_call",
        retry_settings=_build_writer_api_retry_settings(config),
        should_retry=_is_retryable_writer_error,
        run_id=run_id,
    )
    plan = sanitize_writer_section_plan(
        plan=raw_plan,
        question=question,
        context_bundle=context_bundle,
        selected_city_names=selected_city_names,
    )
    return plan, planner_payload


def _resolve_section_token_limit(
    config: AppConfig,
    writer_max_input_tokens: int | None,
) -> int | None:
    """Return the token cap that triggers section-internal batching."""
    limits: list[int] = []
    if config.writer.multi_pass_chunk_tokens > 0:
        limits.append(config.writer.multi_pass_chunk_tokens)
    if writer_max_input_tokens is not None:
        limits.append(writer_max_input_tokens)
    return min(limits) if limits else None


def _write_section_once(
    *,
    agent: Agent,
    question: str,
    analysis_mode: str,
    section: WriterSectionSpec,
    section_context: dict[str, object],
    log_llm_payload: bool,
    config: AppConfig,
    run_id: str | None,
) -> WriterOutput:
    """Run one section writer call."""
    payload = build_section_writer_payload(
        question=question,
        analysis_mode=analysis_mode,
        selected_city_names=section.city_names,
        section=section,
        context_bundle=section_context,
    )
    return call_with_retries(
        lambda: _run_writer_once(
            agent=agent,
            payload=payload,
            max_turns=config.writer.max_turns,
            log_llm_payload=log_llm_payload,
        ),
        operation="writer.section_llm_call",
        retry_settings=_build_writer_api_retry_settings(config),
        should_retry=_is_retryable_writer_error,
        run_id=run_id,
    )


def _write_section_draft(
    *,
    agent: Agent,
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    analysis_mode: str,
    section: WriterSectionSpec,
    log_llm_payload: bool,
    run_id: str | None,
    writer_max_input_tokens: int | None,
) -> tuple[str, dict[str, object]]:
    """Write one planned section, batching within the section only if needed."""
    section_context = build_section_context_bundle(
        context_bundle=context_bundle,
        section=section,
    )
    section_payload = build_section_writer_payload(
        question=question,
        analysis_mode=analysis_mode,
        selected_city_names=section.city_names,
        section=section,
        context_bundle=section_context,
    )
    payload_tokens = count_section_payload_tokens(section_payload)
    section_token_limit = _resolve_section_token_limit(config, writer_max_input_tokens)
    if section_token_limit is not None and payload_tokens > section_token_limit:
        return _write_oversized_section_draft(
            agent=agent,
            question=question,
            section_context=section_context,
            config=config,
            analysis_mode=analysis_mode,
            section=section,
            log_llm_payload=log_llm_payload,
            run_id=run_id,
            writer_max_input_tokens=writer_max_input_tokens,
            payload_tokens=payload_tokens,
            section_token_limit=section_token_limit,
        )

    output = _write_section_once(
        agent=agent,
        question=question,
        analysis_mode=analysis_mode,
        section=section,
        section_context=section_context,
        log_llm_payload=log_llm_payload,
        config=config,
        run_id=run_id,
    )
    return output.content, _build_section_diagnostic(
        section=section,
        payload_tokens=payload_tokens,
        draft_content=output.content,
        batch_count=1,
    )


def _write_oversized_section_draft(
    *,
    agent: Agent,
    question: str,
    section_context: dict[str, object],
    config: AppConfig,
    analysis_mode: str,
    section: WriterSectionSpec,
    log_llm_payload: bool,
    run_id: str | None,
    writer_max_input_tokens: int | None,
    payload_tokens: int,
    section_token_limit: int,
) -> tuple[str, dict[str, object]]:
    """Split an oversized section by the existing city/excerpt batching rules."""
    plan, batches = plan_writer_multi_pass(
        question=question,
        context_bundle=section_context,
        analysis_mode=analysis_mode,
        selected_city_names=section.city_names,
        threshold_tokens=section_token_limit,
        chunk_tokens=config.writer.multi_pass_chunk_tokens,
        max_input_tokens=writer_max_input_tokens,
    )
    if plan is None:
        if writer_max_input_tokens is not None and payload_tokens > writer_max_input_tokens:
            raise ValueError(
                "Writer section payload exceeds the configured LLM input token limit "
                f"and cannot be split: section_id={section.section_id}, "
                f"payload_tokens={payload_tokens}, limit={writer_max_input_tokens}."
            )
        output = _write_section_once(
            agent=agent,
            question=question,
            analysis_mode=analysis_mode,
            section=section,
            section_context=section_context,
            log_llm_payload=log_llm_payload,
            config=config,
            run_id=run_id,
        )
        return output.content, _build_section_diagnostic(
            section=section,
            payload_tokens=payload_tokens,
            draft_content=output.content,
            batch_count=1,
        )

    batch_contents: list[str] = []
    for batch in batches:
        batch_section = _build_batch_section(section, batch.context_bundle)
        batch_output = _write_section_once(
            agent=agent,
            question=question,
            analysis_mode=analysis_mode,
            section=batch_section,
            section_context=batch.context_bundle,
            log_llm_payload=log_llm_payload,
            config=config,
            run_id=run_id,
        )
        batch_contents.append(batch_output.content)

    content = _merge_section_batch_contents(section, batch_contents)
    return content, _build_section_diagnostic(
        section=section,
        payload_tokens=payload_tokens,
        draft_content=content,
        batch_count=plan.batch_count,
    )


def _build_batch_section(
    section: WriterSectionSpec,
    context_bundle: dict[str, object],
) -> WriterSectionSpec:
    """Return a section spec narrowed to one internal batch."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    excerpts = extract_markdown_excerpts(markdown_bundle)
    return section.model_copy(
        update={
            "required_ref_ids": [
                str(excerpt.get("ref_id", "")).strip() for excerpt in excerpts
            ],
            "city_names": [
                str(city)
                for city in markdown_bundle.get("selected_city_names", section.city_names)
                if isinstance(city, str)
            ],
        }
    )


def _merge_section_batch_contents(
    section: WriterSectionSpec,
    batch_contents: list[str],
) -> str:
    """Merge split section drafts without asking the model to add facts."""
    bodies = [
        _strip_matching_section_heading(strip_existing_footer_sections(content), section.title)
        for content in batch_contents
        if content.strip()
    ]
    if not bodies:
        return f"## {section.title}"
    return f"## {section.title}\n\n" + "\n\n".join(bodies)


def _strip_matching_section_heading(content: str, title: str) -> str:
    """Remove the planned section heading from a batch draft when present."""
    lines = content.strip().splitlines()
    if not lines:
        return ""
    first_line = lines[0].strip()
    if first_line.lower() == f"## {title}".lower():
        return "\n".join(lines[1:]).strip()
    return content.strip()


def _build_section_diagnostic(
    *,
    section: WriterSectionSpec,
    payload_tokens: int,
    draft_content: str,
    batch_count: int,
) -> dict[str, object]:
    """Build persisted diagnostics for one section draft."""
    return {
        "section_id": section.section_id,
        "title": section.title,
        "section_type": section.section_type,
        "purpose": section.purpose,
        "required_ref_ids": section.required_ref_ids,
        "city_names": section.city_names,
        "writing_instructions": section.writing_instructions,
        "payload_tokens": payload_tokens,
        "draft_length_chars": len(draft_content),
        "batch_count": batch_count,
    }


def _write_section_drafts(
    *,
    plan: WriterSectionPlan,
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    analysis_mode: str,
    log_llm_payload: bool,
    run_id: str | None,
    writer_max_input_tokens: int | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Write all section drafts and return composer payloads plus diagnostics."""
    section_agent = build_writer_section_agent(config, api_key)
    max_workers = max(config.writer.section_max_workers, 1)

    def _write(section: WriterSectionSpec) -> tuple[dict[str, object], dict[str, object]]:
        content, diagnostic = _write_section_draft(
            agent=section_agent,
            question=question,
            context_bundle=context_bundle,
            config=config,
            analysis_mode=analysis_mode,
            section=section,
            log_llm_payload=log_llm_payload,
            run_id=run_id,
            writer_max_input_tokens=writer_max_input_tokens,
        )
        return (
            {
                "section_id": section.section_id,
                "title": section.title,
                "section_type": section.section_type,
                "required_ref_ids": section.required_ref_ids,
                "city_names": section.city_names,
                "content": strip_existing_footer_sections(content),
            },
            diagnostic,
        )

    if max_workers == 1 or len(plan.sections) <= 1:
        results = [_write(section) for section in plan.sections]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_write, plan.sections))

    section_drafts = [draft for draft, _diagnostic in results]
    diagnostics = [diagnostic for _draft, diagnostic in results]
    return section_drafts, diagnostics


def _compose_section_first_output(
    *,
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    analysis_mode: str,
    selected_city_names: list[str],
    plan: WriterSectionPlan,
    section_drafts: list[dict[str, object]],
    log_llm_payload: bool,
    run_id: str | None,
) -> WriterOutput:
    """Compose final output from section drafts with citation-coverage retries."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    composer_agent = build_writer_section_composer_agent(config, api_key)
    max_attempts = config.writer.max_coverage_attempts
    retry_settings = RetrySettings.bounded(
        max_attempts=max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )

    previous_answer = ""
    missing_city_keys: list[str] = []
    for attempt in range(1, max_attempts + 1):
        payload = build_section_composer_payload(
            question=question,
            analysis_mode=analysis_mode,
            selected_city_names=selected_city_names,
            plan=plan,
            section_drafts=section_drafts,
        )
        if attempt > 1 and previous_answer:
            ref_city_map = extract_ref_city_mapping(markdown_bundle)[1]
            reconsideration_payload: dict[str, object] = {
                "previous_answer": previous_answer
            }
            if missing_city_keys:
                reconsideration_payload["missing_cities"] = [
                    ref_city_map.get(city_key, format_city_display_name(city_key))
                    for city_key in missing_city_keys
                ]
            payload["reconsideration"] = reconsideration_payload

        combined_output = call_with_retries(
            lambda: _run_writer_once(
                agent=composer_agent,
                payload=payload,
                max_turns=config.writer.max_turns,
                log_llm_payload=log_llm_payload,
            ),
            operation="writer.section_composer_llm_call",
            retry_settings=_build_writer_api_retry_settings(config),
            should_retry=_is_retryable_writer_error,
            run_id=run_id,
        )
        (
            content,
            missing_coverage_keys,
            _no_evidence_names,
            city_display_by_key,
            confirmed_city_count,
            required_city_count,
            coverage_ratio,
        ) = _prepare_writer_content(
            content=combined_output.content,
            context_bundle=context_bundle,
            selected_city_names=selected_city_names,
        )
        if not missing_coverage_keys:
            _log_writer_citation_coverage(
                run_id=run_id,
                attempt=attempt,
                max_attempts=max_attempts,
                status="confirmed",
                confirmed_city_count=confirmed_city_count,
                required_city_count=required_city_count,
                coverage_ratio=coverage_ratio,
                analysis_mode=analysis_mode,
            )
            return WriterOutput(
                content=content,
                citation_coverage=_build_citation_coverage(
                    status="confirmed",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    confirmed_city_count=confirmed_city_count,
                    required_city_count=required_city_count,
                    coverage_ratio=coverage_ratio,
                    missing_city_names=[],
                    analysis_mode=analysis_mode,
                ),
            )

        previous_answer = content
        missing_city_keys = missing_coverage_keys
        missing_city_names = [
            city_display_by_key.get(city_key, format_city_display_name(city_key))
            for city_key in missing_city_keys
        ]
        coverage_status = "retrying" if attempt < max_attempts else "exhausted"
        _log_writer_citation_coverage(
            run_id=run_id,
            attempt=attempt,
            max_attempts=max_attempts,
            status=coverage_status,
            confirmed_city_count=confirmed_city_count,
            required_city_count=required_city_count,
            coverage_ratio=coverage_ratio,
            analysis_mode=analysis_mode,
            missing_city_names=missing_city_names,
        )
        if attempt < max_attempts:
            delay_seconds = compute_retry_delay_seconds(attempt, retry_settings)
            log_retry_event(
                operation="writer.section_composer_reconsideration",
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
            operation="writer.section_composer_reconsideration",
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
        return WriterOutput(
            content=content,
            citation_coverage=_build_citation_coverage(
                status="partial",
                attempt=attempt,
                max_attempts=max_attempts,
                confirmed_city_count=confirmed_city_count,
                required_city_count=required_city_count,
                coverage_ratio=coverage_ratio,
                missing_city_names=missing_city_names,
                analysis_mode=analysis_mode,
            ),
        )

    raise RuntimeError("Writer section-first retry loop ended unexpectedly.")


def _persist_writer_section_first(
    *,
    plan: WriterSectionPlan,
    planner_payload: WriterSectionPlannerPayload,
    section_diagnostics: list[dict[str, object]],
    run_logger: RunLogger | None,
    paths: RunPaths | None,
) -> None:
    """Persist section-first writer diagnostics for developer tooling."""
    diagnostics = {
        "strategy": "section_first",
        "analysis_mode": plan.analysis_mode,
        "planner_input_tokens": planner_payload.input_tokens,
        "catalog_truncated": planner_payload.catalog_truncated,
        "section_count": len(plan.sections),
        "sections": section_diagnostics,
    }
    logger.info("WRITER_SECTION_PLAN %s", json.dumps(diagnostics, ensure_ascii=False))
    if run_logger is not None:
        run_logger.record_writer_section_plan(diagnostics)
        run_logger.record_decision(
            {
                "status": "success",
                "reason": "Writer used section-first aggregate planning.",
                "writer_section_plan": diagnostics,
            }
        )
    if run_logger is None or paths is None:
        return

    artifact_path = paths.base_dir / "writer" / "section_first.json"
    write_json(
        artifact_path,
        {
            "plan": plan.model_dump(),
            "diagnostics": diagnostics,
        },
        ensure_ascii=False,
    )
    run_logger.record_artifact("writer_section_plan", artifact_path)


def _write_markdown_section_first(
    *,
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    analysis_mode: str,
    selected_city_names: list[str],
    log_llm_payload: bool,
    run_id: str | None,
    run_logger: RunLogger | None,
    paths: RunPaths | None,
    writer_max_input_tokens: int | None,
) -> WriterOutput:
    """Run aggregate writing through planner, section writers, and composer."""
    plan, planner_payload = _build_writer_section_plan(
        question=question,
        context_bundle=context_bundle,
        config=config,
        api_key=api_key,
        analysis_mode=analysis_mode,
        selected_city_names=selected_city_names,
        log_llm_payload=log_llm_payload,
        run_id=run_id,
    )
    section_drafts, section_diagnostics = _write_section_drafts(
        plan=plan,
        question=question,
        context_bundle=context_bundle,
        config=config,
        api_key=api_key,
        analysis_mode=analysis_mode,
        log_llm_payload=log_llm_payload,
        run_id=run_id,
        writer_max_input_tokens=writer_max_input_tokens,
    )
    _persist_writer_section_first(
        plan=plan,
        planner_payload=planner_payload,
        section_diagnostics=section_diagnostics,
        run_logger=run_logger,
        paths=paths,
    )
    return _compose_section_first_output(
        question=question,
        context_bundle=context_bundle,
        config=config,
        api_key=api_key,
        analysis_mode=analysis_mode,
        selected_city_names=selected_city_names,
        plan=plan,
        section_drafts=section_drafts,
        log_llm_payload=log_llm_payload,
        run_id=run_id,
    )


def write_markdown(
    question: str,
    context_bundle: dict[str, object],
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
    run_id: str | None = None,
    run_logger: RunLogger | None = None,
    paths: RunPaths | None = None,
) -> WriterOutput:
    """Generate the final markdown answer with coverage and multi-pass guardrails."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    selected_city_names = extract_selected_city_names(context_bundle, markdown_bundle)
    analysis_mode = resolve_analysis_mode(context_bundle)
    writer_context_bundle = build_writer_context_bundle(
        context_bundle=context_bundle,
        excerpts=extract_markdown_excerpts(markdown_bundle),
        city_names=selected_city_names,
    )
    writer_max_input_tokens = get_max_input_tokens(
        config.writer.context_window_tokens,
        config.writer.max_output_tokens,
        config.writer.input_token_reserve,
        config.writer.max_input_tokens,
    )
    if (
        analysis_mode == "aggregate"
        and config.writer.section_first_aggregate_enabled
        and _has_writer_visible_evidence(writer_context_bundle)
    ):
        return _write_markdown_section_first(
            question=question,
            context_bundle=writer_context_bundle,
            config=config,
            api_key=api_key,
            analysis_mode=analysis_mode,
            selected_city_names=selected_city_names,
            log_llm_payload=log_llm_payload,
            run_id=run_id,
            run_logger=run_logger,
            paths=paths,
            writer_max_input_tokens=writer_max_input_tokens,
        )

    plan, batches = plan_writer_multi_pass(
        question=question,
        context_bundle=writer_context_bundle,
        analysis_mode=analysis_mode,
        selected_city_names=selected_city_names,
        threshold_tokens=config.writer.multi_pass_threshold_tokens,
        chunk_tokens=config.writer.multi_pass_chunk_tokens,
        max_input_tokens=writer_max_input_tokens,
    )
    if plan is None:
        return _write_markdown_single_bundle(
            question=question,
            context_bundle=writer_context_bundle,
            config=config,
            api_key=api_key,
            analysis_mode=analysis_mode,
            selected_city_names=selected_city_names,
            log_llm_payload=log_llm_payload,
            run_id=run_id,
        )

    batch_outputs: list[WriterOutput] = []
    for batch in batches:
        batch_output = _write_markdown_single_bundle(
            question=question,
            context_bundle=batch.context_bundle,
            config=config,
            api_key=api_key,
            analysis_mode=analysis_mode,
            selected_city_names=batch.city_names,
            log_llm_payload=log_llm_payload,
            run_id=run_id,
        )
        batch_outputs.append(batch_output)

    _persist_writer_multi_pass(
        plan=plan,
        batches=batches,
        batch_outputs=batch_outputs,
        run_logger=run_logger,
        paths=paths,
    )
    combined_content = _combine_writer_drafts(
        question=question,
        analysis_mode=analysis_mode,
        selected_city_names=selected_city_names,
        batch_outputs=batch_outputs,
        batches=batches,
        config=config,
        api_key=api_key,
        log_llm_payload=log_llm_payload,
        run_id=run_id,
    )
    (
        content,
        missing_coverage_keys,
        _no_evidence_names,
        city_display_by_key,
        confirmed_city_count,
        required_city_count,
        coverage_ratio,
    ) = _prepare_writer_content(
        content=combined_content,
        context_bundle=writer_context_bundle,
        selected_city_names=selected_city_names,
    )
    missing_city_names = [
        city_display_by_key.get(city_key, format_city_display_name(city_key))
        for city_key in missing_coverage_keys
    ]
    coverage_status = "confirmed" if not missing_coverage_keys else "partial"
    _log_writer_citation_coverage(
        run_id=run_id,
        attempt=1,
        max_attempts=1,
        status="confirmed" if coverage_status == "confirmed" else "exhausted",
        confirmed_city_count=confirmed_city_count,
        required_city_count=required_city_count,
        coverage_ratio=coverage_ratio,
        analysis_mode=analysis_mode,
        missing_city_names=missing_city_names,
    )
    return WriterOutput(
        content=content,
        citation_coverage=_build_citation_coverage(
            status=coverage_status,
            attempt=1,
            max_attempts=1,
            confirmed_city_count=confirmed_city_count,
            required_city_count=required_city_count,
            coverage_ratio=coverage_ratio,
            missing_city_names=missing_city_names,
            analysis_mode=analysis_mode,
        ),
    )


__all__ = [
    "build_writer_agent",
    "build_writer_combine_agent",
    "build_writer_section_agent",
    "build_writer_section_composer_agent",
    "build_writer_section_planner_agent",
    "write_markdown",
]
