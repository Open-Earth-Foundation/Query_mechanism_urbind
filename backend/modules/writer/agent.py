from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from agents import Agent, function_tool

from backend.modules.writer.models import (
    WriterCitationCoverage,
    WriterMultiPassPlan,
    WriterOutput,
)
from backend.modules.writer.utils.markdown_helpers import (
    append_sections,
    extract_city_coverage_sets,
    extract_markdown_bundle,
    extract_markdown_excerpts,
    extract_ref_city_mapping,
    extract_selected_city_names,
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


def _resolve_writer_combine_prompt_path() -> Path:
    """Resolve the writer prompt used for draft-merging fallback."""
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    return prompts_dir / "writer_system_combine.md"


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
    (
        required_city_keys,
        missing_coverage_keys,
        no_evidence_keys,
        city_display_by_key,
    ) = extract_city_coverage_sets(
        content=content,
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
    prepared_content = append_sections(
        content,
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
        output = _run_writer_once(
            agent=agent,
            payload=payload,
            max_turns=config.writer.max_turns,
            log_llm_payload=log_llm_payload,
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
    combined_output = _run_writer_once(
        agent=combine_agent,
        payload=payload,
        max_turns=config.writer.max_turns,
        log_llm_payload=log_llm_payload,
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
    plan, batches = plan_writer_multi_pass(
        question=question,
        context_bundle=writer_context_bundle,
        analysis_mode=analysis_mode,
        selected_city_names=selected_city_names,
        threshold_tokens=config.writer.multi_pass_threshold_tokens,
        chunk_tokens=config.writer.multi_pass_chunk_tokens,
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
    "write_markdown",
]
