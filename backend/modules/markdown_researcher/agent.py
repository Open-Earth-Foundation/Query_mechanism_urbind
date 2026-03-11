from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agents import Agent, function_tool
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from openai import APIConnectionError

from backend.models import ErrorInfo
from backend.modules.markdown_researcher.models import (
    MarkdownBatchFailure,
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.markdown_researcher.services import (
    build_city_batches,
    resolve_batch_input_token_limit,
    split_documents_by_city,
)
from backend.modules.markdown_researcher.utils import (
    DecisionValidationResult,
    coerce_markdown_result,
    format_batch_failure,
    validate_batch_decisions,
)
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt
from backend.utils.retry import (
    RetrySettings,
    compute_retry_delay_seconds,
    log_retry_event,
    log_retry_exhausted,
)

logger = logging.getLogger(__name__)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return unique non-empty string values while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _validate_or_backfill_batch_decisions(
    *,
    output: MarkdownResearchResult,
    input_chunk_ids: list[str],
    strict_decision_audit: bool,
) -> DecisionValidationResult:
    """Validate batch decisions and backfill missing partitions in non-strict mode."""
    accepted_chunk_ids = _dedupe_preserve_order(output.accepted_chunk_ids)
    rejected_chunk_ids = _dedupe_preserve_order(output.rejected_chunk_ids)

    if not strict_decision_audit:
        input_id_set = set(input_chunk_ids)
        excerpt_source_ids = _dedupe_preserve_order(
            [
                source_chunk_id.strip()
                for excerpt in output.excerpts
                for source_chunk_id in excerpt.source_chunk_ids
                if source_chunk_id.strip() in input_id_set
            ]
        )
        accepted_chunk_ids = _dedupe_preserve_order(
            [*accepted_chunk_ids, *excerpt_source_ids]
        )
        accepted_id_set = set(accepted_chunk_ids)
        rejected_chunk_ids = [
            chunk_id
            for chunk_id in rejected_chunk_ids
            if chunk_id not in accepted_id_set
        ]
        decided_ids = accepted_id_set | set(rejected_chunk_ids)
        rejected_chunk_ids.extend(
            chunk_id for chunk_id in input_chunk_ids if chunk_id not in decided_ids
        )

    return validate_batch_decisions(
        input_chunk_ids=input_chunk_ids,
        accepted_chunk_ids=accepted_chunk_ids,
        rejected_chunk_ids=rejected_chunk_ids,
        excerpts=output.excerpts,
    )


def build_markdown_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the markdown researcher agent with structured output and forced tool calls."""
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "markdown_researcher_system.md"
    )
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(
        config.markdown_researcher.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.markdown_researcher.temperature,
        config.markdown_researcher.max_output_tokens,
        # Grok-specific optional override; unsupported models/providers may reject this.
        reasoning_effort=config.markdown_researcher.reasoning_effort,
    )
    settings.tool_choice = "submit_markdown_excerpts"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=True)
    def submit_markdown_excerpts(
        result: MarkdownResearchResult,
    ) -> MarkdownResearchResult:
        return result

    return Agent(
        name="Markdown Researcher",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_markdown_excerpts],
        output_type=MarkdownResearchResult,
        tool_use_behavior="stop_on_first_tool",
    )


def extract_markdown_excerpts(
    question: str,
    documents: list[dict[str, object]],
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
    run_id: str | None = None,
) -> MarkdownResearchResult:
    """Extract markdown excerpts relevant to the question with graceful partial-failure handling."""
    _thread_local = threading.local()

    def _get_thread_agent() -> Agent:
        local_agent = getattr(_thread_local, "agent", None)
        if local_agent is None:
            local_agent = build_markdown_agent(config, api_key)
            _thread_local.agent = local_agent
        return local_agent

    # Skip city scope selection - process all documents directly
    logger.info("Processing all documents without city scope filtering")

    # Group documents by city
    documents_by_city = split_documents_by_city(documents)

    collected: list[MarkdownExcerpt] = []
    seen: set[tuple[str, str, str]] = set()
    first_error: ErrorInfo | None = None
    failed_batches: list[str] = []
    collected_accepted_ids: list[str] = []
    collected_rejected_ids: list[str] = []
    collected_unresolved_ids: list[str] = []
    batch_failures_payload: list[MarkdownBatchFailure] = []
    any_success = False
    max_turns_exceeded = False

    def _is_retryable_error(exc: Exception) -> bool:
        """Check if an error should trigger a retry attempt."""
        if isinstance(exc, APIConnectionError):
            return True
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return True
        # Treat JSON validation errors as retryable (LLM output may succeed on retry)
        if isinstance(exc, ValueError) and "Invalid JSON" in str(exc):
            return True
        # Malformed model output (e.g. XML-style function calls instead of JSON) is
        # transient and worth retrying.
        if isinstance(exc, ModelBehaviorError):
            return True
        return False

    def _process_city_batch(
        city_name: str,
        batch_index: int,
        batch: list[dict[str, object]],
    ) -> tuple[
        str,
        int,
        list[MarkdownExcerpt],
        list[str],
        list[str],
        list[str],
        MarkdownBatchFailure | None,
        ErrorInfo | None,
        bool,
    ]:
        """Process one city batch and return excerpts."""
        excerpts: list[MarkdownExcerpt] = []
        accepted_chunk_ids: list[str] = []
        rejected_chunk_ids: list[str] = []
        error: ErrorInfo | None = None
        failure_payload: MarkdownBatchFailure | None = None
        success = False

        chunks_payload = [
            {
                "chunk_id": str(document.get("chunk_id", "")),
                "path": str(document.get("path", "")),
                "heading_path": str(document.get("heading_path", "")),
                "block_type": str(document.get("block_type", "")),
                "distance": str(document.get("distance", "")),
                "content": str(document.get("content", "")),
            }
            for document in batch
        ]
        input_chunk_ids = _dedupe_preserve_order(
            [str(item.get("chunk_id", "")).strip() for item in chunks_payload]
        )
        payload = {
            "question": question,
            "city_name": city_name,
            "chunks": chunks_payload,
        }
        retry_settings = RetrySettings.bounded(
            max_attempts=config.retry.max_attempts,
            backoff_base_seconds=config.retry.backoff_base_seconds,
            backoff_max_seconds=config.retry.backoff_max_seconds,
        )
        strict_decision_audit = config.markdown_researcher.strict_decision_audit
        max_attempts = retry_settings.max_attempts
        markdown_max_turns = max(config.markdown_researcher.max_turns, 1)
        run_result = None
        output: MarkdownResearchResult | None = None
        decision_validation: DecisionValidationResult | None = None
        retryable_bad_output_reason: str | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                agent = _get_thread_agent()
                run_result = run_agent_sync(
                    agent,
                    json.dumps(payload, ensure_ascii=False),
                    max_turns=markdown_max_turns,
                    log_llm_payload=log_llm_payload,
                )

                # Get the final output - handle all format variations
                final_output = run_result.final_output
                output = coerce_markdown_result(final_output)
                retryable_bad_output_reason = None
                if output is None:
                    retryable_bad_output_reason = "output_none"
                elif output.status == "error":
                    if output.error is None:
                        retryable_bad_output_reason = "status_error_without_error"
                else:
                    decision_validation = _validate_or_backfill_batch_decisions(
                        input_chunk_ids=input_chunk_ids,
                        output=output,
                        strict_decision_audit=strict_decision_audit,
                    )
                    if not decision_validation.is_valid:
                        retryable_bad_output_reason = (
                            "decision_invariant_failed:"
                            + ",".join(decision_validation.violation_codes)
                        )
                        logger.warning(
                            (
                                "run_id=%s city=%s batch=%s malformed decision payload "
                                "violations=%s overlap=%s unknown_accepted=%s "
                                "unknown_rejected=%s missing=%s unknown_excerpt_sources=%s"
                            ),
                            run_id,
                            city_name,
                            batch_index,
                            decision_validation.violation_codes,
                            decision_validation.overlap_ids,
                            decision_validation.unknown_accepted_ids,
                            decision_validation.unknown_rejected_ids,
                            decision_validation.missing_ids,
                            decision_validation.unknown_excerpt_source_ids,
                        )

                if retryable_bad_output_reason and attempt < max_attempts:
                    delay_seconds = compute_retry_delay_seconds(attempt, retry_settings)
                    log_retry_event(
                        operation="markdown.batch_extraction",
                        run_id=run_id,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error_type="RetryableBadOutput",
                        error_message=retryable_bad_output_reason,
                        next_backoff_seconds=delay_seconds,
                        context={"city_name": city_name, "batch_index": batch_index},
                    )
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue
                if retryable_bad_output_reason and attempt >= max_attempts:
                    log_retry_exhausted(
                        operation="markdown.batch_extraction",
                        run_id=run_id,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error_type="RetryableBadOutput",
                        error_message=retryable_bad_output_reason,
                        context={"city_name": city_name, "batch_index": batch_index},
                    )
                break
            except MaxTurnsExceeded:
                logger.warning(
                    "Markdown extraction for %s batch %s hit max turns limit.",
                    city_name,
                    batch_index,
                )
                error = ErrorInfo(
                    code="MARKDOWN_MAX_TURNS_EXCEEDED",
                    message="Markdown extraction exceeded max turns for this city batch.",
                )
                failure_payload = MarkdownBatchFailure(
                    city_name=city_name,
                    batch_index=batch_index,
                    reason="MARKDOWN_MAX_TURNS_EXCEEDED",
                    unresolved_chunk_ids=input_chunk_ids,
                )
                return (
                    city_name,
                    batch_index,
                    excerpts,
                    accepted_chunk_ids,
                    rejected_chunk_ids,
                    input_chunk_ids,
                    failure_payload,
                    error,
                    success,
                )
            except Exception as exc:  # noqa: BLE001
                if attempt < max_attempts and _is_retryable_error(exc):
                    delay_seconds = compute_retry_delay_seconds(attempt, retry_settings)
                    log_retry_event(
                        operation="markdown.batch_extraction",
                        run_id=run_id,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        next_backoff_seconds=delay_seconds,
                        context={"city_name": city_name, "batch_index": batch_index},
                    )
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)
                    continue
                if _is_retryable_error(exc):
                    log_retry_exhausted(
                        operation="markdown.batch_extraction",
                        run_id=run_id,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        context={"city_name": city_name, "batch_index": batch_index},
                    )
                logger.exception(
                    "Markdown %s batch %s failed with non-retryable error.",
                    city_name,
                    batch_index,
                )
                raise

        if output is None:
            logger.warning(
                "Markdown %s batch %s returned invalid output (coercion failed). result.final_output=%s",
                city_name,
                batch_index,
                (
                    run_result.final_output
                    if hasattr(run_result, "final_output")
                    else run_result
                ),
            )
            error = ErrorInfo(
                code="MARKDOWN_OUTPUT_INVALID",
                message="Markdown researcher did not return structured excerpts.",
            )
            failure_payload = MarkdownBatchFailure(
                city_name=city_name,
                batch_index=batch_index,
                reason="invalid_output_after_retries",
                unresolved_chunk_ids=input_chunk_ids,
            )
            return (
                city_name,
                batch_index,
                excerpts,
                accepted_chunk_ids,
                rejected_chunk_ids,
                input_chunk_ids,
                failure_payload,
                error,
                success,
            )

        if retryable_bad_output_reason == "status_error_without_error":
            logger.warning(
                "Markdown %s batch %s returned status=error with empty error payload.",
                city_name,
                batch_index,
            )
            error = ErrorInfo(
                code="MARKDOWN_OUTPUT_ERROR_EMPTY",
                message="Markdown researcher returned status=error without an error payload.",
            )
            failure_payload = MarkdownBatchFailure(
                city_name=city_name,
                batch_index=batch_index,
                reason="status_error_without_error",
                unresolved_chunk_ids=input_chunk_ids,
            )
            return (
                city_name,
                batch_index,
                excerpts,
                accepted_chunk_ids,
                rejected_chunk_ids,
                input_chunk_ids,
                failure_payload,
                error,
                success,
            )

        if output.status == "error":
            logger.warning(
                "Markdown %s batch %s returned error: %s",
                city_name,
                batch_index,
                output.error,
            )
            error = output.error
            if error is None:
                error = ErrorInfo(
                    code="MARKDOWN_OUTPUT_ERROR_EMPTY",
                    message=(
                        "Markdown researcher returned status=error without details "
                        "after retries."
                    ),
                )
            failure_payload = MarkdownBatchFailure(
                city_name=city_name,
                batch_index=batch_index,
                reason="agent_error_status",
                unresolved_chunk_ids=input_chunk_ids,
            )
            return (
                city_name,
                batch_index,
                excerpts,
                accepted_chunk_ids,
                rejected_chunk_ids,
                input_chunk_ids,
                failure_payload,
                error,
                success,
            )
        if decision_validation is None or not decision_validation.is_valid:
            error = ErrorInfo(
                code="MARKDOWN_DECISION_INVALID",
                message="Markdown researcher returned invalid accepted/rejected decisions.",
            )
            failure_payload = MarkdownBatchFailure(
                city_name=city_name,
                batch_index=batch_index,
                reason="invariant_validation_failed_after_retries",
                unresolved_chunk_ids=input_chunk_ids,
            )
            return (
                city_name,
                batch_index,
                excerpts,
                accepted_chunk_ids,
                rejected_chunk_ids,
                input_chunk_ids,
                failure_payload,
                error,
                success,
            )
        success = True
        accepted_chunk_ids.extend(decision_validation.accepted_ids)
        rejected_chunk_ids.extend(decision_validation.rejected_ids)
        for excerpt in output.excerpts:
            excerpts.append(excerpt)
        return (
            city_name,
            batch_index,
            excerpts,
            accepted_chunk_ids,
            rejected_chunk_ids,
            [],
            None,
            error,
            success,
        )

    batch_max_chunks = max(config.markdown_researcher.batch_max_chunks, 1)
    batch_token_limit = resolve_batch_input_token_limit(config)
    for city_name, city_documents in sorted(documents_by_city.items()):
        logger.info(
            "Preparing markdown for city %s (%d chunks)", city_name, len(city_documents)
        )

    # Build list of city-scoped batches under token/chunk limits.
    all_tasks = build_city_batches(
        documents_by_city=documents_by_city,
        max_batch_input_tokens=batch_token_limit,
        max_batch_chunks=batch_max_chunks,
    )

    if not all_tasks:
        logger.warning("No markdown batches available for extraction.")
        return MarkdownResearchResult(
            status="success",
            excerpts=[],
            error=ErrorInfo(
                code="MARKDOWN_NO_BATCHES",
                message="No markdown batches were available for extraction.",
                details=["No markdown documents were provided."],
            ),
        )

    # Process all city batches with worker-level rate limiting
    configured_workers = max(config.markdown_researcher.max_workers, 1)
    max_workers = min(configured_workers, max(len(all_tasks), 1))

    logger.info(
        "Processing %d markdown batches with %d worker(s) [max_chunks=%d, token_limit=%d]",
        len(all_tasks),
        max_workers,
        batch_max_chunks,
        batch_token_limit,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for city_name, batch_idx, batch in all_tasks:
            future = executor.submit(_process_city_batch, city_name, batch_idx, batch)
            futures[future] = (city_name, batch_idx, batch)

        for future in as_completed(futures):
            try:
                (
                    city_name,
                    batch_idx,
                    excerpts,
                    accepted_chunk_ids,
                    rejected_chunk_ids,
                    unresolved_chunk_ids,
                    failure_payload,
                    error,
                    success,
                ) = future.result()
                if success:
                    any_success = True
                    collected_accepted_ids.extend(accepted_chunk_ids)
                    collected_rejected_ids.extend(rejected_chunk_ids)
                    for excerpt in excerpts:
                        key = (
                            excerpt.city_name.strip().lower(),
                            excerpt.quote.strip(),
                            excerpt.partial_answer.strip(),
                        )
                        if key not in seen:
                            seen.add(key)
                            collected.append(excerpt)
                elif error:
                    failed_batches.append(
                        format_batch_failure(city_name, batch_idx, error.code)
                    )
                    if error.code == "MARKDOWN_MAX_TURNS_EXCEEDED":
                        max_turns_exceeded = True
                    if unresolved_chunk_ids:
                        collected_unresolved_ids.extend(unresolved_chunk_ids)
                    if failure_payload:
                        batch_failures_payload.append(failure_payload)
                    if not first_error:
                        first_error = error
            except Exception as exc:  # noqa: BLE001
                logger.exception("Markdown batch processing failed")
                city_name, batch_idx, batch = futures[future]
                unresolved_chunk_ids = _dedupe_preserve_order(
                    [
                        str(document.get("chunk_id", "")).strip()
                        for document in batch
                    ]
                )
                failed_batches.append(
                    format_batch_failure(city_name, batch_idx, type(exc).__name__)
                )
                if unresolved_chunk_ids:
                    collected_unresolved_ids.extend(unresolved_chunk_ids)
                batch_failures_payload.append(
                    MarkdownBatchFailure(
                        city_name=city_name,
                        batch_index=batch_idx,
                        reason=type(exc).__name__,
                        unresolved_chunk_ids=unresolved_chunk_ids,
                    )
                )
                if not first_error:
                    first_error = ErrorInfo(
                        code="MARKDOWN_BATCH_ERROR",
                        message=f"Markdown batch processing failed: {str(exc)}",
                    )

    deduped_accepted_ids = _dedupe_preserve_order(collected_accepted_ids)
    deduped_rejected_ids = _dedupe_preserve_order(collected_rejected_ids)
    deduped_unresolved_ids = _dedupe_preserve_order(collected_unresolved_ids)
    batch_failures_payload = [
        item
        for item in batch_failures_payload
        if isinstance(item, MarkdownBatchFailure)
    ]

    if any_success:
        logger.info(
            "Markdown extraction completed. Collected %d excerpts%s",
            len(collected),
            " (partial due to max turns)" if max_turns_exceeded else "",
        )
        if failed_batches:
            return MarkdownResearchResult(
                status="success",
                excerpts=collected,
                accepted_chunk_ids=deduped_accepted_ids,
                rejected_chunk_ids=deduped_rejected_ids,
                unresolved_chunk_ids=deduped_unresolved_ids,
                batch_failures=batch_failures_payload,
                error=ErrorInfo(
                    code="MARKDOWN_PARTIAL_BATCH_FAILURE",
                    message="Some markdown city batches failed; returning partial results.",
                    details=failed_batches,
                ),
            )
        return MarkdownResearchResult(
            status="success",
            excerpts=collected,
            accepted_chunk_ids=deduped_accepted_ids,
            rejected_chunk_ids=deduped_rejected_ids,
            unresolved_chunk_ids=deduped_unresolved_ids,
            batch_failures=batch_failures_payload,
        )

    if failed_batches:
        message = "No excerpts extracted; all markdown batches failed."
        if max_turns_exceeded:
            message = (
                "No excerpts extracted; markdown batches exceeded max turns or failed."
            )
        return MarkdownResearchResult(
            status="success",
            excerpts=[],
            accepted_chunk_ids=deduped_accepted_ids,
            rejected_chunk_ids=deduped_rejected_ids,
            unresolved_chunk_ids=deduped_unresolved_ids,
            batch_failures=batch_failures_payload,
            error=ErrorInfo(
                code="MARKDOWN_ALL_BATCHES_FAILED",
                message=message,
                details=failed_batches,
            ),
        )

    if first_error:
        return MarkdownResearchResult(
            status="error",
            accepted_chunk_ids=deduped_accepted_ids,
            rejected_chunk_ids=deduped_rejected_ids,
            unresolved_chunk_ids=deduped_unresolved_ids,
            batch_failures=batch_failures_payload,
            error=first_error,
        )

    return MarkdownResearchResult(
        status="success",
        excerpts=[],
        accepted_chunk_ids=deduped_accepted_ids,
        rejected_chunk_ids=deduped_rejected_ids,
        unresolved_chunk_ids=deduped_unresolved_ids,
        batch_failures=batch_failures_payload,
        error=ErrorInfo(
            code="MARKDOWN_EMPTY_RESULT",
            message="Markdown researcher returned no excerpts.",
            details=[
                "No excerpts were produced and no explicit failures were captured."
            ],
        ),
    )


__all__ = [
    "build_markdown_agent",
    "extract_markdown_excerpts",
]
