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
    MarkdownExcerpt,
    MarkdownResearchResult,
    ThrownExcerpt,
)
from backend.modules.markdown_researcher.services import (
    build_city_batches,
    resolve_batch_input_token_limit,
    split_documents_by_city,
)
from backend.modules.markdown_researcher.utils import (
    coerce_markdown_result,
    format_batch_failure,
    partition_batch_excerpts,
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
    collected_thrown_excerpts: list[ThrownExcerpt] = []
    seen: set[tuple[str, str, str]] = set()
    first_error: ErrorInfo | None = None
    failed_batches: list[str] = []
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
    ) -> tuple[str, int, list[MarkdownExcerpt], list[ThrownExcerpt], ErrorInfo | None, bool]:
        """Process one city batch and return excerpts."""
        excerpts: list[MarkdownExcerpt] = []
        thrown_excerpts: list[ThrownExcerpt] = []
        error: ErrorInfo | None = None
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
        max_attempts = retry_settings.max_attempts
        run_result = None
        output: MarkdownResearchResult | None = None
        retryable_bad_output_reason: str | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                agent = _get_thread_agent()
                run_result = run_agent_sync(
                    agent,
                    json.dumps(payload, ensure_ascii=False),
                    max_turns=config.retry.max_attempts,
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
                return city_name, batch_index, excerpts, thrown_excerpts, error, success
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
                run_result.final_output
                if hasattr(run_result, "final_output")
                else run_result,
            )
            error = ErrorInfo(
                code="MARKDOWN_OUTPUT_INVALID",
                message="Markdown researcher did not return structured excerpts.",
            )
            return city_name, batch_index, excerpts, thrown_excerpts, error, success

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
            return city_name, batch_index, excerpts, thrown_excerpts, error, success

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
            return city_name, batch_index, excerpts, thrown_excerpts, error, success

        valid_chunk_ids = {
            str(document.get("chunk_id", "")).strip()
            for document in batch
            if str(document.get("chunk_id", "")).strip()
        }
        accepted_excerpts, rejected_excerpts = partition_batch_excerpts(
            output.excerpts,
            expected_city_name=city_name,
            batch_index=batch_index,
            valid_chunk_ids=valid_chunk_ids,
        )
        success = True
        for excerpt in accepted_excerpts:
            excerpts.append(excerpt)
        for excerpt in rejected_excerpts:
            thrown_excerpts.append(excerpt)
        return city_name, batch_index, excerpts, thrown_excerpts, error, success

    batch_max_chunks = max(config.markdown_researcher.batch_max_chunks, 1)
    batch_token_limit = resolve_batch_input_token_limit(config)
    for city_name, city_documents in sorted(documents_by_city.items()):
        logger.info("Preparing markdown for city %s (%d chunks)", city_name, len(city_documents))

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
            thrown_excerpts=[],
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
            futures[future] = (city_name, batch_idx)

        for future in as_completed(futures):
            try:
                city_name, batch_idx, excerpts, thrown_excerpts, error, success = future.result()
                for thrown_excerpt in thrown_excerpts:
                    collected_thrown_excerpts.append(thrown_excerpt)
                if success:
                    any_success = True
                    for excerpt in excerpts:
                        key = (
                            excerpt.city_name.strip().lower(),
                            excerpt.quote.strip(),
                            excerpt.partial_answer.strip(),
                        )
                        if key not in seen:
                            seen.add(key)
                            collected.append(excerpt)
                        else:
                            collected_thrown_excerpts.append(
                                ThrownExcerpt(
                                    quote=excerpt.quote,
                                    city_name=excerpt.city_name,
                                    partial_answer=excerpt.partial_answer,
                                    source_chunk_ids=excerpt.source_chunk_ids,
                                    rejection_stage="dedupe",
                                    reason_codes=["duplicate_excerpt"],
                                    batch_index=batch_idx,
                                    expected_city_name=city_name,
                                )
                            )
                elif error:
                    failed_batches.append(
                        format_batch_failure(city_name, batch_idx, error.code)
                    )
                    if error.code == "MARKDOWN_MAX_TURNS_EXCEEDED":
                        max_turns_exceeded = True
                    if not first_error:
                        first_error = error
            except Exception as exc:  # noqa: BLE001
                logger.exception("Markdown batch processing failed")
                city_name, batch_idx = futures[future]
                failed_batches.append(
                    format_batch_failure(city_name, batch_idx, type(exc).__name__)
                )
                if not first_error:
                    first_error = ErrorInfo(
                        code="MARKDOWN_BATCH_ERROR",
                        message=f"Markdown batch processing failed: {str(exc)}",
                    )

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
                error=ErrorInfo(
                    code="MARKDOWN_PARTIAL_BATCH_FAILURE",
                    message="Some markdown city batches failed; returning partial results.",
                    details=failed_batches,
                ),
                thrown_excerpts=collected_thrown_excerpts,
            )
        return MarkdownResearchResult(
            status="success",
            excerpts=collected,
            thrown_excerpts=collected_thrown_excerpts,
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
            error=ErrorInfo(
                code="MARKDOWN_ALL_BATCHES_FAILED",
                message=message,
                details=failed_batches,
            ),
            thrown_excerpts=collected_thrown_excerpts,
        )

    if first_error:
        return MarkdownResearchResult(
            status="error",
            error=first_error,
            thrown_excerpts=collected_thrown_excerpts,
        )

    return MarkdownResearchResult(
        status="success",
        excerpts=[],
        error=ErrorInfo(
            code="MARKDOWN_EMPTY_RESULT",
            message="Markdown researcher returned no excerpts.",
            details=["No excerpts were produced and no explicit failures were captured."],
        ),
        thrown_excerpts=collected_thrown_excerpts,
    )


__all__ = [
    "build_markdown_agent",
    "extract_markdown_excerpts",
]
