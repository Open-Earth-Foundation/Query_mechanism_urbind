from __future__ import annotations

import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agents import Agent, function_tool
from agents.exceptions import MaxTurnsExceeded
from openai import APIConnectionError

from app.models import ErrorInfo
from app.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from app.modules.markdown_researcher.services import (
    split_documents_by_city,
)
from app.modules.markdown_researcher.utils import (
    coerce_markdown_result,
    format_batch_failure,
)
from app.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from app.utils.config import AppConfig
from app.utils.prompts import load_prompt


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
        client_max_retries=0,
    )
    settings = build_model_settings(
        config.markdown_researcher.temperature,
        config.markdown_researcher.max_output_tokens,
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
    documents: list[dict[str, str]],
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
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
        return False

    def _sleep_backoff(attempt: int, base: float, max_delay: float) -> None:
        delay = min(max_delay, base * (2**attempt))
        jitter = random.uniform(0.0, delay * 0.1)
        time.sleep(delay + jitter)

    def _process_city_batch(
        city_name: str,
        batch_index: int,
        batch: list[dict[str, str]],
    ) -> tuple[str, int, list[MarkdownExcerpt], ErrorInfo | None, bool]:
        """Process a single city batch (currently one chunk) and return excerpts."""
        excerpts: list[MarkdownExcerpt] = []
        error: ErrorInfo | None = None
        success = False

        # Current architecture sends one chunk per agent call to keep payloads small and predictable.
        chunk_content = batch[0].get("content", "") if batch else ""
        payload = {
            "question": question,
            "city_name": city_name,
            "content": chunk_content,
        }
        max_retries = max(config.markdown_researcher.max_retries, 0)
        retry_base = max(config.markdown_researcher.retry_base_seconds, 0.1)
        retry_max = max(config.markdown_researcher.retry_max_seconds, retry_base)
        run_result = None
        output: MarkdownResearchResult | None = None
        retryable_bad_output_reason: str | None = None

        for attempt in range(max_retries + 1):
            try:
                agent = _get_thread_agent()
                run_result = run_agent_sync(
                    agent,
                    json.dumps(payload, ensure_ascii=True),
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

                if retryable_bad_output_reason and attempt < max_retries:
                    logger.warning(
                        "Markdown %s batch %s returned retryable bad output (%s) (attempt %d/%d); retrying.",
                        city_name,
                        batch_index,
                        retryable_bad_output_reason,
                        attempt + 1,
                        max_retries + 1,
                    )
                    _sleep_backoff(attempt, retry_base, retry_max)
                    continue
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
                return city_name, batch_index, excerpts, error, success
            except Exception as exc:  # noqa: BLE001
                if attempt < max_retries and _is_retryable_error(exc):
                    logger.warning(
                        "Markdown %s batch %s failed (attempt %d/%d); retrying. error=%s: %s",
                        city_name,
                        batch_index,
                        attempt + 1,
                        max_retries + 1,
                        type(exc).__name__,
                        str(exc),
                        exc_info=True,
                    )
                    _sleep_backoff(attempt, retry_base, retry_max)
                    continue
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
            return city_name, batch_index, excerpts, error, success

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
            return city_name, batch_index, excerpts, error, success

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
            return city_name, batch_index, excerpts, error, success
        success = True
        for excerpt in output.excerpts:
            if excerpt.relevant != "yes":
                continue
            excerpts.append(excerpt)
        return city_name, batch_index, excerpts, error, success

    # Build list of all city batches to process
    all_tasks: list[tuple[str, int, list[dict[str, str]]]] = []
    for city_name, city_documents in sorted(documents_by_city.items()):
        logger.info(
            "Preparing markdown for city %s (%d chunks)",
            city_name,
            len(city_documents),
        )
        # Intentionally one chunk per batch to reduce per-call context size.
        batches = [[document] for document in city_documents]
        for batch_index, batch in enumerate(batches, start=1):
            all_tasks.append((city_name, batch_index, batch))

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
        "Processing %d markdown batches with %d worker(s)",
        len(all_tasks),
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for city_name, batch_idx, batch in all_tasks:
            future = executor.submit(_process_city_batch, city_name, batch_idx, batch)
            futures[future] = (city_name, batch_idx)

        for future in as_completed(futures):
            try:
                city_name, batch_idx, excerpts, error, success = future.result()
                if success:
                    any_success = True
                    for excerpt in excerpts:
                        key = (
                            excerpt.city_name.strip().lower(),
                            excerpt.snippet.strip(),
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
            )
        return MarkdownResearchResult(status="success", excerpts=collected)

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
        )

    if first_error:
        return MarkdownResearchResult(
            status="error",
            error=first_error,
        )

    return MarkdownResearchResult(
        status="success",
        excerpts=[],
        error=ErrorInfo(
            code="MARKDOWN_EMPTY_RESULT",
            message="Markdown researcher returned no excerpts.",
            details=["No excerpts were produced and no explicit failures were captured."],
        ),
    )


__all__ = [
    "build_markdown_agent",
    "extract_markdown_excerpts",
]
