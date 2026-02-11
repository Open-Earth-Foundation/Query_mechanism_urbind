from __future__ import annotations

import ast
import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agents import Agent, function_tool
from agents.exceptions import MaxTurnsExceeded
from openai import APIConnectionError
from pydantic import ValidationError

from backend.models import ErrorInfo
from backend.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.markdown_researcher.services import (
    split_documents_by_city,
)
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt
from backend.utils.tokenization import get_max_input_tokens


logger = logging.getLogger(__name__)


_PY_STRING_LITERAL = r"(?:'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\")"
_EXCERPT_REPR_PATTERN = re.compile(
    rf"MarkdownExcerpt\(\s*snippet=(?P<snippet>{_PY_STRING_LITERAL})\s*,\s*"
    rf"city_name=(?P<city_name>{_PY_STRING_LITERAL})\s*,\s*"
    rf"answer=(?P<answer>{_PY_STRING_LITERAL})\s*,\s*"
    rf"relevant=(?P<relevant>{_PY_STRING_LITERAL})\s*\)",
    re.DOTALL,
)
_STATUS_REPR_PATTERN = re.compile(
    r"status\s*=\s*(?P<status>'success'|'error'|\"success\"|\"error\")"
)
_MISSING_INPUT_MARKERS = (
    "missing input",
    "input not provided",
    "no input",
    "no documents",
    "documents not provided",
    "question not provided",
    "city_name not provided",
    "need the input",
    "wait for input",
    "awaiting input",
    "without input",
)


def _parse_markdown_result_repr(text: str) -> MarkdownResearchResult | None:
    """Parse a MarkdownResearchResult repr-like string into a model."""
    status = "success"
    status_match = _STATUS_REPR_PATTERN.search(text)
    if status_match:
        try:
            status = ast.literal_eval(status_match.group("status"))
        except Exception as exc:
            logger.debug("Failed to parse status from repr: %s", exc)
            status = "success"
    if status not in ("success", "error"):
        status = "success"

    error = None

    excerpts: list[MarkdownExcerpt] = []
    for match in _EXCERPT_REPR_PATTERN.finditer(text):
        try:
            snippet = ast.literal_eval(match.group("snippet"))
            city_name = ast.literal_eval(match.group("city_name"))
            answer = ast.literal_eval(match.group("answer"))
            relevant = ast.literal_eval(match.group("relevant"))
            excerpts.append(
                MarkdownExcerpt(
                    snippet=str(snippet),
                    city_name=str(city_name),
                    answer=str(answer),
                    relevant=str(relevant),
                )
            )
        except Exception as exc:
            logger.debug("Failed to parse excerpt from repr: %s", exc)
            continue

    if not excerpts and not status_match and error is None:
        return None

    return MarkdownResearchResult(status=status, excerpts=excerpts, error=error)


def _coerce_markdown_result(output: object) -> MarkdownResearchResult | None:
    """Coerce raw tool output into a MarkdownResearchResult when possible."""
    if output.__class__.__name__ == "MarkdownResearchResult":
        return output
    if isinstance(output, MarkdownResearchResult):
        return output

    if isinstance(output, str):
        try:
            return MarkdownResearchResult.model_validate_json(output)
        except ValidationError:
            pass
        try:
            parsed = ast.literal_eval(output)
        except (ValueError, SyntaxError):
            parsed = None
        if parsed is not None:
            return _coerce_markdown_result(parsed)
        return _parse_markdown_result_repr(output)

    if isinstance(output, dict):
        if "arguments" in output:
            return _coerce_markdown_result(output["arguments"])
        if "result" in output:
            return _coerce_markdown_result(output["result"])
        try:
            return MarkdownResearchResult.model_validate(output)
        except ValidationError as exc:
            logger.debug("Dict coercion failed: %s", exc)
            return None

    model_dump = getattr(output, "model_dump", None)
    if callable(model_dump):
        return _coerce_markdown_result(model_dump())

    value_dict = getattr(output, "__dict__", None)
    if isinstance(value_dict, dict):
        filtered = {
            key: item
            for key, item in value_dict.items()
            if not str(key).startswith("_")
        }
        return _coerce_markdown_result(filtered)

    parsed_repr = _parse_markdown_result_repr(str(output))
    if parsed_repr is not None:
        return parsed_repr

    logger.debug(
        "No matching coercion strategy for output type: %s", type(output).__name__
    )
    return None


def _get_field(target: object, key: str) -> object | None:
    if isinstance(target, dict):
        return target.get(key)
    return getattr(target, key, None)


def _text_indicates_missing_input(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _MISSING_INPUT_MARKERS)


def _error_indicates_missing_input(error: ErrorInfo | None) -> bool:
    if error is None:
        return False
    parts: list[str] = [error.code, error.message]
    if isinstance(error.details, list):
        parts.extend(str(item) for item in error.details)
    elif error.details is not None:
        parts.append(str(error.details))
    return _text_indicates_missing_input(" ".join(parts))


def _extract_reasoning_texts(run_result: object) -> list[str]:
    raw_responses = _get_field(run_result, "raw_responses")
    if not isinstance(raw_responses, list):
        return []

    reasoning_texts: list[str] = []
    for response in raw_responses:
        output_items = _get_field(response, "output")
        if not isinstance(output_items, list):
            continue
        for item in output_items:
            if _get_field(item, "type") != "reasoning":
                continue
            content = _get_field(item, "content")
            if isinstance(content, list):
                for part in content:
                    text = _get_field(part, "text")
                    if text:
                        reasoning_texts.append(str(text))
            text = _get_field(item, "text")
            if text:
                reasoning_texts.append(str(text))
    return reasoning_texts


def _reasoned_about_content(
    run_result: object,
    question: str,
    city_name: str,
) -> bool:
    reasoning_texts = _extract_reasoning_texts(run_result)
    if not reasoning_texts:
        return False
    combined = " ".join(reasoning_texts)
    if _text_indicates_missing_input(combined):
        return False

    lowered = combined.lower()
    question_tokens = [
        token for token in re.split(r"\W+", question.lower()) if len(token) >= 4
    ]
    overlap_count = sum(1 for token in set(question_tokens) if token in lowered)
    has_city = city_name.lower() in lowered
    has_analysis_terms = any(
        marker in lowered
        for marker in (
            "extract",
            "excerpt",
            "snippet",
            "document",
            "climate initiative",
        )
    )
    return has_analysis_terms and (has_city or overlap_count >= 2)


def _doc_city_name(document: dict[str, str]) -> str:
    city_name = document.get("city_name")
    if city_name:
        return str(city_name)
    path_value = document.get("path", "")
    if path_value:
        return Path(str(path_value)).stem
    return ""


def _format_batch_failure(city_name: str, batch_index: int, reason: str) -> str:
    """Build a compact failure marker for a city batch."""
    return f"{city_name}#batch{batch_index}: {reason}"


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
    max_input_tokens = get_max_input_tokens(
        config.markdown_researcher.context_window_tokens,
        config.markdown_researcher.max_output_tokens,
        config.markdown_researcher.input_token_reserve,
        config.markdown_researcher.max_input_tokens,
    )

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
        """Process a single city batch and return excerpts."""
        excerpts: list[MarkdownExcerpt] = []
        error: ErrorInfo | None = None
        success = False

        payload = {
            "question": question,
            "city_name": city_name,
            "documents": batch,
            "context_window_tokens": config.markdown_researcher.context_window_tokens,
            "max_input_tokens": max_input_tokens,
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
                output = _coerce_markdown_result(final_output)
                retryable_bad_output_reason = None
                if output is None:
                    retryable_bad_output_reason = "output_none"
                elif output.status == "error":
                    if output.error is None:
                        retryable_bad_output_reason = "status_error_without_error"
                    elif _error_indicates_missing_input(output.error):
                        retryable_bad_output_reason = "missing_input_error"
                elif output.error is not None and _error_indicates_missing_input(
                    output.error
                ):
                    retryable_bad_output_reason = "missing_input_error"
                elif not output.excerpts and _reasoned_about_content(
                    run_result, question, city_name
                ):
                    retryable_bad_output_reason = "empty_excerpts_after_reasoning"

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

        if retryable_bad_output_reason == "missing_input_error":
            logger.warning(
                "Markdown %s batch %s reported missing input after retries: %s",
                city_name,
                batch_index,
                output.error,
            )
            error = output.error
            if error is None:
                error = ErrorInfo(
                    code="MARKDOWN_MISSING_INPUT",
                    message=(
                        "Markdown researcher reported missing input after retries."
                    ),
                )
            return city_name, batch_index, excerpts, error, success

        if retryable_bad_output_reason == "empty_excerpts_after_reasoning":
            logger.warning(
                "Markdown %s batch %s returned empty excerpts after reasoning over content.",
                city_name,
                batch_index,
            )
            error = ErrorInfo(
                code="MARKDOWN_EMPTY_EXCERPTS",
                message=(
                    "Markdown researcher produced reasoning about the provided content "
                    "but returned no excerpts."
                ),
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

    # Process all city batches with request-level backoff
    configured_workers = max(config.markdown_researcher.max_workers, 1)
    max_workers = min(configured_workers, max(len(all_tasks), 1))
    request_backoff_base = max(
        config.markdown_researcher.request_backoff_base_seconds, 0.1
    )
    request_backoff_max = max(
        config.markdown_researcher.request_backoff_max_seconds, request_backoff_base
    )

    logger.info(
        "Processing %d markdown batches with %d worker(s), request backoff: %.1f-%.1f seconds",
        len(all_tasks),
        max_workers,
        request_backoff_base,
        request_backoff_max,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task_index, (city_name, batch_idx, batch) in enumerate(all_tasks):
            # Add request-level backoff between submissions to avoid overwhelming the API
            if task_index > 0:
                delay = min(
                    request_backoff_max,
                    request_backoff_base * (2 ** (task_index // max_workers)),
                )
                jitter = random.uniform(0.0, delay * 0.1)
                actual_delay = delay + jitter
                logger.debug(
                    "Request backoff before batch %d/%d: %.2f seconds",
                    task_index + 1,
                    len(all_tasks),
                    actual_delay,
                )
                time.sleep(actual_delay)

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
                            excerpt.answer.strip(),
                        )
                        if key not in seen:
                            seen.add(key)
                            collected.append(excerpt)
                elif error:
                    failed_batches.append(
                        _format_batch_failure(city_name, batch_idx, error.code)
                    )
                    if error.code == "MARKDOWN_MAX_TURNS_EXCEEDED":
                        max_turns_exceeded = True
                    if not first_error:
                        first_error = error
            except Exception as exc:  # noqa: BLE001
                logger.exception("Markdown batch processing failed")
                city_name, batch_idx = futures[future]
                failed_batches.append(
                    _format_batch_failure(city_name, batch_idx, type(exc).__name__)
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
