from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agents import Agent, function_tool
from agents.exceptions import MaxTurnsExceeded
from openai import APIConnectionError
from pydantic import ValidationError

from app.models import ErrorInfo
from app.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from app.modules.markdown_researcher.services import (
    split_documents_by_city,
)
from app.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from app.utils.config import AppConfig
from app.utils.prompts import load_prompt
from app.utils.tokenization import get_max_input_tokens


logger = logging.getLogger(__name__)


def _coerce_markdown_result(output: object) -> MarkdownResearchResult | None:
    """Coerce raw tool output into a MarkdownResearchResult when possible."""
    if isinstance(output, MarkdownResearchResult):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        return _coerce_markdown_result(parsed)
    if isinstance(output, dict):
        try:
            return MarkdownResearchResult.model_validate(output)
        except ValidationError as exc:
            logger.warning("Markdown output validation failed: %s", exc.errors())
            return None
    return None
    return None


def _doc_city_name(document: dict[str, str]) -> str:
    city_name = document.get("city_name")
    if city_name:
        return str(city_name)
    path_value = document.get("path", "")
    if path_value:
        return Path(str(path_value)).stem
    return ""


def build_markdown_agent(config: AppConfig, api_key: str) -> Agent:
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
    )
    settings = build_model_settings(
        config.markdown_researcher.temperature,
        config.markdown_researcher.max_output_tokens,
    )

    @function_tool(strict_mode=False)
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
    run_id: str,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> MarkdownResearchResult:
    """Extract markdown excerpts relevant to the question."""
    agent = build_markdown_agent(config, api_key)
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
    any_success = False
    max_turns_exceeded = False

    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, APIConnectionError):
            return True
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
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
            "run_id": run_id,
            "question": question,
            "city_name": city_name,
            "documents": batch,
            "context_window_tokens": config.markdown_researcher.context_window_tokens,
            "max_input_tokens": max_input_tokens,
        }
        max_retries = max(config.markdown_researcher.max_retries, 0)
        retry_base = max(config.markdown_researcher.retry_base_seconds, 0.1)
        retry_max = max(config.markdown_researcher.retry_max_seconds, retry_base)

        for attempt in range(max_retries + 1):
            try:
                result = run_agent_sync(
                    agent,
                    json.dumps(payload, ensure_ascii=True),
                    log_llm_payload=log_llm_payload,
                )
                break
            except MaxTurnsExceeded:
                logger.warning(
                    "Markdown extraction for %s batch %s hit max turns limit.",
                    city_name,
                    batch_index,
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

        output = _coerce_markdown_result(result.final_output)
        if not output:
            logger.warning(
                "Markdown %s batch %s returned invalid output",
                city_name,
                batch_index,
            )
            error = ErrorInfo(
                code="MARKDOWN_OUTPUT_INVALID",
                message="Markdown researcher did not return structured excerpts.",
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

    # Process all city batches with request-level backoff
    configured_workers = max(config.markdown_researcher.max_workers, 1)
    max_workers = min(configured_workers, max(len(all_tasks), 1))
    request_backoff_base = max(config.markdown_researcher.request_backoff_base_seconds, 0.1)
    request_backoff_max = max(config.markdown_researcher.request_backoff_max_seconds, request_backoff_base)
    
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
                delay = min(request_backoff_max, request_backoff_base * (2 ** (task_index // max_workers)))
                jitter = random.uniform(0.0, delay * 0.1)
                actual_delay = delay + jitter
                logger.debug("Request backoff before batch %d/%d: %.2f seconds", task_index + 1, len(all_tasks), actual_delay)
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
                elif error and not first_error:
                    first_error = error
            except Exception as exc:  # noqa: BLE001
                logger.exception("Markdown batch processing failed")
                if not first_error:
                    first_error = ErrorInfo(
                        code="MARKDOWN_BATCH_ERROR",
                        message=f"Markdown batch processing failed: {str(exc)}",
                    )

    # If we have any successful results, return them (even if partial due to max turns)
    if any_success or collected:
        logger.info(
            "Markdown extraction completed. Collected %d excerpts%s",
            len(collected),
            " (partial due to max turns)" if max_turns_exceeded else "",
        )
        return MarkdownResearchResult(
            run_id=run_id,
            excerpts=collected,
        )

    if first_error:
        return MarkdownResearchResult(
            status="error",
            run_id=run_id,
            error=first_error,
        )

    raise ValueError("Markdown researcher did not return structured excerpts.")


__all__ = [
    "build_markdown_agent",
    "extract_markdown_excerpts",
]
