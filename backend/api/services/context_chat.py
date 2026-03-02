"""LLM-backed context chat over one or more stored run contexts."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from backend.tools.calculator import (
    divide_numbers,
    multiply_numbers,
    subtract_numbers,
    sum_numbers,
)
from backend.utils.retry import RetrySettings, call_with_retries
from backend.utils.config import AppConfig, get_openrouter_api_key
from backend.utils.tokenization import count_tokens, get_encoding

logger = logging.getLogger(__name__)

CHAT_SUM_TOOL_NAME = "sum_numbers"
CHAT_SUBTRACT_TOOL_NAME = "subtract_numbers"
CHAT_MULTIPLY_TOOL_NAME = "multiply_numbers"
CHAT_DIVIDE_TOOL_NAME = "divide_numbers"
CHAT_REF_ID_PATTERN = re.compile(r"^ref_[1-9]\d*$")


def resolve_chat_token_cap(config: AppConfig) -> int:
    """Return the token cap for chat prompts from config."""
    return config.chat.max_context_total_tokens


CHAT_TOOL_DEFINITIONS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": CHAT_SUM_TOOL_NAME,
            "description": "Return arithmetic sum of a list of numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of integer/float values to sum.",
                    }
                },
                "required": ["numbers"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CHAT_SUBTRACT_TOOL_NAME,
            "description": "Subtract one number from another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minuend": {
                        "type": "number",
                        "description": "Starting numeric value.",
                    },
                    "subtrahend": {
                        "type": "number",
                        "description": "Numeric value to subtract from the minuend.",
                    },
                },
                "required": ["minuend", "subtrahend"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CHAT_MULTIPLY_TOOL_NAME,
            "description": "Return arithmetic product of a list of numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of integer/float values to multiply.",
                    }
                },
                "required": ["numbers"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CHAT_DIVIDE_TOOL_NAME,
            "description": "Divide one number by another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dividend": {
                        "type": "number",
                        "description": "Numeric value being divided.",
                    },
                    "divisor": {
                        "type": "number",
                        "description": "Numeric value to divide by. Must not be zero.",
                    },
                },
                "required": ["dividend", "divisor"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(frozen=True)
class ChatContextSource:
    """Single context source used to ground chat replies."""

    run_id: str
    question: str
    final_document: str
    context_bundle: dict[str, Any]


def load_context_bundle(path: Path) -> dict[str, Any]:
    """Load context bundle object from JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Context bundle at {path} is not a JSON object.")


def load_final_document(path: Path) -> str:
    """Load final markdown document."""
    return path.read_text(encoding="utf-8")


def generate_context_chat_reply(
    original_question: str,
    contexts: list[dict[str, Any]],
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int = 0,
    api_key_override: str | None = None,
    citation_catalog: list[dict[str, str]] | None = None,
    retry_missing_citation: bool = False,
    run_id: str | None = None,
) -> str:
    """Generate assistant reply grounded in selected run contexts."""
    resolved_cap = int(token_cap) if token_cap > 0 else resolve_chat_token_cap(config)
    effective_token_cap = max(
        config.chat.min_prompt_token_cap,
        min(resolved_cap, config.chat.max_context_total_tokens),
    )
    timeout = config.chat.provider_timeout_seconds

    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )
    client = OpenAI(
        api_key=api_key,
        base_url=config.openrouter_base_url,
        timeout=timeout,
    )

    normalized_contexts = _normalize_contexts(contexts)
    if not normalized_contexts:
        raise ValueError("At least one chat context source is required.")

    history_limit = max(0, config.chat.max_history_messages)
    bounded_history = _normalize_history(history)
    if history_limit > 0:
        bounded_history = bounded_history[-history_limit:]
    else:
        bounded_history = []

    normalized_citations = _normalize_citation_catalog(citation_catalog)
    prompt_header = _build_system_prompt_header(
        original_question=original_question,
        retry_missing_citation=retry_missing_citation,
    )

    base_request_kwargs: dict[str, Any] = {
        "model": config.chat.model,
        "temperature": float(config.chat.temperature),
        "tools": CHAT_TOOL_DEFINITIONS,
        "tool_choice": "auto",
    }
    if config.chat.reasoning_effort is not None:
        base_request_kwargs["reasoning_effort"] = config.chat.reasoning_effort
    if config.chat.max_output_tokens is not None:
        base_request_kwargs["max_tokens"] = config.chat.max_output_tokens

    retry_settings = RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )

    included_context_ids = [context.run_id for context in normalized_contexts]

    if normalized_citations:
        context_block = _build_fitted_citation_context_block(
            citation_catalog=normalized_citations,
            prompt_header=prompt_header,
            history=bounded_history,
            user_content=user_content,
            token_cap=effective_token_cap,
            prompt_token_buffer=config.chat.prompt_token_buffer,
        )
        system_prompt = _compose_system_prompt(prompt_header, context_block)
        messages = _build_messages(system_prompt, bounded_history, user_content)
        while _estimate_messages_tokens(messages) > effective_token_cap and bounded_history:
            bounded_history = bounded_history[1:]
            context_block = _build_fitted_citation_context_block(
                citation_catalog=normalized_citations,
                prompt_header=prompt_header,
                history=bounded_history,
                user_content=user_content,
                token_cap=effective_token_cap,
                prompt_token_buffer=config.chat.prompt_token_buffer,
            )
            system_prompt = _compose_system_prompt(prompt_header, context_block)
            messages = _build_messages(system_prompt, bounded_history, user_content)
        estimated_prompt_tokens = _estimate_messages_tokens(messages)
        if estimated_prompt_tokens > effective_token_cap:
            raise ValueError(
                "Chat context exceeds token budget after trimming "
                f"({estimated_prompt_tokens} > {effective_token_cap}). "
                "Reduce selected contexts or shorten history/messages."
            )
        logger.info(
            "Context chat request model=%s contexts=%s estimated_prompt_tokens=%d "
            "token_cap=%d effective_token_cap=%d",
            config.chat.model,
            included_context_ids,
            estimated_prompt_tokens,
            resolved_cap,
            effective_token_cap,
        )
        return _run_single_pass(
            client=client,
            messages=messages,
            request_kwargs=base_request_kwargs,
            retry_settings=retry_settings,
            run_id=run_id,
            context_count=len(included_context_ids),
        )

    serialized_contexts = _serialize_all_contexts(normalized_contexts)
    context_tokens = count_tokens(serialized_contexts)
    logger.info(
        "Context chat request model=%s contexts=%s context_tokens=%d "
        "threshold=%d token_cap=%d effective_token_cap=%d",
        config.chat.model,
        included_context_ids,
        context_tokens,
        config.chat.multi_pass_threshold_tokens,
        resolved_cap,
        effective_token_cap,
    )

    if context_tokens <= config.chat.multi_pass_threshold_tokens:
        system_prompt = _compose_system_prompt(prompt_header, serialized_contexts)
        messages = _build_messages(system_prompt, bounded_history, user_content)
        while _estimate_messages_tokens(messages) > effective_token_cap and bounded_history:
            bounded_history = bounded_history[1:]
            messages = _build_messages(system_prompt, bounded_history, user_content)
        estimated_prompt_tokens = _estimate_messages_tokens(messages)
        if estimated_prompt_tokens > effective_token_cap:
            raise ValueError(
                f"Chat context exceeds token budget after trimming history "
                f"({estimated_prompt_tokens} > {effective_token_cap}). "
                "Reduce selected contexts or shorten messages."
            )
        return _run_single_pass(
            client=client,
            messages=messages,
            request_kwargs=base_request_kwargs,
            retry_settings=retry_settings,
            run_id=run_id,
            context_count=len(included_context_ids),
        )

    return _run_multi_pass(
        serialized_contexts=serialized_contexts,
        prompt_header=prompt_header,
        bounded_history=bounded_history,
        user_content=user_content,
        chunk_tokens=config.chat.multi_pass_chunk_tokens,
        effective_token_cap=effective_token_cap,
        prompt_token_buffer=config.chat.prompt_token_buffer,
        client=client,
        request_kwargs=base_request_kwargs,
        retry_settings=retry_settings,
        run_id=run_id,
        context_count=len(included_context_ids),
    )


def _is_retryable_chat_error(exc: Exception) -> bool:
    """Return True for transient provider failures worth retrying."""
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def _extract_response_text(response: Any) -> str:
    """Extract cleaned text content from a chat completion response."""
    if not response.choices:
        raise ValueError("Chat model returned no choices.")
    message = response.choices[0].message
    content = message.content or ""
    if isinstance(content, list):
        content = "".join(
            part.text for part in content if hasattr(part, "text") and isinstance(part.text, str)
        )
    cleaned = str(content).strip()
    if not cleaned:
        raise ValueError("Chat model returned empty content.")
    return cleaned


def _run_single_pass(
    *,
    client: OpenAI,
    messages: list[dict[str, Any]],
    request_kwargs: dict[str, Any],
    retry_settings: RetrySettings,
    run_id: str | None,
    context_count: int,
) -> str:
    """Run a single chat completion pass and return the text reply."""
    response = call_with_retries(
        lambda: _run_chat_completion_with_tools(
            client=client,
            messages=messages,
            request_kwargs=request_kwargs,
            max_tool_rounds=retry_settings.max_attempts,
        ),
        operation="chat.completion",
        retry_settings=retry_settings,
        should_retry=_is_retryable_chat_error,
        run_id=run_id,
        context={"context_count": context_count},
    )
    return _extract_response_text(response)


def _run_multi_pass(
    *,
    serialized_contexts: str,
    prompt_header: str,
    bounded_history: list[dict[str, str]],
    user_content: str,
    chunk_tokens: int,
    effective_token_cap: int,
    prompt_token_buffer: int,
    client: OpenAI,
    request_kwargs: dict[str, Any],
    retry_settings: RetrySettings,
    run_id: str | None,
    context_count: int,
) -> str:
    """Split large context into chunks and refine the answer across multiple passes.

    Pass 1: first chunk → initial answer.
    Pass 2+: previous answer + next chunk → enhanced answer.

    Chunk size is clamped so that every initial pass fits within effective_token_cap.
    Enhancement passes additionally truncate previous_answer when needed.
    """
    # Compute fixed overhead for an initial pass (no context yet).
    # This accounts for prompt_header, history, user_content, and message framing.
    empty_initial_messages = _build_messages(
        _compose_system_prompt(prompt_header, ""),
        bounded_history,
        user_content,
    )
    base_overhead = _estimate_messages_tokens(empty_initial_messages) + prompt_token_buffer
    safe_chunk_tokens = max(1, min(chunk_tokens, effective_token_cap - base_overhead))
    if safe_chunk_tokens < chunk_tokens:
        logger.info(
            "Multi-pass chat: clamping chunk_tokens %d -> %d to fit effective_token_cap=%d",
            chunk_tokens,
            safe_chunk_tokens,
            effective_token_cap,
        )

    chunks = _split_text_by_tokens(serialized_contexts, safe_chunk_tokens)
    logger.info(
        "Multi-pass chat: context split into %d chunks of ~%d tokens each",
        len(chunks),
        safe_chunk_tokens,
    )

    current_answer: str | None = None
    for pass_index, chunk in enumerate(chunks, start=1):
        if current_answer is None:
            system_prompt = _compose_system_prompt(prompt_header, chunk)
        else:
            # Compute how many tokens are available for previous_answer in this pass.
            # Use an empty previous_answer to measure the fixed enhancement overhead.
            empty_enhancement = _compose_enhancement_prompt(
                prompt_header=prompt_header,
                previous_answer="",
                additional_context=chunk,
            )
            empty_enhancement_tokens = _estimate_messages_tokens(
                _build_messages(empty_enhancement, bounded_history, user_content)
            )
            available_for_answer = (
                effective_token_cap - empty_enhancement_tokens - prompt_token_buffer
            )
            if count_tokens(current_answer) > available_for_answer:
                logger.warning(
                    "Multi-pass chat pass %d: truncating previous_answer to %d tokens "
                    "to fit effective_token_cap=%d",
                    pass_index,
                    max(0, available_for_answer),
                    effective_token_cap,
                )
                current_answer = _truncate_to_tokens(current_answer, max(0, available_for_answer))
            system_prompt = _compose_enhancement_prompt(
                prompt_header=prompt_header,
                previous_answer=current_answer,
                additional_context=chunk,
            )

        messages = _build_messages(system_prompt, bounded_history, user_content)
        logger.info("Multi-pass chat: running pass %d/%d", pass_index, len(chunks))
        current_answer = _run_single_pass(
            client=client,
            messages=messages,
            request_kwargs=request_kwargs,
            retry_settings=retry_settings,
            run_id=run_id,
            context_count=context_count,
        )

    if current_answer is None:
        raise ValueError("Multi-pass chat produced no answer.")
    return current_answer


def _split_text_by_tokens(value: str, chunk_tokens: int) -> list[str]:
    """Split text into sequential non-overlapping chunks of ~chunk_tokens tokens."""
    stripped = value.strip()
    if not stripped:
        return []
    encoding = get_encoding()
    tokens = encoding.encode(stripped)
    if len(tokens) <= chunk_tokens:
        return [stripped]
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunks.append(encoding.decode(tokens[start:end]).strip())
        start = end
    return [c for c in chunks if c]


def _compose_enhancement_prompt(
    *,
    prompt_header: str,
    previous_answer: str,
    additional_context: str,
) -> str:
    """Build system prompt for a multi-pass enhancement turn."""
    return (
        f"{prompt_header}\n\n"
        "You previously produced the following answer based on partial context data:\n\n"
        "---\n"
        f"{previous_answer}\n"
        "---\n\n"
        "Below is an additional batch of context data not yet seen. "
        "Enhance and refine your answer by incorporating the new information. "
        "Keep everything from the initial answer that remains accurate, "
        "and add or correct details based on this new data.\n\n"
        "Additional context sources:\n"
        f"{additional_context}"
    )


def _parse_tool_arguments(raw_arguments: str | None) -> dict[str, object]:
    """Parse tool arguments into a JSON object."""
    if not raw_arguments:
        raise ValueError("Tool arguments are empty.")
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("Tool arguments must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return parsed


def _normalize_sum_numbers_args(raw_arguments: str | None) -> list[float]:
    """Parse sum tool arguments and coerce to float list."""
    parsed = _parse_tool_arguments(raw_arguments)
    raw_numbers = parsed.get("numbers")
    if not isinstance(raw_numbers, list):
        raise ValueError("Tool arguments must include list field `numbers`.")
    return [float(value) for value in raw_numbers]


def _normalize_subtract_numbers_args(raw_arguments: str | None) -> tuple[float, float]:
    """Parse subtract tool arguments and coerce to numeric operands."""
    parsed = _parse_tool_arguments(raw_arguments)
    if "minuend" not in parsed:
        raise ValueError("Tool arguments must include numeric field `minuend`.")
    if "subtrahend" not in parsed:
        raise ValueError("Tool arguments must include numeric field `subtrahend`.")
    return (float(parsed["minuend"]), float(parsed["subtrahend"]))


def _normalize_multiply_numbers_args(raw_arguments: str | None) -> list[float]:
    """Parse multiply tool arguments and coerce to float list."""
    parsed = _parse_tool_arguments(raw_arguments)
    raw_numbers = parsed.get("numbers")
    if not isinstance(raw_numbers, list):
        raise ValueError("Tool arguments must include list field `numbers`.")
    return [float(value) for value in raw_numbers]


def _normalize_divide_numbers_args(raw_arguments: str | None) -> tuple[float, float]:
    """Parse divide tool arguments and coerce to numeric operands."""
    parsed = _parse_tool_arguments(raw_arguments)
    if "dividend" not in parsed:
        raise ValueError("Tool arguments must include numeric field `dividend`.")
    if "divisor" not in parsed:
        raise ValueError("Tool arguments must include numeric field `divisor`.")
    return (float(parsed["dividend"]), float(parsed["divisor"]))


def _run_chat_completion_with_tools(
    *,
    client: OpenAI,
    messages: list[dict[str, str]],
    request_kwargs: dict[str, Any],
    max_tool_rounds: int,
) -> Any:
    """Execute chat completion and resolve tool calls in-process."""
    working_messages: list[dict[str, Any]] = [dict(message) for message in messages]
    resolved_max_rounds = max(int(max_tool_rounds), 1)
    for tool_round in range(1, resolved_max_rounds + 1):
        response = client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], working_messages),
            **request_kwargs,
        )
        if not response.choices:
            return response
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return response

        serialized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function_payload = getattr(tool_call, "function", None)
            function_name = str(getattr(function_payload, "name", "") or "").strip()
            function_arguments = str(getattr(function_payload, "arguments", "") or "")
            serialized_tool_calls.append(
                {
                    "id": str(getattr(tool_call, "id", "")),
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_arguments,
                    },
                }
            )

        working_messages.append(
            {
                "role": "assistant",
                "content": str(message.content or ""),
                "tool_calls": serialized_tool_calls,
            }
        )

        for tool_call in serialized_tool_calls:
            tool_name = str(tool_call.get("function", {}).get("name", "")).strip()
            tool_id = str(tool_call.get("id", "")).strip()
            tool_arguments = str(
                tool_call.get("function", {}).get("arguments", "")
            ).strip()
            if not tool_id:
                continue
            logger.info(
                "CHAT_TOOL_CALL round=%d tool=%s tool_call_id=%s",
                tool_round,
                tool_name,
                tool_id,
            )
            tool_result: dict[str, object]
            try:
                if tool_name == CHAT_SUM_TOOL_NAME:
                    numbers = _normalize_sum_numbers_args(tool_arguments)
                    tool_result = {"result": sum_numbers(numbers, source="context_chat")}
                elif tool_name == CHAT_SUBTRACT_TOOL_NAME:
                    minuend, subtrahend = _normalize_subtract_numbers_args(tool_arguments)
                    tool_result = {
                        "result": subtract_numbers(
                            minuend,
                            subtrahend,
                            source="context_chat",
                        )
                    }
                elif tool_name == CHAT_MULTIPLY_TOOL_NAME:
                    numbers = _normalize_multiply_numbers_args(tool_arguments)
                    tool_result = {"result": multiply_numbers(numbers, source="context_chat")}
                elif tool_name == CHAT_DIVIDE_TOOL_NAME:
                    dividend, divisor = _normalize_divide_numbers_args(tool_arguments)
                    tool_result = {
                        "result": divide_numbers(
                            dividend,
                            divisor,
                            source="context_chat",
                        )
                    }
                else:
                    tool_result = {"error": f"Unsupported tool: {tool_name}"}
                    logger.warning(
                        "CHAT_TOOL_CALL_UNSUPPORTED round=%d tool=%s tool_call_id=%s",
                        tool_round,
                        tool_name,
                        tool_id,
                    )
            except Exception as exc:  # noqa: BLE001
                tool_result = {"error": str(exc)}
                logger.exception(
                    "CHAT_TOOL_CALL_ERROR round=%d tool=%s tool_call_id=%s",
                    tool_round,
                    tool_name,
                    tool_id,
                )
            working_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )
    raise ValueError(
        f"Chat tool-call loop exceeded maximum rounds ({resolved_max_rounds})."
    )


def _normalize_contexts(raw_contexts: list[dict[str, Any]]) -> list[ChatContextSource]:
    """Normalize context payloads into typed sources."""
    normalized: list[ChatContextSource] = []
    for item in raw_contexts:
        if not isinstance(item, dict):
            continue
        run_id = item.get("run_id")
        question = item.get("question")
        final_document = item.get("final_document")
        context_bundle = item.get("context_bundle")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        if not isinstance(question, str):
            question = ""
        if not isinstance(final_document, str):
            continue
        if not isinstance(context_bundle, dict):
            continue
        normalized.append(
            ChatContextSource(
                run_id=run_id.strip(),
                question=question.strip(),
                final_document=final_document,
                context_bundle=context_bundle,
            )
        )
    return normalized


def _normalize_citation_catalog(
    raw_catalog: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Normalize citation catalog entries provided by caller."""
    if not isinstance(raw_catalog, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_catalog:
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("ref_id", "")).strip()
        if not ref_id or ref_id in seen or not CHAT_REF_ID_PATTERN.fullmatch(ref_id):
            continue
        city_name = str(item.get("city_name", "")).strip()
        quote = str(item.get("quote", "")).strip()
        partial_answer = str(item.get("partial_answer", "")).strip()
        if not quote and not partial_answer:
            continue
        normalized.append(
            {
                "ref_id": ref_id,
                "city_name": city_name,
                "quote": quote,
                "partial_answer": partial_answer,
            }
        )
        seen.add(ref_id)
    return normalized


def _normalize_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize history to user/assistant content pairs."""
    normalized: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        stripped = content.strip()
        if not stripped:
            continue
        normalized.append({"role": role, "content": stripped})
    return normalized


def _fit_citation_catalog_to_budget(
    citation_catalog: list[dict[str, str]],
    prompt_header: str,
    history: list[dict[str, str]],
    user_content: str,
    token_cap: int,
    prompt_token_buffer: int,
) -> list[dict[str, str]]:
    """Keep only citation entries that fit the strict prompt budget."""
    fixed_messages = [{"role": "user", "content": user_content}] + history
    fixed_tokens = _estimate_messages_tokens(fixed_messages)
    strict_budget = token_cap - fixed_tokens - count_tokens(prompt_header) - prompt_token_buffer
    if strict_budget <= 0:
        return []

    fitted: list[dict[str, str]] = []
    for item in citation_catalog:
        candidate = fitted + [item]
        if count_tokens(_render_citation_catalog_block(candidate)) > strict_budget:
            break
        fitted = candidate
    return fitted


def _build_fitted_citation_context_block(
    citation_catalog: list[dict[str, str]],
    prompt_header: str,
    history: list[dict[str, str]],
    user_content: str,
    token_cap: int,
    prompt_token_buffer: int,
) -> str:
    """Render citation context block after budget-aware catalog pruning."""
    fitted = _fit_citation_catalog_to_budget(
        citation_catalog=citation_catalog,
        prompt_header=prompt_header,
        history=history,
        user_content=user_content,
        token_cap=token_cap,
        prompt_token_buffer=prompt_token_buffer,
    )
    return _render_citation_catalog_block(fitted)


def _render_citation_catalog_block(citation_catalog: list[dict[str, str]]) -> str:
    """Serialize citation catalog into prompt-safe markdown context."""
    if not citation_catalog:
        return (
            "### Citation evidence catalog\n"
            "- No citation entries fit within the prompt token budget for this turn."
        )
    lines = ["### Citation evidence catalog"]
    for item in citation_catalog:
        ref_id = item["ref_id"]
        city_name = item["city_name"] or "(unknown city)"
        quote = item["quote"] or "(empty quote)"
        partial_answer = item["partial_answer"] or "(empty partial answer)"
        lines.append(
            "\n".join(
                [
                    f"- [{ref_id}] City: {city_name}",
                    f"  Quote: {quote}",
                    f"  Partial answer: {partial_answer}",
                ]
            )
        )
    return "\n".join(lines)


def _build_system_prompt_header(
    original_question: str,
    retry_missing_citation: bool,
) -> str:
    """Build stable system prompt prefix."""
    if retry_missing_citation:
        retry_note = (
            "Prior response failed citation requirements. Rewrite the full answer and ensure "
            "every factual claim is immediately followed by one or more valid [ref_n] citations.\n"
        )
    else:
        retry_note = ""
    stripped_original_question = original_question.strip()
    return (
        "You are the Context Analyst for a document-builder workflow.\n"
        "Your job is to answer follow-up questions using only the supplied context sources.\n"
        "Each source comes from a completed run and includes curated citation evidence.\n\n"
        "Rules:\n"
        "1. Ground every factual claim in provided context sources.\n"
        "2. If information is missing or uncertain, say so clearly.\n"
        "3. Compare sources when useful and call out contradictions.\n"
        "4. Always respond in valid markdown. Prefer headings, bullets, and tables for numeric data.\n"
        "5. Never mention internal paths or backend implementation details.\n"
        "6. If a citation evidence catalog is provided, cite factual claims using only [ref_n] tokens present in that catalog.\n"
        "7. Do not invent references and do not use any citation format other than [ref_n].\n"
        "8. If no citation evidence catalog entries are available for this turn, explain that you cannot provide a fully grounded cited answer.\n\n"
        "9. If arithmetic is needed and calculator tools are available, use them instead of mental math.\n\n"
        f"{retry_note}"
        f"Original build question:\n{stripped_original_question}"
    )


def _compose_system_prompt(header: str, context_block: str) -> str:
    """Build final system prompt with context payload."""
    return (
        f"{header}\n\n"
        "Context sources:\n"
        f"{context_block}"
    )



def _serialize_all_contexts(contexts: list[ChatContextSource]) -> str:
    """Serialize all context sources into a single string."""
    sections = [
        _serialize_context(index, context)
        for index, context in enumerate(contexts, start=1)
    ]
    return "\n\n".join(sections)


def _serialize_context(index: int, context: ChatContextSource) -> str:
    """Serialize one context source."""
    serialized_bundle = json.dumps(
        context.context_bundle,
        ensure_ascii=True,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"### Source {index} [run:{context.run_id}]\n"
        f"Run question: {context.question or '(not provided)'}\n\n"
        "Final document markdown:\n"
        "```markdown\n"
        f"{context.final_document.strip()}\n"
        "```\n\n"
        "Context bundle JSON:\n"
        "```json\n"
        f"{serialized_bundle}\n"
        "```"
    )



def _build_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    user_content: str,
) -> list[dict[str, str]]:
    """Build chat completion message list."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def _estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate token usage for chat messages."""
    total = 0
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        total += count_tokens(f"{role}\n{content}") + 6
    return total + 2


def _truncate_to_tokens(value: str, max_tokens: int) -> str:
    """Truncate text to token budget, preserving leading content."""
    if max_tokens <= 0:
        return ""
    encoding = get_encoding()
    tokens = encoding.encode(value)
    if len(tokens) <= max_tokens:
        return value

    suffix = "\n\n[truncated due to prompt token budget]"
    suffix_tokens = encoding.encode(suffix)
    if max_tokens <= len(suffix_tokens):
        return encoding.decode(tokens[:max_tokens])
    head_tokens = tokens[: max_tokens - len(suffix_tokens)]
    return encoding.decode(head_tokens) + suffix


__all__ = [
    "ChatContextSource",
    "resolve_chat_token_cap",
    "generate_context_chat_reply",
    "load_context_bundle",
    "load_final_document",
]
