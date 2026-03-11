"""Provider execution helpers for context chat requests."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from backend.tools.calculator import (
    divide_numbers,
    multiply_numbers,
    subtract_numbers,
    sum_numbers,
)
from backend.utils.config import AppConfig, get_openrouter_api_key
from backend.utils.retry import RetrySettings, call_with_retries

logger = logging.getLogger("backend.api.services.context_chat")

CHAT_SUM_TOOL_NAME = "sum_numbers"
CHAT_SUBTRACT_TOOL_NAME = "subtract_numbers"
CHAT_MULTIPLY_TOOL_NAME = "multiply_numbers"
CHAT_DIVIDE_TOOL_NAME = "divide_numbers"

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


def create_chat_client(config: AppConfig, api_key_override: str | None) -> OpenAI:
    """Build the OpenAI client used for one context-chat request."""
    timeout = config.chat.provider_timeout_seconds
    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )
    return OpenAI(
        api_key=api_key,
        base_url=config.openrouter_base_url,
        timeout=timeout,
    )


def build_request_kwargs(config: AppConfig) -> dict[str, Any]:
    """Build the provider request kwargs for one context-chat request."""
    request_kwargs: dict[str, Any] = {
        "model": config.chat.model,
        "temperature": float(config.chat.temperature),
        "tools": CHAT_TOOL_DEFINITIONS,
        "tool_choice": "auto",
    }
    if config.chat.reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = config.chat.reasoning_effort
    if config.chat.max_output_tokens is not None:
        request_kwargs["max_tokens"] = config.chat.max_output_tokens
    return request_kwargs


def _is_retryable_chat_error(exc: Exception) -> bool:
    """Return whether the provider error is transient and worth retrying."""
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def _extract_response_text(response: Any) -> str:
    """Extract cleaned text content from a chat-completion response."""
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
    """Run one provider request and return the cleaned assistant text."""
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


def _parse_tool_arguments(raw_arguments: str | None) -> dict[str, object]:
    """Parse one tool-call JSON argument payload."""
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
    """Parse sum tool arguments and coerce them to a float list."""
    parsed = _parse_tool_arguments(raw_arguments)
    raw_numbers = parsed.get("numbers")
    if not isinstance(raw_numbers, list):
        raise ValueError("Tool arguments must include list field `numbers`.")
    return [float(value) for value in raw_numbers]


def _normalize_subtract_numbers_args(raw_arguments: str | None) -> tuple[float, float]:
    """Parse subtract tool arguments and coerce them to numeric operands."""
    parsed = _parse_tool_arguments(raw_arguments)
    if "minuend" not in parsed:
        raise ValueError("Tool arguments must include numeric field `minuend`.")
    if "subtrahend" not in parsed:
        raise ValueError("Tool arguments must include numeric field `subtrahend`.")
    return (float(parsed["minuend"]), float(parsed["subtrahend"]))


def _normalize_multiply_numbers_args(raw_arguments: str | None) -> list[float]:
    """Parse multiply tool arguments and coerce them to a float list."""
    parsed = _parse_tool_arguments(raw_arguments)
    raw_numbers = parsed.get("numbers")
    if not isinstance(raw_numbers, list):
        raise ValueError("Tool arguments must include list field `numbers`.")
    return [float(value) for value in raw_numbers]


def _normalize_divide_numbers_args(raw_arguments: str | None) -> tuple[float, float]:
    """Parse divide tool arguments and coerce them to numeric operands."""
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
    """Execute a chat completion request and resolve calculator tool calls in-process."""
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


__all__ = [
    "CHAT_TOOL_DEFINITIONS",
    "OpenAI",
    "build_request_kwargs",
    "create_chat_client",
]
