from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sys
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from agents import Agent, ModelSettings, Runner
from agents.exceptions import MaxTurnsExceeded
from agents.items import ModelResponse, TResponseInputItem
from agents.lifecycle import RunHooksBase
from agents.models.openai_provider import OpenAIProvider
from agents.result import RunResult
from agents.run_context import RunContextWrapper
from agents.tool import Tool
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

logger = logging.getLogger(__name__)

_thread_local = threading.local()

if sys.platform.startswith("win"):
    # Avoid Proactor-specific transport-close races when running many short LLM calls.
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _truncate_text(value: str, max_chars: int = 500) -> str:
    """Trim long strings for log safety."""
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}..."


def _extract_error_payload(response: httpx.Response) -> str | None:
    """Extract a compact error payload from an HTTP response."""
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return None
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        return None
    if isinstance(data, dict):
        data = data.get("error", data)
    return _truncate_text(_dump_payload(data))


def _format_http_error_details(response: httpx.Response) -> str:
    """Format useful error details for HTTP error logging."""
    parts: list[str] = []
    request_id = response.headers.get("x-request-id")
    retry_after = response.headers.get("retry-after")
    should_retry = response.headers.get("x-should-retry")
    error_payload = _extract_error_payload(response)

    if request_id:
        parts.append(f"request_id={request_id}")
    if retry_after:
        parts.append(f"retry_after={retry_after}")
    if should_retry:
        parts.append(f"should_retry={should_retry}")
    if error_payload:
        parts.append(f"error={error_payload}")

    return ", ".join(parts)


async def _log_http_error_response(response: httpx.Response) -> None:
    """Log reasons for HTTP errors, including retry hints."""
    if response.status_code < 400:
        return
    request = response.request
    details = _format_http_error_details(response)
    suffix = f" ({details})" if details else ""
    logger.warning(
        "OpenAI HTTP error response: %s %s -> %s %s%s",
        request.method,
        request.url,
        response.status_code,
        response.reason_phrase,
        suffix,
    )


def _get_openai_http_client() -> httpx.AsyncClient:
    """Return a thread-local HTTP client with error logging hooks."""
    client = getattr(_thread_local, "openai_http_client", None)
    if client is None:
        client = DefaultAsyncHttpxClient(
            event_hooks={"response": [_log_http_error_response]},
        )
        _thread_local.openai_http_client = client
    return client


def _get_thread_openai_client_cache() -> dict[tuple[str, str | None, int | None], AsyncOpenAI]:
    """Return a thread-local cache for OpenAI clients."""
    cache = getattr(_thread_local, "openai_client_cache", None)
    if cache is None:
        cache = {}
        _thread_local.openai_client_cache = cache
    return cache


def _get_openai_client(
    api_key: str,
    base_url: str | None,
    max_retries: int | None = None,
) -> AsyncOpenAI:
    """Return a thread-local cached OpenAI client configured for error logging."""
    cache_key = (api_key, base_url, max_retries)
    cache = _get_thread_openai_client_cache()
    client = cache.get(cache_key)
    if client is None:
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "http_client": _get_openai_http_client(),
        }
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        client = AsyncOpenAI(**kwargs)
        cache[cache_key] = client
    return client


def _safe_serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _safe_serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_serialize(item) for item in value]
    if dataclasses.is_dataclass(value):
        try:
            return _safe_serialize(dataclasses.asdict(value))
        except Exception:  # noqa: BLE001
            return str(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _safe_serialize(model_dump())
        except Exception:  # noqa: BLE001
            pass
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return _safe_serialize(to_dict())
        except Exception:  # noqa: BLE001
            pass
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        try:
            filtered = {key: item for key, item in value_dict.items() if not key.startswith("_")}
            return _safe_serialize(filtered)
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _dump_payload(payload: Any) -> str:
    try:
        return json.dumps(_safe_serialize(payload), ensure_ascii=True)
    except Exception:  # noqa: BLE001
        return str(payload)


def _get_field(target: Any, key: str) -> Any:
    if isinstance(target, Mapping):
        return target.get(key)
    return getattr(target, key, None)


def _extract_text_from_output_item(item: Any) -> list[str]:
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


class CompositeRunHooks(RunHooksBase[Any, Agent[Any]]):
    """Fan-out hook calls to multiple RunHooks instances."""

    def __init__(self, hooks: Sequence[RunHooksBase[Any, Agent[Any]]]) -> None:
        self._hooks = list(hooks)

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        for hook in self._hooks:
            await hook.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        for hook in self._hooks:
            await hook.on_llm_end(context, agent, response)

    async def on_agent_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
    ) -> None:
        for hook in self._hooks:
            await hook.on_agent_start(context, agent)

    async def on_agent_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        output: Any,
    ) -> None:
        for hook in self._hooks:
            await hook.on_agent_end(context, agent, output)

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Agent[Any],
        to_agent: Agent[Any],
    ) -> None:
        for hook in self._hooks:
            await hook.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
    ) -> None:
        for hook in self._hooks:
            await hook.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        tool: Tool,
        result: str,
    ) -> None:
        for hook in self._hooks:
            await hook.on_tool_end(context, agent, tool, result)


class LlmUsageLoggingHooks(RunHooksBase[Any, Agent[Any]]):
    """Logs per-call LLM token usage."""

    def __init__(self) -> None:
        self._turn = 0

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        self._turn += 1
        payload = {
            "event": "llm_usage",
            "turn": self._turn,
            "agent": agent.name,
            "response_id": response.response_id,
            "usage": response.usage,
        }
        logger.info("LLM_USAGE %s", _dump_payload(payload))


class LlmPayloadLoggingHooks(RunHooksBase[Any, Agent[Any]]):
    """Logs full LLM payloads for debugging."""

    def __init__(self, max_chars: int | None = None) -> None:
        self._max_chars = max_chars
        self._turn = 0

    def _log_payload(self, payload: dict[str, Any]) -> None:
        rendered = _dump_payload(payload)
        if self._max_chars is not None:
            rendered = rendered[: self._max_chars]
        logger.info("%s", rendered)

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        self._turn += 1

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        assistant_text: list[str] = []
        for output in response.output:
            assistant_text.extend(_extract_text_from_output_item(output))
        output_items = response.output
        response_payload = {
            "response_id": response.response_id,
            "usage": response.usage,
            "assistant_text": assistant_text,
            "output": output_items,
        }
        payload = {
            "event": "llm_end",
            "turn": self._turn,
            "agent": agent.name,
            "response": response_payload,
        }
        self._log_payload(payload)


def build_model_settings(
    temperature: float | None, max_output_tokens: int | None
) -> ModelSettings:
    settings_kwargs: dict[str, Any] = {"include_usage": True}
    if temperature is not None:
        settings_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        settings_kwargs["max_output_tokens"] = max_output_tokens
    return ModelSettings(**settings_kwargs)


def build_openrouter_model(
    model_name: str,
    api_key: str,
    base_url: str | None,
    client_max_retries: int | None = None,
):
    """Build an OpenRouter-backed model with HTTP error logging enabled."""
    client = _get_openai_client(api_key, base_url, max_retries=client_max_retries)
    provider = OpenAIProvider(openai_client=client)
    return provider.get_model(model_name)


def run_agent_sync(
    agent: Agent,
    input_data: str,
    max_turns: int = 3,
    log_llm_payload: bool = False,
) -> RunResult:
    """Run an agent synchronously with optional max_turns limit.

    Args:
        agent: The agent to run
        input_data: The input data as a JSON string
        max_turns: Maximum number of turns before gracefully stopping (default: 3)
        log_llm_payload: Whether to log full LLM request/response payloads

    Returns:
        The agent's final output

    Raises:
        Only re-raises exceptions other than MaxTurnsExceeded
    """
    hooks_list: list[RunHooksBase[Any, Agent[Any]]] = [LlmUsageLoggingHooks()]
    if log_llm_payload:
        hooks_list.append(LlmPayloadLoggingHooks())
    hooks: RunHooksBase[Any, Agent[Any]] | None = CompositeRunHooks(hooks_list)

    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_local.loop = loop
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(
            Runner.run(agent, input_data, max_turns=max_turns, hooks=hooks)
        )
    except MaxTurnsExceeded as exc:
        logger.warning(
            "Agent exceeded max turns (%d). Gracefully stopping with partial results.",
            max_turns,
        )
        # Return None to signal that the agent hit max turns
        # The caller should handle this appropriately
        raise


__all__ = ["build_model_settings", "build_openrouter_model", "run_agent_sync"]
