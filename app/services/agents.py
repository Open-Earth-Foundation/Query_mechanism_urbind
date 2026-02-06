from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.exceptions import MaxTurnsExceeded
from agents.items import ModelResponse, TResponseInputItem
from agents.lifecycle import RunHooksBase
from agents.models.openai_provider import OpenAIProvider
from agents.result import RunResult
from agents.run_context import RunContextWrapper
from agents.tool import Tool

logger = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


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
        response_payload = {
            "response_id": response.response_id,
            "usage": response.usage,
            "assistant_text": assistant_text,
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


def build_openrouter_model(model_name: str, api_key: str, base_url: str | None):
    provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    return provider.get_model(model_name)


def run_agent_sync(agent: Agent, input_data: str, max_turns: int = 3) -> RunResult:
    """Run an agent synchronously with optional max_turns limit.

    Args:
        agent: The agent to run
        input_data: The input data as a JSON string
        max_turns: Maximum number of turns before gracefully stopping (default: 3)

    Returns:
        The agent's final output

    Raises:
        Only re-raises exceptions other than MaxTurnsExceeded
    """
    hooks_list: list[RunHooksBase[Any, Agent[Any]]] = [LlmUsageLoggingHooks()]
    if _env_truthy("LOG_LLM_PAYLOAD"):
        hooks_list.append(LlmPayloadLoggingHooks())
    hooks: RunHooksBase[Any, Agent[Any]] | None = CompositeRunHooks(hooks_list)
    try:
        return asyncio.run(Runner.run(agent, input_data, max_turns=max_turns, hooks=hooks))
    except MaxTurnsExceeded as exc:
        logger.warning(
            "Agent exceeded max turns (%d). Gracefully stopping with partial results.",
            max_turns,
        )
        # Return None to signal that the agent hit max turns
        # The caller should handle this appropriately
        raise


__all__ = ["build_model_settings", "build_openrouter_model", "run_agent_sync"]
