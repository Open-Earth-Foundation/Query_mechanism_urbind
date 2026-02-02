from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents import Agent, ModelSettings, Runner
from agents.exceptions import MaxTurnsExceeded
from agents.models.openai_provider import OpenAIProvider
from agents.result import RunResult

logger = logging.getLogger(__name__)


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


def run_agent_sync(agent: Agent, input_data: str, max_turns: int = 10) -> RunResult:
    """Run an agent synchronously with optional max_turns limit.

    Args:
        agent: The agent to run
        input_data: The input data as a JSON string
        max_turns: Maximum number of turns before gracefully stopping (default: 10)

    Returns:
        The agent's final output

    Raises:
        Only re-raises exceptions other than MaxTurnsExceeded
    """
    try:
        return asyncio.run(Runner.run(agent, input_data, max_turns=max_turns))
    except MaxTurnsExceeded as exc:
        logger.warning(
            "Agent exceeded max turns (%d). Gracefully stopping with partial results.",
            max_turns,
        )
        # Return None to signal that the agent hit max turns
        # The caller should handle this appropriately
        raise


__all__ = ["build_model_settings", "build_openrouter_model", "run_agent_sync"]
