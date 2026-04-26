"""LLM agent construction and thread-local agent caching for initiative extraction."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from backend.modules.initiative_extractor.models import (
    InitiativeSemanticDedupeGroup,
    InitiativeSemanticDedupeResult,
    InitiativeSegmentStop,
)
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt

_thread_local = threading.local()


def _facade() -> object:
    """Return the compatibility facade module for monkeypatched tests."""
    from backend.modules.initiative_extractor import agent

    return agent


def run_agent_sync(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper so schema/unit tests do not require the Agents SDK at import time."""
    from backend.services.agents import run_agent_sync as run_sync

    return run_sync(*args, **kwargs)


def build_initiative_extractor_agent(config: AppConfig, api_key: str) -> object:
    """Build the initiative extractor agent with structured tool output."""
    from agents import Agent, function_tool
    from backend.services.agents import build_model_settings, build_openrouter_model

    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "initiative_extractor_system.md"
    )
    model = build_openrouter_model(
        config.initiative_extractor.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.initiative_extractor.temperature,
        config.initiative_extractor.max_output_tokens,
        reasoning_effort=config.initiative_extractor.reasoning_effort,
    )
    settings.tool_choice = "required"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=False)
    def submit_initiative_extractions(
        initiatives: list[dict[str, Any]] | None = None,
        segment_data_quality_flags: list[str] | None = None,
        segment_notes: list[str] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return raw tool args so segment-scoped parsing can assign city."""
        return {
            "initiatives": initiatives or [],
            "segment_data_quality_flags": segment_data_quality_flags or [],
            "segment_notes": segment_notes or [],
            "error": error,
        }

    @function_tool(strict_mode=False)
    def stop_initiative_extraction(
        reason: str | None = None,
        segment_data_quality_flags: list[str] | None = None,
        segment_notes: list[str] | None = None,
    ) -> InitiativeSegmentStop:
        return InitiativeSegmentStop.model_validate(
            {
                "reason": reason,
                "segment_data_quality_flags": segment_data_quality_flags or [],
                "segment_notes": segment_notes or [],
            }
        )

    return Agent(
        name="Initiative Extractor",
        instructions=load_prompt(prompt_path),
        model=model,
        model_settings=settings,
        tools=[submit_initiative_extractions, stop_initiative_extraction],
        tool_use_behavior="stop_on_first_tool",
    )


def build_initiative_semantic_dedupe_agent(config: AppConfig, api_key: str) -> object:
    """Build the semantic initiative dedupe agent."""
    from agents import Agent, AgentOutputSchema, function_tool
    from backend.services.agents import build_model_settings, build_openrouter_model

    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "initiative_semantic_dedupe_system.md"
    )
    model = build_openrouter_model(
        config.initiative_extractor.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        config.initiative_extractor.temperature,
        config.initiative_extractor.max_output_tokens,
        reasoning_effort=config.initiative_extractor.reasoning_effort,
    )
    settings.tool_choice = "submit_semantic_dedupe"
    settings.parallel_tool_calls = False

    @function_tool(strict_mode=False)
    def submit_semantic_dedupe(
        duplicate_groups: list[InitiativeSemanticDedupeGroup] | None = None,
        review_notes: list[str] | None = None,
    ) -> InitiativeSemanticDedupeResult:
        return InitiativeSemanticDedupeResult(
            duplicate_groups=duplicate_groups or [],
            review_notes=review_notes or [],
        )

    return Agent(
        name="Initiative Semantic Dedupe",
        instructions=load_prompt(prompt_path),
        model=model,
        model_settings=settings,
        tools=[submit_semantic_dedupe],
        output_type=AgentOutputSchema(
            InitiativeSemanticDedupeResult,
            strict_json_schema=False,
        ),
        tool_use_behavior="stop_on_first_tool",
    )


def _get_thread_agent(config: AppConfig, api_key: str) -> object:
    """Return a thread-local initiative extractor agent."""
    local_agent = getattr(_thread_local, "initiative_extractor_agent", None)
    if local_agent is None:
        local_agent = _facade().build_initiative_extractor_agent(config, api_key)
        _thread_local.initiative_extractor_agent = local_agent
    return local_agent


def _get_thread_semantic_dedupe_agent(config: AppConfig, api_key: str) -> object:
    """Return a thread-local semantic dedupe agent."""
    local_agent = getattr(_thread_local, "initiative_semantic_dedupe_agent", None)
    if local_agent is None:
        local_agent = _facade().build_initiative_semantic_dedupe_agent(config, api_key)
        _thread_local.initiative_semantic_dedupe_agent = local_agent
    return local_agent
