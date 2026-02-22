from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, function_tool

from backend.modules.orchestrator.models import (
    OrchestratorDecision,
    ResearchQuestionRefinement,
)
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt
from backend.utils.tokenization import get_max_input_tokens


def build_orchestrator_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the orchestrator agent."""
    prompt_path = (
        Path(__file__).resolve().parents[2] / "prompts" / "orchestrator_system.md"
    )
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(
        config.orchestrator.model, api_key, config.openrouter_base_url
    )
    settings = build_model_settings(
        config.orchestrator.temperature,
        config.orchestrator.max_output_tokens,
    )

    @function_tool
    def decide_next_action(decision: OrchestratorDecision) -> OrchestratorDecision:
        return decision

    return Agent(
        name="Orchestrator",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[decide_next_action],
        output_type=OrchestratorDecision,
        tool_use_behavior="stop_on_first_tool",
    )


def build_research_question_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the orchestrator agent for refining the research question."""
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "orchestrator_research_question_system.md"
    )
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(
        config.orchestrator.model, api_key, config.openrouter_base_url
    )
    settings = build_model_settings(
        config.orchestrator.temperature,
        config.orchestrator.max_output_tokens,
    )

    @function_tool
    def submit_research_question(
        refinement: ResearchQuestionRefinement,
    ) -> ResearchQuestionRefinement:
        return refinement

    return Agent(
        name="Orchestrator Question Refiner",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_research_question],
        output_type=ResearchQuestionRefinement,
        tool_use_behavior="stop_on_first_tool",
    )


def decide_next_action(
    question: str,
    context_bundle: dict,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> OrchestratorDecision:
    """Run the orchestrator to decide the next action."""
    agent = build_orchestrator_agent(config, api_key)
    payload = {
        "question": question,
        "context_bundle": context_bundle,
        "sql_enabled": config.enable_sql,
        "context_window_tokens": config.orchestrator.context_window_tokens,
        "max_input_tokens": get_max_input_tokens(
            config.orchestrator.context_window_tokens,
            config.orchestrator.max_output_tokens,
            config.orchestrator.input_token_reserve,
            config.orchestrator.max_input_tokens,
        ),
    }
    result = run_agent_sync(
        agent,
        json.dumps(payload, ensure_ascii=False),
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, OrchestratorDecision):
        return output
    raise ValueError("Orchestrator did not return a structured decision.")


def refine_research_question(
    question: str,
    config: AppConfig,
    api_key: str,
    selected_cities: list[str] | None = None,
    log_llm_payload: bool = False,
) -> ResearchQuestionRefinement:
    """Return a lightly refined research question plus retrieval query variants."""
    agent = build_research_question_agent(config, api_key)
    payload = {
        "question": question,
        "selected_cities": selected_cities or [],
        "context_window_tokens": config.orchestrator.context_window_tokens,
        "max_input_tokens": get_max_input_tokens(
            config.orchestrator.context_window_tokens,
            config.orchestrator.max_output_tokens,
            config.orchestrator.input_token_reserve,
            config.orchestrator.max_input_tokens,
        ),
    }
    result = run_agent_sync(
        agent,
        json.dumps(payload, ensure_ascii=False),
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, ResearchQuestionRefinement):
        return output
    raise ValueError("Orchestrator did not return a structured research question.")


__all__ = [
    "build_orchestrator_agent",
    "build_research_question_agent",
    "decide_next_action",
    "refine_research_question",
]
