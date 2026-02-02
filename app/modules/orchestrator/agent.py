from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, function_tool

from app.modules.orchestrator.models import OrchestratorDecision
from app.services.agents import build_model_settings, build_openrouter_model, run_agent_sync
from app.utils.config import AppConfig
from app.utils.prompts import load_prompt
from app.utils.tokenization import get_max_input_tokens


def build_orchestrator_agent(config: AppConfig, api_key: str) -> Agent:
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "orchestrator_system.md"
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(config.orchestrator.model, api_key, config.openrouter_base_url)
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


def decide_next_action(
    question: str,
    context_bundle: dict,
    run_id: str,
    config: AppConfig,
    api_key: str,
) -> OrchestratorDecision:
    agent = build_orchestrator_agent(config, api_key)
    payload = {
        "run_id": run_id,
        "question": question,
        "context_bundle": context_bundle,
        "context_window_tokens": config.orchestrator.context_window_tokens,
        "max_input_tokens": get_max_input_tokens(
            config.orchestrator.context_window_tokens,
            config.orchestrator.max_output_tokens,
            config.orchestrator.input_token_reserve,
            config.orchestrator.max_input_tokens,
        ),
    }
    result = run_agent_sync(agent, json.dumps(payload, ensure_ascii=True))
    output = result.final_output
    if isinstance(output, OrchestratorDecision):
        return output
    raise ValueError("Orchestrator did not return a structured decision.")


__all__ = ["build_orchestrator_agent", "decide_next_action"]
