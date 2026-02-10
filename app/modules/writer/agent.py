from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, function_tool

from app.modules.writer.models import WriterOutput
from app.services.agents import build_model_settings, build_openrouter_model, run_agent_sync
from app.utils.config import AppConfig
from app.utils.prompts import load_prompt


def build_writer_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the writer agent."""
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "writer_system.md"
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(config.writer.model, api_key, config.openrouter_base_url)
    settings = build_model_settings(config.writer.temperature, config.writer.max_output_tokens)

    @function_tool
    def submit_writer_output(output: WriterOutput) -> WriterOutput:
        return output

    return Agent(
        name="Writer",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_writer_output],
        output_type=WriterOutput,
        tool_use_behavior="stop_on_first_tool",
    )


def write_markdown(
    question: str,
    context_bundle: dict,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> WriterOutput:
    """Generate the final markdown answer."""
    agent = build_writer_agent(config, api_key)
    payload = {
        "question": question,
        "context_bundle": context_bundle,
    }
    result = run_agent_sync(
        agent,
        json.dumps(payload, ensure_ascii=True),
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, WriterOutput):
        return output
    raise ValueError("Writer did not return structured output.")


__all__ = ["build_writer_agent", "write_markdown"]
