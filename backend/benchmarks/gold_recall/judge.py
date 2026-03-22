from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, function_tool

from backend.benchmarks.gold_recall.models import FactJudgeDecision
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt

FACT_JUDGE_MODEL = "openai/gpt-5.2"


def build_fact_judge_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the LLM-as-judge agent used for fact presence checks."""
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "benchmark_fact_judge_system.md"
    )
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(
        FACT_JUDGE_MODEL,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        temperature=0.0,
        max_output_tokens=600,
        reasoning_effort="high",
    )

    @function_tool
    def submit_fact_judgement(
        decision: FactJudgeDecision,
    ) -> FactJudgeDecision:
        """Return one structured fact-presence decision unchanged."""
        return decision

    return Agent(
        name="Benchmark Fact Judge",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_fact_judgement],
        output_type=FactJudgeDecision,
        tool_use_behavior="stop_on_first_tool",
    )


def judge_fact_presence(
    *,
    question: str,
    stage_label: str,
    fact: str,
    candidate_text: str,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> FactJudgeDecision:
    """Judge whether one gold fact is stated or directly implied in candidate text."""
    agent = build_fact_judge_agent(config, api_key)
    payload = {
        "question": question,
        "stage_label": stage_label,
        "fact": fact,
        "candidate_text": candidate_text,
    }
    result = run_agent_sync(
        agent,
        json.dumps(payload, ensure_ascii=False),
        max_turns=config.retry.max_attempts,
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, FactJudgeDecision):
        return output
    raise ValueError("Fact judge did not return structured output.")


__all__ = ["FACT_JUDGE_MODEL", "build_fact_judge_agent", "judge_fact_presence"]
