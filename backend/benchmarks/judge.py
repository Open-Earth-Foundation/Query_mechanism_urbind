from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, function_tool

from backend.benchmarks.models import BenchmarkJudgeEvaluation
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt

BENCHMARK_JUDGE_MODEL = "openai/gpt-5.2"


def build_benchmark_judge_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the benchmark LLM-as-judge agent."""
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "benchmark_judge_system.md"
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(
        BENCHMARK_JUDGE_MODEL,
        api_key,
        config.openrouter_base_url,
    )
    settings = build_model_settings(temperature=0.0, max_output_tokens=1200)

    @function_tool
    def submit_benchmark_judgement(
        evaluation: BenchmarkJudgeEvaluation,
    ) -> BenchmarkJudgeEvaluation:
        return evaluation

    return Agent(
        name="Benchmark Judge",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_benchmark_judgement],
        output_type=BenchmarkJudgeEvaluation,
        tool_use_behavior="stop_on_first_tool",
    )


def judge_final_outputs(
    *,
    question: str,
    left_label: str,
    left_text: str,
    right_label: str,
    right_text: str,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> BenchmarkJudgeEvaluation:
    """Compare two final markdown outputs with a numeric rubric."""
    agent = build_benchmark_judge_agent(config, api_key)
    payload = {
        "question": question,
        "left_label": left_label,
        "right_label": right_label,
        "left_text": left_text,
        "right_text": right_text,
    }
    result = run_agent_sync(
        agent,
        json.dumps(payload, ensure_ascii=False),
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, BenchmarkJudgeEvaluation):
        return output
    raise ValueError("Benchmark judge did not return structured output.")


__all__ = ["BENCHMARK_JUDGE_MODEL", "build_benchmark_judge_agent", "judge_final_outputs"]
