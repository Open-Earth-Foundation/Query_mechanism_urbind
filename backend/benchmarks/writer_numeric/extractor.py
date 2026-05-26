from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, function_tool

from backend.benchmarks.writer_numeric.models import (
    WriterMetricExtraction,
    WriterNumberExtraction,
    WriterNumericBenchmarkCase,
)
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt


def build_writer_number_extractor_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the structured extractor used to pull benchmark numbers from final.md."""
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "benchmark_writer_number_extractor_system.md"
    )
    instructions = load_prompt(prompt_path)
    extractor_cfg = config.benchmark_number_extractor
    model = build_openrouter_model(
        extractor_cfg.model,
        api_key,
        config.openrouter_base_url,
        client_max_retries=max(config.retry.max_attempts - 1, 0),
    )
    settings = build_model_settings(
        temperature=extractor_cfg.temperature,
        max_output_tokens=extractor_cfg.max_output_tokens,
        reasoning_effort=extractor_cfg.reasoning_effort,
    )

    @function_tool
    def submit_writer_number_extraction(
        metrics: list[WriterMetricExtraction],
    ) -> WriterNumberExtraction:
        """Return one structured metric list unchanged."""
        return WriterNumberExtraction(metrics=metrics)

    return Agent(
        name="Writer Number Extractor",
        instructions=instructions,
        model=model,
        model_settings=settings,
        tools=[submit_writer_number_extraction],
        output_type=WriterNumberExtraction,
        tool_use_behavior="stop_on_first_tool",
    )


def extract_writer_numbers(
    *,
    case: WriterNumericBenchmarkCase,
    candidate_text: str,
    config: AppConfig,
    api_key: str,
    log_llm_payload: bool = False,
) -> WriterNumberExtraction:
    """Extract one structured numeric decision without exposing baseline answers."""
    agent = build_writer_number_extractor_agent(config, api_key)
    payload = {
        "case_id": case.case_id,
        "question": case.question,
        "selected_cities": case.selected_cities,
        "metrics": [
            {
                "metric_id": metric.metric_id,
                "label": metric.label,
                "unit": metric.unit,
                "display_metadata": metric.display_metadata,
            }
            for metric in case.baseline_metrics
        ],
        "candidate_text": candidate_text,
    }
    result = run_agent_sync(
        agent,
        json.dumps(payload, ensure_ascii=False),
        max_turns=config.retry.max_attempts,
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, WriterNumberExtraction):
        return output
    raise ValueError("Writer number extractor did not return structured output.")


__all__ = ["build_writer_number_extractor_agent", "extract_writer_numbers"]
