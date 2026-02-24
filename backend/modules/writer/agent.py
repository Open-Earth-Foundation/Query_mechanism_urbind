from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from agents import Agent, function_tool

from backend.modules.writer.models import WriterOutput
from backend.services.agents import (
    build_model_settings,
    build_openrouter_model,
    run_agent_sync,
)
from backend.utils.config import AppConfig
from backend.utils.prompts import load_prompt

logger = logging.getLogger(__name__)
_VALID_REF_ID_PATTERN = re.compile(r"^ref_[1-9]\d*$")
_CITATION_PATTERN = re.compile(r"\[(ref_\d+)\]")
_RAW_REFERENCE_TOKEN_PATTERN = re.compile(r"\[(ref_[^\]]+)\]")


def _extract_markdown_bundle(context_bundle: dict[str, object]) -> dict[str, object]:
    """Return markdown bundle from context or an empty dict."""
    markdown_bundle = context_bundle.get("markdown")
    if isinstance(markdown_bundle, dict):
        return markdown_bundle
    return {}


def _extract_excerpt_count(markdown_bundle: dict[str, object]) -> int:
    """Resolve excerpt count with list-length fallback."""
    value = markdown_bundle.get("excerpt_count")
    if isinstance(value, int):
        return max(value, 0)
    excerpts = markdown_bundle.get("excerpts")
    if not isinstance(excerpts, list):
        return 0
    return len([excerpt for excerpt in excerpts if isinstance(excerpt, dict)])


def _extract_expected_ref_ids(context_bundle: dict[str, object]) -> set[str]:
    """Collect valid reference ids from markdown excerpts."""
    markdown_bundle = _extract_markdown_bundle(context_bundle)
    excerpts = markdown_bundle.get("excerpts")
    if not isinstance(excerpts, list):
        return set()

    expected_ids: set[str] = set()
    for excerpt in excerpts:
        if not isinstance(excerpt, dict):
            continue
        ref_id = excerpt.get("ref_id")
        if not isinstance(ref_id, str):
            continue
        candidate = ref_id.strip()
        if _VALID_REF_ID_PATTERN.fullmatch(candidate):
            expected_ids.add(candidate)
    return expected_ids


def _validate_writer_citations(content: str, context_bundle: dict[str, object]) -> None:
    """Emit warnings when writer citations are missing, malformed, or unknown."""
    markdown_bundle = _extract_markdown_bundle(context_bundle)
    excerpt_count = _extract_excerpt_count(markdown_bundle)
    if excerpt_count <= 0:
        return

    expected_ref_ids = _extract_expected_ref_ids(context_bundle)
    cited_ref_ids = {match.group(1) for match in _CITATION_PATTERN.finditer(content)}
    raw_reference_tokens = {
        match.group(1) for match in _RAW_REFERENCE_TOKEN_PATTERN.finditer(content)
    }
    malformed_tokens = sorted(
        token
        for token in raw_reference_tokens
        if not _VALID_REF_ID_PATTERN.fullmatch(token)
    )

    if not cited_ref_ids:
        logger.warning(
            "Writer output contains no [ref_n] citations despite excerpt_count=%d.",
            excerpt_count,
        )

    if not expected_ref_ids:
        logger.warning(
            "Writer context has excerpt_count=%d but no valid ref ids in excerpts.",
            excerpt_count,
        )

    unknown_refs = sorted(ref_id for ref_id in cited_ref_ids if ref_id not in expected_ref_ids)
    if unknown_refs:
        logger.warning(
            "Writer output cites unknown reference ids: %s",
            ", ".join(unknown_refs),
        )

    if malformed_tokens:
        logger.warning(
            "Writer output contains malformed reference tokens: %s",
            ", ".join(malformed_tokens),
        )


def build_writer_agent(config: AppConfig, api_key: str) -> Agent:
    """Build the writer agent."""
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "writer_system.md"
    instructions = load_prompt(prompt_path)
    model = build_openrouter_model(
        config.writer.model, api_key, config.openrouter_base_url
    )
    settings = build_model_settings(
        config.writer.temperature, config.writer.max_output_tokens
    )

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
    context_bundle: dict[str, object],
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
        json.dumps(payload, ensure_ascii=False),
        log_llm_payload=log_llm_payload,
    )
    output = result.final_output
    if isinstance(output, WriterOutput):
        _validate_writer_citations(output.content, context_bundle)
        return output
    raise ValueError("Writer did not return structured output.")


__all__ = ["build_writer_agent", "write_markdown"]
