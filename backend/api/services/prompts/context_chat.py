"""Prompt composition helpers for context chat request assembly."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from string import Template

from backend.api.services.models import ChatContextSource
from backend.utils.prompts import load_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@lru_cache(maxsize=None)
def _load_prompt_template(template_name: str) -> Template:
    """Load one markdown prompt template from the shared prompt directory."""
    prompt_path = _PROMPTS_DIR / template_name
    return Template(load_prompt(prompt_path))


def _render_prompt_template(template_name: str, **substitutions: object) -> str:
    """Render one markdown prompt template with stringified substitutions."""
    normalized_substitutions = {
        key: str(value)
        for key, value in substitutions.items()
    }
    return _load_prompt_template(template_name).substitute(normalized_substitutions)


def build_system_prompt_header(
    original_question: str,
    retry_missing_citation: bool,
) -> str:
    """Build the stable system prompt prefix for one context-chat turn."""
    retry_note_block = ""
    if retry_missing_citation:
        retry_note_block = (
            "- Prior response failed citation requirements. Rewrite the full answer and ensure "
            "every factual claim is immediately followed by one or more valid [ref_n] citations."
        )
    return _render_prompt_template(
        "context_chat_system.md",
        original_question=original_question.strip(),
        retry_note_block=retry_note_block,
    )


def compose_system_prompt(header: str, context_block: str) -> str:
    """Build the final direct-answer system prompt with the serialized context."""
    return f"{header}\n\nContext sources:\n{context_block}"


def compose_evidence_map_prompt(
    *,
    prompt_header: str,
    evidence_block: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """Build the system prompt for one evidence map step."""
    return _render_prompt_template(
        "context_chat_evidence_map_system.md",
        prompt_header=prompt_header,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        evidence_block=evidence_block or "- No evidence items available in this chunk.",
    )


def compose_evidence_reduce_prompt(
    *,
    prompt_header: str,
    analyses_block: str,
    stage_index: int,
    batch_index: int,
    batch_count: int,
) -> str:
    """Build the system prompt for one reduce batch."""
    return _render_prompt_template(
        "context_chat_evidence_reduce_system.md",
        prompt_header=prompt_header,
        stage_index=stage_index,
        batch_index=batch_index,
        batch_count=batch_count,
        analyses_block=analyses_block,
    )


def compose_empty_evidence_prompt(prompt_header: str) -> str:
    """Build the overflow fallback prompt when no compact evidence items exist."""
    return _render_prompt_template(
        "context_chat_empty_evidence_system.md",
        prompt_header=prompt_header,
    )


def render_evidence_items_block(evidence_items: list[dict[str, str]]) -> str:
    """Render compact evidence items into a prompt-safe markdown block."""
    if not evidence_items:
        return "- No evidence items available."
    lines = ["### Evidence items"]
    for item in evidence_items:
        lines.append(
            "\n".join(
                [
                    f"- [{item['ref_id']}] City: {item['city_name'] or '(unknown city)'}",
                    f"  Quote: {item['quote'] or '(empty quote)'}",
                    f"  Partial answer: {item['partial_answer'] or '(empty partial answer)'}",
                ]
            )
        )
    return "\n".join(lines)


def render_partial_answers_block(partial_answers: list[str]) -> str:
    """Render partial analyses into a prompt-safe markdown block."""
    lines: list[str] = []
    for index, answer in enumerate(partial_answers, start=1):
        lines.append(
            "\n".join(
                [
                    f"### Partial analysis {index}",
                    answer.strip() or "(empty partial analysis)",
                ]
            )
        )
    return "\n\n".join(lines)


def render_citation_catalog_block(citation_catalog: list[dict[str, str]]) -> str:
    """Serialize the citation catalog into prompt-safe markdown context."""
    if not citation_catalog:
        return (
            "### Citation evidence catalog\n"
            "- No citation entries fit within the prompt token budget for this turn."
        )
    lines = ["### Citation evidence catalog"]
    for item in citation_catalog:
        lines.append(
            "\n".join(
                [
                    f"- [{item['ref_id']}] City: {item['city_name'] or '(unknown city)'}",
                    f"  Quote: {item['quote'] or '(empty quote)'}",
                    f"  Partial answer: {item['partial_answer'] or '(empty partial answer)'}",
                ]
            )
        )
    return "\n".join(lines)


def serialize_all_contexts(contexts: list[ChatContextSource]) -> str:
    """Serialize all context sources into a single prompt block."""
    sections = [
        serialize_context(index, context)
        for index, context in enumerate(contexts, start=1)
    ]
    return "\n\n".join(sections)


def serialize_context(index: int, context: ChatContextSource) -> str:
    """Serialize one context source for the fallback prompt path."""
    serialized_bundle = json.dumps(
        context.context_bundle,
        ensure_ascii=True,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"### Source {index} [run:{context.run_id}]\n"
        f"Run question: {context.question or '(not provided)'}\n\n"
        "Final document markdown:\n"
        "```markdown\n"
        f"{context.final_document.strip()}\n"
        "```\n\n"
        "Context bundle JSON:\n"
        "```json\n"
        f"{serialized_bundle}\n"
        "```"
    )


__all__ = [
    "build_system_prompt_header",
    "compose_empty_evidence_prompt",
    "compose_evidence_map_prompt",
    "compose_evidence_reduce_prompt",
    "compose_system_prompt",
    "render_citation_catalog_block",
    "render_evidence_items_block",
    "render_partial_answers_block",
    "serialize_all_contexts",
    "serialize_context",
]
