"""Prompt composition helpers for context chat request assembly."""

from __future__ import annotations

import json

from backend.api.services.models import ChatContextSource


def build_system_prompt_header(
    original_question: str,
    retry_missing_citation: bool,
) -> str:
    """Build the stable system prompt prefix for one context-chat turn."""
    if retry_missing_citation:
        retry_note = (
            "Prior response failed citation requirements. Rewrite the full answer and ensure "
            "every factual claim is immediately followed by one or more valid [ref_n] citations.\n"
        )
    else:
        retry_note = ""
    stripped_original_question = original_question.strip()
    return (
        "You are the Context Analyst for a document-builder workflow.\n"
        "Your job is to answer follow-up questions using only the supplied context sources.\n"
        "Each source comes from a completed run and includes curated citation evidence.\n\n"
        "Rules:\n"
        "1. Ground every factual claim in provided context sources.\n"
        "2. If information is missing or uncertain, say so clearly.\n"
        "3. Compare sources when useful and call out contradictions.\n"
        "4. Always respond in valid markdown. Prefer headings, bullets, and tables for numeric data.\n"
        "5. Never mention internal paths or backend implementation details.\n"
        "6. If a citation evidence catalog is provided, cite factual claims using only [ref_n] tokens present in that catalog.\n"
        "7. Do not invent references and do not use any citation format other than [ref_n].\n"
        "8. If no citation evidence catalog entries are available for this turn, explain that you cannot provide a fully grounded cited answer.\n\n"
        "9. If arithmetic is needed and calculator tools are available, use them instead of mental math.\n\n"
        f"{retry_note}"
        f"Original build question:\n{stripped_original_question}"
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
    return (
        f"{prompt_header}\n\n"
        f"You are analyzing evidence chunk {chunk_index} of {total_chunks} for a larger map-reduce answer.\n"
        "Use only the evidence items below.\n"
        "Cite every factual claim with one or more [ref_n] tokens that appear in this chunk.\n"
        "Do not invent citations and do not use any citation format other than [ref_n].\n"
        "If this chunk is not relevant to the latest user question, say so briefly.\n\n"
        "Evidence chunk:\n"
        f"{evidence_block or '- No evidence items available in this chunk.'}"
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
    return (
        f"{prompt_header}\n\n"
        f"You are merging map-reduce summaries at reduce stage {stage_index}, batch {batch_index} of {batch_count}.\n"
        "Use only facts and [ref_n] citations already present in the partial analyses below.\n"
        "Preserve valid citations on factual claims, merge duplicates, and remove contradictions when later analyses correct earlier ones.\n"
        "Do not invent new citations and do not drop necessary citations.\n\n"
        "Partial analyses:\n"
        f"{analyses_block}"
    )


def compose_empty_evidence_prompt(prompt_header: str) -> str:
    """Build the overflow fallback prompt when no compact evidence items exist."""
    return (
        f"{prompt_header}\n\n"
        "No compact evidence items are available for the selected context sources in overflow mode.\n"
        "Explain briefly that the current saved context does not provide extractable evidence for a grounded answer."
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
