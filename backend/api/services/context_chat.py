"""LLM-backed context chat over one or more stored run contexts."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from backend.utils.config import AppConfig, get_openrouter_api_key
from backend.utils.tokenization import count_tokens, get_encoding

logger = logging.getLogger(__name__)

DEFAULT_CHAT_PROMPT_TOKEN_CAP = 250_000
MIN_CHAT_PROMPT_TOKEN_CAP = 20_000
CHAT_PROMPT_TOKEN_CAP_ENV = "CHAT_PROMPT_TOKEN_CAP"
DEFAULT_CHAT_PROVIDER_TIMEOUT_SECONDS = 50.0
CHAT_PROVIDER_TIMEOUT_SECONDS_ENV = "CHAT_PROVIDER_TIMEOUT_SECONDS"
CHAT_PROMPT_TOKEN_BUFFER = 2_000
MIN_CONTEXT_SECTION_TOKENS = 1_200
CHAT_CONTEXT_CHUNK_TARGET_TOKENS = 1_200
CHAT_CONTEXT_CHUNK_OVERLAP_TOKENS = 200
CHAT_CONTEXT_MAX_CHUNKS_PER_SOURCE = 24
CHAT_CONTEXT_BASE_CHUNKS_PER_RUN = 2
CHAT_REF_ID_PATTERN = re.compile(r"^ref_[1-9]\d*$")
CHAT_TERM_PATTERN = re.compile(r"[A-Za-z0-9_]{3,}")
CHAT_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "all",
        "also",
        "and",
        "any",
        "are",
        "been",
        "before",
        "between",
        "but",
        "can",
        "for",
        "from",
        "has",
        "have",
        "how",
        "into",
        "its",
        "more",
        "not",
        "now",
        "only",
        "our",
        "out",
        "over",
        "same",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "use",
        "using",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "your",
    }
)


def _resolve_chat_prompt_token_cap() -> int:
    """Resolve chat prompt token cap from environment with safe bounds."""
    raw_value = os.getenv(CHAT_PROMPT_TOKEN_CAP_ENV, "").strip()
    if not raw_value:
        return DEFAULT_CHAT_PROMPT_TOKEN_CAP

    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %d",
            CHAT_PROMPT_TOKEN_CAP_ENV,
            raw_value,
            DEFAULT_CHAT_PROMPT_TOKEN_CAP,
        )
        return DEFAULT_CHAT_PROMPT_TOKEN_CAP

    if parsed < MIN_CHAT_PROMPT_TOKEN_CAP:
        logger.warning(
            "%s=%d is below minimum %d; using minimum.",
            CHAT_PROMPT_TOKEN_CAP_ENV,
            parsed,
            MIN_CHAT_PROMPT_TOKEN_CAP,
        )
        return MIN_CHAT_PROMPT_TOKEN_CAP
    return parsed


CHAT_PROMPT_TOKEN_CAP = _resolve_chat_prompt_token_cap()


def _resolve_chat_provider_timeout_seconds() -> float:
    """Resolve provider timeout from environment with safe minimum."""
    raw_value = os.getenv(CHAT_PROVIDER_TIMEOUT_SECONDS_ENV, "").strip()
    if not raw_value:
        return DEFAULT_CHAT_PROVIDER_TIMEOUT_SECONDS
    try:
        parsed = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %.1f",
            CHAT_PROVIDER_TIMEOUT_SECONDS_ENV,
            raw_value,
            DEFAULT_CHAT_PROVIDER_TIMEOUT_SECONDS,
        )
        return DEFAULT_CHAT_PROVIDER_TIMEOUT_SECONDS
    return max(5.0, parsed)


CHAT_PROVIDER_TIMEOUT_SECONDS = _resolve_chat_provider_timeout_seconds()


@dataclass(frozen=True)
class ChatContextSource:
    """Single context source used to ground chat replies."""

    run_id: str
    question: str
    final_document: str
    context_bundle: dict[str, Any]


def load_context_bundle(path: Path) -> dict[str, Any]:
    """Load context bundle object from JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Context bundle at {path} is not a JSON object.")


def load_final_document(path: Path) -> str:
    """Load final markdown document."""
    return path.read_text(encoding="utf-8")


def generate_context_chat_reply(
    original_question: str,
    contexts: list[dict[str, Any]],
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int = CHAT_PROMPT_TOKEN_CAP,
    api_key_override: str | None = None,
    citation_catalog: list[dict[str, str]] | None = None,
    retry_missing_citation: bool = False,
) -> str:
    """Generate assistant reply grounded in selected run contexts."""
    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )
    client = OpenAI(
        api_key=api_key,
        base_url=config.openrouter_base_url,
        timeout=CHAT_PROVIDER_TIMEOUT_SECONDS,
    )

    normalized_contexts = _normalize_contexts(contexts)
    if not normalized_contexts:
        raise ValueError("At least one chat context source is required.")

    history_limit = max(0, config.chat.max_history_messages)
    bounded_history = _normalize_history(history)
    if history_limit > 0:
        bounded_history = bounded_history[-history_limit:]
    else:
        bounded_history = []

    normalized_citations = _normalize_citation_catalog(citation_catalog)
    prompt_header = _build_system_prompt_header(
        original_question=original_question,
        allowed_ref_ids=[item["ref_id"] for item in normalized_citations],
        retry_missing_citation=retry_missing_citation,
    )
    context_budget = _compute_context_budget(
        prompt_header=prompt_header,
        history=bounded_history,
        user_content=user_content,
        token_cap=token_cap,
    )
    if normalized_citations:
        context_block = _render_citation_catalog_block(normalized_citations)
        included_context_ids = [context.run_id for context in normalized_contexts]
        excluded_context_ids: list[str] = []
    else:
        query_focus = _build_query_focus_text(user_content, bounded_history)
        context_block, included_context_ids, excluded_context_ids = _render_context_block(
            normalized_contexts,
            context_budget,
            query_focus,
        )
    system_prompt = _compose_system_prompt(prompt_header, context_block)

    messages = _build_messages(system_prompt, bounded_history, user_content)
    while _estimate_messages_tokens(messages) > token_cap and bounded_history:
        bounded_history = bounded_history[1:]
        messages = _build_messages(system_prompt, bounded_history, user_content)

    if _estimate_messages_tokens(messages) > token_cap:
        fixed_tokens = _estimate_messages_tokens(
            [{"role": "user", "content": user_content}] + bounded_history
        )
        max_system_tokens = max(token_cap - fixed_tokens - CHAT_PROMPT_TOKEN_BUFFER, 2_000)
        system_prompt = _truncate_to_tokens(system_prompt, max_system_tokens)
        messages = _build_messages(system_prompt, bounded_history, user_content)

    request_kwargs: dict[str, object] = {
        "model": config.chat.model,
        "messages": messages,
    }
    request_kwargs["temperature"] = float(config.chat.temperature)
    if config.chat.max_output_tokens is not None:
        request_kwargs["max_tokens"] = config.chat.max_output_tokens

    logger.info(
        "Context chat request model=%s contexts=%s excluded=%s estimated_prompt_tokens=%d token_cap=%d",
        config.chat.model,
        included_context_ids,
        excluded_context_ids,
        _estimate_messages_tokens(messages),
        token_cap,
    )
    response = client.chat.completions.create(**request_kwargs)
    if not response.choices:
        raise ValueError("Chat model returned no choices.")
    message = response.choices[0].message
    content = message.content or ""
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if hasattr(part, "text") and isinstance(part.text, str):
                text_parts.append(part.text)
        content = "".join(text_parts)
    cleaned = str(content).strip()
    if not cleaned:
        raise ValueError("Chat model returned empty content.")
    return cleaned


def _normalize_contexts(raw_contexts: list[dict[str, Any]]) -> list[ChatContextSource]:
    """Normalize context payloads into typed sources."""
    normalized: list[ChatContextSource] = []
    for item in raw_contexts:
        if not isinstance(item, dict):
            continue
        run_id = item.get("run_id")
        question = item.get("question")
        final_document = item.get("final_document")
        context_bundle = item.get("context_bundle")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        if not isinstance(question, str):
            question = ""
        if not isinstance(final_document, str):
            continue
        if not isinstance(context_bundle, dict):
            continue
        normalized.append(
            ChatContextSource(
                run_id=run_id.strip(),
                question=question.strip(),
                final_document=final_document,
                context_bundle=context_bundle,
            )
        )
    return normalized


def _normalize_citation_catalog(
    raw_catalog: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """Normalize citation catalog entries provided by caller."""
    if not isinstance(raw_catalog, list):
        return []
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_catalog:
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("ref_id", "")).strip()
        if not ref_id or ref_id in seen or not CHAT_REF_ID_PATTERN.fullmatch(ref_id):
            continue
        city_name = str(item.get("city_name", "")).strip()
        quote = str(item.get("quote", "")).strip()
        partial_answer = str(item.get("partial_answer", "")).strip()
        if not quote and not partial_answer:
            continue
        normalized.append(
            {
                "ref_id": ref_id,
                "city_name": city_name,
                "quote": quote,
                "partial_answer": partial_answer,
            }
        )
        seen.add(ref_id)
    return normalized


def _normalize_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize history to user/assistant content pairs."""
    normalized: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        stripped = content.strip()
        if not stripped:
            continue
        normalized.append({"role": role, "content": stripped})
    return normalized


def _render_citation_catalog_block(citation_catalog: list[dict[str, str]]) -> str:
    """Serialize citation catalog into prompt-safe markdown context."""
    lines = ["### Citation evidence catalog"]
    for item in citation_catalog:
        ref_id = item["ref_id"]
        city_name = item["city_name"] or "(unknown city)"
        quote = item["quote"] or "(empty quote)"
        partial_answer = item["partial_answer"] or "(empty partial answer)"
        lines.append(
            "\n".join(
                [
                    f"- [{ref_id}] City: {city_name}",
                    f"  Quote: {quote}",
                    f"  Partial answer: {partial_answer}",
                ]
            )
        )
    return "\n".join(lines)


def _build_system_prompt_header(
    original_question: str,
    allowed_ref_ids: list[str],
    retry_missing_citation: bool,
) -> str:
    """Build stable system prompt prefix."""
    allowed_refs_text = ", ".join(allowed_ref_ids) if allowed_ref_ids else "(none provided)"
    retry_note = (
        "Prior response failed citation requirements. Rewrite the full answer and ensure "
        "every factual claim is immediately followed by one or more valid [ref_n] citations.\n"
        if retry_missing_citation
        else ""
    )
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
        "6. Cite factual claims using only [ref_n] tokens from the allowed reference list.\n"
        "7. Do not invent references and do not use any citation format other than [ref_n].\n"
        f"Allowed references for this turn: {allowed_refs_text}\n\n"
        f"{retry_note}"
        f"Original build question:\n{original_question.strip()}"
    )


def _compose_system_prompt(header: str, context_block: str) -> str:
    """Build final system prompt with context payload."""
    return (
        f"{header}\n\n"
        "Context sources:\n"
        f"{context_block}"
    )


def _compute_context_budget(
    prompt_header: str,
    history: list[dict[str, str]],
    user_content: str,
    token_cap: int,
) -> int:
    """Compute remaining token budget for serialized contexts."""
    fixed_messages = [{"role": "user", "content": user_content}] + history
    fixed_tokens = _estimate_messages_tokens(fixed_messages)
    header_tokens = count_tokens(prompt_header)
    remaining = token_cap - fixed_tokens - header_tokens - CHAT_PROMPT_TOKEN_BUFFER
    return max(remaining, 8_000)


@dataclass(frozen=True)
class _ScoredContextChunk:
    """Single scored context chunk candidate used for prompt pooling."""

    run_id: str
    run_question: str
    source_label: str
    language: str
    ordinal: int
    content: str
    score: float


def _build_query_focus_text(user_content: str, history: list[dict[str, str]]) -> str:
    """Build compact query focus text from latest user inputs."""
    recent_user_messages = [
        item["content"] for item in history if item.get("role") == "user"
    ]
    recent_user_messages = recent_user_messages[-2:]
    return "\n".join(recent_user_messages + [user_content.strip()])


def _render_context_block(
    contexts: list[ChatContextSource],
    max_tokens: int,
    query_focus: str,
) -> tuple[str, list[str], list[str]]:
    """Serialize contexts fully when possible, otherwise switch to pooled excerpts."""
    full_sections = [
        _serialize_context(index, context)
        for index, context in enumerate(contexts, start=1)
    ]
    full_tokens = sum(count_tokens(section) for section in full_sections)
    if full_tokens <= max_tokens:
        return (
            "\n\n".join(full_sections),
            [context.run_id for context in contexts],
            [],
        )
    return _render_pooled_context_block(contexts, max_tokens, query_focus)


def _render_pooled_context_block(
    contexts: list[ChatContextSource],
    max_tokens: int,
    query_focus: str,
) -> tuple[str, list[str], list[str]]:
    """Render a pooled context block with high-signal excerpts from each run."""
    query_terms = _extract_terms(query_focus)
    candidates_by_run: dict[str, list[_ScoredContextChunk]] = {}
    for context in contexts:
        candidates_by_run[context.run_id] = _build_scored_chunks(context, query_terms)

    sections: list[str] = []
    selected_ids: list[str] = []
    remaining = max_tokens
    used_chunk_ids: set[tuple[str, str, int]] = set()

    for context in contexts:
        run_candidates = candidates_by_run.get(context.run_id, [])
        picked_for_run = 0
        for chunk in run_candidates:
            chunk_id = (chunk.run_id, chunk.source_label, chunk.ordinal)
            if chunk_id in used_chunk_ids:
                continue
            serialized = _serialize_chunk(chunk)
            serialized_tokens = count_tokens(serialized)
            if serialized_tokens <= remaining:
                sections.append(serialized)
                selected_ids.append(chunk.run_id)
                used_chunk_ids.add(chunk_id)
                remaining -= serialized_tokens
                picked_for_run += 1
            elif picked_for_run == 0 and remaining >= MIN_CONTEXT_SECTION_TOKENS:
                sections.append(_truncate_to_tokens(serialized, remaining))
                selected_ids.append(chunk.run_id)
                used_chunk_ids.add(chunk_id)
                remaining = 0
                picked_for_run += 1
                break
            if picked_for_run >= CHAT_CONTEXT_BASE_CHUNKS_PER_RUN:
                break
        if remaining < MIN_CONTEXT_SECTION_TOKENS:
            break

    remaining_candidates: list[_ScoredContextChunk] = []
    for run_candidates in candidates_by_run.values():
        for chunk in run_candidates:
            chunk_id = (chunk.run_id, chunk.source_label, chunk.ordinal)
            if chunk_id not in used_chunk_ids:
                remaining_candidates.append(chunk)
    remaining_candidates.sort(key=lambda chunk: (-chunk.score, chunk.run_id, chunk.ordinal))

    for chunk in remaining_candidates:
        if remaining < MIN_CONTEXT_SECTION_TOKENS:
            break
        serialized = _serialize_chunk(chunk)
        serialized_tokens = count_tokens(serialized)
        if serialized_tokens > remaining:
            continue
        sections.append(serialized)
        selected_ids.append(chunk.run_id)
        remaining -= serialized_tokens

    if not sections:
        pooled_text = (
            "No context sources could fit within token budget. "
            "Ask the user to reduce the selected contexts."
        )
        return pooled_text, [], [context.run_id for context in contexts]

    included_ids = _dedupe_preserve_order(selected_ids)
    included_set = set(included_ids)
    excluded_ids = [context.run_id for context in contexts if context.run_id not in included_set]
    sections.insert(
        0,
        "Context pooling mode: full documents plus bundles exceeded token budget, "
        "so this prompt includes top-ranked excerpts selected from all chosen runs.",
    )
    if excluded_ids:
        sections.append(
            "Runs excluded due to token budget: "
            + ", ".join(f"[run:{run_id}]" for run_id in excluded_ids)
        )
    return "\n\n".join(sections), included_ids, excluded_ids


def _build_scored_chunks(
    context: ChatContextSource,
    query_terms: set[str],
) -> list[_ScoredContextChunk]:
    """Build scored chunk candidates from final markdown and context bundle."""
    chunks: list[_ScoredContextChunk] = []
    final_chunks = _chunk_text_by_tokens(
        context.final_document.strip(),
        target_tokens=CHAT_CONTEXT_CHUNK_TARGET_TOKENS,
        overlap_tokens=CHAT_CONTEXT_CHUNK_OVERLAP_TOKENS,
        max_chunks=CHAT_CONTEXT_MAX_CHUNKS_PER_SOURCE,
    )
    for index, chunk_text in enumerate(final_chunks, start=1):
        chunks.append(
            _ScoredContextChunk(
                run_id=context.run_id,
                run_question=context.question,
                source_label="final_document",
                language="markdown",
                ordinal=index,
                content=chunk_text,
                score=_score_chunk(chunk_text, query_terms, base_weight=1.0),
            )
        )

    serialized_bundle = json.dumps(
        context.context_bundle,
        ensure_ascii=True,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    bundle_chunks = _chunk_text_by_tokens(
        serialized_bundle,
        target_tokens=CHAT_CONTEXT_CHUNK_TARGET_TOKENS,
        overlap_tokens=CHAT_CONTEXT_CHUNK_OVERLAP_TOKENS,
        max_chunks=CHAT_CONTEXT_MAX_CHUNKS_PER_SOURCE,
    )
    for index, chunk_text in enumerate(bundle_chunks, start=1):
        chunks.append(
            _ScoredContextChunk(
                run_id=context.run_id,
                run_question=context.question,
                source_label="context_bundle",
                language="json",
                ordinal=index,
                content=chunk_text,
                score=_score_chunk(chunk_text, query_terms, base_weight=0.9),
            )
        )

    chunks.sort(key=lambda chunk: (-chunk.score, chunk.source_label, chunk.ordinal))
    return chunks


def _serialize_chunk(chunk: _ScoredContextChunk) -> str:
    """Serialize one selected context chunk."""
    return (
        f"### Source [run:{chunk.run_id}] {chunk.source_label} excerpt {chunk.ordinal}\n"
        f"Run question: {chunk.run_question or '(not provided)'}\n"
        f"```{chunk.language}\n"
        f"{chunk.content}\n"
        "```"
    )


def _serialize_context(index: int, context: ChatContextSource) -> str:
    """Serialize one context source."""
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


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return values de-duplicated while preserving input order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _extract_terms(value: str) -> set[str]:
    """Extract normalized keyword terms for simple lexical scoring."""
    terms: set[str] = set()
    for match in CHAT_TERM_PATTERN.findall(value.lower()):
        if match in CHAT_STOPWORDS:
            continue
        terms.add(match)
    return terms


def _score_chunk(content: str, query_terms: set[str], base_weight: float) -> float:
    """Score one chunk against query terms with overlap and frequency signals."""
    if not query_terms:
        return base_weight
    chunk_terms = _extract_terms(content)
    overlap = len(chunk_terms & query_terms)
    lowered_content = content.lower()
    frequency_score = 0.0
    for term in query_terms:
        frequency_score += min(3.0, float(lowered_content.count(term)))
    return base_weight + (overlap * 2.5) + (frequency_score * 0.35)


def _chunk_text_by_tokens(
    value: str,
    target_tokens: int,
    overlap_tokens: int,
    max_chunks: int,
) -> list[str]:
    """Split text into overlapping token chunks for retrieval-style pooling."""
    stripped = value.strip()
    if not stripped:
        return []
    encoding = get_encoding()
    tokens = encoding.encode(stripped)
    if not tokens:
        return []
    if len(tokens) <= target_tokens:
        return [stripped]

    step = max(1, target_tokens - overlap_tokens)
    chunks: list[str] = []
    start_index = 0
    while start_index < len(tokens) and len(chunks) < max_chunks:
        end_index = min(start_index + target_tokens, len(tokens))
        chunk_text = encoding.decode(tokens[start_index:end_index]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end_index >= len(tokens):
            break
        start_index += step
    return chunks


def _build_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    user_content: str,
) -> list[dict[str, str]]:
    """Build chat completion message list."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def _estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate token usage for chat messages."""
    total = 0
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        total += count_tokens(f"{role}\n{content}") + 6
    return total + 2


def _truncate_to_tokens(value: str, max_tokens: int) -> str:
    """Truncate text to token budget, preserving leading content."""
    if max_tokens <= 0:
        return ""
    encoding = get_encoding()
    tokens = encoding.encode(value)
    if len(tokens) <= max_tokens:
        return value

    suffix = "\n\n[truncated due to prompt token budget]"
    suffix_tokens = encoding.encode(suffix)
    if max_tokens <= len(suffix_tokens):
        return encoding.decode(tokens[:max_tokens])
    head_tokens = tokens[: max_tokens - len(suffix_tokens)]
    return encoding.decode(head_tokens) + suffix


__all__ = [
    "ChatContextSource",
    "CHAT_PROMPT_TOKEN_CAP",
    "generate_context_chat_reply",
    "load_context_bundle",
    "load_final_document",
]
