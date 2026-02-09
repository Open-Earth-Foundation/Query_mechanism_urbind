"""LLM-backed context chat over one or more stored run contexts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.utils.config import AppConfig, get_openrouter_api_key
from app.utils.tokenization import count_tokens, get_encoding

logger = logging.getLogger(__name__)

CHAT_PROMPT_TOKEN_CAP = 300_000
CHAT_PROMPT_TOKEN_BUFFER = 2_000
MIN_CONTEXT_SECTION_TOKENS = 1_200


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
) -> str:
    """Generate assistant reply grounded in selected run contexts."""
    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )
    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)

    normalized_contexts = _normalize_contexts(contexts)
    if not normalized_contexts:
        raise ValueError("At least one chat context source is required.")

    history_limit = max(0, config.chat.max_history_messages)
    bounded_history = _normalize_history(history)
    if history_limit > 0:
        bounded_history = bounded_history[-history_limit:]
    else:
        bounded_history = []

    prompt_header = _build_system_prompt_header(original_question)
    context_budget = _compute_context_budget(
        prompt_header=prompt_header,
        history=bounded_history,
        user_content=user_content,
        token_cap=token_cap,
    )
    context_block, included_context_ids, excluded_context_ids = _render_context_block(
        normalized_contexts,
        context_budget,
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
    if config.chat.temperature is not None:
        request_kwargs["temperature"] = config.chat.temperature
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


def _build_system_prompt_header(original_question: str) -> str:
    """Build stable system prompt prefix."""
    return (
        "You are the Context Analyst for a document-builder workflow.\n"
        "Your job is to answer follow-up questions using only the supplied context sources.\n"
        "Each source comes from a completed run and includes: final markdown document + context bundle JSON.\n\n"
        "Rules:\n"
        "1. Ground every factual claim in provided context sources.\n"
        "2. If information is missing or uncertain, say so clearly.\n"
        "3. Compare sources when useful and call out contradictions.\n"
        "4. Use concise, practical markdown output.\n"
        "5. Never mention internal paths or backend implementation details.\n"
        "6. When citing evidence, reference source run ids like [run:20260209_1757].\n\n"
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


def _render_context_block(
    contexts: list[ChatContextSource],
    max_tokens: int,
) -> tuple[str, list[str], list[str]]:
    """Serialize selected contexts with token-bound truncation."""
    sections: list[str] = []
    included_ids: list[str] = []
    excluded_ids: list[str] = []
    remaining = max_tokens

    for index, context in enumerate(contexts, start=1):
        serialized = _serialize_context(index, context)
        serialized_tokens = count_tokens(serialized)
        if serialized_tokens <= remaining:
            sections.append(serialized)
            included_ids.append(context.run_id)
            remaining -= serialized_tokens
            continue
        if remaining >= MIN_CONTEXT_SECTION_TOKENS:
            truncated = _truncate_to_tokens(serialized, remaining)
            sections.append(truncated)
            included_ids.append(context.run_id)
            remaining = 0
            continue
        excluded_ids.append(context.run_id)

    if not sections:
        sections.append(
            "No context sources could fit within token budget. "
            "Ask the user to reduce the selected contexts."
        )

    if excluded_ids:
        sections.append(
            "Excluded due to token budget: "
            + ", ".join(f"[run:{run_id}]" for run_id in excluded_ids)
        )

    return "\n\n".join(sections), included_ids, excluded_ids


def _serialize_context(index: int, context: ChatContextSource) -> str:
    """Serialize one context source."""
    serialized_bundle = json.dumps(
        context.context_bundle,
        ensure_ascii=True,
        default=str,
        indent=2,
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
