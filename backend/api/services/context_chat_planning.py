"""Planning and prompt-window helpers for context chat."""

from __future__ import annotations

from bisect import bisect_right
import re
from typing import Any, Literal

from backend.api.services.models import (
    ChatContextSource,
    CitationCatalogTokenCache,
    ContextChatPlan,
    ContextWindowEstimate,
    PreparedContextChatRequest,
)
import backend.api.services.prompts.context_chat as context_chat_prompts
from backend.api.services.utils.context_chat import (
    build_messages as _build_messages,
    estimate_messages_tokens as _estimate_messages_tokens,
)
from backend.utils.city_normalization import format_city_stem
from backend.utils.config import AppConfig
from backend.utils.tokenization import count_tokens

CHAT_REF_ID_PATTERN = re.compile(r"^ref_[1-9]\d*$")


def resolve_chat_token_cap(config: AppConfig) -> int:
    """Return the configured prompt token cap for context chat."""
    return config.chat.max_context_total_tokens


def plan_context_chat_request(
    original_question: str,
    contexts: list[dict[str, Any]],
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int = 0,
    citation_catalog: list[dict[str, str]] | None = None,
    citation_prefix_tokens: list[int] | None = None,
    retry_missing_citation: bool = False,
) -> ContextChatPlan:
    """Plan whether one context-chat request can stay direct or must split."""
    prepared = _prepare_context_chat_request(
        original_question=original_question,
        contexts=contexts,
        history=history,
        user_content=user_content,
        config=config,
        token_cap=token_cap,
        citation_catalog=citation_catalog,
        citation_prefix_tokens=citation_prefix_tokens,
        retry_missing_citation=retry_missing_citation,
    )
    return ContextChatPlan(
        mode=prepared.mode,
        context_ids=prepared.context_ids,
        resolved_token_cap=prepared.resolved_cap,
        effective_token_cap=prepared.effective_token_cap,
        estimated_prompt_tokens=prepared.estimated_prompt_tokens,
        context_tokens=prepared.context_tokens,
        split_reason=prepared.split_reason,
        context_window_kind=prepared.context_window_kind,
        context_block_tokens=prepared.context_block_tokens,
        prompt_header_tokens=prepared.prompt_header_tokens,
        history_tokens=prepared.history_tokens,
        user_tokens=prepared.user_tokens,
        citation_catalog_entry_count=prepared.citation_catalog_entry_count,
        fitted_citation_entry_count=prepared.fitted_citation_entry_count,
        fitted_citation_ref_ids=prepared.fitted_citation_ref_ids,
    )


def estimate_context_window(
    original_question: str,
    contexts: list[dict[str, Any]],
    config: AppConfig,
    token_cap: int = 0,
    citation_catalog: list[dict[str, str]] | None = None,
) -> ContextWindowEstimate:
    """Estimate the context-window size for one context-chat request."""
    prepared = _prepare_context_chat_request(
        original_question=original_question,
        contexts=contexts,
        history=[],
        user_content="",
        config=config,
        token_cap=token_cap,
        citation_catalog=citation_catalog,
        citation_prefix_tokens=None,
        retry_missing_citation=False,
    )
    return ContextWindowEstimate(
        mode=prepared.mode,
        resolved_token_cap=prepared.resolved_cap,
        effective_token_cap=prepared.effective_token_cap,
        context_window_kind=prepared.context_window_kind,
        context_window_tokens=_estimate_full_context_window_tokens(
            normalized_contexts=prepared.normalized_contexts,
            normalized_citations=prepared.normalized_citations,
            context_window_kind=prepared.context_window_kind,
        ),
        fitted_context_window_tokens=prepared.context_block_tokens,
        estimated_prompt_tokens=prepared.estimated_prompt_tokens,
        citation_catalog_entry_count=prepared.citation_catalog_entry_count,
        fitted_citation_entry_count=prepared.fitted_citation_entry_count,
    )


def build_citation_catalog_from_contexts(
    contexts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the prompt citation catalog from stored context bundles."""
    normalized_contexts = _normalize_contexts(contexts)
    synthetic_index = 1
    citation_catalog: list[dict[str, str]] = []
    for context in normalized_contexts:
        markdown_payload = context.context_bundle.get("markdown")
        if not isinstance(markdown_payload, dict):
            continue
        raw_excerpts = markdown_payload.get("excerpts")
        if not isinstance(raw_excerpts, list):
            continue
        for excerpt_index, raw_excerpt in enumerate(raw_excerpts):
            if not isinstance(raw_excerpt, dict):
                continue
            source_ref_id = str(raw_excerpt.get("ref_id", "")).strip()
            if not source_ref_id:
                source_ref_id = f"ref_{excerpt_index + 1}"
            if not CHAT_REF_ID_PATTERN.fullmatch(source_ref_id):
                continue
            quote = str(raw_excerpt.get("quote", "")).strip()
            partial_answer = str(raw_excerpt.get("partial_answer", "")).strip()
            if not quote and not partial_answer:
                continue
            citation_catalog.append(
                {
                    "ref_id": f"ref_{synthetic_index}",
                    "city_name": format_city_stem(
                        str(raw_excerpt.get("city_name", "")).strip()
                    ),
                    "quote": quote,
                    "partial_answer": partial_answer,
                }
            )
            synthetic_index += 1
    return citation_catalog


def build_citation_catalog_token_cache(
    citation_catalog: list[dict[str, str]] | None,
) -> CitationCatalogTokenCache:
    """Build exact prefix-token accounting for one ordered citation catalog."""
    normalized_catalog = _normalize_citation_catalog(citation_catalog)
    if not normalized_catalog:
        return CitationCatalogTokenCache(
            ordered_ref_ids=[],
            prefix_tokens=[],
            total_tokens=count_tokens(context_chat_prompts.render_citation_catalog_block([])),
        )
    prefix_tokens: list[int] = []
    for index in range(1, len(normalized_catalog) + 1):
        prefix_tokens.append(
            count_tokens(context_chat_prompts.render_citation_catalog_block(normalized_catalog[:index]))
        )
    return CitationCatalogTokenCache(
        ordered_ref_ids=[item["ref_id"] for item in normalized_catalog],
        prefix_tokens=prefix_tokens,
        total_tokens=prefix_tokens[-1],
    )


def _prepare_context_chat_request(
    *,
    original_question: str,
    contexts: list[dict[str, Any]],
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int,
    citation_catalog: list[dict[str, str]] | None,
    citation_prefix_tokens: list[int] | None,
    retry_missing_citation: bool,
) -> PreparedContextChatRequest:
    """Normalize one chat request and resolve the direct-vs-split strategy."""
    resolved_cap = int(token_cap) if token_cap > 0 else resolve_chat_token_cap(config)
    effective_token_cap = max(
        config.chat.min_prompt_token_cap,
        min(resolved_cap, config.chat.max_context_total_tokens),
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

    prompt_header = context_chat_prompts.build_system_prompt_header(
        original_question=original_question,
        retry_missing_citation=retry_missing_citation,
    )
    normalized_citations = _normalize_citation_catalog(citation_catalog)
    context_ids = [context.run_id for context in normalized_contexts]
    user_tokens = _estimate_messages_tokens([{"role": "user", "content": user_content}])
    prompt_header_tokens = count_tokens(prompt_header)
    bounded_history_tokens = _estimate_messages_tokens(bounded_history)

    if normalized_citations:
        fitted_citations = _fit_citation_catalog_to_budget(
            citation_catalog=normalized_citations,
            prompt_header=prompt_header,
            history=bounded_history,
            user_content=user_content,
            token_cap=effective_token_cap,
            prompt_token_buffer=config.chat.prompt_token_buffer,
            citation_prefix_tokens=citation_prefix_tokens,
        )
        fitted_ref_ids = [item["ref_id"] for item in fitted_citations]
        fitted_catalog_tokens = count_tokens(
            context_chat_prompts.render_citation_catalog_block(fitted_citations)
        )
        if len(fitted_citations) < len(normalized_citations):
            return PreparedContextChatRequest(
                prompt_header=prompt_header,
                normalized_contexts=normalized_contexts,
                normalized_citations=normalized_citations,
                bounded_history=bounded_history,
                user_content=user_content,
                resolved_cap=resolved_cap,
                effective_token_cap=effective_token_cap,
                context_ids=context_ids,
                mode="split",
                split_reason=(
                    "Citation evidence catalog exceeds direct prompt budget and requires "
                    "overflow map-reduce."
                ),
                context_window_kind="citation_catalog",
                context_block_tokens=fitted_catalog_tokens,
                prompt_header_tokens=prompt_header_tokens,
                history_tokens=bounded_history_tokens,
                user_tokens=user_tokens,
                citation_catalog_entry_count=len(normalized_citations),
                fitted_citation_entry_count=len(fitted_citations),
                fitted_citation_ref_ids=fitted_ref_ids,
            )
        return _prepare_direct_prompt_request(
            prompt_header=prompt_header,
            context_block=context_chat_prompts.render_citation_catalog_block(
                fitted_citations
            ),
            normalized_contexts=normalized_contexts,
            normalized_citations=normalized_citations,
            bounded_history=bounded_history,
            user_content=user_content,
            resolved_cap=resolved_cap,
            effective_token_cap=effective_token_cap,
            context_ids=context_ids,
            overflow_reason_prefix="Chat context exceeds token budget after trimming",
            context_window_kind="citation_catalog",
            context_block_tokens=fitted_catalog_tokens,
            prompt_header_tokens=prompt_header_tokens,
            user_tokens=user_tokens,
            citation_catalog_entry_count=len(normalized_citations),
            fitted_citation_entry_count=len(fitted_citations),
            fitted_citation_ref_ids=fitted_ref_ids,
        )

    serialized_contexts = context_chat_prompts.serialize_all_contexts(normalized_contexts)
    context_tokens = count_tokens(serialized_contexts)
    if context_tokens > config.chat.multi_pass_threshold_tokens:
        return PreparedContextChatRequest(
            prompt_header=prompt_header,
            normalized_contexts=normalized_contexts,
            normalized_citations=normalized_citations,
            bounded_history=bounded_history,
            user_content=user_content,
            resolved_cap=resolved_cap,
            effective_token_cap=effective_token_cap,
            context_ids=context_ids,
            mode="split",
            context_tokens=context_tokens,
            split_reason=(
                "Serialized context exceeds direct prompt threshold and requires "
                "overflow map-reduce."
            ),
            context_window_kind="serialized_contexts",
            context_block_tokens=context_tokens,
            prompt_header_tokens=prompt_header_tokens,
            history_tokens=bounded_history_tokens,
            user_tokens=user_tokens,
        )
    return _prepare_direct_prompt_request(
        prompt_header=prompt_header,
        context_block=serialized_contexts,
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
        bounded_history=bounded_history,
        user_content=user_content,
        resolved_cap=resolved_cap,
        effective_token_cap=effective_token_cap,
        context_ids=context_ids,
        context_tokens=context_tokens,
        overflow_reason_prefix="Chat context exceeds token budget after trimming history",
        context_window_kind="serialized_contexts",
        context_block_tokens=context_tokens,
        prompt_header_tokens=prompt_header_tokens,
        user_tokens=user_tokens,
    )


def _prepare_direct_prompt_request(
    *,
    prompt_header: str,
    context_block: str,
    normalized_contexts: list[ChatContextSource],
    normalized_citations: list[dict[str, str]],
    bounded_history: list[dict[str, str]],
    user_content: str,
    resolved_cap: int,
    effective_token_cap: int,
    context_ids: list[str],
    overflow_reason_prefix: str,
    context_tokens: int | None = None,
    context_window_kind: Literal["citation_catalog", "serialized_contexts"],
    context_block_tokens: int,
    prompt_header_tokens: int,
    user_tokens: int,
    citation_catalog_entry_count: int | None = None,
    fitted_citation_entry_count: int | None = None,
    fitted_citation_ref_ids: list[str] | None = None,
) -> PreparedContextChatRequest:
    """Build the direct prompt shape or return a split-mode overflow reason."""
    system_prompt = context_chat_prompts.compose_system_prompt(prompt_header, context_block)
    working_history = list(bounded_history)
    messages = _build_messages(system_prompt, working_history, user_content)
    while _estimate_messages_tokens(messages) > effective_token_cap and working_history:
        working_history = working_history[1:]
        messages = _build_messages(system_prompt, working_history, user_content)
    estimated_prompt_tokens = _estimate_messages_tokens(messages)
    history_tokens = _estimate_messages_tokens(working_history)
    if estimated_prompt_tokens > effective_token_cap:
        return PreparedContextChatRequest(
            prompt_header=prompt_header,
            normalized_contexts=normalized_contexts,
            normalized_citations=normalized_citations,
            bounded_history=bounded_history,
            user_content=user_content,
            resolved_cap=resolved_cap,
            effective_token_cap=effective_token_cap,
            context_ids=context_ids,
            mode="split",
            context_tokens=context_tokens,
            split_reason=(
                f"{overflow_reason_prefix} ({estimated_prompt_tokens} > {effective_token_cap}). "
                "Reduce selected contexts or shorten history/messages."
            ),
            context_window_kind=context_window_kind,
            context_block_tokens=context_block_tokens,
            prompt_header_tokens=prompt_header_tokens,
            history_tokens=history_tokens,
            user_tokens=user_tokens,
            citation_catalog_entry_count=citation_catalog_entry_count,
            fitted_citation_entry_count=fitted_citation_entry_count,
            fitted_citation_ref_ids=fitted_citation_ref_ids,
        )
    return PreparedContextChatRequest(
        prompt_header=prompt_header,
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
        bounded_history=bounded_history,
        user_content=user_content,
        resolved_cap=resolved_cap,
        effective_token_cap=effective_token_cap,
        context_ids=context_ids,
        mode="direct",
        direct_system_prompt=system_prompt,
        direct_history=working_history,
        estimated_prompt_tokens=estimated_prompt_tokens,
        context_tokens=context_tokens,
        context_window_kind=context_window_kind,
        context_block_tokens=context_block_tokens,
        prompt_header_tokens=prompt_header_tokens,
        history_tokens=history_tokens,
        user_tokens=user_tokens,
        citation_catalog_entry_count=citation_catalog_entry_count,
        fitted_citation_entry_count=fitted_citation_entry_count,
        fitted_citation_ref_ids=fitted_citation_ref_ids,
    )


def _normalize_contexts(raw_contexts: list[dict[str, Any]]) -> list[ChatContextSource]:
    """Normalize raw context payloads into typed context sources."""
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
    """Normalize citation catalog entries provided by the caller."""
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
    """Normalize raw history entries to user and assistant content pairs."""
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


def _fit_citation_catalog_to_budget(
    citation_catalog: list[dict[str, str]],
    prompt_header: str,
    history: list[dict[str, str]],
    user_content: str,
    token_cap: int,
    prompt_token_buffer: int,
    citation_prefix_tokens: list[int] | None = None,
) -> list[dict[str, str]]:
    """Keep only citation entries that fit the strict prompt budget."""
    fixed_messages = [{"role": "user", "content": user_content}] + history
    fixed_tokens = _estimate_messages_tokens(fixed_messages)
    strict_budget = token_cap - fixed_tokens - count_tokens(prompt_header) - prompt_token_buffer
    if strict_budget <= 0:
        return []

    normalized_prefix_tokens = _normalize_citation_prefix_tokens(
        citation_prefix_tokens,
        expected_entries=len(citation_catalog),
    )
    if normalized_prefix_tokens is not None:
        fitted_count = bisect_right(normalized_prefix_tokens, strict_budget)
        return citation_catalog[:fitted_count]

    fitted: list[dict[str, str]] = []
    for item in citation_catalog:
        candidate = fitted + [item]
        if (
            count_tokens(context_chat_prompts.render_citation_catalog_block(candidate))
            > strict_budget
        ):
            break
        fitted = candidate
    return fitted


def _normalize_citation_prefix_tokens(
    citation_prefix_tokens: list[int] | None,
    *,
    expected_entries: int,
) -> list[int] | None:
    """Validate cached prefix-token accounting against the current citation order."""
    if not isinstance(citation_prefix_tokens, list):
        return None
    if len(citation_prefix_tokens) != expected_entries:
        return None
    normalized: list[int] = []
    previous = -1
    for raw_value in citation_prefix_tokens:
        if not isinstance(raw_value, int) or raw_value < 0 or raw_value < previous:
            return None
        normalized.append(raw_value)
        previous = raw_value
    return normalized


def _estimate_full_context_window_tokens(
    *,
    normalized_contexts: list[ChatContextSource],
    normalized_citations: list[dict[str, str]],
    context_window_kind: Literal["citation_catalog", "serialized_contexts"] | None,
) -> int | None:
    """Count full context-window tokens before any budget fitting or trimming."""
    if context_window_kind == "citation_catalog":
        return count_tokens(
            context_chat_prompts.render_citation_catalog_block(normalized_citations)
        )
    if context_window_kind == "serialized_contexts":
        return count_tokens(context_chat_prompts.serialize_all_contexts(normalized_contexts))
    return None


__all__ = [
    "CHAT_REF_ID_PATTERN",
    "build_citation_catalog_from_contexts",
    "build_citation_catalog_token_cache",
    "estimate_context_window",
    "plan_context_chat_request",
    "resolve_chat_token_cap",
]
