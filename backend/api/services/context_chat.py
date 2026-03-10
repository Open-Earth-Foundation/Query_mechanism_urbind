"""LLM-backed context chat over one or more stored run contexts."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from backend.tools.calculator import (
    divide_numbers,
    multiply_numbers,
    subtract_numbers,
    sum_numbers,
)
from backend.utils.city_normalization import format_city_stem
from backend.utils.retry import RetrySettings, call_with_retries
from backend.utils.config import AppConfig, get_openrouter_api_key
from backend.utils.json_io import read_json_object, write_json
from backend.utils.tokenization import count_tokens, get_encoding

logger = logging.getLogger(__name__)

CHAT_SUM_TOOL_NAME = "sum_numbers"
CHAT_SUBTRACT_TOOL_NAME = "subtract_numbers"
CHAT_MULTIPLY_TOOL_NAME = "multiply_numbers"
CHAT_DIVIDE_TOOL_NAME = "divide_numbers"
CHAT_REF_ID_PATTERN = re.compile(r"^ref_[1-9]\d*$")
CHAT_EVIDENCE_CACHE_FILENAME = "evidence_chunks.json"


def resolve_chat_token_cap(config: AppConfig) -> int:
    """Return the token cap for chat prompts from config."""
    return config.chat.max_context_total_tokens


CHAT_TOOL_DEFINITIONS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": CHAT_SUM_TOOL_NAME,
            "description": "Return arithmetic sum of a list of numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of integer/float values to sum.",
                    }
                },
                "required": ["numbers"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CHAT_SUBTRACT_TOOL_NAME,
            "description": "Subtract one number from another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minuend": {
                        "type": "number",
                        "description": "Starting numeric value.",
                    },
                    "subtrahend": {
                        "type": "number",
                        "description": "Numeric value to subtract from the minuend.",
                    },
                },
                "required": ["minuend", "subtrahend"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CHAT_MULTIPLY_TOOL_NAME,
            "description": "Return arithmetic product of a list of numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of integer/float values to multiply.",
                    }
                },
                "required": ["numbers"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": CHAT_DIVIDE_TOOL_NAME,
            "description": "Divide one number by another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dividend": {
                        "type": "number",
                        "description": "Numeric value being divided.",
                    },
                    "divisor": {
                        "type": "number",
                        "description": "Numeric value to divide by. Must not be zero.",
                    },
                },
                "required": ["dividend", "divisor"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass(frozen=True)
class ChatContextSource:
    """Single context source used to ground chat replies."""

    run_id: str
    question: str
    final_document: str
    context_bundle: dict[str, Any]


@dataclass(frozen=True)
class ContextChatPlan:
    """Preflight strategy decision for one chat request."""

    mode: Literal["direct", "split"]
    context_ids: list[str]
    resolved_token_cap: int
    effective_token_cap: int
    estimated_prompt_tokens: int | None = None
    context_tokens: int | None = None
    split_reason: str | None = None
    context_window_kind: Literal["citation_catalog", "serialized_contexts"] | None = None
    context_block_tokens: int | None = None
    prompt_header_tokens: int | None = None
    history_tokens: int | None = None
    user_tokens: int | None = None
    citation_catalog_entry_count: int | None = None
    fitted_citation_entry_count: int | None = None
    fitted_citation_ref_ids: list[str] | None = None


@dataclass(frozen=True)
class ContextWindowEstimate:
    """Context-window estimate derived from the same prompt assembly path as chat."""

    mode: Literal["direct", "split"]
    resolved_token_cap: int
    effective_token_cap: int
    context_window_kind: Literal["citation_catalog", "serialized_contexts"] | None = None
    context_window_tokens: int | None = None
    fitted_context_window_tokens: int | None = None
    estimated_prompt_tokens: int | None = None
    citation_catalog_entry_count: int | None = None
    fitted_citation_entry_count: int | None = None


@dataclass(frozen=True)
class CitationCatalogTokenCache:
    """Cached token accounting for one ordered citation catalog."""

    ordered_ref_ids: list[str]
    prefix_tokens: list[int]
    total_tokens: int


@dataclass(frozen=True)
class _PreparedContextChatRequest:
    """Normalized chat inputs and the resolved direct-vs-split strategy."""

    prompt_header: str
    normalized_contexts: list[ChatContextSource]
    normalized_citations: list[dict[str, str]]
    bounded_history: list[dict[str, str]]
    user_content: str
    resolved_cap: int
    effective_token_cap: int
    context_ids: list[str]
    mode: Literal["direct", "split"]
    direct_system_prompt: str | None = None
    direct_history: list[dict[str, str]] | None = None
    estimated_prompt_tokens: int | None = None
    context_tokens: int | None = None
    split_reason: str | None = None
    context_window_kind: Literal["citation_catalog", "serialized_contexts"] | None = None
    context_block_tokens: int | None = None
    prompt_header_tokens: int | None = None
    history_tokens: int | None = None
    user_tokens: int | None = None
    citation_catalog_entry_count: int | None = None
    fitted_citation_entry_count: int | None = None
    fitted_citation_ref_ids: list[str] | None = None


def load_context_bundle(path: Path) -> dict[str, Any]:
    """Load context bundle object from JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Context bundle at {path} is not a JSON object.")


def load_final_document(path: Path) -> str:
    """Load final markdown document."""
    return path.read_text(encoding="utf-8")


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
    """Plan whether one chat request can stay direct or must switch to split mode."""
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
    """Estimate the context-window tokens for chat selection and diagnostics."""
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
            total_tokens=count_tokens(_render_citation_catalog_block([])),
        )
    prefix_tokens: list[int] = []
    for index in range(1, len(normalized_catalog) + 1):
        prefix_tokens.append(count_tokens(_render_citation_catalog_block(normalized_catalog[:index])))
    return CitationCatalogTokenCache(
        ordered_ref_ids=[item["ref_id"] for item in normalized_catalog],
        prefix_tokens=prefix_tokens,
        total_tokens=prefix_tokens[-1],
    )


def generate_context_chat_reply(
    original_question: str,
    contexts: list[dict[str, Any]],
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int = 0,
    api_key_override: str | None = None,
    citation_catalog: list[dict[str, str]] | None = None,
    citation_prefix_tokens: list[int] | None = None,
    retry_missing_citation: bool = False,
    run_id: str | None = None,
) -> str:
    """Generate assistant reply grounded in selected run contexts."""
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
    timeout = config.chat.provider_timeout_seconds
    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )
    client = OpenAI(
        api_key=api_key,
        base_url=config.openrouter_base_url,
        timeout=timeout,
    )

    base_request_kwargs: dict[str, Any] = {
        "model": config.chat.model,
        "temperature": float(config.chat.temperature),
        "tools": CHAT_TOOL_DEFINITIONS,
        "tool_choice": "auto",
    }
    if config.chat.reasoning_effort is not None:
        base_request_kwargs["reasoning_effort"] = config.chat.reasoning_effort
    if config.chat.max_output_tokens is not None:
        base_request_kwargs["max_tokens"] = config.chat.max_output_tokens

    retry_settings = RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )
    if prepared.mode == "direct":
        if prepared.context_tokens is not None:
            logger.info(
                "Context chat direct request model=%s contexts=%s context_tokens=%d estimated_prompt_tokens=%d "
                "threshold=%d token_cap=%d effective_token_cap=%d context_window_kind=%s "
                "context_block_tokens=%s prompt_header_tokens=%s history_tokens=%s user_tokens=%s "
                "citation_entries=%s fitted_citation_entries=%s",
                config.chat.model,
                prepared.context_ids,
                prepared.context_tokens,
                prepared.estimated_prompt_tokens or 0,
                config.chat.multi_pass_threshold_tokens,
                prepared.resolved_cap,
                prepared.effective_token_cap,
                prepared.context_window_kind,
                prepared.context_block_tokens,
                prepared.prompt_header_tokens,
                prepared.history_tokens,
                prepared.user_tokens,
                prepared.citation_catalog_entry_count,
                prepared.fitted_citation_entry_count,
            )
        else:
            logger.info(
                "Context chat direct request model=%s contexts=%s estimated_prompt_tokens=%d "
                "token_cap=%d effective_token_cap=%d context_window_kind=%s context_block_tokens=%s "
                "prompt_header_tokens=%s history_tokens=%s user_tokens=%s citation_entries=%s "
                "fitted_citation_entries=%s",
                config.chat.model,
                prepared.context_ids,
                prepared.estimated_prompt_tokens or 0,
                prepared.resolved_cap,
                prepared.effective_token_cap,
                prepared.context_window_kind,
                prepared.context_block_tokens,
                prepared.prompt_header_tokens,
                prepared.history_tokens,
                prepared.user_tokens,
                prepared.citation_catalog_entry_count,
                prepared.fitted_citation_entry_count,
            )
        return _run_single_pass(
            client=client,
            messages=_build_messages(
                prepared.direct_system_prompt or "",
                list(prepared.direct_history or []),
                prepared.user_content,
            ),
            request_kwargs=base_request_kwargs,
            retry_settings=retry_settings,
            run_id=run_id,
            context_count=len(prepared.context_ids),
        )

    logger.info(
        "Context chat overflow fallback run_id=%s contexts=%s error=%s",
        run_id,
        prepared.context_ids,
        prepared.split_reason,
    )
    return _run_overflow_evidence_map_reduce(
        prompt_header=prepared.prompt_header,
        normalized_contexts=prepared.normalized_contexts,
        normalized_citations=prepared.normalized_citations,
        bounded_history=prepared.bounded_history,
        user_content=prepared.user_content,
        effective_token_cap=prepared.effective_token_cap,
        config=config,
        client=client,
        request_kwargs=base_request_kwargs,
        retry_settings=retry_settings,
        run_id=run_id,
        context_count=len(prepared.context_ids),
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
) -> _PreparedContextChatRequest:
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

    prompt_header = _build_system_prompt_header(
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
        fitted_catalog_tokens = count_tokens(_render_citation_catalog_block(fitted_citations))
        if len(fitted_citations) < len(normalized_citations):
            return _PreparedContextChatRequest(
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
            context_block=_render_citation_catalog_block(fitted_citations),
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

    serialized_contexts = _serialize_all_contexts(normalized_contexts)
    context_tokens = count_tokens(serialized_contexts)
    if context_tokens > config.chat.multi_pass_threshold_tokens:
        return _PreparedContextChatRequest(
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
) -> _PreparedContextChatRequest:
    """Build the direct prompt shape or return a split-mode overflow reason."""
    system_prompt = _compose_system_prompt(prompt_header, context_block)
    working_history = list(bounded_history)
    messages = _build_messages(system_prompt, working_history, user_content)
    while _estimate_messages_tokens(messages) > effective_token_cap and working_history:
        working_history = working_history[1:]
        messages = _build_messages(system_prompt, working_history, user_content)
    estimated_prompt_tokens = _estimate_messages_tokens(messages)
    history_tokens = _estimate_messages_tokens(working_history)
    if estimated_prompt_tokens > effective_token_cap:
        return _PreparedContextChatRequest(
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
    return _PreparedContextChatRequest(
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


def _run_overflow_evidence_map_reduce(
    *,
    prompt_header: str,
    normalized_contexts: list[ChatContextSource],
    normalized_citations: list[dict[str, str]],
    bounded_history: list[dict[str, str]],
    user_content: str,
    effective_token_cap: int,
    config: AppConfig,
    client: OpenAI,
    request_kwargs: dict[str, Any],
    retry_settings: RetrySettings,
    run_id: str | None,
    context_count: int,
) -> str:
    """Answer from compact evidence chunks when the direct prompt would overflow."""
    cache_payload = _load_or_build_evidence_cache(
        run_id=run_id,
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
        config=config,
    )
    cached_chunks = cache_payload.get("chunks")
    cache_chunk_count = len(cached_chunks) if isinstance(cached_chunks, list) else 0
    evidence_items = _flatten_evidence_cache_items(cache_payload)
    logger.info(
        "Context chat split mode enabled run_id=%s context_count=%d evidence_items=%d "
        "cache_chunks=%d effective_token_cap=%d",
        run_id,
        context_count,
        len(evidence_items),
        cache_chunk_count,
        effective_token_cap,
    )
    if not evidence_items:
        logger.info("Context chat split mode empty-evidence fallback run_id=%s", run_id)
        return _run_overflow_empty_evidence_answer(
            prompt_header=prompt_header,
            bounded_history=bounded_history,
            user_content=user_content,
            effective_token_cap=effective_token_cap,
            prompt_token_buffer=config.chat.prompt_token_buffer,
            client=client,
            request_kwargs=request_kwargs,
            retry_settings=retry_settings,
            run_id=run_id,
            context_count=context_count,
        )

    map_history, map_budget = _resolve_prompt_budget(
        prompt_factory=lambda: _compose_evidence_map_prompt(
            prompt_header=prompt_header,
            evidence_block="",
            chunk_index=1,
            total_chunks=1,
        ),
        history=bounded_history,
        user_content=user_content,
        effective_token_cap=effective_token_cap,
        prompt_token_buffer=config.chat.prompt_token_buffer,
    )
    evidence_blocks = _build_request_evidence_blocks(
        evidence_items=evidence_items,
        block_budget=max(map_budget, 1),
    )
    partial_answers: list[str] = []
    total_blocks = len(evidence_blocks)
    logger.info(
        "Context chat split mode map phase run_id=%s map_chunks=%d map_budget=%d",
        run_id,
        total_blocks,
        map_budget,
    )
    for chunk_index, evidence_block in enumerate(evidence_blocks, start=1):
        system_prompt = _compose_evidence_map_prompt(
            prompt_header=prompt_header,
            evidence_block=evidence_block,
            chunk_index=chunk_index,
            total_chunks=total_blocks,
        )
        partial_answers.append(
            _run_single_pass(
                client=client,
                messages=_build_messages(system_prompt, map_history, user_content),
                request_kwargs=request_kwargs,
                retry_settings=retry_settings,
                run_id=run_id,
                context_count=context_count,
            )
        )
    return _run_reduce_passes(
        partial_answers=partial_answers,
        prompt_header=prompt_header,
        bounded_history=bounded_history,
        user_content=user_content,
        effective_token_cap=effective_token_cap,
        prompt_token_buffer=config.chat.prompt_token_buffer,
        client=client,
        request_kwargs=request_kwargs,
        retry_settings=retry_settings,
        run_id=run_id,
        context_count=context_count,
    )


def _run_overflow_empty_evidence_answer(
    *,
    prompt_header: str,
    bounded_history: list[dict[str, str]],
    user_content: str,
    effective_token_cap: int,
    prompt_token_buffer: int,
    client: OpenAI,
    request_kwargs: dict[str, Any],
    retry_settings: RetrySettings,
    run_id: str | None,
    context_count: int,
) -> str:
    """Use the LLM to explain that no compact evidence is available in overflow mode."""
    working_history, _budget = _resolve_prompt_budget(
        prompt_factory=lambda: _compose_empty_evidence_prompt(prompt_header),
        history=bounded_history,
        user_content=user_content,
        effective_token_cap=effective_token_cap,
        prompt_token_buffer=prompt_token_buffer,
    )
    system_prompt = _compose_empty_evidence_prompt(prompt_header)
    return _run_single_pass(
        client=client,
        messages=_build_messages(system_prompt, working_history, user_content),
        request_kwargs=request_kwargs,
        retry_settings=retry_settings,
        run_id=run_id,
        context_count=context_count,
    )


def _run_reduce_passes(
    *,
    partial_answers: list[str],
    prompt_header: str,
    bounded_history: list[dict[str, str]],
    user_content: str,
    effective_token_cap: int,
    prompt_token_buffer: int,
    client: OpenAI,
    request_kwargs: dict[str, Any],
    retry_settings: RetrySettings,
    run_id: str | None,
    context_count: int,
) -> str:
    """Recursively merge map outputs until one final answer remains."""
    pending_answers = [answer.strip() for answer in partial_answers if answer.strip()]
    if not pending_answers:
        return _run_overflow_empty_evidence_answer(
            prompt_header=prompt_header,
            bounded_history=bounded_history,
            user_content=user_content,
            effective_token_cap=effective_token_cap,
            prompt_token_buffer=prompt_token_buffer,
            client=client,
            request_kwargs=request_kwargs,
            retry_settings=retry_settings,
            run_id=run_id,
            context_count=context_count,
        )

    stage_index = 1
    while len(pending_answers) > 1:
        reduce_history, reduce_budget = _resolve_prompt_budget(
            prompt_factory=lambda: _compose_evidence_reduce_prompt(
                prompt_header=prompt_header,
                analyses_block="",
                stage_index=stage_index,
                batch_index=1,
                batch_count=1,
            ),
            history=bounded_history,
            user_content=user_content,
            effective_token_cap=effective_token_cap,
            prompt_token_buffer=prompt_token_buffer,
        )
        grouped_blocks = _group_partial_answers(
            partial_answers=pending_answers,
            block_budget=max(reduce_budget, 1),
        )
        next_round: list[str] = []
        total_groups = len(grouped_blocks)
        logger.info(
            "Context chat split mode reduce phase run_id=%s stage=%d input_partials=%d "
            "reduce_batches=%d reduce_budget=%d",
            run_id,
            stage_index,
            len(pending_answers),
            total_groups,
            reduce_budget,
        )
        for batch_index, analyses_block in enumerate(grouped_blocks, start=1):
            system_prompt = _compose_evidence_reduce_prompt(
                prompt_header=prompt_header,
                analyses_block=analyses_block,
                stage_index=stage_index,
                batch_index=batch_index,
                batch_count=total_groups,
            )
            next_round.append(
                _run_single_pass(
                    client=client,
                    messages=_build_messages(system_prompt, reduce_history, user_content),
                    request_kwargs=request_kwargs,
                    retry_settings=retry_settings,
                    run_id=run_id,
                    context_count=context_count,
                )
            )
        pending_answers = [answer.strip() for answer in next_round if answer.strip()]
        stage_index += 1

    return pending_answers[0]


def _load_or_build_evidence_cache(
    *,
    run_id: str | None,
    normalized_contexts: list[ChatContextSource],
    normalized_citations: list[dict[str, str]],
    config: AppConfig,
) -> dict[str, object]:
    """Load a per-run evidence cache when valid or rebuild it from compact evidence."""
    evidence_items = _resolve_overflow_evidence_items(
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
    )
    source_signature = _build_evidence_source_signature(
        normalized_contexts=normalized_contexts,
        evidence_items=evidence_items,
    )
    if run_id:
        cache_path = _chat_evidence_cache_path(config.runs_dir, run_id)
        cached_payload = _read_json_object(cache_path)
        if (
            isinstance(cached_payload, dict)
            and str(cached_payload.get("source_signature", "")).strip() == source_signature
        ):
            logger.info("Context chat evidence cache hit run_id=%s", run_id)
            return cached_payload

    chunk_groups = _chunk_evidence_items(
        evidence_items=evidence_items,
        block_budget=max(config.chat.multi_pass_chunk_tokens, 1),
    )
    payload: dict[str, object] = {
        "source_signature": source_signature,
        "evidence_count": len(evidence_items),
        "chunks": [
            {
                "chunk_id": f"chunk_{index}",
                "ref_ids": [item["ref_id"] for item in chunk],
                "items": chunk,
            }
            for index, chunk in enumerate(chunk_groups, start=1)
        ],
    }
    if run_id:
        cache_path = _chat_evidence_cache_path(config.runs_dir, run_id)
        _write_json_object(cache_path, payload)
        logger.info("Context chat evidence cache built run_id=%s chunks=%d", run_id, len(chunk_groups))
    return payload


def _resolve_overflow_evidence_items(
    *,
    normalized_contexts: list[ChatContextSource],
    normalized_citations: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Choose stable compact evidence items for overflow prompts."""
    if normalized_citations:
        return _normalize_evidence_items(normalized_citations)
    return _extract_evidence_items_from_contexts(normalized_contexts)


def _normalize_evidence_items(
    normalized_citations: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Normalize compact evidence items while preserving stable ref order."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in normalized_citations:
        ref_id = str(item.get("ref_id", "")).strip()
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        items.append(
            {
                "ref_id": ref_id,
                "city_name": str(item.get("city_name", "")).strip(),
                "quote": str(item.get("quote", "")).strip(),
                "partial_answer": str(item.get("partial_answer", "")).strip(),
            }
        )
    return items


def _extract_evidence_items_from_contexts(
    normalized_contexts: list[ChatContextSource],
) -> list[dict[str, str]]:
    """Fallback overflow evidence extraction from stored markdown excerpts."""
    extracted: list[dict[str, str]] = []
    seen: set[str] = set()
    for context in normalized_contexts:
        markdown_payload = context.context_bundle.get("markdown")
        if not isinstance(markdown_payload, dict):
            continue
        raw_excerpts = markdown_payload.get("excerpts")
        if not isinstance(raw_excerpts, list):
            continue
        for raw_excerpt in raw_excerpts:
            if not isinstance(raw_excerpt, dict):
                continue
            ref_id = str(raw_excerpt.get("ref_id", "")).strip()
            if not ref_id or ref_id in seen or not CHAT_REF_ID_PATTERN.fullmatch(ref_id):
                continue
            quote = str(raw_excerpt.get("quote", "")).strip()
            partial_answer = str(raw_excerpt.get("partial_answer", "")).strip()
            if not quote and not partial_answer:
                continue
            extracted.append(
                {
                    "ref_id": ref_id,
                    "city_name": str(raw_excerpt.get("city_name", "")).strip(),
                    "quote": quote,
                    "partial_answer": partial_answer,
                }
            )
            seen.add(ref_id)
    return extracted


def _build_evidence_source_signature(
    *,
    normalized_contexts: list[ChatContextSource],
    evidence_items: list[dict[str, str]],
) -> str:
    """Build a stable cache signature for the active context sources and evidence."""
    payload = {
        "context_ids": [context.run_id for context in normalized_contexts],
        "evidence_items": evidence_items,
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _flatten_evidence_cache_items(cache_payload: dict[str, object]) -> list[dict[str, str]]:
    """Flatten cached chunk payload back into a stable evidence item list."""
    chunks = cache_payload.get("chunks")
    if not isinstance(chunks, list):
        return []
    items: list[dict[str, str]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        raw_items = chunk.get("items")
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "ref_id": str(item.get("ref_id", "")).strip(),
                    "city_name": str(item.get("city_name", "")).strip(),
                    "quote": str(item.get("quote", "")).strip(),
                    "partial_answer": str(item.get("partial_answer", "")).strip(),
                }
            )
    return [item for item in items if item["ref_id"]]


def _build_request_evidence_blocks(
    *,
    evidence_items: list[dict[str, str]],
    block_budget: int,
) -> list[str]:
    """Build rendered evidence blocks that fit within the request budget."""
    chunks = _chunk_evidence_items(
        evidence_items=evidence_items,
        block_budget=block_budget,
    )
    if not chunks:
        return []
    return [_render_evidence_items_block(chunk) for chunk in chunks]


def _chunk_evidence_items(
    *,
    evidence_items: list[dict[str, str]],
    block_budget: int,
) -> list[list[dict[str, str]]]:
    """Group evidence items into token-bounded blocks while preserving order."""
    if block_budget <= 0:
        return []
    chunks: list[list[dict[str, str]]] = []
    current_chunk: list[dict[str, str]] = []
    for raw_item in evidence_items:
        item = _fit_evidence_item_to_budget(raw_item, block_budget)
        candidate = current_chunk + [item]
        if current_chunk and count_tokens(_render_evidence_items_block(candidate)) > block_budget:
            chunks.append(current_chunk)
            current_chunk = [item]
            continue
        current_chunk = candidate
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _fit_evidence_item_to_budget(
    item: dict[str, str],
    block_budget: int,
) -> dict[str, str]:
    """Trim oversized quote fields so one evidence item always fits in a block."""
    candidate = dict(item)
    if count_tokens(_render_evidence_items_block([candidate])) <= block_budget:
        return candidate

    quote_tokens = max(block_budget // 2, 1)
    partial_tokens = max(block_budget // 3, 1)
    candidate["quote"] = _truncate_to_tokens(candidate["quote"], quote_tokens)
    candidate["partial_answer"] = _truncate_to_tokens(
        candidate["partial_answer"],
        partial_tokens,
    )
    while count_tokens(_render_evidence_items_block([candidate])) > block_budget:
        if candidate["quote"] and len(candidate["quote"]) >= len(candidate["partial_answer"]):
            candidate["quote"] = _truncate_to_tokens(
                candidate["quote"],
                max(count_tokens(candidate["quote"]) - 32, 1),
            )
            continue
        if candidate["partial_answer"]:
            candidate["partial_answer"] = _truncate_to_tokens(
                candidate["partial_answer"],
                max(count_tokens(candidate["partial_answer"]) - 24, 1),
            )
            continue
        if candidate["city_name"]:
            candidate["city_name"] = _truncate_to_tokens(
                candidate["city_name"],
                max(count_tokens(candidate["city_name"]) - 8, 0),
            )
            continue
        if candidate["quote"] or candidate["partial_answer"] or candidate["city_name"]:
            candidate["quote"] = ""
            candidate["partial_answer"] = ""
            candidate["city_name"] = ""
            continue
        break
    return candidate


def _group_partial_answers(
    *,
    partial_answers: list[str],
    block_budget: int,
) -> list[str]:
    """Group partial answers into token-bounded reduce batches."""
    if block_budget <= 0:
        return []
    groups: list[list[str]] = []
    current_group: list[str] = []
    for answer in partial_answers:
        candidate = current_group + [answer]
        if current_group and count_tokens(_render_partial_answers_block(candidate)) > block_budget:
            groups.append(current_group)
            current_group = [answer]
            continue
        current_group = candidate
    if current_group:
        groups.append(current_group)
    return [_render_partial_answers_block(group) for group in groups]


def _resolve_prompt_budget(
    *,
    prompt_factory: Callable[[], str],
    history: list[dict[str, str]],
    user_content: str,
    effective_token_cap: int,
    prompt_token_buffer: int,
) -> tuple[list[dict[str, str]], int]:
    """Trim history until the prompt has a positive remaining budget for context blocks."""
    working_history = list(history)
    while True:
        system_prompt = prompt_factory()
        prompt_tokens = _estimate_messages_tokens(
            _build_messages(system_prompt, working_history, user_content)
        )
        remaining_budget = effective_token_cap - prompt_tokens - prompt_token_buffer
        if remaining_budget > 0 or not working_history:
            return working_history, remaining_budget
        working_history = working_history[1:]


def _compose_evidence_map_prompt(
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


def _compose_evidence_reduce_prompt(
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


def _compose_empty_evidence_prompt(prompt_header: str) -> str:
    """Build the overflow fallback prompt when no compact evidence items exist."""
    return (
        f"{prompt_header}\n\n"
        "No compact evidence items are available for the selected context sources in overflow mode.\n"
        "Explain briefly that the current saved context does not provide extractable evidence for a grounded answer."
    )


def _render_evidence_items_block(evidence_items: list[dict[str, str]]) -> str:
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


def _render_partial_answers_block(partial_answers: list[str]) -> str:
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


def _chat_evidence_cache_path(runs_dir: Path, run_id: str) -> Path:
    """Return the cached compact evidence artifact path for one run."""
    return runs_dir / run_id / "chat_cache" / CHAT_EVIDENCE_CACHE_FILENAME


def _read_json_object(path: Path) -> dict[str, object] | None:
    """Read one JSON object from disk with safe fallback."""
    payload = read_json_object(
        path,
        logger=logger,
        error_prefix="Failed to read JSON object",
    )
    if isinstance(payload, dict):
        return payload
    return None


def _write_json_object(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON object artifact with stable formatting."""
    write_json(path, payload, ensure_ascii=False)


def _is_retryable_chat_error(exc: Exception) -> bool:
    """Return True for transient provider failures worth retrying."""
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def _extract_response_text(response: Any) -> str:
    """Extract cleaned text content from a chat completion response."""
    if not response.choices:
        raise ValueError("Chat model returned no choices.")
    message = response.choices[0].message
    content = message.content or ""
    if isinstance(content, list):
        content = "".join(
            part.text for part in content if hasattr(part, "text") and isinstance(part.text, str)
        )
    cleaned = str(content).strip()
    if not cleaned:
        raise ValueError("Chat model returned empty content.")
    return cleaned


def _run_single_pass(
    *,
    client: OpenAI,
    messages: list[dict[str, Any]],
    request_kwargs: dict[str, Any],
    retry_settings: RetrySettings,
    run_id: str | None,
    context_count: int,
) -> str:
    """Run a single chat completion pass and return the text reply."""
    response = call_with_retries(
        lambda: _run_chat_completion_with_tools(
            client=client,
            messages=messages,
            request_kwargs=request_kwargs,
            max_tool_rounds=retry_settings.max_attempts,
        ),
        operation="chat.completion",
        retry_settings=retry_settings,
        should_retry=_is_retryable_chat_error,
        run_id=run_id,
        context={"context_count": context_count},
    )
    return _extract_response_text(response)


def _parse_tool_arguments(raw_arguments: str | None) -> dict[str, object]:
    """Parse tool arguments into a JSON object."""
    if not raw_arguments:
        raise ValueError("Tool arguments are empty.")
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("Tool arguments must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return parsed


def _normalize_sum_numbers_args(raw_arguments: str | None) -> list[float]:
    """Parse sum tool arguments and coerce to float list."""
    parsed = _parse_tool_arguments(raw_arguments)
    raw_numbers = parsed.get("numbers")
    if not isinstance(raw_numbers, list):
        raise ValueError("Tool arguments must include list field `numbers`.")
    return [float(value) for value in raw_numbers]


def _normalize_subtract_numbers_args(raw_arguments: str | None) -> tuple[float, float]:
    """Parse subtract tool arguments and coerce to numeric operands."""
    parsed = _parse_tool_arguments(raw_arguments)
    if "minuend" not in parsed:
        raise ValueError("Tool arguments must include numeric field `minuend`.")
    if "subtrahend" not in parsed:
        raise ValueError("Tool arguments must include numeric field `subtrahend`.")
    return (float(parsed["minuend"]), float(parsed["subtrahend"]))


def _normalize_multiply_numbers_args(raw_arguments: str | None) -> list[float]:
    """Parse multiply tool arguments and coerce to float list."""
    parsed = _parse_tool_arguments(raw_arguments)
    raw_numbers = parsed.get("numbers")
    if not isinstance(raw_numbers, list):
        raise ValueError("Tool arguments must include list field `numbers`.")
    return [float(value) for value in raw_numbers]


def _normalize_divide_numbers_args(raw_arguments: str | None) -> tuple[float, float]:
    """Parse divide tool arguments and coerce to numeric operands."""
    parsed = _parse_tool_arguments(raw_arguments)
    if "dividend" not in parsed:
        raise ValueError("Tool arguments must include numeric field `dividend`.")
    if "divisor" not in parsed:
        raise ValueError("Tool arguments must include numeric field `divisor`.")
    return (float(parsed["dividend"]), float(parsed["divisor"]))


def _run_chat_completion_with_tools(
    *,
    client: OpenAI,
    messages: list[dict[str, str]],
    request_kwargs: dict[str, Any],
    max_tool_rounds: int,
) -> Any:
    """Execute chat completion and resolve tool calls in-process."""
    working_messages: list[dict[str, Any]] = [dict(message) for message in messages]
    resolved_max_rounds = max(int(max_tool_rounds), 1)
    for tool_round in range(1, resolved_max_rounds + 1):
        response = client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], working_messages),
            **request_kwargs,
        )
        if not response.choices:
            return response
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return response

        serialized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function_payload = getattr(tool_call, "function", None)
            function_name = str(getattr(function_payload, "name", "") or "").strip()
            function_arguments = str(getattr(function_payload, "arguments", "") or "")
            serialized_tool_calls.append(
                {
                    "id": str(getattr(tool_call, "id", "")),
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_arguments,
                    },
                }
            )

        working_messages.append(
            {
                "role": "assistant",
                "content": str(message.content or ""),
                "tool_calls": serialized_tool_calls,
            }
        )

        for tool_call in serialized_tool_calls:
            tool_name = str(tool_call.get("function", {}).get("name", "")).strip()
            tool_id = str(tool_call.get("id", "")).strip()
            tool_arguments = str(
                tool_call.get("function", {}).get("arguments", "")
            ).strip()
            if not tool_id:
                continue
            logger.info(
                "CHAT_TOOL_CALL round=%d tool=%s tool_call_id=%s",
                tool_round,
                tool_name,
                tool_id,
            )
            tool_result: dict[str, object]
            try:
                if tool_name == CHAT_SUM_TOOL_NAME:
                    numbers = _normalize_sum_numbers_args(tool_arguments)
                    tool_result = {"result": sum_numbers(numbers, source="context_chat")}
                elif tool_name == CHAT_SUBTRACT_TOOL_NAME:
                    minuend, subtrahend = _normalize_subtract_numbers_args(tool_arguments)
                    tool_result = {
                        "result": subtract_numbers(
                            minuend,
                            subtrahend,
                            source="context_chat",
                        )
                    }
                elif tool_name == CHAT_MULTIPLY_TOOL_NAME:
                    numbers = _normalize_multiply_numbers_args(tool_arguments)
                    tool_result = {"result": multiply_numbers(numbers, source="context_chat")}
                elif tool_name == CHAT_DIVIDE_TOOL_NAME:
                    dividend, divisor = _normalize_divide_numbers_args(tool_arguments)
                    tool_result = {
                        "result": divide_numbers(
                            dividend,
                            divisor,
                            source="context_chat",
                        )
                    }
                else:
                    tool_result = {"error": f"Unsupported tool: {tool_name}"}
                    logger.warning(
                        "CHAT_TOOL_CALL_UNSUPPORTED round=%d tool=%s tool_call_id=%s",
                        tool_round,
                        tool_name,
                        tool_id,
                    )
            except Exception as exc:  # noqa: BLE001
                tool_result = {"error": str(exc)}
                logger.exception(
                    "CHAT_TOOL_CALL_ERROR round=%d tool=%s tool_call_id=%s",
                    tool_round,
                    tool_name,
                    tool_id,
                )
            working_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )
    raise ValueError(
        f"Chat tool-call loop exceeded maximum rounds ({resolved_max_rounds})."
    )


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
        if count_tokens(_render_citation_catalog_block(candidate)) > strict_budget:
            break
        fitted = candidate
    return fitted


def _render_citation_catalog_block(citation_catalog: list[dict[str, str]]) -> str:
    """Serialize citation catalog into prompt-safe markdown context."""
    if not citation_catalog:
        return (
            "### Citation evidence catalog\n"
            "- No citation entries fit within the prompt token budget for this turn."
        )
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
    """Count full context-window tokens before any budget fitting or history trimming."""
    if context_window_kind == "citation_catalog":
        return count_tokens(_render_citation_catalog_block(normalized_citations))
    if context_window_kind == "serialized_contexts":
        return count_tokens(_serialize_all_contexts(normalized_contexts))
    return None


def _build_system_prompt_header(
    original_question: str,
    retry_missing_citation: bool,
) -> str:
    """Build stable system prompt prefix."""
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


def _compose_system_prompt(header: str, context_block: str) -> str:
    """Build final system prompt with context payload."""
    return (
        f"{header}\n\n"
        "Context sources:\n"
        f"{context_block}"
    )



def _serialize_all_contexts(contexts: list[ChatContextSource]) -> str:
    """Serialize all context sources into a single string."""
    sections = [
        _serialize_context(index, context)
        for index, context in enumerate(contexts, start=1)
    ]
    return "\n\n".join(sections)


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
    "ContextChatPlan",
    "ContextWindowEstimate",
    "CitationCatalogTokenCache",
    "build_citation_catalog_from_contexts",
    "build_citation_catalog_token_cache",
    "estimate_context_window",
    "resolve_chat_token_cap",
    "plan_context_chat_request",
    "generate_context_chat_reply",
    "load_context_bundle",
    "load_final_document",
]
