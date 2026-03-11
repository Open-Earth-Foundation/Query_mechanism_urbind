"""Stable public facade for LLM-backed context chat."""

from __future__ import annotations

import logging
from typing import Any

import backend.api.services.context_chat_evidence as context_chat_evidence
import backend.api.services.context_chat_execution as context_chat_execution
import backend.api.services.context_chat_io as context_chat_io
import backend.api.services.context_chat_planning as context_chat_planning
from backend.api.services.models import (
    ChatContextSource,
    CitationCatalogTokenCache,
    ContextChatPlan,
    ContextWindowEstimate,
)
from backend.api.services.utils.context_chat import build_messages as _build_messages
from backend.utils.config import AppConfig
from backend.utils.retry import RetrySettings

logger = logging.getLogger(__name__)

CHAT_EVIDENCE_CACHE_SCHEMA_VERSION = (
    context_chat_evidence.CHAT_EVIDENCE_CACHE_SCHEMA_VERSION
)
CHAT_TOOL_DEFINITIONS = context_chat_execution.CHAT_TOOL_DEFINITIONS

build_citation_catalog_from_contexts = context_chat_planning.build_citation_catalog_from_contexts
build_citation_catalog_token_cache = context_chat_planning.build_citation_catalog_token_cache
estimate_context_window = context_chat_planning.estimate_context_window
load_context_bundle = context_chat_io.load_context_bundle
load_final_document = context_chat_io.load_final_document
plan_context_chat_request = context_chat_planning.plan_context_chat_request
resolve_chat_token_cap = context_chat_planning.resolve_chat_token_cap


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
    """Generate an assistant reply grounded in the selected run contexts."""
    prepared = context_chat_planning._prepare_context_chat_request(
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
    client = context_chat_execution.create_chat_client(config, api_key_override)
    request_kwargs = context_chat_execution.build_request_kwargs(config)
    retry_settings = RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )
    if prepared.mode == "direct":
        _log_direct_request_metrics(prepared, config)
        return context_chat_execution._run_single_pass(
            client=client,
            messages=_build_messages(
                prepared.direct_system_prompt or "",
                list(prepared.direct_history or []),
                prepared.user_content,
            ),
            request_kwargs=request_kwargs,
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
    return context_chat_evidence._run_overflow_evidence_map_reduce(
        prompt_header=prepared.prompt_header,
        normalized_contexts=prepared.normalized_contexts,
        normalized_citations=prepared.normalized_citations,
        bounded_history=prepared.bounded_history,
        user_content=prepared.user_content,
        effective_token_cap=prepared.effective_token_cap,
        config=config,
        client=client,
        request_kwargs=request_kwargs,
        retry_settings=retry_settings,
        run_id=run_id,
        context_count=len(prepared.context_ids),
    )


def _log_direct_request_metrics(
    prepared: context_chat_planning.PreparedContextChatRequest,
    config: AppConfig,
) -> None:
    """Log the resolved prompt-window metrics for a direct request."""
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
        return
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
