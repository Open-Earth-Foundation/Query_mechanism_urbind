"""Reply generation and job processing helpers for chat operations."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import re
import time
from typing import Any, Callable
from uuid import uuid4

from openai import APIStatusError, APITimeoutError, AuthenticationError

from backend.api.models import SendChatMessageCompletedResponse
from backend.api.services.chat_context_loader import LoadedChatSource
from backend.api.services.chat_session_helpers import ChatCitationEntry
from backend.api.services.context_chat import generate_context_chat_reply, plan_context_chat_request
from backend.api.services.chat_followup_research import run_chat_followup_search, ChatFollowupSearchResult
from backend.modules.orchestrator.utils.references import is_valid_ref_id
from backend.utils.city_normalization import format_city_stem
from backend.utils.config import AppConfig
from backend.utils.retry import (
    RetrySettings,
    compute_retry_delay_seconds,
    log_retry_event,
    log_retry_exhausted,
)
from backend.utils.tokenization import count_tokens
from backend.modules.writer.utils.markdown_helpers import (
    extract_markdown_bundle,
    extract_markdown_excerpts as extract_bundle_excerpts,
    extract_selected_city_names,
)

logger = logging.getLogger(__name__)

_REF_TOKEN_PATTERN = re.compile(r"\[(ref_[1-9]\d*)\]")


def build_chat_sources(
    contexts: list[Any],  # list[LoadedContext]
    followup_bundles: list[Any],  # list[LoadedFollowupBundle]
) -> list[LoadedChatSource]:
    """Build reply-generation context sources from runs and follow-up bundles."""
    sources = [_as_chat_source_from_context(context) for context in contexts]
    sources.extend(
        _as_chat_source_from_followup_bundle(bundle) for bundle in followup_bundles
    )
    return sources


def _as_chat_source_from_context(context: Any) -> LoadedChatSource:
    """Convert one loaded run context into the generic chat-source shape."""
    return LoadedChatSource(
        source_type="run",
        source_id=context.run_id,
        question=context.question,
        final_document=context.final_document,
        context_bundle=context.context_bundle,
    )


def _as_chat_source_from_followup_bundle(bundle: Any) -> LoadedChatSource:
    """Convert one loaded follow-up bundle into the generic chat-source shape."""
    return LoadedChatSource(
        source_type="followup_bundle",
        source_id=bundle.bundle_id,
        question=str(bundle.context_bundle.get("research_question", "")).strip(),
        final_document="",
        context_bundle=bundle.context_bundle,
    )


def build_chat_citation_entries(
    sources: list[LoadedChatSource],
) -> list[ChatCitationEntry]:
    """Build deterministic synthetic chat citations from all loaded chat sources."""
    entries: list[ChatCitationEntry] = []
    synthetic_index = 1
    for source in sources:
        markdown_payload = source.context_bundle.get("markdown")
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
            if not is_valid_ref_id(source_ref_id):
                continue
            quote = str(raw_excerpt.get("quote", "")).strip()
            partial_answer = str(raw_excerpt.get("partial_answer", "")).strip()
            if not quote and not partial_answer:
                continue
            entries.append(
                ChatCitationEntry(
                    ref_id=f"ref_{synthetic_index}",
                    city_name=format_city_stem(str(raw_excerpt.get("city_name", "")).strip()),
                    quote=quote,
                    partial_answer=partial_answer,
                    source_type=source.source_type,
                    source_id=source.source_id,
                    source_ref_id=source_ref_id,
                )
            )
            synthetic_index += 1
    return entries


def build_llm_citation_catalog(
    entries: list[ChatCitationEntry],
) -> list[dict[str, str]]:
    """Build LLM-facing citation catalog with no internal identifiers."""
    return [
        {
            "ref_id": entry.ref_id,
            "city_name": entry.city_name,
            "quote": entry.quote,
            "partial_answer": entry.partial_answer,
        }
        for entry in entries
    ]


def _sample_ids(ids: list[str], *, limit: int = 6) -> list[str]:
    """Return a short sample of ids for structured logging."""
    if len(ids) <= limit:
        return ids
    return [*ids[:limit], "..."]


def _summarize_citation_sources(
    entries: list[ChatCitationEntry],
    *,
    selected_ref_ids: list[str] | None = None,
) -> list[dict[str, object]]:
    """Group citation entries by source for prompt-window diagnostics."""
    selected_ref_set = None if selected_ref_ids is None else set(selected_ref_ids)
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        if selected_ref_set is not None and entry.ref_id not in selected_ref_set:
            continue
        group_key = (entry.source_type, entry.source_id)
        bucket = grouped.get(group_key)
        if bucket is None:
            bucket = {
                "source_type": entry.source_type,
                "source_id": entry.source_id,
                "entry_count": 0,
                "ref_ids": [],
                "source_ref_ids": [],
            }
            grouped[group_key] = bucket
        bucket["entry_count"] = int(bucket["entry_count"]) + 1
        cast_refs = bucket["ref_ids"]
        cast_source_refs = bucket["source_ref_ids"]
        if isinstance(cast_refs, list):
            cast_refs.append(entry.ref_id)
        if isinstance(cast_source_refs, list):
            cast_source_refs.append(entry.source_ref_id)

    summaries: list[dict[str, object]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        ref_ids = [
            ref_id for ref_id in bucket.get("ref_ids", []) if isinstance(ref_id, str) and ref_id
        ]
        source_ref_ids = [
            ref_id
            for ref_id in bucket.get("source_ref_ids", [])
            if isinstance(ref_id, str) and ref_id
        ]
        summaries.append(
            {
                "source_type": bucket["source_type"],
                "source_id": bucket["source_id"],
                "entry_count": bucket["entry_count"],
                "sample_ref_ids": _sample_ids(ref_ids),
                "sample_source_ref_ids": _sample_ids(source_ref_ids),
            }
        )
    return summaries


def _extract_ordered_ref_ids(content: str) -> list[str]:
    """Extract citation refs preserving mention order with de-duplication."""
    ordered: list[str] = []
    seen: set[str] = set()
    for match in _REF_TOKEN_PATTERN.finditer(content):
        ref_id = match.group(1).strip()
        if ref_id in seen:
            continue
        seen.add(ref_id)
        ordered.append(ref_id)
    return ordered


def _resolve_assistant_citations(
    content: str,
    entries_by_ref_id: dict[str, ChatCitationEntry],
) -> tuple[list[dict[str, object]], bool]:
    """Resolve assistant citations and return validity status."""
    ordered_refs = _extract_ordered_ref_ids(content)
    resolved: list[dict[str, object]] = []
    for ref_id in ordered_refs:
        entry = entries_by_ref_id.get(ref_id)
        if entry is None:
            continue
        resolved.append(
            {
                "ref_id": entry.ref_id,
                "city_name": entry.city_name,
                "source_type": entry.source_type,
                "source_id": entry.source_id,
                "source_ref_id": entry.source_ref_id,
            }
        )
    return resolved, len(resolved) > 0


def build_router_payload(
    *,
    user_message: str,
    original_question: str,
    history: list[dict[str, str]],
    selected_run_ids: list[str],
    followup_bundles: list[Any],  # list[LoadedFollowupBundle]
    sources: list[LoadedChatSource],
) -> dict[str, object]:
    """Build the compact payload sent to the follow-up router."""
    return {
        "user_message": user_message,
        "original_question": original_question,
        "history": history,
        "selected_run_ids": selected_run_ids,
        "selected_followup_bundle_ids": [bundle.bundle_id for bundle in followup_bundles],
        "contexts": [_build_router_context_payload(source) for source in sources],
    }


_ROUTER_MAX_EXCERPTS_PER_SOURCE = 50


def _build_router_context_payload(source: LoadedChatSource) -> dict[str, object]:
    """Build one compact context summary for the follow-up router.

    Excerpts are capped at _ROUTER_MAX_EXCERPTS_PER_SOURCE to keep the router
    payload small — the router needs routing signal, not the full citation text.
    """
    markdown_bundle = extract_markdown_bundle(source.context_bundle)
    excerpts = extract_bundle_excerpts(markdown_bundle)
    selected_city_names = extract_selected_city_names(source.context_bundle, markdown_bundle)
    inspected_raw = markdown_bundle.get("inspected_city_names")
    inspected_city_names = selected_city_names
    if isinstance(inspected_raw, list):
        normalized = [str(item).strip() for item in inspected_raw if isinstance(item, str)]
        inspected_city_names = [name for name in normalized if name]
    capped_excerpts = excerpts[:_ROUTER_MAX_EXCERPTS_PER_SOURCE]
    return {
        "source_type": source.source_type,
        "source_id": source.source_id,
        "question": source.question,
        "selected_city_names": selected_city_names,
        "inspected_city_names": inspected_city_names,
        "excerpt_count": len(excerpts),
        "excerpts": [
            {
                "city_name": format_city_stem(str(excerpt.get("city_name", "")).strip()),
                "quote": str(excerpt.get("quote", "")).strip(),
                "partial_answer": str(excerpt.get("partial_answer", "")).strip(),
            }
            for excerpt in capped_excerpts
            if isinstance(excerpt, dict)
        ],
    }


def _build_context_model_payloads(sources: list[LoadedChatSource]) -> list[dict[str, object]]:
    """Build raw source payloads consumed by the context-chat service."""
    return [
        {
            "run_id": source.source_id,
            "question": source.question,
            "final_document": source.final_document,
            "context_bundle": source.context_bundle,
        }
        for source in sources
    ]


def _summarize_session_source_tokens(
    loaded_contexts: list[Any],  # list[LoadedContext]
    loaded_followup_bundles: list[Any],  # list[LoadedFollowupBundle]
) -> int:
    """Return canonical prompt-context tokens for the session-selected chat sources."""
    return sum(context.prompt_context_tokens for context in loaded_contexts) + sum(
        bundle.prompt_context_tokens for bundle in loaded_followup_bundles
    )


def log_chat_router_preflight(
    *,
    run_id: str,
    conversation_id: str,
    selected_run_ids: list[str],
    loaded_contexts: list[Any],
    loaded_followup_bundles: list[Any],
    router_payload: dict[str, object],
) -> None:
    """Log router payload sizing before the follow-up routing model runs."""
    router_payload_tokens = count_tokens(json.dumps(router_payload, ensure_ascii=False))
    context_summaries: list[dict[str, object]] = []
    raw_contexts = router_payload.get("contexts")
    if isinstance(raw_contexts, list):
        for item in raw_contexts:
            if not isinstance(item, dict):
                continue
            rendered_tokens = count_tokens(json.dumps(item, ensure_ascii=False))
            context_summaries.append(
                {
                    "source_type": item.get("source_type"),
                    "source_id": item.get("source_id"),
                    "excerpt_count": item.get("excerpt_count"),
                    "payload_tokens": rendered_tokens,
                }
            )
    logger.info(
        "Context chat router preflight run_id=%s conversation_id=%s contexts=%s "
        "followup_bundles=%s session_context_tokens=%d router_payload_tokens=%d "
        "router_context_summaries=%s",
        run_id,
        conversation_id,
        selected_run_ids,
        [bundle.bundle_id for bundle in loaded_followup_bundles],
        _summarize_session_source_tokens(loaded_contexts, loaded_followup_bundles),
        router_payload_tokens,
        context_summaries,
    )


def log_context_reply_plan(
    *,
    run_id: str,
    conversation_id: str,
    selected_run_ids: list[str],
    loaded_contexts: list[Any],
    loaded_followup_bundles: list[Any],
    sources: list[LoadedChatSource],
    plan: Any,  # ContextChatPlan
) -> None:
    """Log the resolved direct-vs-split decision for one chat answer attempt."""
    citation_entries = build_chat_citation_entries(sources)
    full_catalog_sources = _summarize_citation_sources(citation_entries)
    prompt_window_sources = _summarize_citation_sources(
        citation_entries,
        selected_ref_ids=plan.fitted_citation_ref_ids,
    )
    logger.info(
        "Context chat reply plan run_id=%s conversation_id=%s mode=%s contexts=%s "
        "followup_bundles=%s session_context_tokens=%d estimated_prompt_tokens=%s "
        "context_tokens=%s token_cap=%d effective_token_cap=%d split_reason=%s "
        "context_window_kind=%s context_block_tokens=%s prompt_header_tokens=%s "
        "history_tokens=%s user_tokens=%s citation_entries=%s fitted_citation_entries=%s "
        "prompt_window_sources=%s full_catalog_sources=%s",
        run_id,
        conversation_id,
        plan.mode,
        selected_run_ids,
        [bundle.bundle_id for bundle in loaded_followup_bundles],
        _summarize_session_source_tokens(loaded_contexts, loaded_followup_bundles),
        plan.estimated_prompt_tokens,
        plan.context_tokens,
        plan.resolved_token_cap,
        plan.effective_token_cap,
        plan.split_reason,
        plan.context_window_kind,
        plan.context_block_tokens,
        plan.prompt_header_tokens,
        plan.history_tokens,
        plan.user_tokens,
        plan.citation_catalog_entry_count,
        plan.fitted_citation_entry_count,
        prompt_window_sources,
        full_catalog_sources,
    )


def log_chat_request_summary(
    *,
    run_id: str,
    conversation_id: str,
    selected_run_ids: list[str],
    loaded_contexts: list[Any],
    loaded_followup_bundles: list[Any],
    history: list[dict[str, str]],
    token_cap: int,
    followup_search_enabled: bool,
    clarification_city: str | None = None,
) -> None:
    """Log one concise chat-source summary before any chat LLM call starts."""
    logger.info(
        "Context chat request summary run_id=%s conversation_id=%s selected_runs=%s "
        "followup_bundles=%s history_messages=%d session_context_tokens=%d token_cap=%d "
        "followup_search_enabled=%s clarification_city=%s",
        run_id,
        conversation_id,
        selected_run_ids,
        [bundle.bundle_id for bundle in loaded_followup_bundles],
        len(history),
        _summarize_session_source_tokens(loaded_contexts, loaded_followup_bundles),
        token_cap,
        followup_search_enabled,
        clarification_city,
    )


def build_context_reply_plan(
    *,
    original_question: str,
    sources: list[LoadedChatSource],
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int,
    citation_prefix_tokens: list[int] | None = None,
) -> Any:  # ContextChatPlan
    """Build the citation catalog and preflight strategy for one answer-from-context turn."""
    citation_entries = build_chat_citation_entries(sources)
    llm_citation_catalog = build_llm_citation_catalog(citation_entries)
    return plan_context_chat_request(
        original_question=original_question,
        contexts=_build_context_model_payloads(sources),
        history=history,
        user_content=user_content,
        config=config,
        token_cap=token_cap,
        citation_catalog=llm_citation_catalog,
        citation_prefix_tokens=citation_prefix_tokens,
    )


def answer_from_context_reply(
    *,
    run_id: str,
    original_question: str,
    sources: list[LoadedChatSource],
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int,
    api_key_override: str | None,
    citation_prefix_tokens: list[int] | None = None,
) -> tuple[str, list[dict[str, object]], str | None]:
    """Generate a grounded reply and resolve synthetic citations."""
    citation_entries = build_chat_citation_entries(sources)
    citation_entries_by_ref_id = {entry.ref_id: entry for entry in citation_entries}
    llm_citation_catalog = build_llm_citation_catalog(citation_entries)
    chat_kwargs: dict[str, object] = {
        "original_question": original_question,
        "contexts": _build_context_model_payloads(sources),
        "history": history,
        "user_content": user_content,
        "config": config,
        "token_cap": token_cap,
        "citation_catalog": llm_citation_catalog,
        "run_id": run_id,
    }
    if citation_prefix_tokens is not None:
        chat_kwargs["citation_prefix_tokens"] = citation_prefix_tokens
    if api_key_override is not None:
        chat_kwargs["api_key_override"] = api_key_override

    retry_settings = RetrySettings.bounded(
        max_attempts=config.retry.max_attempts,
        backoff_base_seconds=config.retry.backoff_base_seconds,
        backoff_max_seconds=config.retry.backoff_max_seconds,
    )
    max_attempts = retry_settings.max_attempts
    assistant_text = ""
    assistant_citations: list[dict[str, object]] = []
    assistant_citation_warning: str | None = None
    has_valid_citations = not citation_entries
    for attempt in range(1, max_attempts + 1):
        attempt_kwargs = dict(chat_kwargs)
        if attempt > 1:
            attempt_kwargs["retry_missing_citation"] = True
        assistant_text = generate_context_chat_reply(**attempt_kwargs)
        assistant_citations, has_valid_citations = _resolve_assistant_citations(
            assistant_text,
            citation_entries_by_ref_id,
        )
        if has_valid_citations:
            break
        if attempt < max_attempts:
            delay_seconds = compute_retry_delay_seconds(attempt, retry_settings)
            log_retry_event(
                operation="chat.citation_coverage",
                run_id=run_id,
                attempt=attempt,
                max_attempts=max_attempts,
                error_type="MissingCitationCoverage",
                error_message="Assistant response is missing valid [ref_n] citations.",
                next_backoff_seconds=delay_seconds,
                context={"source_count": len(sources)},
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            continue
        log_retry_exhausted(
            operation="chat.citation_coverage",
            run_id=run_id,
            attempt=attempt,
            max_attempts=max_attempts,
            error_type="MissingCitationCoverage",
            error_message="Assistant response is missing valid [ref_n] citations.",
            context={"source_count": len(sources)},
        )
        assistant_citation_warning = "Assistant response is missing valid [ref_n] citations."
        logger.warning(
            "Context chat response missing valid [ref_n] citations run_id=%s attempts=%d",
            run_id,
            max_attempts,
        )
    return assistant_text, assistant_citations, assistant_citation_warning


def build_out_of_scope_reply() -> str:
    """Return the deterministic out-of-scope assistant reply."""
    return (
        "This follow-up is outside the current city-report context. "
        "Start a new chat for that topic."
    )


def build_city_clarification_reply() -> str:
    """Return the deterministic city-clarification assistant reply."""
    return (
        "I can refresh the context for one city at a time. "
        "Please ask about exactly one city."
    )


def build_followup_failure_reply(target_city: str | None) -> str:
    """Return the deterministic failure reply for unsuccessful follow-up search."""
    city_label = format_city_stem(target_city or "") or "that city"
    return (
        f"I could not refresh the context for {city_label} just now, "
        "so I cannot answer this reliably."
    )


def build_unavailable_city_reply(target_city: str | None) -> str:
    """Return the deterministic reply when a city is not available for follow-up search."""
    city_label = format_city_stem(target_city or "") or "that city"
    return f"We don't have {city_label} in the current city list. Please choose another one."


def is_unavailable_city_result(search_result: ChatFollowupSearchResult) -> bool:
    """Return true when a follow-up search failed because the city is unavailable."""
    from backend.api.services.chat_followup_research import CHAT_FOLLOWUP_CITY_UNAVAILABLE
    return search_result.error_code == CHAT_FOLLOWUP_CITY_UNAVAILABLE


def build_routing_metadata(
    *,
    action: str,
    reason: str,
    target_city: str | None = None,
    bundle_id: str | None = None,
    pending_user_message: str | None = None,
) -> dict[str, object]:
    """Build assistant routing metadata for persistence."""
    payload: dict[str, object] = {
        "action": action,
        "reason": reason,
    }
    if target_city:
        payload["target_city"] = target_city
    if bundle_id:
        payload["bundle_id"] = bundle_id
    if pending_user_message:
        payload["pending_user_message"] = pending_user_message
    return payload


def next_turn_index(history: list[dict[str, str]]) -> int:
    """Return the next user-turn index for bundle id generation."""
    user_turns = sum(1 for item in history if item.get("role") == "user")
    return user_turns + 1


def build_chat_job_processor(
    *,
    run_store: Any,  # RunStore
    chat_memory_store: Any,  # ChatMemoryStore
    config_path: Any,  # Path
) -> Callable[[Any], Any]:  # Callable[[StartChatJobCommand], ChatJobResult]
    """Build the split-mode chat job worker used by the background executor."""
    from pathlib import Path
    from backend.utils.config import load_config
    from backend.api.services.chat_context_loader import load_context_for_run_id

    def _processor(command: Any) -> Any:  # command: StartChatJobCommand -> ChatJobResult
        from backend.api.services.chat_jobs import ChatJobResult
        
        session = chat_memory_store.get_session(command.run_id, command.conversation_id)
        config = load_config(config_path)
        config.enable_sql = False
        loaded_contexts = [
            load_context_for_run_id(run_store, context_run_id, config)
            for context_run_id in command.context_run_ids
        ]
        if not loaded_contexts:
            raise ValueError("No valid run contexts remain for the queued split-mode chat job.")
        from backend.api.services.chat_context_loader import load_followup_bundles_by_ids
        loaded_followup_bundles = load_followup_bundles_by_ids(
            run_store=run_store,
            run_id=command.run_id,
            conversation_id=command.conversation_id,
            bundle_ids=command.followup_bundle_ids,
            config=config,
        )
        sources = build_chat_sources(loaded_contexts, loaded_followup_bundles)
        from backend.api.services.chat_session_helpers import as_session_prompt_context_cache
        prompt_cache = as_session_prompt_context_cache(
            session,
            context_run_ids=[context.run_id for context in loaded_contexts],
            followup_bundle_ids=[bundle.bundle_id for bundle in loaded_followup_bundles],
        )
        assistant_text, assistant_citations, assistant_citation_warning = (
            answer_from_context_reply(
                run_id=command.run_id,
                original_question=command.original_question,
                sources=sources,
                history=command.history,
                user_content=command.user_content,
                config=config,
                token_cap=command.token_cap,
                api_key_override=command.api_key_override,
                citation_prefix_tokens=(
                    prompt_cache.citation_prefix_tokens if prompt_cache is not None else None
                ),
            )
        )
        return ChatJobResult(
            assistant_content=assistant_text,
            assistant_citations=assistant_citations or None,
            assistant_citation_warning=assistant_citation_warning,
        )

    return _processor


def queue_split_context_chat_job(
    *,
    run_id: str,
    conversation_id: str,
    payload_content: str,
    original_question: str,
    effective_user_content: str,
    history: list[dict[str, str]],
    context_run_ids: list[str],
    followup_bundle_ids: list[str],
    token_cap: int,
    assistant_routing: dict[str, object] | None,
    api_key_override: str | None,
    chat_memory_store: Any,  # ChatMemoryStore
    chat_job_executor: Any,  # ChatJobExecutor
) -> Any:  # SendChatMessageResponse
    """Persist one split-mode user turn and queue the assistant answer in the background."""
    from backend.api.services.chat_jobs import StartChatJobCommand
    from backend.api.models import ChatMessageJobAcceptedResponse
    from backend.api.services.chat_session_helpers import as_message, as_chat_routing

    job_id = uuid4().hex
    _, pending_job = chat_memory_store.create_pending_job(
        run_id=run_id,
        conversation_id=conversation_id,
        job_id=job_id,
    )
    try:
        _, user_message = chat_memory_store.append_user_message(
            run_id=run_id,
            conversation_id=conversation_id,
            content=payload_content,
        )
    except Exception:
        chat_memory_store.clear_pending_job(
            run_id=run_id,
            conversation_id=conversation_id,
            job_id=job_id,
        )
        raise
    command = StartChatJobCommand(
        run_id=run_id,
        conversation_id=conversation_id,
        job_id=job_id,
        job_number=int(pending_job["job_number"]),
        original_question=original_question,
        user_content=effective_user_content,
        history=history,
        context_run_ids=context_run_ids,
        followup_bundle_ids=followup_bundle_ids,
        token_cap=token_cap,
        assistant_routing=assistant_routing,
        api_key_override=api_key_override,
    )
    try:
        chat_job_executor.submit(command)
    except Exception:  # noqa: BLE001
        from backend.api.services.chat_jobs import build_chat_job_failure_message
        
        logger.exception(
            "Context chat split job submission failed run_id=%s conversation_id=%s job_id=%s job_number=%s",
            run_id,
            conversation_id,
            job_id,
            pending_job["job_number"],
        )
        chat_memory_store.clear_pending_job(
            run_id=run_id,
            conversation_id=conversation_id,
            job_id=job_id,
        )
        _, assistant_message = chat_memory_store.append_assistant_message(
            run_id=run_id,
            conversation_id=conversation_id,
            content=build_chat_job_failure_message(),
        )
        return SendChatMessageCompletedResponse(
            mode="completed",
            run_id=run_id,
            conversation_id=conversation_id,
            user_message=as_message(user_message),
            assistant_message=as_message(assistant_message),
        )

    pending_handle = as_pending_job(run_id, conversation_id, pending_job)
    if pending_handle is None:
        raise ValueError("Queued split-mode chat job is missing pending-job metadata.")
    return ChatMessageJobAcceptedResponse(
        mode="queued",
        run_id=run_id,
        conversation_id=conversation_id,
        user_message=as_message(user_message),
        job=pending_handle,
        routing=as_chat_routing(assistant_routing),
    )


__all__ = [
    "build_chat_sources",
    "build_chat_citation_entries",
    "build_llm_citation_catalog",
    "build_router_payload",
    "build_context_reply_plan",
    "answer_from_context_reply",
    "build_out_of_scope_reply",
    "build_city_clarification_reply",
    "build_followup_failure_reply",
    "build_unavailable_city_reply",
    "is_unavailable_city_result",
    "build_routing_metadata",
    "next_turn_index",
    "log_chat_router_preflight",
    "log_context_reply_plan",
    "log_chat_request_summary",
    "build_chat_job_processor",
    "queue_split_context_chat_job",
]
