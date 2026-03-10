"""Session management helpers for chat operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
import logging

from fastapi import HTTPException, status

from backend.api.models import ChatSessionResponse, ChatJobStatusResponse
from backend.api.services.chat_context_loader import (
    LoadedContext,
    LoadedFollowupBundle,
    LoadedChatSource,
    load_context_for_run_id,
    load_followup_bundle,
)
from backend.api.services.context_chat import (
    build_citation_catalog_token_cache,
    estimate_context_window,
    ContextChatPlan,
)
from backend.api.services.context_prompt_cache import PromptContextKind
from backend.api.services.run_store import RunStore, SUCCESS_STATUSES
from backend.api.services.chat_memory import ChatMemoryStore
from backend.utils.config import AppConfig
from backend.utils.tokenization import count_tokens
import json

logger = logging.getLogger(__name__)

_SESSION_PROMPT_CONTEXT_CACHE_KEY = "prompt_context_cache"


@dataclass(frozen=True)
class ChatCitationEntry:
    """Deterministic chat citation mapping entry."""

    ref_id: str
    city_name: str
    quote: str
    partial_answer: str
    source_type: Literal["run", "followup_bundle"]
    source_id: str
    source_ref_id: str


@dataclass(frozen=True)
class SessionPromptContextCache:
    """Combined prompt-context cache for one exact session source selection."""

    context_run_ids: list[str]
    followup_bundle_ids: list[str]
    mode: Literal["direct", "split"]
    prompt_context_tokens: int
    prompt_context_kind: PromptContextKind
    citation_catalog_entry_count: int
    citation_ref_ids_in_order: list[str]
    citation_prefix_tokens: list[int]


def as_datetime(value: object) -> datetime:
    """Parse session timestamp value."""
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError("Invalid timestamp in chat session payload.")


def as_chat_citations(value: object) -> list[dict[str, str]] | None:
    """Normalize optional assistant citation metadata from persisted messages."""
    if not isinstance(value, list):
        return None
    citations: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("ref_id", "")).strip()
        city_name = str(item.get("city_name", "")).strip()
        source_type = str(item.get("source_type", "")).strip()
        source_id = str(item.get("source_id", "")).strip()
        source_ref_id = str(item.get("source_ref_id", "")).strip()
        if not ref_id or source_type not in {"run", "followup_bundle"} or not source_id:
            continue
        if not source_ref_id:
            continue
        citations.append(
            {
                "ref_id": ref_id,
                "city_name": city_name,
                "source_type": source_type,
                "source_id": source_id,
                "source_ref_id": source_ref_id,
            }
        )
    return citations if citations else None


def as_chat_routing(value: object) -> dict[str, Any] | None:
    """Normalize optional assistant routing metadata."""
    if not isinstance(value, dict):
        return None
    action = str(value.get("action", "")).strip()
    reason = str(value.get("reason", "")).strip()
    if action not in {
        "answer_from_context",
        "search_single_city",
        "out_of_scope",
        "needs_city_clarification",
    }:
        return None
    if not reason:
        return None
    target_city = str(value.get("target_city", "")).strip()
    bundle_id = str(value.get("bundle_id", "")).strip()
    pending_user_message = str(value.get("pending_user_message", "")).strip()
    return {
        "action": action,
        "reason": reason,
        "target_city": target_city or None,
        "bundle_id": bundle_id or None,
        "pending_user_message": pending_user_message or None,
    }


def build_chat_job_status_url(run_id: str, conversation_id: str, job_id: str) -> str:
    """Build the relative polling path for one chat job."""
    return (
        f"/api/v1/runs/{run_id}/chat/sessions/{conversation_id}"
        f"/jobs/{job_id}"
    )


def as_pending_job(
    run_id: str,
    conversation_id: str,
    value: object,
) -> dict[str, Any] | None:
    """Normalize the persisted pending-job payload into the API model."""
    if not isinstance(value, dict):
        return None
    job_id = str(value.get("job_id", "")).strip()
    status_value = str(value.get("status", "")).strip()
    job_number = value.get("job_number")
    if not job_id or status_value not in {"queued", "running", "completed", "failed"}:
        return None
    if not isinstance(job_number, int) or job_number <= 0:
        return None
    return {
        "job_id": job_id,
        "job_number": job_number,
        "status": status_value,
        "status_url": build_chat_job_status_url(run_id, conversation_id, job_id),
    }


def as_message(payload: object) -> Any:
    """Convert stored message payload into API model."""
    from backend.api.models import ChatMessage
    
    if not isinstance(payload, dict):
        raise ValueError("Invalid message payload.")
    role = payload.get("role")
    content = payload.get("content")
    created_at = payload.get("created_at")
    citation_warning = payload.get("citation_warning")
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid message role.")
    if not isinstance(content, str):
        raise ValueError("Invalid message content.")
    citations = as_chat_citations(payload.get("citations"))
    routing = as_chat_routing(payload.get("routing"))
    return ChatMessage(
        role=role,
        content=content,
        created_at=as_datetime(created_at),
        citations=citations,
        citation_warning=str(citation_warning) if isinstance(citation_warning, str) else None,
        routing=routing,
    )


def as_chat_job_status_response(record: Any) -> ChatJobStatusResponse:
    """Convert one persisted chat-job record into the polling response model."""
    return ChatJobStatusResponse(
        run_id=record.run_id,
        conversation_id=record.conversation_id,
        job_id=record.job_id,
        job_number=record.job_number,
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        finish_reason=record.finish_reason,
        error=record.error,
    )


def selected_context_run_ids(
    session: dict[str, object], fallback_run_id: str
) -> list[str]:
    """Extract selected context run ids from session payload."""
    raw = session.get("context_run_ids")
    if not isinstance(raw, list):
        return [fallback_run_id]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        deduped.append(cleaned)
        seen.add(cleaned)
    if not deduped:
        return [fallback_run_id]
    return deduped


def selected_followup_bundles(session: dict[str, object]) -> list[dict[str, str]]:
    """Extract normalized follow-up bundle metadata from session payload."""
    raw = session.get("followup_bundles")
    if not isinstance(raw, list):
        return []
    bundles: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        bundle_id = str(item.get("bundle_id", "")).strip()
        city_key = str(item.get("city_key", "")).strip()
        target_city = str(item.get("target_city", "")).strip()
        created_at = str(item.get("created_at", "")).strip()
        if not bundle_id or bundle_id in seen or not city_key or not target_city or not created_at:
            continue
        bundles.append(
            {
                "bundle_id": bundle_id,
                "city_key": city_key,
                "target_city": target_city,
                "created_at": created_at,
            }
        )
        seen.add(bundle_id)
    return bundles


def pending_job_payload(session: dict[str, object]) -> dict[str, object] | None:
    """Return the raw pending-job payload when present."""
    raw = session.get("pending_job")
    if isinstance(raw, dict):
        return raw
    return None


def apply_followup_bundle_token_cap(
    bundles: list[LoadedFollowupBundle],
    token_cap: int,
    starting_total: int,
) -> tuple[list[LoadedFollowupBundle], list[str]]:
    """Apply prompt-budget cap to auto-attached follow-up bundles only."""
    included: list[LoadedFollowupBundle] = []
    excluded: list[str] = []
    running_total = starting_total
    for bundle in bundles:
        next_total = running_total + bundle.prompt_context_tokens
        if next_total <= token_cap:
            included.append(bundle)
            running_total = next_total
        else:
            excluded.append(bundle.bundle_id)
    return included, excluded


def resolve_session_contexts(
    run_store: RunStore,
    session: dict[str, object],
    conversation_id: str,
    fallback_run_id: str,
    config: AppConfig,
    token_cap: int,
) -> tuple[
    list[str],
    list[LoadedContext],
    list[str],
    list[LoadedFollowupBundle],
    list[str],
]:
    """Resolve selected contexts for a chat session.

    Manual run contexts remain selected even when they exceed the direct prompt cap.
    The cap is still applied to auto-attached follow-up bundles so chat-owned searches
    do not grow without bound.
    """
    selected_ids = selected_context_run_ids(session, fallback_run_id)
    try:
        base_context = load_context_for_run_id(run_store, fallback_run_id, config)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "The base run context is no longer usable for chat. "
                f"Fix run artifacts for `{fallback_run_id}` and retry."
            ),
        ) from exc

    loaded_contexts: list[LoadedContext] = []
    excluded: list[str] = []
    for context_run_id in selected_ids:
        if context_run_id == fallback_run_id:
            continue
        try:
            loaded_contexts.append(load_context_for_run_id(run_store, context_run_id, config))
        except ValueError:
            excluded.append(context_run_id)
    included = [base_context, *loaded_contexts]
    included_total = sum(context.prompt_context_tokens for context in included)

    loaded_followup_bundles: list[LoadedFollowupBundle] = []
    excluded_followup_bundle_ids: list[str] = []
    for bundle_meta in selected_followup_bundles(session):
        try:
            loaded_followup_bundles.append(
                load_followup_bundle(
                    run_store=run_store,
                    run_id=fallback_run_id,
                    conversation_id=conversation_id,
                    bundle_id=bundle_meta["bundle_id"],
                    config=config,
                    bundle_meta=bundle_meta,
                )
            )
        except ValueError:
            excluded_followup_bundle_ids.append(bundle_meta["bundle_id"])

    included_followups, cap_excluded_followups = apply_followup_bundle_token_cap(
        loaded_followup_bundles,
        token_cap=token_cap,
        starting_total=included_total,
    )
    excluded_followup_bundle_ids.extend(cap_excluded_followups)
    return selected_ids, included, excluded, included_followups, excluded_followup_bundle_ids


def normalize_prompt_context_kind(value: object) -> PromptContextKind | None:
    """Return the normalized prompt-context kind when valid."""
    if value in {"citation_catalog", "serialized_contexts"}:
        return value
    return None


def as_session_prompt_context_cache(
    session: dict[str, object],
    *,
    context_run_ids: list[str],
    followup_bundle_ids: list[str],
) -> SessionPromptContextCache | None:
    """Return the persisted session prompt cache when it matches the active source set."""
    raw_cache = session.get(_SESSION_PROMPT_CONTEXT_CACHE_KEY)
    if not isinstance(raw_cache, dict):
        return None
    raw_context_run_ids = raw_cache.get("context_run_ids")
    raw_followup_bundle_ids = raw_cache.get("followup_bundle_ids")
    raw_mode = raw_cache.get("mode")
    raw_prompt_context_tokens = raw_cache.get("prompt_context_tokens")
    raw_prompt_context_kind = normalize_prompt_context_kind(
        raw_cache.get("prompt_context_kind")
    )
    raw_citation_catalog_entry_count = raw_cache.get("citation_catalog_entry_count")
    raw_ref_ids = raw_cache.get("citation_ref_ids_in_order")
    raw_prefix_tokens = raw_cache.get("citation_prefix_tokens")
    if raw_mode not in {"direct", "split"}:
        return None
    if not isinstance(raw_prompt_context_tokens, int) or raw_prompt_context_tokens < 0:
        return None
    if raw_prompt_context_kind is None:
        return None
    if not isinstance(raw_citation_catalog_entry_count, int) or raw_citation_catalog_entry_count < 0:
        return None
    if not isinstance(raw_context_run_ids, list) or not isinstance(raw_followup_bundle_ids, list):
        return None
    cached_context_run_ids = [
        value.strip()
        for value in raw_context_run_ids
        if isinstance(value, str) and value.strip()
    ]
    cached_followup_bundle_ids = [
        value.strip()
        for value in raw_followup_bundle_ids
        if isinstance(value, str) and value.strip()
    ]
    if cached_context_run_ids != context_run_ids or cached_followup_bundle_ids != followup_bundle_ids:
        return None
    if not isinstance(raw_ref_ids, list) or not isinstance(raw_prefix_tokens, list):
        return None
    ref_ids = [value for value in raw_ref_ids if isinstance(value, str) and value.strip()]
    prefix_tokens = [value for value in raw_prefix_tokens if isinstance(value, int) and value >= 0]
    if len(ref_ids) != len(raw_ref_ids) or len(prefix_tokens) != len(raw_prefix_tokens):
        return None
    if len(ref_ids) != raw_citation_catalog_entry_count:
        return None
    if len(prefix_tokens) != raw_citation_catalog_entry_count:
        return None
    return SessionPromptContextCache(
        context_run_ids=cached_context_run_ids,
        followup_bundle_ids=cached_followup_bundle_ids,
        mode=raw_mode,
        prompt_context_tokens=raw_prompt_context_tokens,
        prompt_context_kind=raw_prompt_context_kind,
        citation_catalog_entry_count=raw_citation_catalog_entry_count,
        citation_ref_ids_in_order=ref_ids,
        citation_prefix_tokens=prefix_tokens,
    )


def session_prompt_context_cache_payload(
    cache: SessionPromptContextCache,
) -> dict[str, object]:
    """Serialize one session prompt-context cache for persistence."""
    return {
        "context_run_ids": cache.context_run_ids,
        "followup_bundle_ids": cache.followup_bundle_ids,
        "mode": cache.mode,
        "prompt_context_tokens": cache.prompt_context_tokens,
        "prompt_context_kind": cache.prompt_context_kind,
        "citation_catalog_entry_count": cache.citation_catalog_entry_count,
        "citation_ref_ids_in_order": cache.citation_ref_ids_in_order,
        "citation_prefix_tokens": cache.citation_prefix_tokens,
    }


def build_session_prompt_context_cache(
    *,
    original_question: str,
    sources: list[LoadedChatSource],
    context_run_ids: list[str],
    followup_bundle_ids: list[str],
    config: AppConfig,
    token_cap: int,
    build_citation_catalog_fn,
    estimate_context_window_fn,
) -> SessionPromptContextCache:
    """Compute the combined prompt-context cache for the active session sources."""
    citation_entries = build_citation_catalog_fn(sources)
    llm_citation_catalog = [
        {
            "ref_id": entry.ref_id,
            "city_name": entry.city_name,
            "quote": entry.quote,
            "partial_answer": entry.partial_answer,
        }
        for entry in citation_entries
    ]
    citation_token_cache = build_citation_catalog_token_cache(llm_citation_catalog)
    context_models = [
        {
            "run_id": source.source_id,
            "question": source.question,
            "final_document": source.final_document,
            "context_bundle": source.context_bundle,
        }
        for source in sources
    ]
    prompt_estimate = estimate_context_window_fn(
        original_question=original_question,
        contexts=context_models,
        config=config,
        token_cap=token_cap,
        citation_catalog=llm_citation_catalog,
    )
    prompt_context_kind = normalize_prompt_context_kind(prompt_estimate.context_window_kind)
    if prompt_context_kind is None:
        prompt_context_kind = (
            "citation_catalog" if llm_citation_catalog else "serialized_contexts"
        )
    return SessionPromptContextCache(
        context_run_ids=list(context_run_ids),
        followup_bundle_ids=list(followup_bundle_ids),
        mode=prompt_estimate.mode,
        prompt_context_tokens=prompt_estimate.context_window_tokens or 0,
        prompt_context_kind=prompt_context_kind,
        citation_catalog_entry_count=len(llm_citation_catalog),
        citation_ref_ids_in_order=citation_token_cache.ordered_ref_ids,
        citation_prefix_tokens=citation_token_cache.prefix_tokens,
    )


def get_or_build_session_prompt_context_cache(
    *,
    run_id: str,
    conversation_id: str,
    session: dict[str, object],
    store: ChatMemoryStore,
    original_question: str,
    sources: list[LoadedChatSource],
    context_run_ids: list[str],
    followup_bundle_ids: list[str],
    config: AppConfig,
    token_cap: int,
    build_citation_catalog_fn,
    estimate_context_window_fn,
) -> tuple[dict[str, object], SessionPromptContextCache]:
    """Return cached session prompt metrics or compute and persist them once."""
    cached = as_session_prompt_context_cache(
        session,
        context_run_ids=context_run_ids,
        followup_bundle_ids=followup_bundle_ids,
    )
    if cached is not None:
        logger.info(
            "Session prompt cache hit run_id=%s conversation_id=%s contexts=%s followup_bundles=%s prompt_context_tokens=%d prompt_context_kind=%s",
            run_id,
            conversation_id,
            context_run_ids,
            followup_bundle_ids,
            cached.prompt_context_tokens,
            cached.prompt_context_kind,
        )
        return session, cached

    cache = build_session_prompt_context_cache(
        original_question=original_question,
        sources=sources,
        context_run_ids=context_run_ids,
        followup_bundle_ids=followup_bundle_ids,
        config=config,
        token_cap=token_cap,
        build_citation_catalog_fn=build_citation_catalog_fn,
        estimate_context_window_fn=estimate_context_window_fn,
    )
    session = store.update_prompt_context_cache(
        run_id=run_id,
        conversation_id=conversation_id,
        prompt_context_cache=session_prompt_context_cache_payload(cache),
    )
    logger.info(
        "Session prompt cache miss run_id=%s conversation_id=%s contexts=%s followup_bundles=%s prompt_context_tokens=%d prompt_context_kind=%s",
        run_id,
        conversation_id,
        context_run_ids,
        followup_bundle_ids,
        cache.prompt_context_tokens,
        cache.prompt_context_kind,
    )
    return session, cache


def build_session_contexts_response(
    run_id: str,
    conversation_id: str,
    run_store: RunStore,
    session: dict[str, object],
    store: ChatMemoryStore,
    config: AppConfig,
    token_cap: int,
    build_context_summary_fn,
    build_followup_bundle_summary_fn,
    build_chat_sources_fn,
    build_citation_catalog_fn,
    estimate_context_window_fn,
) -> Any:
    """Build session context payload for API response."""
    (
        selected_ids,
        included_contexts,
        excluded_ids,
        followup_bundles,
        excluded_followups,
    ) = resolve_session_contexts(
        run_store,
        session,
        conversation_id=conversation_id,
        fallback_run_id=run_id,
        config=config,
        token_cap=token_cap,
    )
    sources = build_chat_sources_fn(included_contexts, followup_bundles)
    session, prompt_cache = get_or_build_session_prompt_context_cache(
        run_id=run_id,
        conversation_id=conversation_id,
        session=session,
        store=store,
        original_question=included_contexts[0].question if included_contexts else "",
        sources=sources,
        context_run_ids=[context.run_id for context in included_contexts],
        followup_bundle_ids=[bundle.bundle_id for bundle in followup_bundles],
        config=config,
        token_cap=token_cap,
        build_citation_catalog_fn=build_citation_catalog_fn,
        estimate_context_window_fn=estimate_context_window_fn,
    )
    total_tokens = sum(context.total_tokens for context in included_contexts) + sum(
        bundle.total_tokens for bundle in followup_bundles
    )
    
    from backend.api.models import ChatSessionContextsResponse
    
    return ChatSessionContextsResponse(
        run_id=run_id,
        conversation_id=conversation_id,
        context_run_ids=selected_ids,
        contexts=[build_context_summary_fn(context) for context in included_contexts],
        followup_bundles=[build_followup_bundle_summary_fn(bundle) for bundle in followup_bundles],
        total_tokens=total_tokens,
        prompt_context_tokens=prompt_cache.prompt_context_tokens,
        prompt_context_kind=prompt_cache.prompt_context_kind,
        token_cap=token_cap,
        excluded_context_run_ids=excluded_ids,
        excluded_followup_bundle_ids=excluded_followups,
        is_capped=(
            len(excluded_ids) > 0
            or len(excluded_followups) > 0
            or prompt_cache.mode == "split"
            or prompt_cache.prompt_context_tokens > token_cap
        ),
    )


def as_session_response(payload: dict[str, object]) -> ChatSessionResponse:
    """Convert stored session payload into API model."""
    messages_raw = payload.get("messages")
    message_models: list[dict[str, Any]] = []
    if isinstance(messages_raw, list):
        message_models = [as_message(item) for item in messages_raw]

    run_id = payload.get("run_id")
    conversation_id = payload.get("conversation_id")
    if not isinstance(run_id, str) or not isinstance(conversation_id, str):
        raise ValueError("Invalid session payload identifiers.")
    return ChatSessionResponse(
        run_id=run_id,
        conversation_id=conversation_id,
        created_at=as_datetime(payload.get("created_at")),
        updated_at=as_datetime(payload.get("updated_at")),
        pending_job=as_pending_job(run_id, conversation_id, payload.get("pending_job")),
        messages=message_models,
    )


__all__ = [
    "ChatCitationEntry",
    "SessionPromptContextCache",
    "as_datetime",
    "as_message",
    "as_chat_citations",
    "as_chat_routing",
    "as_pending_job",
    "as_chat_job_status_response",
    "build_chat_job_status_url",
    "selected_context_run_ids",
    "selected_followup_bundles",
    "pending_job_payload",
    "apply_followup_bundle_token_cap",
    "resolve_session_contexts",
    "normalize_prompt_context_kind",
    "as_session_prompt_context_cache",
    "session_prompt_context_cache_payload",
    "build_session_prompt_context_cache",
    "get_or_build_session_prompt_context_cache",
    "build_session_contexts_response",
    "as_session_response",
]
