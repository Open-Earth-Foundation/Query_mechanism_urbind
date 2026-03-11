"""High-level send-message orchestration for run-scoped context chat."""

from __future__ import annotations

import logging

from backend.api.models import (
    SendChatMessageCompletedResponse,
    SendChatMessageRequest,
)
from backend.api.services.chat_followup_flow import refresh_followup_context, resolve_followup_decision
from backend.api.services.chat_errors import ChatNoContextSourcesError
from backend.api.services.chat_jobs import ChatJobExecutor
from backend.api.services.chat_memory import (
    ChatMemoryStore,
    ChatSessionPendingJobError,
)
from backend.api.services.chat_reply_helpers import (
    answer_from_context_reply,
    build_chat_citation_entries,
    build_chat_sources,
    build_city_clarification_reply,
    build_context_reply_plan,
    build_out_of_scope_reply,
    build_routing_metadata,
    log_chat_request_summary,
    log_context_reply_plan,
)
from backend.api.services.chat_session_helpers import (
    as_message,
    get_or_build_session_prompt_context_cache,
    pending_job_payload,
    resolve_session_contexts,
)
from backend.api.services.chat_split_flow import queue_split_context_chat_job
from backend.api.services.context_chat import (
    estimate_context_window,
    resolve_chat_token_cap,
)
from backend.api.services.models import (
    ChatSendServiceResult,
    ContextChatPlan,
    LoadedChatSessionState,
)
from backend.api.services.run_store import RunRecord, RunStore
from backend.modules.orchestrator.models import ChatFollowupDecision
from backend.utils.config import AppConfig, get_openrouter_api_key

logger = logging.getLogger(__name__)


def process_send_chat_message(
    *,
    run_id: str,
    conversation_id: str,
    payload: SendChatMessageRequest,
    run_store: RunStore,
    run_record: RunRecord,
    store: ChatMemoryStore,
    chat_job_executor: ChatJobExecutor,
    config: AppConfig,
    api_key_override: str | None,
) -> ChatSendServiceResult:
    """Process one chat message against the selected run and session contexts."""
    session = store.get_session(run_id, conversation_id)
    pending_job = pending_job_payload(session)
    if pending_job is not None:
        raise ChatSessionPendingJobError(
            "This chat session already has a pending long-context answer. "
            "Wait for it to finish before sending another message."
        )

    token_cap = resolve_chat_token_cap(config)
    state = _load_chat_session_state(
        run_id=run_id,
        conversation_id=conversation_id,
        run_store=run_store,
        run_record=run_record,
        store=store,
        session=session,
        config=config,
        token_cap=token_cap,
    )
    history = _build_history(state.session)
    user_content = payload.content.strip()
    clarification_city = (payload.clarification_city or "").strip()
    assistant_text = ""
    assistant_citations: list[dict[str, object]] = []
    assistant_citation_warning: str | None = None
    assistant_routing: dict[str, object] | None = None

    log_chat_request_summary(
        run_id=run_id,
        conversation_id=conversation_id,
        selected_run_ids=state.selected_run_ids,
        loaded_contexts=state.loaded_contexts,
        loaded_followup_bundles=state.loaded_followup_bundles,
        history=history,
        token_cap=token_cap,
        followup_search_enabled=config.chat.followup_search_enabled,
        clarification_city=clarification_city or None,
    )

    if not config.chat.followup_search_enabled:
        direct_result = _maybe_queue_split_reply(
            run_id=run_id,
            conversation_id=conversation_id,
            payload=payload,
            run_record=run_record,
            store=store,
            chat_job_executor=chat_job_executor,
            state=state,
            history=history,
            user_content=user_content,
            config=config,
            token_cap=token_cap,
            assistant_routing=None,
            api_key_override=api_key_override,
        )
        if direct_result is not None:
            return direct_result
        assistant_text, assistant_citations, assistant_citation_warning = (
            answer_from_context_reply(
                run_id=run_id,
                original_question=run_record.question,
                sources=state.sources,
                history=history,
                user_content=user_content,
                config=config,
                token_cap=token_cap,
                api_key_override=api_key_override,
                citation_prefix_tokens=state.prompt_cache.citation_prefix_tokens,
            )
        )
    else:
        effective_api_key = (
            api_key_override if api_key_override is not None else get_openrouter_api_key()
        )
        routing_decision = resolve_followup_decision(
            clarification_city=clarification_city or None,
            user_message=user_content,
            original_question=run_record.question,
            history=history,
            selected_run_ids=state.selected_run_ids,
            loaded_contexts=state.loaded_contexts,
            loaded_followup_bundles=state.loaded_followup_bundles,
            sources=state.sources,
            config=config,
            effective_api_key=effective_api_key,
            run_id=run_id,
            conversation_id=conversation_id,
        )
        assistant_text, assistant_routing, state = _resolve_followup_reply(
            run_id=run_id,
            conversation_id=conversation_id,
            run_store=run_store,
            run_record=run_record,
            store=store,
            state=state,
            history=history,
            user_content=user_content,
            routing_decision=routing_decision,
            config=config,
            token_cap=token_cap,
            effective_api_key=effective_api_key,
        )
        if assistant_text:
            assistant_citations = []
            assistant_citation_warning = None
        else:
            queued_result = _maybe_queue_split_reply(
                run_id=run_id,
                conversation_id=conversation_id,
                payload=payload,
                run_record=run_record,
                store=store,
                chat_job_executor=chat_job_executor,
                state=state,
                history=history,
                user_content=user_content,
                config=config,
                token_cap=token_cap,
                assistant_routing=assistant_routing,
                api_key_override=api_key_override,
            )
            if queued_result is not None:
                return queued_result
            assistant_text, assistant_citations, assistant_citation_warning = (
                answer_from_context_reply(
                    run_id=run_id,
                    original_question=run_record.question,
                    sources=state.sources,
                    history=history,
                    user_content=user_content,
                    config=config,
                    token_cap=token_cap,
                    api_key_override=api_key_override,
                    citation_prefix_tokens=state.prompt_cache.citation_prefix_tokens,
                )
            )

    _, user_message, assistant_message = store.append_turn(
        run_id=run_id,
        conversation_id=conversation_id,
        user_content=user_content,
        assistant_content=assistant_text,
        assistant_citations=assistant_citations or None,
        assistant_citation_warning=assistant_citation_warning,
        assistant_routing=assistant_routing,
    )
    if state.excluded_context_ids:
        filtered_ids = [context.run_id for context in state.loaded_contexts]
        store.update_context_runs(run_id, conversation_id, filtered_ids or [run_id])

    return ChatSendServiceResult(
        response=SendChatMessageCompletedResponse(
            mode="completed",
            run_id=run_id,
            conversation_id=conversation_id,
            user_message=as_message(user_message),
            assistant_message=as_message(assistant_message),
        )
    )
def _build_history(session: dict[str, object]) -> list[dict[str, str]]:
    """Extract persisted transcript turns into the router/chat history format."""
    history: list[dict[str, str]] = []
    messages = session.get("messages")
    if not isinstance(messages, list):
        return history
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            history.append({"role": role, "content": content})
    return history


def _load_chat_session_state(
    *,
    run_id: str,
    conversation_id: str,
    run_store: RunStore,
    run_record: RunRecord,
    store: ChatMemoryStore,
    session: dict[str, object],
    config: AppConfig,
    token_cap: int,
) -> LoadedChatSessionState:
    """Load active contexts, prune invalid follow-up bundles, and build prompt cache.

    Raises:
        ChatNoContextSourcesError: When the session no longer has any usable contexts.
    """
    (
        selected_run_ids,
        loaded_contexts,
        excluded_context_ids,
        loaded_followup_bundles,
        excluded_followup_bundle_ids,
    ) = resolve_session_contexts(
        run_store,
        session,
        conversation_id=conversation_id,
        fallback_run_id=run_id,
        config=config,
        token_cap=token_cap,
    )
    if not loaded_contexts:
        raise ChatNoContextSourcesError(
            "No usable context sources are selected for this session. "
            "Open context manager and select at least one completed run."
        )
    if excluded_followup_bundle_ids:
        session = store.prune_followup_bundles(
            run_id,
            conversation_id,
            [bundle.bundle_id for bundle in loaded_followup_bundles],
        )
        (
            selected_run_ids,
            loaded_contexts,
            excluded_context_ids,
            loaded_followup_bundles,
            excluded_followup_bundle_ids,
        ) = resolve_session_contexts(
            run_store,
            session,
            conversation_id=conversation_id,
            fallback_run_id=run_id,
            config=config,
            token_cap=token_cap,
        )
    sources = build_chat_sources(loaded_contexts, loaded_followup_bundles)
    session, prompt_cache = get_or_build_session_prompt_context_cache(
        run_id=run_id,
        conversation_id=conversation_id,
        session=session,
        store=store,
        original_question=run_record.question,
        sources=sources,
        context_run_ids=[context.run_id for context in loaded_contexts],
        followup_bundle_ids=[bundle.bundle_id for bundle in loaded_followup_bundles],
        config=config,
        token_cap=token_cap,
        build_citation_catalog_fn=build_chat_citation_entries,
        estimate_context_window_fn=estimate_context_window,
    )
    return LoadedChatSessionState(
        session=session,
        selected_run_ids=selected_run_ids,
        loaded_contexts=loaded_contexts,
        excluded_context_ids=excluded_context_ids,
        loaded_followup_bundles=loaded_followup_bundles,
        excluded_followup_bundle_ids=excluded_followup_bundle_ids,
        sources=sources,
        prompt_cache=prompt_cache,
    )


def _maybe_queue_split_reply(
    *,
    run_id: str,
    conversation_id: str,
    payload: SendChatMessageRequest,
    run_record: RunRecord,
    store: ChatMemoryStore,
    chat_job_executor: ChatJobExecutor,
    state: LoadedChatSessionState,
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int,
    assistant_routing: dict[str, object] | None,
    api_key_override: str | None,
) -> ChatSendServiceResult | None:
    """Queue the reply when the direct prompt plan exceeds the configured cap."""
    plan = _build_and_log_context_reply_plan(
        run_id=run_id,
        conversation_id=conversation_id,
        run_record=run_record,
        state=state,
        history=history,
        user_content=user_content,
        config=config,
        token_cap=token_cap,
    )
    if plan.mode != "split":
        return None
    logger.info(
        "Context chat split job accepted run_id=%s conversation_id=%s contexts=%s followup_bundles=%s estimated_prompt_tokens=%s context_tokens=%s split_reason=%s",
        run_id,
        conversation_id,
        state.selected_run_ids,
        [bundle.bundle_id for bundle in state.loaded_followup_bundles],
        plan.estimated_prompt_tokens,
        plan.context_tokens,
        plan.split_reason,
    )
    response = queue_split_context_chat_job(
        run_id=run_id,
        conversation_id=conversation_id,
        user_content=payload.content,
        original_question=run_record.question,
        effective_user_content=user_content,
        history=history,
        context_run_ids=state.selected_run_ids,
        followup_bundle_ids=[bundle.bundle_id for bundle in state.loaded_followup_bundles],
        token_cap=token_cap,
        assistant_routing=assistant_routing,
        api_key_override=api_key_override,
        chat_memory_store=store,
        chat_job_executor=chat_job_executor,
    )
    return ChatSendServiceResult(response=response, queued=response.mode == "queued")


def _build_and_log_context_reply_plan(
    *,
    run_id: str,
    conversation_id: str,
    run_record: RunRecord,
    state: LoadedChatSessionState,
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int,
) -> ContextChatPlan:
    """Build and log the prompt plan for the current session state."""
    plan = build_context_reply_plan(
        original_question=run_record.question,
        sources=state.sources,
        history=history,
        user_content=user_content,
        config=config,
        token_cap=token_cap,
        citation_prefix_tokens=state.prompt_cache.citation_prefix_tokens,
    )
    log_context_reply_plan(
        run_id=run_id,
        conversation_id=conversation_id,
        selected_run_ids=state.selected_run_ids,
        loaded_contexts=state.loaded_contexts,
        loaded_followup_bundles=state.loaded_followup_bundles,
        sources=state.sources,
        plan=plan,
    )
    return plan


def _resolve_followup_reply(
    *,
    run_id: str,
    conversation_id: str,
    run_store: RunStore,
    run_record: RunRecord,
    store: ChatMemoryStore,
    state: LoadedChatSessionState,
    history: list[dict[str, str]],
    user_content: str,
    routing_decision: ChatFollowupDecision | None,
    config: AppConfig,
    token_cap: int,
    effective_api_key: str,
) -> tuple[str | None, dict[str, object] | None, LoadedChatSessionState]:
    """Resolve the follow-up branch before final answer generation."""
    if routing_decision is None or routing_decision.action == "answer_from_context":
        assistant_routing = None
        if routing_decision is not None:
            assistant_routing = build_routing_metadata(
                action=routing_decision.action,
                reason=routing_decision.reason,
                target_city=routing_decision.target_city,
            )
        return None, assistant_routing, state

    if routing_decision.action == "out_of_scope":
        return (
            build_out_of_scope_reply(),
            build_routing_metadata(
                action=routing_decision.action,
                reason=routing_decision.reason,
            ),
            state,
        )

    if routing_decision.action == "needs_city_clarification":
        return (
            build_city_clarification_reply(),
            build_routing_metadata(
                action=routing_decision.action,
                reason=routing_decision.reason,
                pending_user_message=user_content,
            ),
            state,
        )

    target_city = routing_decision.target_city
    if not target_city:
        return (
            build_city_clarification_reply(),
            build_routing_metadata(
                action="needs_city_clarification",
                reason="Router requested a city search but did not provide exactly one city.",
                pending_user_message=user_content,
            ),
            state,
        )

    refresh = refresh_followup_context(
        run_store=run_store,
        store=store,
        session=state.session,
        run_id=run_id,
        conversation_id=conversation_id,
        history=history,
        user_message=user_content,
        target_city=target_city,
        rewritten_question=routing_decision.rewritten_question or user_content,
        routing_decision=routing_decision,
        config=config,
        api_key=effective_api_key,
    )
    if refresh.assistant_text is not None:
        return refresh.assistant_text, refresh.assistant_routing, state

    reloaded_state = _load_chat_session_state(
        run_id=run_id,
        conversation_id=conversation_id,
        run_store=run_store,
        run_record=run_record,
        store=store,
        session=refresh.session,
        config=config,
        token_cap=token_cap,
    )
    return None, refresh.assistant_routing, reloaded_state


__all__ = ["ChatSendServiceResult", "process_send_chat_message"]
