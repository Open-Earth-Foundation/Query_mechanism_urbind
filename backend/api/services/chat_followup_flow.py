"""Follow-up routing and city-refresh helpers for context chat."""

from __future__ import annotations

import logging

from backend.api.services.chat_followup_research import run_chat_followup_search
from backend.api.services.chat_memory import ChatMemoryStore
from backend.api.services.chat_reply_helpers import (
    build_followup_failure_reply,
    build_router_payload,
    build_routing_metadata,
    build_unavailable_city_reply,
    is_unavailable_city_result,
    log_chat_router_preflight,
    next_turn_index,
)
from backend.api.services.models import (
    FollowupSearchResolution,
    LoadedChatSource,
    LoadedContext,
    LoadedFollowupBundle,
)
from backend.api.services.run_store import RunStore
from backend.modules.orchestrator.agent import route_chat_followup
from backend.modules.orchestrator.models import ChatFollowupDecision
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)


def resolve_followup_decision(
    *,
    clarification_city: str | None,
    user_message: str,
    original_question: str,
    history: list[dict[str, str]],
    selected_run_ids: list[str],
    loaded_contexts: list[LoadedContext],
    loaded_followup_bundles: list[LoadedFollowupBundle],
    sources: list[LoadedChatSource],
    config: AppConfig,
    effective_api_key: str,
    run_id: str,
    conversation_id: str,
) -> ChatFollowupDecision | None:
    """Resolve the next follow-up action, bypassing the router for chosen cities."""
    resolved_city = (clarification_city or "").strip()
    if resolved_city:
        return ChatFollowupDecision(
            action="search_single_city",
            reason="User selected a single city after clarification request.",
            target_city=resolved_city,
            rewritten_question=user_message,
        )

    router_payload = build_router_payload(
        user_message=user_message,
        original_question=original_question,
        history=history,
        sources=sources,
        max_history_messages=config.chat.followup_router_max_history_messages,
        max_excerpts_per_source=config.chat.followup_router_max_excerpts_per_source,
    )
    log_chat_router_preflight(
        run_id=run_id,
        conversation_id=conversation_id,
        selected_run_ids=selected_run_ids,
        loaded_contexts=loaded_contexts,
        loaded_followup_bundles=loaded_followup_bundles,
        router_payload=router_payload,
    )
    try:
        return route_chat_followup(
            payload=router_payload,
            config=config,
            api_key=effective_api_key,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Chat follow-up router failed; falling back to answer_from_context. run_id=%s conversation_id=%s error=%s",
            run_id,
            conversation_id,
            exc,
        )
        return None


def refresh_followup_context(
    *,
    run_store: RunStore,
    store: ChatMemoryStore,
    session: dict[str, object],
    run_id: str,
    conversation_id: str,
    history: list[dict[str, str]],
    user_message: str,
    target_city: str,
    rewritten_question: str,
    routing_decision: ChatFollowupDecision,
    config: AppConfig,
    api_key: str,
) -> FollowupSearchResolution:
    """Run one city-specific follow-up search and attach the resulting bundle when successful."""
    search_result = run_chat_followup_search(
        runs_dir=run_store.runs_dir,
        run_id=run_id,
        conversation_id=conversation_id,
        turn_index=next_turn_index(history),
        question=rewritten_question,
        target_city=target_city,
        config=config,
        api_key=api_key,
    )
    assistant_routing = build_routing_metadata(
        action=routing_decision.action,
        reason=routing_decision.reason,
        target_city=target_city,
        bundle_id=search_result.bundle_id,
    )
    if search_result.status == "success" and search_result.excerpt_count > 0:
        updated_session = store.attach_followup_bundle(
            run_id=run_id,
            conversation_id=conversation_id,
            bundle_id=search_result.bundle_id,
            city_key=target_city.strip().casefold(),
            target_city=search_result.target_city,
            created_at=search_result.created_at.isoformat(),
            max_followup_bundles=config.chat.max_auto_followup_bundles,
        )
        return FollowupSearchResolution(
            session=updated_session,
            assistant_routing=assistant_routing,
        )
    if is_unavailable_city_result(search_result):
        return FollowupSearchResolution(
            session=session,
            assistant_text=build_unavailable_city_reply(target_city),
            assistant_routing=build_routing_metadata(
                action="needs_city_clarification",
                reason="Requested city is not available in the current searchable city list.",
                pending_user_message=user_message,
            ),
        )
    return FollowupSearchResolution(
        session=session,
        assistant_text=build_followup_failure_reply(target_city),
        assistant_routing=assistant_routing,
    )


__all__ = ["FollowupSearchResolution", "refresh_followup_context", "resolve_followup_decision"]
