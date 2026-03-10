"""Run-scoped context chat endpoints."""

from __future__ import annotations

from pathlib import Path
import logging

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from openai import APIStatusError, APITimeoutError, AuthenticationError

from backend.api.models import (
    ChatContextCatalogResponse,
    ChatContextSummary,
    ChatFollowupBundleSummary,
    ChatFollowupReferenceListResponse,
    ChatSessionContextsResponse,
    ChatSessionListResponse,
    ChatSessionResponse,
    CreateChatSessionRequest,
    SendChatMessageCompletedResponse,
    SendChatMessageRequest,
    SendChatMessageResponse,
    UpdateChatContextsRequest,
)
from backend.api.services import (
    ChatJobExecutor,
    ChatMemoryStore,
    ChatSessionExistsError,
    ChatSessionNotFoundError,
    ChatSessionPendingJobError,
    RunRecord,
    RunStore,
    build_reference_item,
    followup_bundle_dir,
    load_reference_records,
    resolve_chat_token_cap,
)
from backend.api.services.chat_context_loader import (
    fast_context_summary,
    list_available_context_summaries,
    load_context_for_run_id,
    load_followup_bundle,
    validate_context_run_id,
)
from backend.api.services.chat_session_helpers import (
    as_pending_job,
    as_session_response,
    build_session_contexts_response,
    pending_job_payload,
    resolve_session_contexts,
    selected_followup_bundles,
)
from backend.api.services.chat_reply_helpers import (
    answer_from_context_reply,
    build_chat_job_processor,
    build_chat_sources,
    build_city_clarification_reply,
    build_context_reply_plan,
    build_followup_failure_reply,
    build_out_of_scope_reply,
    build_routing_metadata,
    build_unavailable_city_reply,
    is_unavailable_city_result,
    log_chat_request_summary,
    log_context_reply_plan,
    log_chat_router_preflight,
    next_turn_index,
    queue_split_context_chat_job,
)
from backend.modules.orchestrator.agent import route_chat_followup
from backend.modules.orchestrator.utils.references import is_valid_ref_id
from backend.utils.config import AppConfig, get_openrouter_api_key, load_config
from backend.utils.retry import compute_retry_delay_seconds, log_retry_event
from backend.api.services.chat_followup_research import run_chat_followup_search

router = APIRouter()
logger = logging.getLogger(__name__)


def _resolve_api_key_override(raw: str | None) -> str | None:
    """Normalize optional API key header value."""
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    return cleaned


def _get_run_store(request: Request) -> RunStore:
    """Return run store from app state."""
    run_store = getattr(request.app.state, "run_store", None)
    if not isinstance(run_store, RunStore):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Run store is not initialized.",
        )
    return run_store


def _get_chat_memory_store(request: Request) -> ChatMemoryStore:
    """Return chat memory store from app state."""
    store = getattr(request.app.state, "chat_memory_store", None)
    if not isinstance(store, ChatMemoryStore):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat memory store is not initialized.",
        )
    return store


def _get_chat_job_store(request: Request):
    """Return chat job store from app state."""
    store = getattr(request.app.state, "chat_job_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat job store is not initialized.",
        )
    return store


def _get_chat_job_executor(request: Request) -> ChatJobExecutor:
    """Return chat job executor from app state."""
    executor = getattr(request.app.state, "chat_job_executor", None)
    if not isinstance(executor, ChatJobExecutor):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat job executor is not initialized.",
        )
    return executor


def _load_request_config(request: Request) -> AppConfig:
    """Load AppConfig using the config path resolved at API startup."""
    config_path = getattr(request.app.state, "config_path", Path("llm_config.yaml"))
    return load_config(Path(config_path))


def _require_chat_ready_run(run_id: str, request: Request) -> tuple[RunStore, RunRecord]:
    """Require run existence and completed status before chat access."""
    from backend.api.services import SUCCESS_STATUSES
    
    run_store = _get_run_store(request)
    run_record = run_store.get_run(run_id)
    if run_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run `{run_id}` was not found.",
        )
    if run_record.status not in SUCCESS_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run `{run_id}` is not ready for chat (status: `{run_record.status}`).",
        )
    return run_store, run_record


@router.get(
    "/chat/contexts",
    response_model=ChatContextCatalogResponse,
    name="list_chat_contexts",
)
def list_chat_contexts(request: Request) -> ChatContextCatalogResponse:
    """List all available completed run contexts for chat selection."""
    run_store = _get_run_store(request)
    config = _load_request_config(request)
    summaries = list_available_context_summaries(run_store, config)
    token_cap = resolve_chat_token_cap(config)
    return ChatContextCatalogResponse(
        contexts=summaries,
        total=len(summaries),
        token_cap=token_cap,
    )


@router.get(
    "/runs/{run_id}/chat/sessions",
    response_model=ChatSessionListResponse,
    name="list_chat_sessions",
)
def list_chat_sessions(run_id: str, request: Request) -> ChatSessionListResponse:
    """List chat sessions for a completed run."""
    _require_chat_ready_run(run_id, request)
    store = _get_chat_memory_store(request)
    conversations = store.list_sessions(run_id)
    return ChatSessionListResponse(
        run_id=run_id,
        conversations=conversations,
        total=len(conversations),
    )


@router.post(
    "/runs/{run_id}/chat/sessions",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    name="create_chat_session",
)
def create_chat_session(
    run_id: str,
    payload: CreateChatSessionRequest,
    request: Request,
) -> ChatSessionResponse:
    """Create a chat session scoped to a completed run."""
    _require_chat_ready_run(run_id, request)
    store = _get_chat_memory_store(request)
    try:
        session = store.create_session(run_id, conversation_id=payload.conversation_id)
    except ChatSessionExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return as_session_response(session)


@router.get(
    "/runs/{run_id}/chat/sessions/{conversation_id}",
    response_model=ChatSessionResponse,
    name="get_chat_session",
)
def get_chat_session(
    run_id: str,
    conversation_id: str,
    request: Request,
) -> ChatSessionResponse:
    """Fetch a chat session transcript."""
    _require_chat_ready_run(run_id, request)
    store = _get_chat_memory_store(request)
    try:
        session = store.get_session(run_id, conversation_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return as_session_response(session)


@router.get(
    "/runs/{run_id}/chat/sessions/{conversation_id}/jobs/{job_id}",
    response_model=None,
    name="get_chat_job_status",
)
def get_chat_job_status(
    run_id: str,
    conversation_id: str,
    job_id: str,
    request: Request,
):
    """Return persisted status for one queued split-mode chat job."""
    _require_chat_ready_run(run_id, request)
    store = _get_chat_job_store(request)
    record = store.get_job(run_id, conversation_id, job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Chat job `{job_id}` was not found for conversation "
                f"`{conversation_id}` in run `{run_id}`."
            ),
        )
    from backend.api.services.chat_session_helpers import as_chat_job_status_response
    return as_chat_job_status_response(record)


@router.get(
    "/runs/{run_id}/chat/sessions/{conversation_id}/contexts",
    response_model=ChatSessionContextsResponse,
    name="get_chat_session_contexts",
)
def get_chat_session_contexts(
    run_id: str,
    conversation_id: str,
    request: Request,
) -> ChatSessionContextsResponse:
    """Return selected context runs for a chat session."""
    run_store, _ = _require_chat_ready_run(run_id, request)
    store = _get_chat_memory_store(request)
    try:
        session = store.get_session(run_id, conversation_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    config = _load_request_config(request)
    from backend.api.services.chat_reply_helpers import build_chat_citation_entries
    from backend.api.services.context_chat import estimate_context_window
    return build_session_contexts_response(
        run_id,
        conversation_id,
        run_store,
        session,
        store,
        config,
        resolve_chat_token_cap(config),
        lambda context: context.to_summary(),
        lambda bundle: bundle.to_summary(),
        build_chat_sources,
        build_chat_citation_entries,
        estimate_context_window,
    )


@router.get(
    "/runs/{run_id}/chat/sessions/{conversation_id}/followups/{bundle_id}/references",
    response_model=ChatFollowupReferenceListResponse,
    response_model_exclude_none=True,
    name="list_chat_followup_references",
)
def list_chat_followup_references(
    run_id: str,
    conversation_id: str,
    bundle_id: str,
    request: Request,
    ref_id: str | None = Query(
        default=None,
        description="Optional `ref_n` id filter. When omitted, all references are returned.",
    ),
    include_quote: bool = Query(
        default=False,
        description=(
            "When false (default), only lightweight citation fields are returned. "
            "Set true to include quote payload for click-to-inspect UX."
        ),
    ),
) -> ChatFollowupReferenceListResponse:
    """Return references for one persisted chat follow-up bundle."""
    _run_store, _ = _require_chat_ready_run(run_id, request)
    store = _get_chat_memory_store(request)
    try:
        session = store.get_session(run_id, conversation_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    normalized_ref_id = (ref_id or "").strip()
    if normalized_ref_id and not is_valid_ref_id(normalized_ref_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ref_id must match format `ref_<positive_integer>`.",
        )

    known_bundle_ids = {bundle["bundle_id"] for bundle in selected_followup_bundles(session)}
    if bundle_id not in known_bundle_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Follow-up bundle `{bundle_id}` was not found for this session.",
        )

    artifact_dir = followup_bundle_dir(
        runs_dir=_run_store.runs_dir,
        run_id=run_id,
        conversation_id=conversation_id,
        bundle_id=bundle_id,
    )
    records = load_reference_records(artifact_dir, bundle_id)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No markdown references available for follow-up bundle `{bundle_id}`.",
        )

    items = [
        build_reference_item(record=item, include_quote=include_quote)
        for item in records
        if is_valid_ref_id(str(item.get("ref_id", "")).strip())
    ]
    items.sort(key=lambda item: (item.excerpt_index, item.ref_id))
    if normalized_ref_id:
        items = [item for item in items if item.ref_id == normalized_ref_id]
        if not items:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference `{normalized_ref_id}` was not found for bundle `{bundle_id}`.",
            )

    return ChatFollowupReferenceListResponse(
        run_id=run_id,
        conversation_id=conversation_id,
        bundle_id=bundle_id,
        reference_count=len(items),
        references=items,
    )


@router.put(
    "/runs/{run_id}/chat/sessions/{conversation_id}/contexts",
    response_model=ChatSessionContextsResponse,
    name="update_chat_session_contexts",
)
def update_chat_session_contexts(
    run_id: str,
    conversation_id: str,
    payload: UpdateChatContextsRequest,
    request: Request,
) -> ChatSessionContextsResponse:
    """Update selected context runs for a chat session."""
    run_store, _ = _require_chat_ready_run(run_id, request)
    store = _get_chat_memory_store(request)
    try:
        session = store.get_session(run_id, conversation_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    pending_job = as_pending_job(run_id, conversation_id, pending_job_payload(session))
    if pending_job is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This chat session already has a pending long-context answer. "
                "Wait for it to finish before changing contexts."
            ),
        )

    config = _load_request_config(request)
    token_cap = resolve_chat_token_cap(config)

    requested: list[str] = [run_id]
    seen: set[str] = {run_id}
    for raw_id in payload.context_run_ids:
        cleaned = raw_id.strip()
        if not cleaned or cleaned in seen:
            continue
        requested.append(cleaned)
        seen.add(cleaned)

    missing: list[str] = []
    for run_id_value in requested:
        try:
            validate_context_run_id(run_store, run_id_value)
        except ValueError:
            missing.append(run_id_value)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Some selected context runs are unavailable or not completed: "
                + ", ".join(missing)
            ),
        )

    try:
        session = store.update_context_runs(
            run_id=run_id,
            conversation_id=conversation_id,
            context_run_ids=requested,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Build a lightweight response using per-run sidecar summaries.
    # The combined session prompt cache is expensive to compute and only needed
    # when a message is sent, so we skip it here and let it build lazily on demand.
    context_summaries: list[ChatContextSummary] = []
    excluded_context_ids: list[str] = []
    for rid in requested:
        try:
            context_summaries.append(fast_context_summary(run_store, rid, config))
        except ValueError:
            excluded_context_ids.append(rid)

    followup_summaries: list[ChatFollowupBundleSummary] = []
    excluded_followup_ids: list[str] = []
    running_prompt_tokens = sum(s.prompt_context_tokens for s in context_summaries)
    for bundle_meta in selected_followup_bundles(session):
        try:
            loaded = load_followup_bundle(
                run_store=run_store,
                run_id=run_id,
                conversation_id=conversation_id,
                bundle_id=bundle_meta["bundle_id"],
                config=config,
                bundle_meta=bundle_meta,
            )
            next_total = running_prompt_tokens + loaded.prompt_context_tokens
            if next_total <= token_cap:
                followup_summaries.append(loaded.to_summary())
                running_prompt_tokens = next_total
            else:
                excluded_followup_ids.append(bundle_meta["bundle_id"])
        except ValueError:
            excluded_followup_ids.append(bundle_meta["bundle_id"])

    total_tokens = sum(s.total_tokens for s in context_summaries) + sum(
        fb.total_tokens for fb in followup_summaries
    )
    return ChatSessionContextsResponse(
        run_id=run_id,
        conversation_id=conversation_id,
        context_run_ids=requested,
        contexts=context_summaries,
        followup_bundles=followup_summaries,
        total_tokens=total_tokens,
        prompt_context_tokens=running_prompt_tokens,
        prompt_context_kind=context_summaries[0].prompt_context_kind if context_summaries else None,
        token_cap=token_cap,
        excluded_context_run_ids=excluded_context_ids,
        excluded_followup_bundle_ids=excluded_followup_ids,
        is_capped=(
            len(excluded_context_ids) > 0
            or len(excluded_followup_ids) > 0
            or running_prompt_tokens > token_cap
        ),
    )


@router.post(
    "/runs/{run_id}/chat/sessions/{conversation_id}/messages",
    response_model=SendChatMessageResponse,
    name="send_chat_message",
)
def send_chat_message(
    run_id: str,
    conversation_id: str,
    payload: SendChatMessageRequest,
    response: Response,
    request: Request,
    x_openrouter_api_key: str | None = Header(
        default=None, alias="X-OpenRouter-Api-Key"
    ),
) -> SendChatMessageResponse:
    """Send chat message and return assistant response."""
    run_store, run_record = _require_chat_ready_run(run_id, request)
    store = _get_chat_memory_store(request)
    try:
        session = store.get_session(run_id, conversation_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    pending_job = as_pending_job(run_id, conversation_id, pending_job_payload(session))
    if pending_job is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This chat session already has a pending long-context answer. "
                "Wait for it to finish before sending another message."
            ),
        )

    config = _load_request_config(request)
    config.enable_sql = False
    token_cap = resolve_chat_token_cap(config)
    chat_job_executor = _get_chat_job_executor(request)

    (
        selected_run_ids,
        loaded_contexts,
        excluded_ids,
        loaded_followup_bundles,
        excluded_followup_ids,
    ) = resolve_session_contexts(
        run_store,
        session,
        conversation_id=conversation_id,
        fallback_run_id=run_id,
        config=config,
        token_cap=token_cap,
    )
    if not loaded_contexts:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "No usable context sources are selected for this session. "
                "Open context manager and select at least one completed run."
            ),
        )
    if excluded_followup_ids:
        session = store.prune_followup_bundles(
            run_id,
            conversation_id,
            [bundle.bundle_id for bundle in loaded_followup_bundles],
        )
        (
            selected_run_ids,
            loaded_contexts,
            excluded_ids,
            loaded_followup_bundles,
            excluded_followup_ids,
        ) = resolve_session_contexts(
            run_store,
            session,
            conversation_id=conversation_id,
            fallback_run_id=run_id,
            config=config,
            token_cap=token_cap,
        )

    history: list[dict[str, str]] = []
    messages = session.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                history.append({"role": role, "content": content})

    api_key_override = _resolve_api_key_override(x_openrouter_api_key)
    clarification_city = (payload.clarification_city or "").strip()
    clarification_question = (payload.clarification_question or "").strip()
    effective_user_content = (
        clarification_question
        if clarification_city and clarification_question
        else payload.content
    )
    sources = build_chat_sources(loaded_contexts, loaded_followup_bundles)
    
    from backend.api.services.chat_session_helpers import get_or_build_session_prompt_context_cache
    from backend.api.services.chat_reply_helpers import build_chat_citation_entries
    from backend.api.services.context_chat import estimate_context_window
    
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
    assistant_citations: list[dict[str, object]] = []
    assistant_citation_warning: str | None = None
    assistant_routing: dict[str, object] | None = None
    assistant_text = ""
    try:
        log_chat_request_summary(
            run_id=run_id,
            conversation_id=conversation_id,
            selected_run_ids=selected_run_ids,
            loaded_contexts=loaded_contexts,
            loaded_followup_bundles=loaded_followup_bundles,
            history=history,
            token_cap=token_cap,
            followup_search_enabled=config.chat.followup_search_enabled,
            clarification_city=clarification_city or None,
        )
        effective_api_key = (
            api_key_override
            if api_key_override is not None
            else get_openrouter_api_key()
        )
        if not config.chat.followup_search_enabled:
            plan = build_context_reply_plan(
                original_question=run_record.question,
                sources=sources,
                history=history,
                user_content=effective_user_content,
                config=config,
                token_cap=token_cap,
                citation_prefix_tokens=prompt_cache.citation_prefix_tokens,
            )
            log_context_reply_plan(
                run_id=run_id,
                conversation_id=conversation_id,
                selected_run_ids=selected_run_ids,
                loaded_contexts=loaded_contexts,
                loaded_followup_bundles=loaded_followup_bundles,
                sources=sources,
                plan=plan,
            )
            if getattr(plan, "mode", "direct") == "split":
                logger.info(
                    "Context chat split job accepted run_id=%s conversation_id=%s contexts=%s "
                    "followup_bundles=%s estimated_prompt_tokens=%s context_tokens=%s split_reason=%s",
                    run_id,
                    conversation_id,
                    selected_run_ids,
                    [bundle.bundle_id for bundle in loaded_followup_bundles],
                    getattr(plan, "estimated_prompt_tokens", None),
                    getattr(plan, "context_tokens", None),
                    getattr(plan, "split_reason", None),
                )
                queued_response = queue_split_context_chat_job(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    payload_content=payload.content,
                    original_question=run_record.question,
                    effective_user_content=effective_user_content,
                    history=history,
                    context_run_ids=selected_run_ids,
                    followup_bundle_ids=[bundle.bundle_id for bundle in loaded_followup_bundles],
                    token_cap=token_cap,
                    assistant_routing=assistant_routing,
                    api_key_override=api_key_override,
                    chat_memory_store=store,
                    chat_job_executor=chat_job_executor,
                )
                if queued_response.mode == "queued":
                    response.status_code = status.HTTP_202_ACCEPTED
                return queued_response
            assistant_text, assistant_citations, assistant_citation_warning = (
                answer_from_context_reply(
                    run_id=run_id,
                    original_question=run_record.question,
                    sources=sources,
                    history=history,
                    user_content=effective_user_content,
                    config=config,
                    token_cap=token_cap,
                    api_key_override=api_key_override,
                    citation_prefix_tokens=prompt_cache.citation_prefix_tokens,
                )
            )
        else:
            from backend.modules.orchestrator.models import ChatFollowupDecision

            # Compute the plan before routing. Split-mode sessions are queued
            # immediately — no value routing when the answer is generated async.
            plan = build_context_reply_plan(
                original_question=run_record.question,
                sources=sources,
                history=history,
                user_content=effective_user_content,
                config=config,
                token_cap=token_cap,
                citation_prefix_tokens=prompt_cache.citation_prefix_tokens,
            )
            log_context_reply_plan(
                run_id=run_id,
                conversation_id=conversation_id,
                selected_run_ids=selected_run_ids,
                loaded_contexts=loaded_contexts,
                loaded_followup_bundles=loaded_followup_bundles,
                sources=sources,
                plan=plan,
            )
            if getattr(plan, "mode", "direct") == "split":
                logger.info(
                    "Context chat split job accepted run_id=%s conversation_id=%s contexts=%s "
                    "followup_bundles=%s estimated_prompt_tokens=%s context_tokens=%s split_reason=%s",
                    run_id,
                    conversation_id,
                    selected_run_ids,
                    [bundle.bundle_id for bundle in loaded_followup_bundles],
                    getattr(plan, "estimated_prompt_tokens", None),
                    getattr(plan, "context_tokens", None),
                    getattr(plan, "split_reason", None),
                )
                queued_response = queue_split_context_chat_job(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    payload_content=payload.content,
                    original_question=run_record.question,
                    effective_user_content=effective_user_content,
                    history=history,
                    context_run_ids=selected_run_ids,
                    followup_bundle_ids=[bundle.bundle_id for bundle in loaded_followup_bundles],
                    token_cap=token_cap,
                    assistant_routing=assistant_routing,
                    api_key_override=api_key_override,
                    chat_memory_store=store,
                    chat_job_executor=chat_job_executor,
                )
                if queued_response.mode == "queued":
                    response.status_code = status.HTTP_202_ACCEPTED
                return queued_response

            if clarification_city:
                routing_decision = ChatFollowupDecision(
                    action="search_single_city",
                    reason="User selected a single city after clarification request.",
                    target_city=clarification_city,
                    rewritten_question=effective_user_content,
                )
            else:
                router_payload = _build_router_payload(
                    user_message=payload.content,
                    original_question=run_record.question,
                    history=history,
                    selected_run_ids=selected_run_ids,
                    followup_bundles=loaded_followup_bundles,
                    sources=sources,
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
                    routing_decision = route_chat_followup(
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
                    routing_decision = None

            if routing_decision is None or routing_decision.action == "answer_from_context":
                if routing_decision is not None:
                    assistant_routing = build_routing_metadata(
                        action=routing_decision.action,
                        reason=routing_decision.reason,
                        target_city=routing_decision.target_city,
                    )
                assistant_text, assistant_citations, assistant_citation_warning = (
                    answer_from_context_reply(
                        run_id=run_id,
                        original_question=run_record.question,
                        sources=sources,
                        history=history,
                        user_content=effective_user_content,
                        config=config,
                        token_cap=token_cap,
                        api_key_override=api_key_override,
                        citation_prefix_tokens=prompt_cache.citation_prefix_tokens,
                    )
                )
            elif routing_decision.action == "out_of_scope":
                assistant_text = build_out_of_scope_reply()
                assistant_routing = build_routing_metadata(
                    action=routing_decision.action,
                    reason=routing_decision.reason,
                )
            elif routing_decision.action == "needs_city_clarification":
                assistant_text = build_city_clarification_reply()
                assistant_routing = build_routing_metadata(
                    action=routing_decision.action,
                    reason=routing_decision.reason,
                    pending_user_message=payload.content,
                )
            else:
                target_city = routing_decision.target_city
                if not target_city:
                    assistant_text = build_city_clarification_reply()
                    assistant_routing = build_routing_metadata(
                        action="needs_city_clarification",
                        reason="Router requested a city search but did not provide exactly one city.",
                        pending_user_message=payload.content,
                    )
                else:
                    search_result = run_chat_followup_search(
                        runs_dir=run_store.runs_dir,
                        run_id=run_id,
                        conversation_id=conversation_id,
                        turn_index=next_turn_index(history),
                        question=routing_decision.rewritten_question or effective_user_content,
                        target_city=target_city,
                        config=config,
                        api_key=effective_api_key,
                    )
                    assistant_routing = build_routing_metadata(
                        action=routing_decision.action,
                        reason=routing_decision.reason,
                        target_city=target_city,
                        bundle_id=search_result.bundle_id,
                    )
                    if search_result.status == "success" and search_result.excerpt_count > 0:
                        session = store.attach_followup_bundle(
                            run_id=run_id,
                            conversation_id=conversation_id,
                            bundle_id=search_result.bundle_id,
                            city_key=target_city.strip().casefold(),
                            target_city=search_result.target_city,
                            created_at=search_result.created_at.isoformat(),
                            max_followup_bundles=config.chat.max_auto_followup_bundles,
                        )
                        (
                            selected_run_ids,
                            loaded_contexts,
                            excluded_ids,
                            loaded_followup_bundles,
                            excluded_followup_ids,
                        ) = resolve_session_contexts(
                            run_store,
                            session,
                            conversation_id=conversation_id,
                            fallback_run_id=run_id,
                            config=config,
                            token_cap=token_cap,
                        )
                        if excluded_followup_ids:
                            session = store.prune_followup_bundles(
                                run_id,
                                conversation_id,
                                [bundle.bundle_id for bundle in loaded_followup_bundles],
                            )
                            (
                                selected_run_ids,
                                loaded_contexts,
                                excluded_ids,
                                loaded_followup_bundles,
                                excluded_followup_ids,
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
                            followup_bundle_ids=[
                                bundle.bundle_id for bundle in loaded_followup_bundles
                            ],
                            config=config,
                            token_cap=token_cap,
                            build_citation_catalog_fn=build_chat_citation_entries,
                            estimate_context_window_fn=estimate_context_window,
                        )
                        plan = build_context_reply_plan(
                            original_question=run_record.question,
                            sources=sources,
                            history=history,
                            user_content=effective_user_content,
                            config=config,
                            token_cap=token_cap,
                            citation_prefix_tokens=prompt_cache.citation_prefix_tokens,
                        )
                        log_context_reply_plan(
                            run_id=run_id,
                            conversation_id=conversation_id,
                            selected_run_ids=selected_run_ids,
                            loaded_contexts=loaded_contexts,
                            loaded_followup_bundles=loaded_followup_bundles,
                            sources=sources,
                            plan=plan,
                        )
                        if getattr(plan, "mode", "direct") == "split":
                            logger.info(
                                "Context chat split job accepted run_id=%s conversation_id=%s contexts=%s "
                                "followup_bundles=%s estimated_prompt_tokens=%s context_tokens=%s split_reason=%s",
                                run_id,
                                conversation_id,
                                selected_run_ids,
                                [bundle.bundle_id for bundle in loaded_followup_bundles],
                                getattr(plan, "estimated_prompt_tokens", None),
                                getattr(plan, "context_tokens", None),
                                getattr(plan, "split_reason", None),
                            )
                            queued_response = queue_split_context_chat_job(
                                run_id=run_id,
                                conversation_id=conversation_id,
                                payload_content=payload.content,
                                original_question=run_record.question,
                                effective_user_content=effective_user_content,
                                history=history,
                                context_run_ids=selected_run_ids,
                                followup_bundle_ids=[
                                    bundle.bundle_id for bundle in loaded_followup_bundles
                                ],
                                token_cap=token_cap,
                                assistant_routing=assistant_routing,
                                api_key_override=api_key_override,
                                chat_memory_store=store,
                                chat_job_executor=chat_job_executor,
                            )
                            if queued_response.mode == "queued":
                                response.status_code = status.HTTP_202_ACCEPTED
                            return queued_response
                        assistant_text, assistant_citations, assistant_citation_warning = (
                            answer_from_context_reply(
                                run_id=run_id,
                                original_question=run_record.question,
                                sources=sources,
                                history=history,
                                user_content=effective_user_content,
                                config=config,
                                token_cap=token_cap,
                                api_key_override=api_key_override,
                                citation_prefix_tokens=prompt_cache.citation_prefix_tokens,
                            )
                        )
                    else:
                        if is_unavailable_city_result(search_result):
                            assistant_text = build_unavailable_city_reply(target_city)
                            assistant_routing = build_routing_metadata(
                                action="needs_city_clarification",
                                reason=(
                                    "Requested city is not available in the current searchable city list."
                                ),
                                pending_user_message=effective_user_content,
                            )
                        else:
                            assistant_text = build_followup_failure_reply(target_city)
    except APITimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                "Chat request timed out at provider. "
                "Reduce selected contexts or retry."
            ),
        ) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Authentication failed for provided API key. "
                "Switch to a valid OpenRouter key and retry."
            ),
        ) from exc
    except APIStatusError as exc:
        if exc.status_code in {401, 403}:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "API key rejected by provider (401/403). "
                    "Switch to another OpenRouter key and retry."
                ),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Chat provider error: {exc.status_code}",
        ) from exc
    except EnvironmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"{exc}. Provide your own OpenRouter key via UI "
                "or configure server OPENROUTER_API_KEY."
            ),
        ) from exc
    except ChatSessionPendingJobError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    try:
        _, user_message, assistant_message = store.append_turn(
            run_id=run_id,
            conversation_id=conversation_id,
            user_content=payload.content,
            assistant_content=assistant_text,
            assistant_citations=assistant_citations or None,
            assistant_citation_warning=assistant_citation_warning,
            assistant_routing=assistant_routing,
        )
    except ChatSessionPendingJobError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if excluded_ids:
        filtered_ids = [context.run_id for context in loaded_contexts]
        store.update_context_runs(run_id, conversation_id, filtered_ids or [run_id])

    from backend.api.services.chat_session_helpers import as_message
    
    return SendChatMessageCompletedResponse(
        mode="completed",
        run_id=run_id,
        conversation_id=conversation_id,
        user_message=as_message(user_message),
        assistant_message=as_message(assistant_message),
    )


def _build_router_payload(
    *,
    user_message: str,
    original_question: str,
    history: list[dict[str, str]],
    selected_run_ids: list[str],
    followup_bundles,
    sources,
):
    """Build the compact payload sent to the follow-up router."""
    from backend.api.services.chat_reply_helpers import build_router_payload
    return build_router_payload(
        user_message=user_message,
        original_question=original_question,
        history=history,
        selected_run_ids=selected_run_ids,
        followup_bundles=followup_bundles,
        sources=sources,
    )


__all__ = ["build_chat_job_processor", "router"]
