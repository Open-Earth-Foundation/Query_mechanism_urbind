"""Run-scoped context chat endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from openai import APIStatusError, AuthenticationError

from app.api.models import (
    ChatContextCatalogResponse,
    ChatContextSummary,
    ChatMessage,
    ChatSessionContextsResponse,
    ChatSessionListResponse,
    ChatSessionResponse,
    CreateChatSessionRequest,
    SendChatMessageRequest,
    SendChatMessageResponse,
    UpdateChatContextsRequest,
)
from app.api.services import (
    CHAT_PROMPT_TOKEN_CAP,
    ChatMemoryStore,
    ChatSessionExistsError,
    ChatSessionNotFoundError,
    RunRecord,
    RunStore,
    SUCCESS_STATUSES,
    generate_context_chat_reply,
    load_context_bundle,
    load_final_document,
)
from app.utils.config import load_config
from app.utils.tokenization import count_tokens

router = APIRouter()


def _resolve_api_key_override(raw: str | None) -> str | None:
    """Normalize optional API key header value."""
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    return cleaned


@dataclass(frozen=True)
class _LoadedContext:
    """Loaded context artifacts with token accounting."""

    run_id: str
    question: str
    status: str
    started_at: datetime
    final_output_path: Path
    context_bundle_path: Path
    final_document: str
    context_bundle: dict[str, Any]
    document_tokens: int
    bundle_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by both context artifacts."""
        return self.document_tokens + self.bundle_tokens

    def to_summary(self) -> ChatContextSummary:
        """Convert to API summary model."""
        return ChatContextSummary(
            run_id=self.run_id,
            question=self.question,
            status=self.status,  # type: ignore[arg-type]
            started_at=self.started_at,
            final_output_path=str(self.final_output_path),
            context_bundle_path=str(self.context_bundle_path),
            document_tokens=self.document_tokens,
            bundle_tokens=self.bundle_tokens,
            total_tokens=self.total_tokens,
        )


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


def _require_chat_ready_run(run_id: str, request: Request) -> tuple[RunStore, RunRecord]:
    """Require run existence and completed status before chat access."""
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


def _resolve_final_output_path(run_store: RunStore, run_id: str, raw_path: Path | None) -> Path:
    """Resolve final output path or raise when missing."""
    candidate = raw_path if raw_path is not None else run_store.runs_dir / run_id / "final.md"
    if not candidate.exists():
        raise ValueError(f"Final output is missing for run `{run_id}`.")
    return candidate


def _resolve_context_bundle_path(
    run_store: RunStore, run_id: str, raw_path: Path | None
) -> Path:
    """Resolve context bundle path or raise when missing."""
    candidate = (
        raw_path
        if raw_path is not None
        else run_store.runs_dir / run_id / "context_bundle.json"
    )
    if not candidate.exists():
        raise ValueError(f"Context bundle is missing for run `{run_id}`.")
    return candidate


def _load_context_for_record(run_store: RunStore, run_record: RunRecord) -> _LoadedContext:
    """Load context material for one completed run."""
    final_path = _resolve_final_output_path(
        run_store, run_record.run_id, run_record.final_output_path
    )
    context_path = _resolve_context_bundle_path(
        run_store, run_record.run_id, run_record.context_bundle_path
    )
    final_document = load_final_document(final_path)
    context_bundle = load_context_bundle(context_path)
    return _LoadedContext(
        run_id=run_record.run_id,
        question=run_record.question,
        status=run_record.status,
        started_at=run_record.started_at,
        final_output_path=final_path,
        context_bundle_path=context_path,
        final_document=final_document,
        context_bundle=context_bundle,
        document_tokens=count_tokens(final_document),
        bundle_tokens=count_tokens(
            context_path.read_text(encoding="utf-8")
        ),
    )


def _load_context_for_run_id(run_store: RunStore, run_id: str) -> _LoadedContext:
    """Load context material for a specific run id."""
    run_record = run_store.get_run(run_id)
    if run_record is None:
        raise ValueError(f"Context run `{run_id}` was not found.")
    if run_record.status not in SUCCESS_STATUSES:
        raise ValueError(
            f"Context run `{run_id}` is not ready (status: `{run_record.status}`)."
        )
    return _load_context_for_record(run_store, run_record)


def _available_contexts(run_store: RunStore) -> list[_LoadedContext]:
    """List all completed runs that have usable chat context artifacts."""
    contexts: list[_LoadedContext] = []
    for run_record in run_store.list_runs():
        if run_record.status not in SUCCESS_STATUSES:
            continue
        try:
            contexts.append(_load_context_for_record(run_store, run_record))
        except ValueError:
            continue
    return contexts


def _as_datetime(value: object) -> datetime:
    """Parse session timestamp value."""
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError("Invalid timestamp in chat session payload.")


def _as_message(payload: object) -> ChatMessage:
    """Convert stored message payload into API model."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid message payload.")
    role = payload.get("role")
    content = payload.get("content")
    created_at = payload.get("created_at")
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid message role.")
    if not isinstance(content, str):
        raise ValueError("Invalid message content.")
    return ChatMessage(
        role=role,
        content=content,
        created_at=_as_datetime(created_at),
    )


def _as_session_response(payload: dict[str, object]) -> ChatSessionResponse:
    """Convert stored session payload into API model."""
    messages_raw = payload.get("messages")
    message_models: list[ChatMessage] = []
    if isinstance(messages_raw, list):
        message_models = [_as_message(item) for item in messages_raw]

    run_id = payload.get("run_id")
    conversation_id = payload.get("conversation_id")
    if not isinstance(run_id, str) or not isinstance(conversation_id, str):
        raise ValueError("Invalid session payload identifiers.")
    return ChatSessionResponse(
        run_id=run_id,
        conversation_id=conversation_id,
        created_at=_as_datetime(payload.get("created_at")),
        updated_at=_as_datetime(payload.get("updated_at")),
        messages=message_models,
    )


def _selected_context_run_ids(
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


def _apply_context_token_cap(
    contexts: list[_LoadedContext], token_cap: int
) -> tuple[list[_LoadedContext], list[str]]:
    """Apply sequential token cap to selected contexts."""
    included: list[_LoadedContext] = []
    excluded: list[str] = []
    running_total = 0
    for context in contexts:
        next_total = running_total + context.total_tokens
        if next_total <= token_cap:
            included.append(context)
            running_total = next_total
        else:
            excluded.append(context.run_id)
    return included, excluded


def _resolve_session_contexts(
    run_store: RunStore,
    session: dict[str, object],
    fallback_run_id: str,
) -> tuple[list[str], list[_LoadedContext], list[str]]:
    """Resolve selected contexts for a chat session with token cap applied."""
    selected_ids = _selected_context_run_ids(session, fallback_run_id)
    loaded_contexts: list[_LoadedContext] = []
    excluded: list[str] = []
    for context_run_id in selected_ids:
        try:
            loaded_contexts.append(_load_context_for_run_id(run_store, context_run_id))
        except ValueError:
            excluded.append(context_run_id)
    included, cap_excluded = _apply_context_token_cap(loaded_contexts, CHAT_PROMPT_TOKEN_CAP)
    excluded.extend(cap_excluded)
    return selected_ids, included, excluded


def _build_session_contexts_response(
    run_id: str,
    conversation_id: str,
    run_store: RunStore,
    session: dict[str, object],
) -> ChatSessionContextsResponse:
    """Build session context payload for API response."""
    selected_ids, included_contexts, excluded_ids = _resolve_session_contexts(
        run_store,
        session,
        fallback_run_id=run_id,
    )
    total_tokens = sum(context.total_tokens for context in included_contexts)
    return ChatSessionContextsResponse(
        run_id=run_id,
        conversation_id=conversation_id,
        context_run_ids=selected_ids,
        contexts=[context.to_summary() for context in included_contexts],
        total_tokens=total_tokens,
        token_cap=CHAT_PROMPT_TOKEN_CAP,
        excluded_context_run_ids=excluded_ids,
        is_capped=len(excluded_ids) > 0,
    )


@router.get(
    "/chat/contexts",
    response_model=ChatContextCatalogResponse,
    name="list_chat_contexts",
)
def list_chat_contexts(request: Request) -> ChatContextCatalogResponse:
    """List all available completed run contexts for chat selection."""
    run_store = _get_run_store(request)
    contexts = _available_contexts(run_store)
    return ChatContextCatalogResponse(
        contexts=[context.to_summary() for context in contexts],
        total=len(contexts),
        token_cap=CHAT_PROMPT_TOKEN_CAP,
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
    return _as_session_response(session)


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
    return _as_session_response(session)


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
    return _build_session_contexts_response(run_id, conversation_id, run_store, session)


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
        _ = store.get_session(run_id, conversation_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    available = {context.run_id: context for context in _available_contexts(run_store)}
    requested: list[str] = []
    seen: set[str] = set()
    for raw_id in payload.context_run_ids:
        cleaned = raw_id.strip()
        if not cleaned or cleaned in seen:
            continue
        requested.append(cleaned)
        seen.add(cleaned)
    if not requested:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="context_run_ids must include at least one run id.",
        )

    missing = [run_id_value for run_id_value in requested if run_id_value not in available]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Some selected context runs are unavailable or not completed: "
                + ", ".join(missing)
            ),
        )

    total_tokens = sum(available[run_id_value].total_tokens for run_id_value in requested)
    if total_tokens > CHAT_PROMPT_TOKEN_CAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Selected contexts total {total_tokens} tokens, "
                f"which exceeds the {CHAT_PROMPT_TOKEN_CAP} token cap."
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

    return _build_session_contexts_response(run_id, conversation_id, run_store, session)


@router.post(
    "/runs/{run_id}/chat/sessions/{conversation_id}/messages",
    response_model=SendChatMessageResponse,
    name="send_chat_message",
)
def send_chat_message(
    run_id: str,
    conversation_id: str,
    payload: SendChatMessageRequest,
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

    selected_ids, loaded_contexts, excluded_ids = _resolve_session_contexts(
        run_store,
        session,
        fallback_run_id=run_id,
    )
    if not loaded_contexts:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "No usable context sources are selected for this session. "
                "Open context manager and select at least one completed run."
            ),
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

    config_path = getattr(request.app.state, "config_path", Path("llm_config.yaml"))
    config = load_config(Path(config_path))
    config.enable_sql = False

    api_key_override = _resolve_api_key_override(x_openrouter_api_key)
    chat_kwargs: dict[str, object] = {
        "original_question": run_record.question,
        "contexts": [
            {
                "run_id": context.run_id,
                "question": context.question,
                "final_document": context.final_document,
                "context_bundle": context.context_bundle,
            }
            for context in loaded_contexts
        ],
        "history": history,
        "user_content": payload.content,
        "config": config,
        "token_cap": CHAT_PROMPT_TOKEN_CAP,
    }
    if api_key_override is not None:
        chat_kwargs["api_key_override"] = api_key_override
    try:
        assistant_text = generate_context_chat_reply(**chat_kwargs)
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
    _, user_message, assistant_message = store.append_turn(
        run_id=run_id,
        conversation_id=conversation_id,
        user_content=payload.content,
        assistant_content=assistant_text,
    )

    if excluded_ids:
        # Persist selected ids without unavailable/capped ids to keep session healthy.
        filtered_ids = [context.run_id for context in loaded_contexts]
        store.update_context_runs(run_id, conversation_id, filtered_ids or [run_id])

    return SendChatMessageResponse(
        run_id=run_id,
        conversation_id=conversation_id,
        user_message=_as_message(user_message),
        assistant_message=_as_message(assistant_message),
    )


__all__ = ["router"]
