"""Run-scoped context chat endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import re
import time
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from openai import APIStatusError, APITimeoutError, AuthenticationError

from backend.api.models import (
    ChatCitation,
    ChatFollowupBundleSummary,
    ChatFollowupReferenceListResponse,
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
from backend.api.services import (
    CHAT_FOLLOWUP_CITY_UNAVAILABLE,
    ChatMemoryStore,
    ChatSessionExistsError,
    ChatSessionNotFoundError,
    RunRecord,
    RunStore,
    SUCCESS_STATUSES,
    build_reference_item,
    followup_bundle_dir,
    generate_context_chat_reply,
    load_reference_records,
    load_context_bundle,
    load_final_document,
    resolve_chat_token_cap,
    run_chat_followup_search,
)
from backend.modules.orchestrator.agent import route_chat_followup
from backend.modules.orchestrator.models import ChatFollowupDecision
from backend.modules.orchestrator.utils.references import is_valid_ref_id
from backend.utils.city_normalization import format_city_stem
from backend.utils.config import AppConfig, get_openrouter_api_key, load_config
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

router = APIRouter()
logger = logging.getLogger(__name__)
_REF_TOKEN_PATTERN = re.compile(r"\[(ref_[1-9]\d*)\]")


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


@dataclass(frozen=True)
class _LoadedFollowupBundle:
    """Loaded chat-owned follow-up bundle."""

    bundle_id: str
    target_city: str
    created_at: datetime
    context_bundle_path: Path
    context_bundle: dict[str, Any]
    bundle_tokens: int
    excerpt_count: int

    @property
    def total_tokens(self) -> int:
        """Total prompt-budget cost for this follow-up bundle."""
        return self.bundle_tokens

    def to_summary(self) -> ChatFollowupBundleSummary:
        """Convert to API summary model."""
        return ChatFollowupBundleSummary(
            bundle_id=self.bundle_id,
            target_city=self.target_city,
            excerpt_count=self.excerpt_count,
            total_tokens=self.total_tokens,
            created_at=self.created_at,
        )


@dataclass(frozen=True)
class _LoadedChatSource:
    """Generic chat source consumed by citation building and reply generation."""

    source_type: Literal["run", "followup_bundle"]
    source_id: str
    question: str
    final_document: str
    context_bundle: dict[str, Any]


@dataclass(frozen=True)
class _ChatCitationEntry:
    """Deterministic chat citation mapping entry."""

    ref_id: str
    city_name: str
    quote: str
    partial_answer: str
    source_type: Literal["run", "followup_bundle"]
    source_id: str
    source_ref_id: str


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


def _load_request_config(request: Request) -> AppConfig:
    """Load AppConfig using the config path resolved at API startup."""
    config_path = getattr(request.app.state, "config_path", Path("llm_config.yaml"))
    return load_config(Path(config_path))


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
    run_dir = run_store.runs_dir / run_id
    candidates: list[Path] = []
    if raw_path is not None:
        candidates.append(raw_path)
        candidates.append(run_dir / raw_path.name)
    candidates.append(run_dir / "final.md")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"Final output is missing for run `{run_id}`.")


def _resolve_context_bundle_path(
    run_store: RunStore, run_id: str, raw_path: Path | None
) -> Path:
    """Resolve context bundle path or raise when missing."""
    run_dir = run_store.runs_dir / run_id
    candidates: list[Path] = []
    if raw_path is not None:
        candidates.append(raw_path)
        candidates.append(run_dir / raw_path.name)
    candidates.append(run_dir / "context_bundle.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"Context bundle is missing for run `{run_id}`.")


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


def _load_followup_bundle(
    *,
    run_store: RunStore,
    run_id: str,
    conversation_id: str,
    bundle_meta: dict[str, str],
) -> _LoadedFollowupBundle:
    """Load one persisted follow-up bundle referenced by the session."""
    bundle_id = bundle_meta["bundle_id"]
    bundle_dir = followup_bundle_dir(
        runs_dir=run_store.runs_dir,
        run_id=run_id,
        conversation_id=conversation_id,
        bundle_id=bundle_id,
    )
    context_bundle_path = bundle_dir / "context_bundle.json"
    if not context_bundle_path.exists():
        raise ValueError(f"Follow-up bundle `{bundle_id}` is missing context_bundle.json.")
    bundle_text = context_bundle_path.read_text(encoding="utf-8")
    context_bundle = load_context_bundle(context_bundle_path)
    markdown_bundle = extract_markdown_bundle(context_bundle)
    excerpt_count = len(extract_bundle_excerpts(markdown_bundle))
    target_city = str(context_bundle.get("target_city", "")).strip() or bundle_meta["target_city"]
    created_at_raw = str(context_bundle.get("created_at", "")).strip() or bundle_meta["created_at"]
    try:
        created_at = datetime.fromisoformat(created_at_raw)
    except ValueError as exc:
        raise ValueError(f"Follow-up bundle `{bundle_id}` has invalid created_at.") from exc
    return _LoadedFollowupBundle(
        bundle_id=bundle_id,
        target_city=target_city,
        created_at=created_at,
        context_bundle_path=context_bundle_path,
        context_bundle=context_bundle,
        bundle_tokens=count_tokens(bundle_text),
        excerpt_count=excerpt_count,
    )


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
    citation_warning = payload.get("citation_warning")
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid message role.")
    if not isinstance(content, str):
        raise ValueError("Invalid message content.")
    citations = _as_chat_citations(payload.get("citations"))
    routing = _as_chat_routing(payload.get("routing"))
    return ChatMessage(
        role=role,
        content=content,
        created_at=_as_datetime(created_at),
        citations=citations,
        citation_warning=str(citation_warning) if isinstance(citation_warning, str) else None,
        routing=routing,
    )


def _as_chat_citations(value: object) -> list[ChatCitation] | None:
    """Normalize optional assistant citation metadata from persisted messages."""
    if not isinstance(value, list):
        return None
    citations: list[ChatCitation] = []
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
            ChatCitation(
                ref_id=ref_id,
                city_name=city_name,
                source_type=source_type,
                source_id=source_id,
                source_ref_id=source_ref_id,
            )
        )
    return citations if citations else None


def _as_chat_routing(value: object) -> dict[str, Any] | None:
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


def _selected_followup_bundles(session: dict[str, object]) -> list[dict[str, str]]:
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


def _apply_additional_context_token_cap(
    *,
    base_context: _LoadedContext,
    additional_contexts: list[_LoadedContext],
    token_cap: int,
) -> tuple[list[_LoadedContext], list[str]]:
    """Keep the base context pinned while capping only extra selected runs."""
    included: list[_LoadedContext] = [base_context]
    excluded: list[str] = []
    running_total = base_context.total_tokens
    for context in additional_contexts:
        next_total = running_total + context.total_tokens
        if next_total <= token_cap:
            included.append(context)
            running_total = next_total
        else:
            excluded.append(context.run_id)
    return included, excluded


def _apply_followup_bundle_token_cap(
    bundles: list[_LoadedFollowupBundle],
    token_cap: int,
    starting_total: int,
) -> tuple[list[_LoadedFollowupBundle], list[str]]:
    """Apply prompt-budget cap to auto-attached follow-up bundles only."""
    included: list[_LoadedFollowupBundle] = []
    excluded: list[str] = []
    running_total = starting_total
    for bundle in bundles:
        next_total = running_total + bundle.total_tokens
        if next_total <= token_cap:
            included.append(bundle)
            running_total = next_total
        else:
            excluded.append(bundle.bundle_id)
    return included, excluded


def _resolve_session_contexts(
    run_store: RunStore,
    session: dict[str, object],
    conversation_id: str,
    fallback_run_id: str,
    token_cap: int,
) -> tuple[
    list[str],
    list[_LoadedContext],
    list[str],
    list[_LoadedFollowupBundle],
    list[str],
]:
    """Resolve selected contexts for a chat session with token cap applied."""
    selected_ids = _selected_context_run_ids(session, fallback_run_id)
    try:
        base_context = _load_context_for_run_id(run_store, fallback_run_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "The base run context is no longer usable for chat. "
                f"Fix run artifacts for `{fallback_run_id}` and retry."
            ),
        ) from exc

    loaded_contexts: list[_LoadedContext] = []
    excluded: list[str] = []
    for context_run_id in selected_ids:
        if context_run_id == fallback_run_id:
            continue
        try:
            loaded_contexts.append(_load_context_for_run_id(run_store, context_run_id))
        except ValueError:
            excluded.append(context_run_id)
    included, cap_excluded = _apply_additional_context_token_cap(
        base_context=base_context,
        additional_contexts=loaded_contexts,
        token_cap=token_cap,
    )
    excluded.extend(cap_excluded)
    included_total = sum(context.total_tokens for context in included)

    loaded_followup_bundles: list[_LoadedFollowupBundle] = []
    excluded_followup_bundle_ids: list[str] = []
    for bundle_meta in _selected_followup_bundles(session):
        try:
            loaded_followup_bundles.append(
                _load_followup_bundle(
                    run_store=run_store,
                    run_id=fallback_run_id,
                    conversation_id=conversation_id,
                    bundle_meta=bundle_meta,
                )
            )
        except ValueError:
            excluded_followup_bundle_ids.append(bundle_meta["bundle_id"])

    included_followups, cap_excluded_followups = _apply_followup_bundle_token_cap(
        loaded_followup_bundles,
        token_cap=token_cap,
        starting_total=included_total,
    )
    excluded_followup_bundle_ids.extend(cap_excluded_followups)
    return selected_ids, included, excluded, included_followups, excluded_followup_bundle_ids


def _build_session_contexts_response(
    run_id: str,
    conversation_id: str,
    run_store: RunStore,
    session: dict[str, object],
    token_cap: int,
) -> ChatSessionContextsResponse:
    """Build session context payload for API response."""
    (
        selected_ids,
        included_contexts,
        excluded_ids,
        followup_bundles,
        excluded_followups,
    ) = _resolve_session_contexts(
        run_store,
        session,
        conversation_id=conversation_id,
        fallback_run_id=run_id,
        token_cap=token_cap,
    )
    total_tokens = sum(context.total_tokens for context in included_contexts) + sum(
        bundle.total_tokens for bundle in followup_bundles
    )
    return ChatSessionContextsResponse(
        run_id=run_id,
        conversation_id=conversation_id,
        context_run_ids=selected_ids,
        contexts=[context.to_summary() for context in included_contexts],
        followup_bundles=[bundle.to_summary() for bundle in followup_bundles],
        total_tokens=total_tokens,
        token_cap=token_cap,
        excluded_context_run_ids=excluded_ids,
        excluded_followup_bundle_ids=excluded_followups,
        is_capped=(
            len(excluded_ids) > 0
            or len(excluded_followups) > 0
            or total_tokens > token_cap
        ),
    )


def _build_chat_sources(
    contexts: list[_LoadedContext],
    followup_bundles: list[_LoadedFollowupBundle],
) -> list[_LoadedChatSource]:
    """Build reply-generation context sources from runs and follow-up bundles."""
    sources = [
        _LoadedChatSource(
            source_type="run",
            source_id=context.run_id,
            question=context.question,
            final_document=context.final_document,
            context_bundle=context.context_bundle,
        )
        for context in contexts
    ]
    sources.extend(
        _LoadedChatSource(
            source_type="followup_bundle",
            source_id=bundle.bundle_id,
            question=str(bundle.context_bundle.get("research_question", "")).strip(),
            final_document="",
            context_bundle=bundle.context_bundle,
        )
        for bundle in followup_bundles
    )
    return sources


def _build_chat_citation_entries(
    sources: list[_LoadedChatSource],
) -> list[_ChatCitationEntry]:
    """Build deterministic synthetic chat citations from all loaded chat sources."""
    entries: list[_ChatCitationEntry] = []
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
                _ChatCitationEntry(
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


def _build_llm_citation_catalog(
    entries: list[_ChatCitationEntry],
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
    entries_by_ref_id: dict[str, _ChatCitationEntry],
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


def _build_router_payload(
    *,
    user_message: str,
    original_question: str,
    history: list[dict[str, str]],
    selected_run_ids: list[str],
    followup_bundles: list[_LoadedFollowupBundle],
    sources: list[_LoadedChatSource],
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


def _build_router_context_payload(source: _LoadedChatSource) -> dict[str, object]:
    """Build one compact context summary for the follow-up router."""
    markdown_bundle = extract_markdown_bundle(source.context_bundle)
    excerpts = extract_bundle_excerpts(markdown_bundle)
    selected_city_names = extract_selected_city_names(source.context_bundle, markdown_bundle)
    inspected_raw = markdown_bundle.get("inspected_city_names")
    inspected_city_names = selected_city_names
    if isinstance(inspected_raw, list):
        normalized = [str(item).strip() for item in inspected_raw if isinstance(item, str)]
        inspected_city_names = [name for name in normalized if name]
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
            for excerpt in excerpts
            if isinstance(excerpt, dict)
        ],
    }


def _answer_from_context_reply(
    *,
    run_id: str,
    original_question: str,
    sources: list[_LoadedChatSource],
    history: list[dict[str, str]],
    user_content: str,
    config: AppConfig,
    token_cap: int,
    api_key_override: str | None,
) -> tuple[str, list[dict[str, object]], str | None]:
    """Generate a grounded reply and resolve synthetic citations."""
    citation_entries = _build_chat_citation_entries(sources)
    citation_entries_by_ref_id = {entry.ref_id: entry for entry in citation_entries}
    llm_citation_catalog = _build_llm_citation_catalog(citation_entries)
    chat_kwargs: dict[str, object] = {
        "original_question": original_question,
        "contexts": [
            {
                "run_id": source.source_id,
                "question": source.question,
                "final_document": source.final_document,
                "context_bundle": source.context_bundle,
            }
            for source in sources
        ],
        "history": history,
        "user_content": user_content,
        "config": config,
        "token_cap": token_cap,
        "citation_catalog": llm_citation_catalog,
        "run_id": run_id,
    }
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


def _build_out_of_scope_reply() -> str:
    """Return the deterministic out-of-scope assistant reply."""
    return (
        "This follow-up is outside the current city-report context. "
        "Start a new chat for that topic."
    )


def _build_city_clarification_reply() -> str:
    """Return the deterministic city-clarification assistant reply."""
    return (
        "I can refresh the context for one city at a time. "
        "Please ask about exactly one city."
    )


def _build_followup_failure_reply(target_city: str | None) -> str:
    """Return the deterministic failure reply for unsuccessful follow-up search."""
    city_label = format_city_stem(target_city or "") or "that city"
    return (
        f"I could not refresh the context for {city_label} just now, "
        "so I cannot answer this reliably."
    )


def _build_unavailable_city_reply(target_city: str | None) -> str:
    """Return the deterministic reply when a city is not available for follow-up search."""
    city_label = format_city_stem(target_city or "") or "that city"
    return f"We don't have {city_label} in the current city list. Please choose another one."


def _is_unavailable_city_result(search_result: ChatFollowupSearchResult) -> bool:
    """Return true when a follow-up search failed because the city is unavailable."""
    return search_result.error_code == CHAT_FOLLOWUP_CITY_UNAVAILABLE


def _build_routing_metadata(
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


def _next_turn_index(history: list[dict[str, str]]) -> int:
    """Return the next user-turn index for bundle id generation."""
    user_turns = sum(1 for item in history if item.get("role") == "user")
    return user_turns + 1


@router.get(
    "/chat/contexts",
    response_model=ChatContextCatalogResponse,
    name="list_chat_contexts",
)
def list_chat_contexts(request: Request) -> ChatContextCatalogResponse:
    """List all available completed run contexts for chat selection."""
    run_store = _get_run_store(request)
    contexts = _available_contexts(run_store)
    config = _load_request_config(request)
    return ChatContextCatalogResponse(
        contexts=[context.to_summary() for context in contexts],
        total=len(contexts),
        token_cap=resolve_chat_token_cap(config),
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
    config = _load_request_config(request)
    return _build_session_contexts_response(
        run_id,
        conversation_id,
        run_store,
        session,
        resolve_chat_token_cap(config),
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

    known_bundle_ids = {bundle["bundle_id"] for bundle in _selected_followup_bundles(session)}
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
        _ = store.get_session(run_id, conversation_id)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    config = _load_request_config(request)
    token_cap = resolve_chat_token_cap(config)

    available = {context.run_id: context for context in _available_contexts(run_store)}
    requested: list[str] = [run_id]
    seen: set[str] = set()
    seen.add(run_id)
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

    base_tokens = available[run_id].total_tokens
    extra_run_ids = [run_id_value for run_id_value in requested if run_id_value != run_id]
    extra_tokens = sum(available[run_id_value].total_tokens for run_id_value in extra_run_ids)
    remaining_budget = max(token_cap - base_tokens, 0)

    if base_tokens > token_cap and extra_run_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Base context `{run_id}` already uses {base_tokens} tokens, "
                f"which exceeds the {token_cap} token cap. Remove extra contexts and retry."
            ),
        )

    if extra_tokens > remaining_budget:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Selected extra contexts total {extra_tokens} tokens, "
                f"which exceeds the remaining {remaining_budget} token budget "
                f"after pinning base context `{run_id}`."
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

    return _build_session_contexts_response(run_id, conversation_id, run_store, session, token_cap)


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

    config = _load_request_config(request)
    config.enable_sql = False
    token_cap = resolve_chat_token_cap(config)

    (
        selected_run_ids,
        loaded_contexts,
        excluded_ids,
        loaded_followup_bundles,
        excluded_followup_ids,
    ) = _resolve_session_contexts(
        run_store,
        session,
        conversation_id=conversation_id,
        fallback_run_id=run_id,
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
        ) = _resolve_session_contexts(
            run_store,
            session,
            conversation_id=conversation_id,
            fallback_run_id=run_id,
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
    sources = _build_chat_sources(loaded_contexts, loaded_followup_bundles)
    assistant_citations: list[dict[str, object]] = []
    assistant_citation_warning: str | None = None
    assistant_routing: dict[str, object] | None = None
    assistant_text = ""
    try:
        effective_api_key = (
            api_key_override
            if api_key_override is not None
            else get_openrouter_api_key()
        )
        if not config.chat.followup_search_enabled:
            assistant_text, assistant_citations, assistant_citation_warning = (
                _answer_from_context_reply(
                    run_id=run_id,
                    original_question=run_record.question,
                    sources=sources,
                    history=history,
                    user_content=effective_user_content,
                    config=config,
                    token_cap=token_cap,
                    api_key_override=api_key_override,
                )
            )
        else:
            if clarification_city:
                routing_decision = ChatFollowupDecision(
                    action="search_single_city",
                    reason="User selected a single city after clarification request.",
                    target_city=clarification_city,
                    rewritten_question=effective_user_content,
                )
            else:
                try:
                    routing_decision = route_chat_followup(
                        payload=_build_router_payload(
                            user_message=payload.content,
                            original_question=run_record.question,
                            history=history,
                            selected_run_ids=selected_run_ids,
                            followup_bundles=loaded_followup_bundles,
                            sources=sources,
                        ),
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
                    assistant_routing = _build_routing_metadata(
                        action=routing_decision.action,
                        reason=routing_decision.reason,
                        target_city=routing_decision.target_city,
                    )
                assistant_text, assistant_citations, assistant_citation_warning = (
                    _answer_from_context_reply(
                        run_id=run_id,
                        original_question=run_record.question,
                        sources=sources,
                        history=history,
                        user_content=effective_user_content,
                        config=config,
                        token_cap=token_cap,
                        api_key_override=api_key_override,
                    )
                )
            elif routing_decision.action == "out_of_scope":
                assistant_text = _build_out_of_scope_reply()
                assistant_routing = _build_routing_metadata(
                    action=routing_decision.action,
                    reason=routing_decision.reason,
                )
            elif routing_decision.action == "needs_city_clarification":
                assistant_text = _build_city_clarification_reply()
                assistant_routing = _build_routing_metadata(
                    action=routing_decision.action,
                    reason=routing_decision.reason,
                    pending_user_message=payload.content,
                )
            else:
                target_city = routing_decision.target_city
                if not target_city:
                    assistant_text = _build_city_clarification_reply()
                    assistant_routing = _build_routing_metadata(
                        action="needs_city_clarification",
                        reason="Router requested a city search but did not provide exactly one city.",
                        pending_user_message=payload.content,
                    )
                else:
                    search_result = run_chat_followup_search(
                        runs_dir=run_store.runs_dir,
                        run_id=run_id,
                        conversation_id=conversation_id,
                        turn_index=_next_turn_index(history),
                        question=routing_decision.rewritten_question or effective_user_content,
                        target_city=target_city,
                        config=config,
                        api_key=effective_api_key,
                    )
                    assistant_routing = _build_routing_metadata(
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
                        ) = _resolve_session_contexts(
                            run_store,
                            session,
                            conversation_id=conversation_id,
                            fallback_run_id=run_id,
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
                            ) = _resolve_session_contexts(
                                run_store,
                                session,
                                conversation_id=conversation_id,
                                fallback_run_id=run_id,
                                token_cap=token_cap,
                            )
                        sources = _build_chat_sources(loaded_contexts, loaded_followup_bundles)
                        assistant_text, assistant_citations, assistant_citation_warning = (
                            _answer_from_context_reply(
                                run_id=run_id,
                                original_question=run_record.question,
                                sources=sources,
                                history=history,
                                user_content=effective_user_content,
                                config=config,
                                token_cap=token_cap,
                                api_key_override=api_key_override,
                            )
                        )
                    else:
                        if _is_unavailable_city_result(search_result):
                            assistant_text = _build_unavailable_city_reply(target_city)
                            assistant_routing = _build_routing_metadata(
                                action="needs_city_clarification",
                                reason=(
                                    "Requested city is not available in the current searchable city list."
                                ),
                                pending_user_message=effective_user_content,
                            )
                        else:
                            assistant_text = _build_followup_failure_reply(target_city)
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
    _, user_message, assistant_message = store.append_turn(
        run_id=run_id,
        conversation_id=conversation_id,
        user_content=payload.content,
        assistant_content=assistant_text,
        assistant_citations=assistant_citations or None,
        assistant_citation_warning=assistant_citation_warning,
        assistant_routing=assistant_routing,
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
