"""Shared service-layer data models for chat and prompt-context helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from backend.api.models import (
    ChatContextSummary,
    ChatFollowupBundleSummary,
    SendChatMessageResponse,
)

PromptContextKind = Literal["citation_catalog", "serialized_contexts"]


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
    context_window_kind: PromptContextKind | None = None
    context_block_tokens: int | None = None
    prompt_header_tokens: int | None = None
    history_tokens: int | None = None
    user_tokens: int | None = None
    citation_catalog_entry_count: int | None = None
    fitted_citation_entry_count: int | None = None
    fitted_citation_ref_ids: list[str] | None = None


@dataclass(frozen=True)
class ContextWindowEstimate:
    """Context-window estimate derived from the chat prompt assembly path."""

    mode: Literal["direct", "split"]
    resolved_token_cap: int
    effective_token_cap: int
    context_window_kind: PromptContextKind | None = None
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
class PreparedContextChatRequest:
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
    context_window_kind: PromptContextKind | None = None
    context_block_tokens: int | None = None
    prompt_header_tokens: int | None = None
    history_tokens: int | None = None
    user_tokens: int | None = None
    citation_catalog_entry_count: int | None = None
    fitted_citation_entry_count: int | None = None
    fitted_citation_ref_ids: list[str] | None = None


@dataclass(frozen=True)
class TokenSidecar:
    """All token metrics for one run, stored in a tiny sidecar file."""

    document_tokens: int
    bundle_tokens: int
    prompt_context_tokens: int
    prompt_context_kind: PromptContextKind


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


@dataclass(frozen=True)
class LoadedContext:
    """Loaded context artifacts with token accounting."""

    run_id: str
    question: str
    status: str
    started_at: datetime
    final_output_path: Path
    context_bundle_path: Path
    markdown_excerpts_path: Path
    final_document: str
    context_bundle: dict[str, Any]
    document_tokens: int
    bundle_tokens: int
    prompt_context_tokens: int
    prompt_context_kind: PromptContextKind

    @property
    def total_tokens(self) -> int:
        """Return tokens consumed by the stored document and bundle artifacts."""
        return self.document_tokens + self.bundle_tokens

    def to_summary(self) -> ChatContextSummary:
        """Convert the loaded context to the API summary model."""
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
            prompt_context_tokens=self.prompt_context_tokens,
            prompt_context_kind=self.prompt_context_kind,
        )


@dataclass(frozen=True)
class LoadedFollowupBundle:
    """Loaded chat-owned follow-up bundle."""

    bundle_id: str
    target_city: str
    created_at: datetime
    context_bundle_path: Path
    markdown_excerpts_path: Path
    context_bundle: dict[str, Any]
    bundle_tokens: int
    excerpt_count: int
    prompt_context_tokens: int
    prompt_context_kind: PromptContextKind

    @property
    def total_tokens(self) -> int:
        """Return the prompt-budget cost for this follow-up bundle."""
        return self.bundle_tokens

    def to_summary(self) -> ChatFollowupBundleSummary:
        """Convert the loaded follow-up bundle to the API summary model."""
        return ChatFollowupBundleSummary(
            bundle_id=self.bundle_id,
            target_city=self.target_city,
            excerpt_count=self.excerpt_count,
            total_tokens=self.total_tokens,
            prompt_context_tokens=self.prompt_context_tokens,
            prompt_context_kind=self.prompt_context_kind,
            created_at=self.created_at,
        )


@dataclass(frozen=True)
class LoadedChatSource:
    """Generic chat source consumed by citation building and reply generation."""

    source_type: Literal["run", "followup_bundle"]
    source_id: str
    question: str
    final_document: str
    context_bundle: dict[str, Any]


@dataclass(frozen=True)
class FollowupSearchResolution:
    """Result of trying to refresh follow-up context for one target city."""

    session: dict[str, object]
    assistant_routing: dict[str, object]
    assistant_text: str | None = None


@dataclass(frozen=True)
class LoadedChatSessionState:
    """Resolved session sources and prompt cache used for one reply attempt."""

    session: dict[str, object]
    selected_run_ids: list[str]
    loaded_contexts: list[LoadedContext]
    excluded_context_ids: list[str]
    loaded_followup_bundles: list[LoadedFollowupBundle]
    excluded_followup_bundle_ids: list[str]
    sources: list[LoadedChatSource]
    prompt_cache: SessionPromptContextCache


@dataclass(frozen=True)
class ChatSendServiceResult:
    """Send-message service result with queue metadata for the HTTP layer."""

    response: SendChatMessageResponse
    queued: bool = False


__all__ = [
    "ChatCitationEntry",
    "ChatContextSource",
    "ChatSendServiceResult",
    "CitationCatalogTokenCache",
    "ContextChatPlan",
    "ContextWindowEstimate",
    "FollowupSearchResolution",
    "LoadedChatSessionState",
    "LoadedChatSource",
    "LoadedContext",
    "LoadedFollowupBundle",
    "PreparedContextChatRequest",
    "PromptContextKind",
    "SessionPromptContextCache",
    "TokenSidecar",
]
