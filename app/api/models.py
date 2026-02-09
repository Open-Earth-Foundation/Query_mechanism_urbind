"""Pydantic models for FastAPI run lifecycle endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal[
    "queued",
    "running",
    "completed",
    "completed_with_gaps",
    "failed",
    "stopped",
]


class RunError(BaseModel):
    """Error payload returned for failed runs."""

    code: str
    message: str


class CreateRunRequest(BaseModel):
    """Request body for starting a new backend run."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    run_id: str | None = None
    cities: list[str] | None = None
    config_path: str | None = None
    markdown_path: str | None = None
    log_llm_payload: bool = False


class CreateRunResponse(BaseModel):
    """Response body after accepting a run start request."""

    run_id: str
    status: RunStatus
    status_url: str
    output_url: str
    context_url: str


class RunStatusResponse(BaseModel):
    """Response body for run status polling."""

    run_id: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    finish_reason: str | None = None
    error: RunError | None = None


class RunOutputResponse(BaseModel):
    """Response body for final markdown output retrieval."""

    run_id: str
    status: RunStatus
    content: str
    final_output_path: str


class RunContextResponse(BaseModel):
    """Response body for context bundle retrieval."""

    run_id: str
    status: RunStatus
    context_bundle: dict[str, object]
    context_bundle_path: str


class CityListResponse(BaseModel):
    """Response body listing available cities from markdown sources."""

    cities: list[str]
    total: int
    markdown_dir: str


class CityGroup(BaseModel):
    """Predefined city group exposed to frontend selection."""

    id: str
    name: str
    description: str | None = None
    cities: list[str]


class CityGroupListResponse(BaseModel):
    """Response body listing predefined city groups."""

    groups: list[CityGroup]
    total: int
    groups_path: str


ChatRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    """Single chat message in persisted conversation memory."""

    role: ChatRole
    content: str
    created_at: datetime


class CreateChatSessionRequest(BaseModel):
    """Request body for creating a run-scoped chat session."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = None


class ChatSessionResponse(BaseModel):
    """Response body with chat session metadata and transcript."""

    run_id: str
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessage]


class ChatSessionListResponse(BaseModel):
    """Response body listing sessions for a run."""

    run_id: str
    conversations: list[str]
    total: int


class ChatContextSummary(BaseModel):
    """Stored output/context pair available for chat grounding."""

    run_id: str
    question: str
    status: RunStatus
    started_at: datetime
    final_output_path: str
    context_bundle_path: str
    document_tokens: int
    bundle_tokens: int
    total_tokens: int


class ChatContextCatalogResponse(BaseModel):
    """Response body listing available run contexts for chat."""

    contexts: list[ChatContextSummary]
    total: int
    token_cap: int


class UpdateChatContextsRequest(BaseModel):
    """Request body for updating chat session context selection."""

    model_config = ConfigDict(extra="forbid")

    context_run_ids: list[str]


class ChatSessionContextsResponse(BaseModel):
    """Response body describing selected contexts for a chat session."""

    run_id: str
    conversation_id: str
    context_run_ids: list[str]
    contexts: list[ChatContextSummary]
    total_tokens: int
    token_cap: int
    excluded_context_run_ids: list[str]
    is_capped: bool


class SendChatMessageRequest(BaseModel):
    """Request body for sending a message to context chat."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)


class SendChatMessageResponse(BaseModel):
    """Response body after assistant generates a reply."""

    run_id: str
    conversation_id: str
    user_message: ChatMessage
    assistant_message: ChatMessage


__all__ = [
    "RunStatus",
    "RunError",
    "CreateRunRequest",
    "CreateRunResponse",
    "RunStatusResponse",
    "RunOutputResponse",
    "RunContextResponse",
    "CityListResponse",
    "CityGroup",
    "CityGroupListResponse",
    "ChatRole",
    "ChatMessage",
    "CreateChatSessionRequest",
    "ChatSessionResponse",
    "ChatSessionListResponse",
    "ChatContextSummary",
    "ChatContextCatalogResponse",
    "UpdateChatContextsRequest",
    "ChatSessionContextsResponse",
    "SendChatMessageRequest",
    "SendChatMessageResponse",
]
