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
AnalysisMode = Literal["aggregate", "city_by_city"]
QueryMode = Literal["standard", "dev"]
VectorStoreWarmupStatus = Literal[
    "pending",
    "skipped",
    "checking",
    "stale",
    "running",
    "completed",
    "failed",
]


class RunError(BaseModel):
    """Error payload returned for failed runs."""

    code: str
    message: str


class CreateRunRequest(BaseModel):
    """Request body for starting a new backend run."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    query_mode: QueryMode = "standard"
    query_2: str | None = None
    query_3: str | None = None
    run_id: str | None = None
    cities: list[str] | None = None
    config_path: str | None = None
    markdown_path: str | None = None
    log_llm_payload: bool = False
    analysis_mode: AnalysisMode = "aggregate"
    enrichment_enabled: bool | None = None
    web_research_enabled: bool | None = None


class CreateRunResponse(BaseModel):
    """Response body after accepting a run start request."""

    run_id: str
    status: RunStatus
    status_url: str
    output_url: str
    context_url: str


class VectorStoreWarmupResponse(BaseModel):
    """Status payload for the API startup vector-store warm-up."""

    status: VectorStoreWarmupStatus
    enabled: bool
    auto_update_on_run: bool
    update_mode: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    message: str
    error: str | None = None
    stats: dict[str, object] | None = None
    job_name: str | None = None
    latest_artifact: str | None = None


class PipelineStepItem(BaseModel):
    """Single sub-item inside a pipeline progress step."""

    text: str
    item_type: str | None = None
    title: str | None = None
    domain: str | None = None
    url: str | None = None
    count: int | None = None
    metadata: dict[str, object] | None = None


class PipelineStep(BaseModel):
    """One step in the pipeline progress tracker."""

    id: str
    label: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    items: list[PipelineStepItem] = Field(default_factory=list)


class RunStatusResponse(BaseModel):
    """Response body for run status polling."""

    run_id: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    finish_reason: str | None = None
    error: RunError | None = None
    steps: list[PipelineStep] | None = None


class RunDiagnosticsArtifactPaths(BaseModel):
    """Run-local artifact labels available for one diagnostics view."""

    run_log: str | None = None
    run_summary: str | None = None
    error_log: str | None = None


class RunWriterCitationCoverage(BaseModel):
    """Writer citation-coverage diagnostics exposed to developer tooling."""

    status: Literal["confirmed", "partial", "retrying", "exhausted"]
    attempt: int | None = None
    max_attempts: int | None = None
    coverage_confirmed: int
    coverage_required: int
    coverage_ratio: str
    missing_cities: list[str] = Field(default_factory=list)
    analysis_mode: str | None = None


class RunWriterMultiPassBatch(BaseModel):
    """One writer batch used by the writer multi-pass fallback."""

    batch_index: int
    city_names: list[str] = Field(default_factory=list)
    excerpt_count: int
    payload_tokens: int


class RunWriterMultiPass(BaseModel):
    """Writer multi-pass diagnostics exposed to developer tooling."""

    strategy: Literal["split_by_city"]
    combine_strategy: Literal["draft_merge"]
    analysis_mode: str
    payload_tokens: int
    threshold_tokens: int
    batch_count: int
    batches: list[RunWriterMultiPassBatch] = Field(default_factory=list)


class RunDiagnosticsResponse(BaseModel):
    """Response body for developer-facing run diagnostics."""

    run_id: str
    question: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    finish_reason: str | None = None
    error: RunError | None = None
    artifacts: RunDiagnosticsArtifactPaths
    writer_citation_coverage: RunWriterCitationCoverage | None = None
    writer_multi_pass: RunWriterMultiPass | None = None
    llm_usage: dict[str, object] | None = None
    retry_summary: dict[str, object] | None = None
    warning_entries: list[str] = Field(default_factory=list)
    log_tail: list[str] = Field(default_factory=list)
    error_log_text: str | None = None


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


class ArtifactFieldSource(BaseModel):
    """One evidence/source record backing (or failing to back) a field."""

    source_id: str | None = None
    title: str | None = None
    quote: str | None = None
    has_evidence: bool = False


class ArtifactFieldEstimate(BaseModel):
    """Estimate range + method metadata for an estimated field."""

    low: float | str | None = None
    mid: float | str | None = None
    high: float | str | None = None
    method: str | None = None
    confidence: str | None = None
    basis: str | None = None


class ArtifactField(BaseModel):
    """Normalized per-(city, field) audit record for the artifact card grid."""

    city: str
    field: str
    type: str | None = None
    scope: str | None = None
    status: Literal["estimated", "non_estimable", "unresolved"]
    explanation: str | None = None
    gap_description: str | None = None
    recommendation: str | None = None
    estimate: ArtifactFieldEstimate | None = None
    sources: list[ArtifactFieldSource] = Field(default_factory=list)
    reason: str | None = None
    reason_label: str | None = None


class ArtifactStage(BaseModel):
    """Lightweight per-stage summary from the unified stage records."""

    stage_number: int | None = None
    name: str
    status: str = "completed"
    metrics: dict[str, object] = Field(default_factory=dict)
    inputs: dict[str, object] = Field(default_factory=dict)


class EnrichmentStep(BaseModel):
    """One enrichment sub-step summary (gap analysis / external+web / assumptions)."""

    key: str
    label: str
    summary: str
    metrics: dict[str, object] = Field(default_factory=dict)
    warn: str | None = None


class RunArtifactsResponse(BaseModel):
    """Response body for the per-step / per-field enrichment audit view."""

    run_id: str
    status: RunStatus
    fields: list[ArtifactField] = Field(default_factory=list)
    stages: list[ArtifactStage] = Field(default_factory=list)
    enrichment_steps: list[EnrichmentStep] = Field(default_factory=list)


class RunReferenceResponse(BaseModel):
    """Response body for one run-scoped markdown reference lookup."""

    run_id: str
    ref_id: str
    excerpt_index: int
    city_name: str
    quote: str
    partial_answer: str
    source_chunk_ids: list[str]


class RunReferenceItem(BaseModel):
    """Single run-scoped reference item from list endpoint."""

    ref_id: str
    excerpt_index: int
    city_name: str
    quote: str | None = None
    partial_answer: str | None = None
    source_chunk_ids: list[str] | None = None


class RunReferenceListResponse(BaseModel):
    """Response body for run-scoped markdown references."""

    run_id: str
    reference_count: int
    references: list[RunReferenceItem]


class SourceChunkItem(BaseModel):
    """Single resolved markdown chunk used for source expansion UI."""

    chunk_id: str
    content: str
    city_name: str | None = None
    source_path: str | None = None
    heading_path: str | None = None
    block_type: str | None = None


class SourceChunkListResponse(BaseModel):
    """Response body for run-scoped source chunk lookup."""

    run_id: str
    chunk_count: int
    chunks: list[SourceChunkItem]


class RunSummary(BaseModel):
    """Minimal run metadata used by run picker UI."""

    run_id: str
    question: str
    status: RunStatus
    picker_timestamp: str


class RunListResponse(BaseModel):
    """Response body listing known runs from backend storage."""

    runs: list[RunSummary]
    total: int


class MissingDataItem(BaseModel):
    """Single missing-data item inferred from run output and context."""

    city: str = Field(min_length=1)
    missing_description: str = Field(min_length=1)
    proposed_number: float | int | str | None = None


class AssumptionsPayload(BaseModel):
    """Editable assumptions payload submitted by frontend users."""

    model_config = ConfigDict(extra="forbid")

    items: list[MissingDataItem] = Field(min_length=1)
    rewrite_instructions: str | None = None
    persist_artifacts: bool = False


class RegenerationResult(BaseModel):
    """Response body after assumptions-based document regeneration."""

    run_id: str
    revised_output_path: str | None = None
    revised_content: str
    assumptions_path: str | None = None


class CityListResponse(BaseModel):
    """Response body listing available cities from markdown sources."""

    cities: list[str]
    total: int
    markdown_dir: str


class CityMarkdownResponse(BaseModel):
    """Response body for one city's raw markdown source document."""

    city_name: str
    content: str
    source_paths: list[str]


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
ChatJobStatus = Literal["queued", "running", "completed", "failed"]
ChatFollowupAction = Literal[
    "answer_from_context",
    "search_single_city",
    "out_of_scope",
    "needs_city_clarification",
]
ChatCitationSourceType = Literal["run", "followup_bundle"]


class ChatCitation(BaseModel):
    """Assistant citation metadata for deterministic frontend resolution."""

    ref_id: str
    city_name: str
    source_type: ChatCitationSourceType
    source_id: str
    source_ref_id: str


class ChatRoutingMetadata(BaseModel):
    """Routing metadata persisted for assistant follow-up messages."""

    action: ChatFollowupAction
    reason: str
    target_city: str | None = None
    bundle_id: str | None = None
    pending_user_message: str | None = None


class ChatMessage(BaseModel):
    """Single chat message in persisted conversation memory."""

    role: ChatRole
    content: str
    created_at: datetime
    citations: list[ChatCitation] | None = None
    citation_warning: str | None = None
    routing: ChatRoutingMetadata | None = None


class CreateChatSessionRequest(BaseModel):
    """Request body for creating a run-scoped chat session."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = None


class ChatJobHandle(BaseModel):
    """Minimal chat-job handle returned for polling and session resume."""

    job_id: str
    job_number: int
    status: ChatJobStatus
    status_url: str


class ChatSessionResponse(BaseModel):
    """Response body with chat session metadata and transcript."""

    run_id: str
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    pending_job: ChatJobHandle | None = None
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
    prompt_context_tokens: int
    prompt_context_kind: Literal["citation_catalog", "serialized_contexts"] | None = None


class ChatFollowupBundleSummary(BaseModel):
    """Auto-attached follow-up excerpt bundle for one chat session."""

    bundle_id: str
    target_city: str
    excerpt_count: int
    total_tokens: int
    prompt_context_tokens: int
    prompt_context_kind: Literal["citation_catalog", "serialized_contexts"] | None = None
    created_at: datetime


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
    followup_bundles: list[ChatFollowupBundleSummary] = Field(default_factory=list)
    total_tokens: int
    prompt_context_tokens: int
    prompt_context_kind: Literal["citation_catalog", "serialized_contexts"] | None = None
    token_cap: int
    excluded_context_run_ids: list[str]
    excluded_followup_bundle_ids: list[str] = Field(default_factory=list)
    is_capped: bool


class SendChatMessageRequest(BaseModel):
    """Request body for sending a message to context chat."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    clarification_city: str | None = None


class SendChatMessageCompletedResponse(BaseModel):
    """Response body after assistant generates a synchronous reply."""

    mode: Literal["completed"] = "completed"
    run_id: str
    conversation_id: str
    user_message: ChatMessage
    assistant_message: ChatMessage


class ChatMessageJobAcceptedResponse(BaseModel):
    """Response body after accepting one queued split-mode chat job."""

    mode: Literal["queued"] = "queued"
    run_id: str
    conversation_id: str
    user_message: ChatMessage
    job: ChatJobHandle
    routing: ChatRoutingMetadata | None = None


SendChatMessageResponse = SendChatMessageCompletedResponse | ChatMessageJobAcceptedResponse


class ChatJobStatusResponse(BaseModel):
    """Response body for chat-job polling."""

    run_id: str
    conversation_id: str
    job_id: str
    job_number: int
    status: ChatJobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    finish_reason: str | None = None
    error: RunError | None = None


class ChatFollowupReferenceListResponse(BaseModel):
    """Response body for one chat follow-up bundle reference list."""

    run_id: str
    conversation_id: str
    bundle_id: str
    reference_count: int
    references: list[RunReferenceItem]


__all__ = [
    "RunStatus",
    "AnalysisMode",
    "RunError",
    "PipelineStepItem",
    "PipelineStep",
    "CreateRunRequest",
    "CreateRunResponse",
    "RunStatusResponse",
    "RunDiagnosticsArtifactPaths",
    "RunWriterCitationCoverage",
    "RunWriterMultiPassBatch",
    "RunWriterMultiPass",
    "RunDiagnosticsResponse",
    "RunOutputResponse",
    "RunContextResponse",
    "RunReferenceResponse",
    "RunReferenceItem",
    "RunReferenceListResponse",
    "SourceChunkItem",
    "SourceChunkListResponse",
    "RunSummary",
    "RunListResponse",
    "MissingDataItem",
    "AssumptionsPayload",
    "RegenerationResult",
    "CityListResponse",
    "CityGroup",
    "CityGroupListResponse",
    "ChatRole",
    "ChatJobStatus",
    "ChatFollowupAction",
    "ChatCitationSourceType",
    "ChatCitation",
    "ChatRoutingMetadata",
    "ChatMessage",
    "CreateChatSessionRequest",
    "ChatJobHandle",
    "ChatSessionResponse",
    "ChatSessionListResponse",
    "ChatContextSummary",
    "ChatFollowupBundleSummary",
    "ChatContextCatalogResponse",
    "UpdateChatContextsRequest",
    "ChatSessionContextsResponse",
    "SendChatMessageRequest",
    "SendChatMessageCompletedResponse",
    "ChatMessageJobAcceptedResponse",
    "SendChatMessageResponse",
    "ChatJobStatusResponse",
    "ChatFollowupReferenceListResponse",
]
