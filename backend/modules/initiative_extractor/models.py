from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from backend.models import ErrorInfo

JsonScalar: TypeAlias = str | int | float | bool | None
# Intentionally loose: LLM-extracted numeric payloads are audit data, while the
# surrounding initiative and artifact shape remains validated.
JsonValue: TypeAlias = Any


class InitiativeNumbers(BaseModel):
    """Current and planned numeric facts extracted from the source."""

    model_config = ConfigDict(extra="forbid")

    current: dict[str, JsonValue] = Field(default_factory=dict)
    planned: dict[str, JsonValue] = Field(default_factory=dict)


class InitiativeExtraction(BaseModel):
    """Canonical v1 initiative extraction shape without TEF classification."""

    model_config = ConfigDict(extra="forbid")

    city: str
    initiative_name: str
    general_description: str | None = None
    objective_text: str | None = None
    implementation_text: str | None = None
    planned_outputs_text: str | None = None
    delivery_text: str | None = None
    funding_text: str | None = None
    timeline_text: str | None = None
    numbers: InitiativeNumbers = Field(default_factory=InitiativeNumbers)


class InitiativeSourceRef(BaseModel):
    """Line-level source reference for one extracted initiative."""

    model_config = ConfigDict(extra="ignore")

    source_document: str
    segment_id: str
    start_line: int
    end_line: int
    section_heading: str | None = None
    quote: str | None = None


class InitiativeExtractionCandidate(BaseModel):
    """Pipeline initiative wrapper plus artifact metadata."""

    model_config = ConfigDict(extra="ignore")

    initiative: InitiativeExtraction
    document_local_code: str | None = None
    source_refs: list[InitiativeSourceRef] = Field(default_factory=list)
    data_quality_flags: list[str] = Field(default_factory=list)
    number_context: dict[str, JsonValue] = Field(default_factory=dict)
    number_deferred: dict[str, JsonValue] = Field(default_factory=dict)
    number_uncertain: dict[str, JsonValue] = Field(default_factory=dict)
    extraction_notes: list[str] = Field(default_factory=list)


class InitiativeSegmentExtraction(BaseModel):
    """Structured result for one document segment."""

    model_config = ConfigDict(extra="ignore")

    initiatives: list[InitiativeExtraction] = Field(default_factory=list)
    segment_data_quality_flags: list[str] = Field(default_factory=list)
    segment_notes: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None


class InitiativeSegmentStop(BaseModel):
    """Structured stop signal for an exhausted document segment."""

    model_config = ConfigDict(extra="ignore")

    reason: str | None = None
    segment_data_quality_flags: list[str] = Field(default_factory=list)
    segment_notes: list[str] = Field(default_factory=list)


class InitiativeExtractionRecord(InitiativeExtractionCandidate):
    """Deduplicated initiative record written to artifacts."""

    record_id: str
    source_document: str


class InitiativeDocumentSegment(BaseModel):
    """Line-aware markdown segment sent to the extractor."""

    segment_id: str
    city_name: str
    source_document: str
    source_path: str
    start_line: int
    end_line: int
    heading_path: str | None = None
    content: str
    token_count: int
    parent_segment_id: str | None = None


class InitiativeRawSegmentResult(BaseModel):
    """Raw per-segment extraction result before deduplication."""

    segment_id: str
    source_document: str
    status: Literal["success", "error"]
    initiatives: list[InitiativeExtractionCandidate] = Field(default_factory=list)
    segment_data_quality_flags: list[str] = Field(default_factory=list)
    segment_notes: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None
    action_heavy: bool = False
    extraction_iterations: int = 1
    extraction_complete: bool = False
    stop_reason: str | None = None


class InitiativeReviewItem(BaseModel):
    """Review item produced by validation, coverage audit, or deduplication."""

    review_type: str
    severity: Literal["info", "warning", "error"] = "warning"
    message: str
    source_document: str | None = None
    segment_id: str | None = None
    record_id: str | None = None
    document_local_code: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class InitiativeSemanticDedupeGroup(BaseModel):
    """LLM-proposed group of records that describe the same initiative."""

    model_config = ConfigDict(extra="ignore")

    canonical_record_id: str
    duplicate_record_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str | None = None


class InitiativeSemanticDedupeResult(BaseModel):
    """Structured semantic dedupe result for one record batch."""

    model_config = ConfigDict(extra="ignore")

    duplicate_groups: list[InitiativeSemanticDedupeGroup] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


class InitiativeExtractionRunResult(BaseModel):
    """Summary returned by the extraction pipeline."""

    run_id: str
    output_dir: str
    documents_count: int
    segments_count: int
    raw_initiatives_count: int
    deduped_initiatives_count: int
    review_items_count: int


__all__ = [
    "InitiativeDocumentSegment",
    "InitiativeExtraction",
    "InitiativeExtractionCandidate",
    "InitiativeExtractionRecord",
    "InitiativeExtractionRunResult",
    "InitiativeNumbers",
    "InitiativeRawSegmentResult",
    "InitiativeReviewItem",
    "InitiativeSemanticDedupeGroup",
    "InitiativeSemanticDedupeResult",
    "InitiativeSegmentExtraction",
    "InitiativeSegmentStop",
    "InitiativeSourceRef",
    "JsonScalar",
    "JsonValue",
]
