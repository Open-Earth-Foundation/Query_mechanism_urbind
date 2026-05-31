from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

WriterCitationCoverageStatus = Literal["confirmed", "partial"]
WriterContextSourceKind = Literal[
    "ccc_excerpt",
    "ccc_source_chunk",
    "external_markdown_claim",
    "external_markdown_resolution",
    "external_no_evidence",
    "web_finding",
    "assumption",
    "non_estimable",
    "enriched_field",
    "freshness_result",
]


class WriterCitationCoverage(BaseModel):
    """Structured writer citation-coverage diagnostics for one final draft."""

    status: WriterCitationCoverageStatus
    attempt: int
    max_attempts: int
    coverage_confirmed: int
    coverage_required: int
    coverage_ratio: str
    missing_cities: list[str] = Field(default_factory=list)
    analysis_mode: str


class WriterOutput(BaseModel):
    """Final writer output plus optional citation-coverage diagnostics."""

    content: str
    citation_coverage: WriterCitationCoverage | None = None


class WriterSectionSpec(BaseModel):
    """One question-specific section planned for aggregate writing."""

    section_id: str
    title: str
    section_type: str
    purpose: str
    required_ref_ids: list[str] = Field(default_factory=list)
    city_names: list[str] = Field(default_factory=list)
    writing_instructions: str


class WriterSectionPlan(BaseModel):
    """Structured section-first plan emitted by the aggregate writer planner."""

    strategy: Literal["section_first"] = "section_first"
    analysis_mode: Literal["aggregate"] = "aggregate"
    sections: list[WriterSectionSpec] = Field(default_factory=list)


class WriterContextItem(BaseModel):
    """One searchable, writer-visible context item."""

    item_id: str
    source_kind: WriterContextSourceKind
    city_name: str = ""
    city_key: str = ""
    source_id: str = ""
    ref_id: str = ""
    field: str = ""
    text: str
    quote: str = ""
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class WriterContextSourceSummary(BaseModel):
    """Compact source summary exposed to the research curator."""

    source_kind: WriterContextSourceKind
    count: int
    cities: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)


class WriterContextSearchHit(BaseModel):
    """One bounded regex hit over writer-visible context."""

    search_id: str
    hit_id: str
    item_id: str
    source_kind: WriterContextSourceKind
    city_name: str = ""
    city_key: str = ""
    source_id: str = ""
    ref_id: str = ""
    field: str = ""
    matched_text: str
    snippet: str
    line_start: int | None = None
    line_end: int | None = None


class WriterSavedEvidence(BaseModel):
    """One context excerpt saved by the optional research curator."""

    saved_id: str
    ref_id: str
    item_id: str
    source_kind: WriterContextSourceKind
    city_name: str = ""
    city_key: str = ""
    source_id: str = ""
    field: str = ""
    quote: str = ""
    text: str
    reason: str = ""
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class WriterMissingEvidenceRecord(BaseModel):
    """One missing-evidence note recorded by the research curator."""

    missing_id: str
    city_name: str = ""
    city_key: str = ""
    source_kind: WriterContextSourceKind | None = None
    field: str = ""
    reason: str
    searched_patterns: list[str] = Field(default_factory=list)


class WriterEvidenceSelection(BaseModel):
    """Final curator summary emitted after tool-based evidence saving."""

    status: Literal["saved_evidence", "no_relevant_evidence", "needs_excerpt_fallback"]
    saved_evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class WriterMultiPassBatch(BaseModel):
    """One writer batch emitted by the multi-pass fallback planner."""

    batch_index: int
    city_names: list[str] = Field(default_factory=list)
    excerpt_count: int
    payload_tokens: int


class WriterMultiPassPlan(BaseModel):
    """Structured writer multi-pass diagnostics for developer tooling."""

    strategy: Literal["split_by_city"] = "split_by_city"
    combine_strategy: Literal["draft_merge"] = "draft_merge"
    analysis_mode: str
    payload_tokens: int
    threshold_tokens: int
    batch_count: int
    batches: list[WriterMultiPassBatch] = Field(default_factory=list)


__all__ = [
    "WriterCitationCoverage",
    "WriterCitationCoverageStatus",
    "WriterContextItem",
    "WriterContextSearchHit",
    "WriterContextSourceKind",
    "WriterContextSourceSummary",
    "WriterEvidenceSelection",
    "WriterMissingEvidenceRecord",
    "WriterMultiPassBatch",
    "WriterMultiPassPlan",
    "WriterOutput",
    "WriterSavedEvidence",
    "WriterSectionPlan",
    "WriterSectionSpec",
]
