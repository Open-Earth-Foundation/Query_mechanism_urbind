"""Pydantic models for the web research enrichment and assumptions modelling layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Type literals
# ---------------------------------------------------------------------------
GapClassification = Literal["estimable_numerical", "derivable_from_ratio", "non_estimable"]
EstimationMethod = Literal["national_regional_average", "peer_city_proxy", "expert_heuristic_scaling"]
FieldStatus = Literal["resolved", "partially_resolved", "still_missing"]
EnrichedFieldSource = Literal["ccc", "web", "external_markdown", "estimated", "none"]
FreshnessClassification = Literal["consistent", "superseded", "uncertain", "cancelled"]
ExternalClaimRole = Literal["confirms_ccc", "fills_missing", "challenges_ccc", "unresolved"]
ExternalResolutionAction = Literal[
    "confirm",
    "fill",
    "conflict_review_required",
    "unresolved",
]

# ---------------------------------------------------------------------------
# Gap Analysis models (Agent 1 output)
# ---------------------------------------------------------------------------


class FieldClassification(BaseModel):
    field: str
    classification: GapClassification
    searchable: bool
    rationale: str


class CityGap(BaseModel):
    city: str
    blank_fields: list[str]
    stale_flags: list[str]
    search_priority: Literal["high", "medium", "low"]


class GapManifest(BaseModel):
    query_fields: list[FieldClassification]
    city_gaps: list[CityGap]
    non_estimable_fields: list[str]


# ---------------------------------------------------------------------------
# Web Research models (Agent 3 output — Phase 2)
# ---------------------------------------------------------------------------


class WebFinding(BaseModel):
    city: str
    field: str
    value: str | float | int | None
    unit: str | None = None
    source_url: str
    source_type: str  # e.g. "operator_press_release", "government_report"
    source_date: str | None = None
    extraction_confidence: float


class SearchBatch(BaseModel):
    batch_id: str
    cities: list[str]
    target_fields: list[str]
    search_type: str
    queries: list[str]
    budget: dict[str, object]
    priority: str


# ---------------------------------------------------------------------------
# Governed external Markdown source search models
# ---------------------------------------------------------------------------


class SourceMetadata(BaseModel):
    """Metadata for one approved external Markdown source."""

    source_id: str
    title: str
    upstream_group: str
    geographic_scope: str = "city"
    city: list[str] = Field(default_factory=list)
    country: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    description: str
    source_type: str
    publisher: str | None = None
    verticals: list[str] = Field(default_factory=list)
    tef_sectors: list[str] = Field(default_factory=list)
    tef_transitions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    data_years: list[int] = Field(default_factory=list)
    target_years: list[int] = Field(default_factory=list)
    source_url: str | None = None


class TagOptions(BaseModel):
    """Distinct metadata values available to external-source search agents."""

    cities: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    publication_years: list[int] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    verticals: list[str] = Field(default_factory=list)
    tef_sectors: list[str] = Field(default_factory=list)


class SourceSummary(BaseModel):
    """Compact candidate-source summary returned before text search."""

    source_id: str
    title: str
    city: list[str] = Field(default_factory=list)
    country: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    source_type: str
    verticals: list[str] = Field(default_factory=list)
    tef_sectors: list[str] = Field(default_factory=list)
    description: str


class SearchHit(BaseModel):
    """One bounded regex hit over a tagged external Markdown source."""

    search_id: str
    hit_id: str
    source_id: str
    title: str
    city: list[str] = Field(default_factory=list)
    line_start: int
    line_end: int
    matched_text: str
    snippet: str
    heading_path: list[str] = Field(default_factory=list)
    truncated: bool = False


class EvidenceCandidateInput(BaseModel):
    """LLM-selected hit metadata for saving evidence into the run basket."""

    hit_id: str
    city: str
    field: str
    reason: str
    confidence: float


class EvidenceCandidate(BaseModel):
    """Saved external-source evidence candidate for one run."""

    candidate_id: str
    hit_id: str
    source_id: str
    title: str
    city: str
    field: str
    matched_text: str
    quote: str
    line_start: int
    line_end: int
    heading_path: list[str] = Field(default_factory=list)
    confidence: float
    reason: str
    source_type: str
    publication_year: int | None = None
    source_url: str | None = None


class NoEvidenceRecord(BaseModel):
    """Audit record for a searched field where no usable evidence was found."""

    record_id: str
    city: str
    field: str
    searched_source_ids: list[str]
    search_summary: str


class ExternalEvidenceClaim(BaseModel):
    """Structured claim extracted from saved external Markdown evidence."""

    city: str
    field: str
    value: str | float | int | None
    unit: str | None = None
    source_id: str
    source_type: str
    publication_year: int | None = None
    line_start: int
    line_end: int
    quote: str
    confidence: float
    claim_role: ExternalClaimRole
    candidate_id: str | None = None
    source_url: str | None = None
    rationale: str | None = None


class ExternalSourceAgentResult(BaseModel):
    """Final LLM output for one external-source research task."""

    claims: list[ExternalEvidenceClaim] = Field(default_factory=list)
    no_evidence: list[NoEvidenceRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ExternalEvidenceResolution(BaseModel):
    """Resolver decision for how external evidence interacts with CCC evidence."""

    city: str
    field: str
    action: ExternalResolutionAction
    ccc_value: str | float | int | None = None
    external_value: str | float | int | None = None
    unit: str | None = None
    source_id: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    quote: str | None = None
    confidence: float | None = None
    rationale: str


# ---------------------------------------------------------------------------
# Freshness models (Agent 4 output — Phase 2)
# ---------------------------------------------------------------------------


class FreshnessResult(BaseModel):
    city: str
    field: str
    ccc_value: str | None
    web_value: str | None
    classification: FreshnessClassification
    reason: str
    web_source_url: str | None = None


# ---------------------------------------------------------------------------
# Enriched field (Agent 5 output)
# ---------------------------------------------------------------------------


class EnrichedField(BaseModel):
    city: str
    field: str
    status: FieldStatus
    value: str | float | int | None = None
    source: EnrichedFieldSource = "none"
    provenance: dict[str, object] = Field(default_factory=dict)
    freshness_flag: str | None = None


# ---------------------------------------------------------------------------
# Assumptions models (Agent 6 output)
# ---------------------------------------------------------------------------


class EstimateRange(BaseModel):
    low: float | str
    mid: float | str
    high: float | str


class AssumptionRecord(BaseModel):
    city: str
    field_name: str
    gap_description: str
    method_used: EstimationMethod
    estimate: EstimateRange
    confidence: Literal["HIGH", "MEDIUM", "LOW", "VERY_LOW"]
    reference_data: str
    rationale: str
    basis: str
    is_replaceable: bool = True


class NonEstimableRecord(BaseModel):
    city: str
    field_name: str
    gap_description: str
    status: str = "NON_ESTIMABLE"
    explanation: str
    recommendation: str  # Door Opener recommendation


# ---------------------------------------------------------------------------
# Enrichment bundle (Agent 7 output)
# ---------------------------------------------------------------------------


class EnrichmentMeta(BaseModel):
    created_at: datetime
    gap_analyst_model: str
    assumptions_estimator_model: str
    total_gaps: int
    estimable_count: int
    non_estimable_count: int
    web_findings_count: int = 0
    external_evidence_count: int = 0
    elapsed_seconds: float


class EnrichmentBundle(BaseModel):
    gap_manifest: GapManifest
    enriched_fields: list[EnrichedField] = Field(default_factory=list)
    web_findings: list[WebFinding] = Field(default_factory=list)
    external_evidence: list[ExternalEvidenceClaim] = Field(default_factory=list)
    external_resolutions: list[ExternalEvidenceResolution] = Field(default_factory=list)
    external_no_evidence: list[NoEvidenceRecord] = Field(default_factory=list)
    freshness_results: list[FreshnessResult] = Field(default_factory=list)
    assumptions: list[AssumptionRecord] = Field(default_factory=list)
    non_estimable: list[NonEstimableRecord] = Field(default_factory=list)
    saturation_warning: str | None = None
    meta: EnrichmentMeta


# ---------------------------------------------------------------------------
# LLM response envelopes (parsing only)
# ---------------------------------------------------------------------------


class _GapManifestEnvelope(BaseModel):
    query_fields: list[FieldClassification] = Field(default_factory=list)
    city_gaps: list[CityGap] = Field(default_factory=list)
    non_estimable_fields: list[str] = Field(default_factory=list)


class _AssumptionsEnvelope(BaseModel):
    assumptions: list[AssumptionRecord] = Field(default_factory=list)
    non_estimable: list[NonEstimableRecord] = Field(default_factory=list)


__all__ = [
    "GapClassification",
    "EstimationMethod",
    "FieldStatus",
    "EnrichedFieldSource",
    "FreshnessClassification",
    "ExternalClaimRole",
    "ExternalResolutionAction",
    "FieldClassification",
    "CityGap",
    "GapManifest",
    "WebFinding",
    "SearchBatch",
    "SourceMetadata",
    "TagOptions",
    "SourceSummary",
    "SearchHit",
    "EvidenceCandidateInput",
    "EvidenceCandidate",
    "NoEvidenceRecord",
    "ExternalEvidenceClaim",
    "ExternalSourceAgentResult",
    "ExternalEvidenceResolution",
    "FreshnessResult",
    "EnrichedField",
    "EstimateRange",
    "AssumptionRecord",
    "NonEstimableRecord",
    "EnrichmentMeta",
    "EnrichmentBundle",
    "_GapManifestEnvelope",
    "_AssumptionsEnvelope",
]
