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
FreshnessClassification = Literal["consistent", "superseded", "uncertain", "cancelled"]

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
    source: Literal["ccc", "web", "estimated", "none"] = "none"
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
    elapsed_seconds: float


class EnrichmentBundle(BaseModel):
    gap_manifest: GapManifest
    enriched_fields: list[EnrichedField] = Field(default_factory=list)
    web_findings: list[WebFinding] = Field(default_factory=list)
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
    "FreshnessClassification",
    "FieldClassification",
    "CityGap",
    "GapManifest",
    "WebFinding",
    "SearchBatch",
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
