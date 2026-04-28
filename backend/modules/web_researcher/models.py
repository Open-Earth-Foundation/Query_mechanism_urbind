"""Pydantic models for the web research enrichment and assumptions modelling layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Type literals
# ---------------------------------------------------------------------------
GapClassification = Literal["estimable_numerical", "derivable_from_ratio", "non_estimable"]
EstimationMethod = Literal[
    "national_regional_average",
    "peer_city_proxy",
    "expert_heuristic_scaling",
    "structured_lookup",
]
FieldStatus = Literal["resolved", "partially_resolved", "still_missing"]
FreshnessClassification = Literal["consistent", "superseded", "uncertain", "cancelled"]
# (SourceTier defined below alongside WebFinding so it stays close to its consumer.)

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


class FieldDecomposition(BaseModel):
    """Output of Phase 0 — fields decomposed and classified, no per-city detection.

    Used by the new two-pass gap analyst flow (orchestrator Phase 0 + Phase 2).
    Phase 1 fan-out runs between the decomposition and per-city gap detection
    so that local data (markdown, structured lookups, vector benchmarks) can
    enrich the context bundle before per-city gaps are computed.
    """

    query_fields: list[FieldClassification]
    non_estimable_fields: list[str]


class StructuredLookupResult(BaseModel):
    """One (city, field) value resolved by a structured-lookup ingestion.

    Emitted by Phase 1 fan-out and merged into ``context_bundle["phase1"]``
    so that Phase 2 gap detection sees the values without an LLM call.
    """

    source_id: str
    ingestion_id: str
    city: str
    field: str
    value: float | int | str | None = None
    unit: str | None = None
    asof: str | None = None
    extra: dict[str, object] = Field(default_factory=dict)


class BenchmarkExcerptRecord(BaseModel):
    """One benchmark excerpt retrieved by Phase 1 fan-out via similarity search."""

    chunk_id: str
    source_id: str
    ingestion_id: str
    source_path: str
    tier: str
    doc_slug: str
    heading_path: str
    block_type: str
    raw_text: str
    distance: float
    chunk_index: int | None = None


class Phase1Artefacts(BaseModel):
    """Aggregate of everything Phase 1 fan-out produced for one run."""

    structured_lookups: list[StructuredLookupResult] = Field(default_factory=list)
    benchmark_excerpts: list[BenchmarkExcerptRecord] = Field(default_factory=list)
    queried_cities: list[str] = Field(default_factory=list)
    queried_fields: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Web Research models (Agent 3 output — Phase 2)
# ---------------------------------------------------------------------------


SourceTier = Literal["tier1", "open"]


class WebFinding(BaseModel):
    city: str
    field: str
    value: str | float | int | None
    unit: str | None = None
    source_url: str
    source_type: str  # e.g. "operator_press_release", "government_report"
    source_date: str | None = None
    extraction_confidence: float
    # Provenance — populated when the search worker uses the tier-1 pre-pass.
    source_id: str | None = None
    source_tier: SourceTier | None = None


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
    # Populated when the field's value was resolved through a manifest-declared
    # source (tier-1 web allowlist entry, structured lookup, or benchmark
    # collection).  ``source_id`` references a stable handle in
    # ``backend/data/sources_manifest.yaml`` or ``tier1_web_sources.yaml``;
    # the writer uses this for human-readable attribution.
    source_id: str | None = None
    source_tier: SourceTier | None = None
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


class _FieldDecompositionEnvelope(BaseModel):
    query_fields: list[FieldClassification] = Field(default_factory=list)
    non_estimable_fields: list[str] = Field(default_factory=list)


class _CityGapsEnvelope(BaseModel):
    city_gaps: list[CityGap] = Field(default_factory=list)


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
    "FieldDecomposition",
    "StructuredLookupResult",
    "BenchmarkExcerptRecord",
    "Phase1Artefacts",
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
    "_FieldDecompositionEnvelope",
    "_CityGapsEnvelope",
    "_AssumptionsEnvelope",
]
