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
FieldStatus = Literal[
    "resolved",
    # ``bundled_only`` — the city reports an aggregate / bundled value but
    # the question asks for a disaggregated line that's not present (e.g. the
    # CCC says "total fleet CAPEX = €100M" but we asked about per-vehicle
    # CAPEX).  The bundled total is recorded in ``provenance.bundled_value``
    # so the estimator can compute the disaggregated line via per-unit
    # cost ratios from peers.
    "bundled_only",
    "partially_resolved",
    "still_missing",
]
FreshnessClassification = Literal["consistent", "superseded", "uncertain", "cancelled"]
# Scope classifies *what kind of actor or asset* a field measures.  Two fields
# with different scopes must not be summed into a headline total — e.g.
# municipal-fleet CAPEX and public-transport CAPEX live on different ledgers.
# The writer enforces per-scope subtotals based on this tag.
Scope = Literal[
    "municipal",
    "public_transport",
    "private",
    "mixed",
    "unscoped",
]
# (SourceTier defined below alongside WebFinding so it stays close to its consumer.)

# ---------------------------------------------------------------------------
# Gap Analysis models (Agent 1 output)
# ---------------------------------------------------------------------------


class FieldClassification(BaseModel):
    field: str
    classification: GapClassification
    searchable: bool
    rationale: str
    scope: Scope = "unscoped"


class CityGap(BaseModel):
    city: str
    blank_fields: list[str]
    stale_flags: list[str]
    # Fields where the city reports a parent / aggregate value (e.g. "total
    # fleet CAPEX") but the requested disaggregated line (e.g. per-vehicle
    # CAPEX) is not present.  These flow into the estimator as
    # ``status="bundled_only"`` and use peer per-unit ratios to derive the
    # missing line rather than being treated as fully resolved.
    bundled_fields: list[str] = Field(default_factory=list)
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


class FinancingBlock(BaseModel):
    """Optional split of a value across funding sources.

    Each component is a numeric share *of the same unit as the parent value*
    (typically currency).  Components left as ``None`` mean the source is
    not separately disclosed; ``gap`` is the unfunded remainder when known.
    The writer surfaces this as a column in the augmented data table.
    """

    federal: float | None = None
    state: float | None = None
    eu: float | None = None
    operator: float | None = None
    gap: float | None = None
    notes: str | None = None


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
    # Copied from the matching FieldClassification at merge time.  Used by
    # the writer to enforce per-scope subtotals (no cross-scope summing).
    scope: Scope = "unscoped"
    # Optional funding-source split (federal / state / EU / operator / gap).
    # Surfaced by the writer as a "Financing" column in the augmented data
    # table when at least one component is populated.
    financing: FinancingBlock | None = None


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


class DerivedMetric(BaseModel):
    """A ratio computed once from the resolved dataset.

    Two flavors today:
    - ``per_capita`` — value / city population (city-proper, from URBAN AUDIT).
    - ``per_unit`` — numerator / denominator within the same city + scope
      (e.g. total_capex / vehicle_count → per-vehicle CAPEX).

    Scope safety: the metric carries the parent field's ``scope``.  The
    writer must NOT aggregate derived metrics across different scopes.
    """

    city: str
    metric: str  # human-readable label, e.g. "per_capita_capex" or "per_vehicle_capex"
    kind: Literal["per_capita", "per_unit"]
    value: float
    unit: str | None = None
    numerator_field: str
    numerator_value: float
    denominator_field: str  # e.g. "city_population" or "vehicle_count"
    denominator_value: float
    scope: Scope = "unscoped"
    notes: str | None = None


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
    derived_metrics: list[DerivedMetric] = Field(default_factory=list)
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
    "Scope",
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
    "FinancingBlock",
    "EstimateRange",
    "AssumptionRecord",
    "NonEstimableRecord",
    "DerivedMetric",
    "EnrichmentMeta",
    "EnrichmentBundle",
    "_GapManifestEnvelope",
    "_FieldDecompositionEnvelope",
    "_CityGapsEnvelope",
    "_AssumptionsEnvelope",
]
