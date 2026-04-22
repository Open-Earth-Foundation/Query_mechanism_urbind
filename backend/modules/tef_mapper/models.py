from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.models import ErrorInfo

TefSectorKey = Literal["transport", "industry", "afolu", "buildings", "energy", "waste"]
TefTargetType = Literal["transition_element", "subcategory"]


class TefSectorCard(BaseModel):
    """Compact TEF root-sector card loaded from the local catalog."""

    model_config = ConfigDict(extra="forbid")

    sector: TefSectorKey
    path: str
    path_code: str
    label: str
    description: str
    total_transition_count: int
    child_subcategory_count: int
    card_text: str


class TefSubsectorCard(BaseModel):
    """Compact TEF category card loaded from the local catalog."""

    model_config = ConfigDict(extra="forbid")

    path: str
    path_code: str
    label: str
    sector: TefSectorKey
    parent_path: str
    depth: int
    direct_transition_count: int
    total_transition_count: int
    has_transition_elements: bool
    description: str
    card_text: str


class TefTransitionElement(BaseModel):
    """Compact TEF Transition Element record loaded from the local catalog."""

    model_config = ConfigDict(extra="forbid")

    tef_id: str
    title: str
    sector: TefSectorKey
    path: str
    path_code: str
    path_labels: list[str]
    type: str | None = None
    unit_of_measure: str | None = None
    sustainability: str | None = None
    description: str | None = None
    long_name: str | None = None
    short_name: str | None = None
    shift_from: list[str] = Field(default_factory=list)
    shift_to: list[str] = Field(default_factory=list)
    carbon_causal_chains: list[str] = Field(default_factory=list)
    ipcc_mitigation_method: str | None = None
    tef_source_path: str


class TefSectorAlternative(BaseModel):
    """Alternative TEF root sector considered by the sector router."""

    model_config = ConfigDict(extra="forbid")

    sector: TefSectorKey
    path: str
    confidence: float = Field(ge=0.0, le=1.0)


class TefSectorRoute(BaseModel):
    """LLM output for root-sector routing."""

    model_config = ConfigDict(extra="forbid")

    sector: TefSectorKey
    selected_path: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    rationale: str
    alternatives: list[TefSectorAlternative] = Field(default_factory=list)


class TefPathAlternative(BaseModel):
    """Alternative TEF category path considered by the router."""

    model_config = ConfigDict(extra="forbid")

    path: str
    confidence: float = Field(ge=0.0, le=1.0)


class TefSubsectorRoute(BaseModel):
    """LLM output for one category-routing step."""

    model_config = ConfigDict(extra="forbid")

    selected_path: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    rationale: str
    alternatives: list[TefPathAlternative] = Field(default_factory=list)


class TefTransitionMatch(BaseModel):
    """One proposed mapping from an initiative to a TEF Transition Element."""

    model_config = ConfigDict(extra="forbid")

    tef_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_primary: bool
    rationale: str


class TefTransitionMapping(BaseModel):
    """LLM output for final Transition Element mapping."""

    model_config = ConfigDict(extra="forbid")

    needs_review: bool
    matches: list[TefTransitionMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_single_primary(self) -> TefTransitionMapping:
        """Ensure at most one Transition Element match is marked primary."""
        primary_count = sum(1 for match in self.matches if match.is_primary)
        if primary_count > 1:
            raise ValueError("At most one Transition Element match may be primary.")
        return self


class TefSectorRouteRecord(BaseModel):
    """Artifact row for one initiative sector-routing pass."""

    model_config = ConfigDict(extra="forbid")

    initiative_record_id: str
    source_document: str
    status: Literal["success", "error"]
    route: TefSectorRoute | None = None
    error: ErrorInfo | None = None


class TefSubsectorRouteRecord(BaseModel):
    """Artifact row for one initiative category-routing step."""

    model_config = ConfigDict(extra="forbid")

    initiative_record_id: str
    source_document: str
    parent_path: str
    candidate_paths: list[str]
    status: Literal["success", "error"]
    route: TefSubsectorRoute | None = None
    error: ErrorInfo | None = None


class TefTransitionMappingRecord(BaseModel):
    """Artifact row for one initiative Transition Element mapping pass."""

    model_config = ConfigDict(extra="forbid")

    initiative_record_id: str
    source_document: str
    selected_path: str
    candidate_tef_ids: list[str]
    status: Literal["success", "error"]
    mapping: TefTransitionMapping | None = None
    error: ErrorInfo | None = None


class TefMappingReviewItem(BaseModel):
    """Review item emitted for manual review of the JSON mapping artifacts."""

    model_config = ConfigDict(extra="forbid")

    review_type: str
    severity: Literal["info", "warning", "error"] = "warning"
    message: str
    initiative_record_id: str | None = None
    source_document: str | None = None
    target_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TefFinalMappingRecord(BaseModel):
    """Durable JSON row mapping one initiative to a TEF target."""

    model_config = ConfigDict(extra="forbid")

    initiative_record_id: str
    city: str
    source_document: str
    document_local_code: str | None = None
    initiative_name: str
    target_type: TefTargetType
    target_id: str
    target_path: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_primary: bool
    needs_review: bool
    rationale: str
    sector_route: dict[str, Any]
    subsector_routes: list[dict[str, Any]] = Field(default_factory=list)
    mapper_version: str
    tef_source_version: str
    extraction_run_id: str | None = None


class TefInitiativeMappingResult(BaseModel):
    """Complete staged TEF mapping result for one initiative record."""

    model_config = ConfigDict(extra="forbid")

    initiative_record_id: str
    source_document: str
    status: Literal["success", "error"]
    sector_route_record: TefSectorRouteRecord | None = None
    subsector_route_records: list[TefSubsectorRouteRecord] = Field(default_factory=list)
    transition_mapping_record: TefTransitionMappingRecord | None = None
    final_mappings: list[TefFinalMappingRecord] = Field(default_factory=list)
    review_items: list[TefMappingReviewItem] = Field(default_factory=list)
    error: ErrorInfo | None = None


class TefMappingRunResult(BaseModel):
    """Summary returned by the artifact-first TEF mapping pipeline."""

    run_id: str
    output_dir: str
    initiatives_count: int
    mapped_initiatives_count: int
    final_mappings_count: int
    review_items_count: int


__all__ = [
    "TefFinalMappingRecord",
    "TefInitiativeMappingResult",
    "TefMappingReviewItem",
    "TefMappingRunResult",
    "TefPathAlternative",
    "TefSectorAlternative",
    "TefSectorCard",
    "TefSectorKey",
    "TefSectorRoute",
    "TefSectorRouteRecord",
    "TefSubsectorCard",
    "TefSubsectorRoute",
    "TefSubsectorRouteRecord",
    "TefTargetType",
    "TefTransitionElement",
    "TefTransitionMapping",
    "TefTransitionMappingRecord",
    "TefTransitionMatch",
]
