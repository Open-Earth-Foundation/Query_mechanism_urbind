from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

WriterCitationCoverageStatus = Literal["confirmed", "partial"]


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
    "WriterMultiPassBatch",
    "WriterMultiPassPlan",
    "WriterOutput",
    "WriterSectionPlan",
    "WriterSectionSpec",
]
