"""Pydantic models for calculator planning, extraction, and grouped summaries."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.modules.orchestrator.utils.references import is_valid_ref_id

RecordRole = Literal[
    "atomic",
    "reported_total",
    "target",
    "share_percent",
    "context",
]
YearPolicy = Literal["ignore_year", "separate_by_year"]
CalculatorStatus = Literal["success", "partial", "empty", "error"]
WorkerStatus = Literal["records", "done"]

_CATEGORY_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class CalculationCategory(BaseModel):
    """Planner-defined additive category extracted from excerpt evidence."""

    category_key: str
    label: str
    description: str
    operation: Literal["sum"] = "sum"
    preferred_unit: str
    year_policy: YearPolicy
    inclusion_rule: str
    exclusion_rule: str
    sum_reported_total_into_target: bool = False

    @field_validator("category_key")
    @classmethod
    def _validate_category_key(cls, value: str) -> str:
        """Validate that planner category keys stay in snake_case."""
        candidate = value.strip()
        if not _CATEGORY_KEY_PATTERN.fullmatch(candidate):
            raise ValueError("category_key must be unique snake_case.")
        return candidate


class CalculationPlan(BaseModel):
    """Planner output defining up to ten additive calculation categories."""

    categories: list[CalculationCategory] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def _validate_categories(self) -> "CalculationPlan":
        """Validate category count and uniqueness constraints."""
        if len(self.categories) > 10:
            raise ValueError("Calculator planner may return at most 10 categories.")
        seen: set[str] = set()
        for category in self.categories:
            if category.category_key in seen:
                raise ValueError("Calculator planner categories must be unique.")
            seen.add(category.category_key)
        return self


class CalculationRecord(BaseModel):
    """One numeric fact extracted for a category from excerpt evidence."""

    category_key: str
    city: str
    value: float
    unit: str
    note: str
    ref_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    year: int | None = None
    record_role: RecordRole

    @field_validator("category_key")
    @classmethod
    def _validate_record_category_key(cls, value: str) -> str:
        """Validate that extracted records reference a snake_case category key."""
        candidate = value.strip()
        if not _CATEGORY_KEY_PATTERN.fullmatch(candidate):
            raise ValueError("category_key must be unique snake_case.")
        return candidate

    @field_validator("city", "unit", "note")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        """Validate that required text fields are non-empty after trimming."""
        candidate = value.strip()
        if not candidate:
            raise ValueError("String fields must be non-empty.")
        return candidate

    @field_validator("ref_ids")
    @classmethod
    def _validate_ref_ids(cls, value: list[str]) -> list[str]:
        """Validate, trim, and de-duplicate supporting reference ids."""
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            candidate = item.strip()
            if not candidate or candidate in seen:
                continue
            if not is_valid_ref_id(candidate):
                raise ValueError(f"Invalid ref_id `{candidate}`.")
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    @field_validator("source_chunk_ids")
    @classmethod
    def _validate_source_chunk_ids(cls, value: list[str]) -> list[str]:
        """Validate, trim, and de-duplicate supporting source chunk ids."""
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            candidate = item.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized


class CalculationWorkerOutput(BaseModel):
    """Structured output from one calculator worker pass."""

    status: WorkerStatus
    category_key: str
    records: list[CalculationRecord] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def _validate_done_payload(self) -> "CalculationWorkerOutput":
        """Require `done` payloads to stop without returning new records."""
        if self.status == "done" and self.records:
            raise ValueError("status=done must not include records.")
        return self


class CalculationGroupSummary(BaseModel):
    """Deterministic grouped total for one category and normalized unit bucket."""

    normalized_unit: str
    display_unit: str
    year: int | None = None
    current_total: float
    current_terms: list[CalculationRecord] = Field(default_factory=list)
    target_total: float = 0.0
    target_terms: list[CalculationRecord] = Field(default_factory=list)
    non_additive_records: list[CalculationRecord] = Field(default_factory=list)
    current_record_count: int = 0
    target_record_count: int = 0
    current_city_count: int = 0
    target_city_count: int = 0
    selected_city_count: int = 0
    current_coverage_ratio: str = "0/0"
    target_coverage_ratio: str = "0/0"
    cities_with_current_records: list[str] = Field(default_factory=list)
    cities_with_target_records: list[str] = Field(default_factory=list)
    cities_with_only_non_additive_records: list[str] = Field(default_factory=list)
    cities_with_no_usable_records: list[str] = Field(default_factory=list)
    ref_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class CalculationCategorySummary(BaseModel):
    """Writer-facing grouped summary for one planned calculator category."""

    category: CalculationCategory
    status: CalculatorStatus
    note: str = ""
    record_count: int = 0
    current_record_count: int = 0
    target_record_count: int = 0
    records: list[CalculationRecord] = Field(default_factory=list)
    groups: list[CalculationGroupSummary] = Field(default_factory=list)


class CalculationRunSummary(BaseModel):
    """Writer-facing output from the calculator stage."""

    status: CalculatorStatus
    note: str = ""
    selected_city_names: list[str] = Field(default_factory=list)
    category_count: int = 0
    categories: list[CalculationCategorySummary] = Field(default_factory=list)


__all__ = [
    "CalculationCategory",
    "CalculationCategorySummary",
    "CalculationGroupSummary",
    "CalculationPlan",
    "CalculationRecord",
    "CalculationRunSummary",
    "CalculationWorkerOutput",
    "CalculatorStatus",
    "RecordRole",
    "WorkerStatus",
    "YearPolicy",
]
