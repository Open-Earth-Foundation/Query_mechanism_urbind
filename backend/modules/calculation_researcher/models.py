from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.modules.orchestrator.utils.references import is_valid_ref_id

CalculationOperation = Literal["sum", "subtract", "multiply", "divide"]
CalculationYearRule = Literal[
    "same_year_only",
    "latest_available_per_city",
    "user_specified_year",
]


class CalculationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calculation_goal: str
    metric_name: str
    operation: CalculationOperation
    city_scope: list[str] = Field(min_length=1)
    inclusion_rule: str
    exclusion_rule: str
    year_rule: CalculationYearRule
    target_year: int | None = None
    unit_rule: str
    notes_for_subagent: str = ""

    @model_validator(mode="after")
    def _validate_year_configuration(self) -> "CalculationRequest":
        if self.year_rule == "user_specified_year" and self.target_year is None:
            raise ValueError("target_year is required when year_rule=user_specified_year.")
        if self.year_rule != "user_specified_year" and self.target_year is not None:
            raise ValueError("target_year must be null unless year_rule=user_specified_year.")
        return self


class IncludedCityValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city_name: str
    year: int | None = None
    value: float
    unit: str
    ref_ids: list[str] = Field(min_length=1)
    evidence_note: str = ""

    @model_validator(mode="after")
    def _validate_ref_ids(self) -> "IncludedCityValue":
        invalid = [ref_id for ref_id in self.ref_ids if not is_valid_ref_id(ref_id)]
        if invalid:
            raise ValueError(f"Invalid ref ids: {', '.join(invalid)}")
        return self


class ExcludedPolicyCity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city_name: str
    reason_no_numeric: str
    policy_summary: str
    ref_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ref_ids(self) -> "ExcludedPolicyCity":
        invalid = [ref_id for ref_id in self.ref_ids if not is_valid_ref_id(ref_id)]
        if invalid:
            raise ValueError(f"Invalid ref ids: {', '.join(invalid)}")
        return self


class CalculationAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str
    ref_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ref_ids(self) -> "CalculationAssumption":
        invalid = [ref_id for ref_id in self.ref_ids if not is_valid_ref_id(ref_id)]
        if invalid:
            raise ValueError(f"Invalid ref ids: {', '.join(invalid)}")
        return self


class CalculationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class CalculationSubagentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "partial", "error"]
    metric_name: str
    operation: CalculationOperation
    total_value: float | None = None
    unit: str | None = None
    coverage_observed: int = 0
    coverage_total: int = 0
    included_cities: list[IncludedCityValue] = Field(default_factory=list)
    excluded_policy_cities: list[ExcludedPolicyCity] = Field(default_factory=list)
    assumptions: list[CalculationAssumption] = Field(default_factory=list)
    final_ref_ids: list[str] = Field(default_factory=list)
    error: CalculationError | None = None

    @model_validator(mode="after")
    def _validate_ref_ids(self) -> "CalculationSubagentOutput":
        invalid = [ref_id for ref_id in self.final_ref_ids if not is_valid_ref_id(ref_id)]
        if invalid:
            raise ValueError(f"Invalid final_ref_ids: {', '.join(invalid)}")
        return self


__all__ = [
    "CalculationAssumption",
    "CalculationError",
    "CalculationOperation",
    "CalculationRequest",
    "CalculationSubagentOutput",
    "CalculationYearRule",
    "ExcludedPolicyCity",
    "IncludedCityValue",
]
