from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PipelineMode = Literal["ccc_only", "full_pipeline"]
RequestedMode = Literal["ccc_only", "full_pipeline", "both"]
ComparisonStatus = Literal["match", "mismatch", "missing"]

_DYNAMIC_SCOPE_TOKENS = {"*", "all", "all_cities", "poland"}


def _looks_dynamic_scope_token(value: str) -> bool:
    """Return True when one selected-city entry looks like a dynamic scope token."""
    normalized = value.strip().lower()
    if normalized in _DYNAMIC_SCOPE_TOKENS:
        return True
    return normalized.startswith(("group:", "city_group:", "country:", "all_"))


class MetricComponent(BaseModel):
    """One manually curated component used to justify a baseline metric."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    value: int | float | str
    source_note: str = Field(min_length=1)


class BaselineMetric(BaseModel):
    """One benchmarked numeric target extracted from the writer output."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    expected_value: int | float | str
    components: list[MetricComponent] = Field(min_length=1)
    display_metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class WriterNumericBenchmarkCase(BaseModel):
    """One real writer question with a frozen city scope and numeric baseline."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    selected_cities: list[str] = Field(min_length=1)
    baseline_metrics: list[BaselineMetric] = Field(min_length=1)
    requires_explicit_include: bool = False

    @model_validator(mode="after")
    def _validate_case(self) -> "WriterNumericBenchmarkCase":
        """Enforce case-level invariants."""
        if any(_looks_dynamic_scope_token(city) for city in self.selected_cities):
            raise ValueError(
                "selected_cities must be an explicit city list, not a dynamic scope token."
            )
        seen_metric_ids: set[str] = set()
        for metric in self.baseline_metrics:
            if metric.metric_id in seen_metric_ids:
                raise ValueError(
                    f"Duplicate metric_id found in case {self.case_id}: {metric.metric_id}"
                )
            seen_metric_ids.add(metric.metric_id)
        return self


class WriterNumericBenchmarkDataset(BaseModel):
    """Versioned writer numeric benchmark dataset."""

    model_config = ConfigDict(extra="forbid")

    version: int
    default_mode: PipelineMode
    notes: list[str] = Field(default_factory=list)
    cases: list[WriterNumericBenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dataset(self) -> "WriterNumericBenchmarkDataset":
        """Enforce dataset-level invariants."""
        if self.version != 1:
            raise ValueError("Only writer numeric benchmark dataset version=1 is supported.")
        seen_case_ids: set[str] = set()
        for case in self.cases:
            if case.case_id in seen_case_ids:
                raise ValueError(f"Duplicate case_id found: {case.case_id}")
            seen_case_ids.add(case.case_id)
        return self


class WriterMetricExtraction(BaseModel):
    """One extractor decision for one expected benchmark metric."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(min_length=1)
    found: bool
    raw_snippet: str | None = None
    normalized_value: str | int | float | None = None
    unit: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _validate_extraction(self) -> "WriterMetricExtraction":
        """Require a snippet and normalized value when a metric is found."""
        if self.found and self.normalized_value is None:
            raise ValueError("Found metrics must include normalized_value.")
        if self.found and not isinstance(self.raw_snippet, str):
            raise ValueError("Found metrics must include raw_snippet.")
        if self.found and not self.raw_snippet.strip():
            raise ValueError("Found metrics must include raw_snippet.")
        return self


class WriterNumberExtraction(BaseModel):
    """Structured extractor output returned by the LLM tool call."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[WriterMetricExtraction] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_metrics(self) -> "WriterNumberExtraction":
        """Reject duplicate metric ids inside one extraction result."""
        seen_metric_ids: set[str] = set()
        for metric in self.metrics:
            if metric.metric_id in seen_metric_ids:
                raise ValueError(f"Duplicate extracted metric_id: {metric.metric_id}")
            seen_metric_ids.add(metric.metric_id)
        return self


class MetricComparison(BaseModel):
    """Deterministic benchmark comparison for one metric."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    expected_value: int | float | str
    expected_normalized: str | None
    extracted_value: int | float | str | None
    extracted_normalized: str | None
    found: bool
    status: ComparisonStatus
    raw_snippet: str | None = None
    notes: str = ""


class WriterNumericCaseResult(BaseModel):
    """Persisted result for one case and one pipeline mode."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    mode: PipelineMode
    question: str = Field(min_length=1)
    selected_cities: list[str] = Field(min_length=1)
    run_id: str = Field(min_length=1)
    run_dir: str = Field(min_length=1)
    final_output_path: str = Field(min_length=1)
    context_bundle_path: str = Field(min_length=1)
    extracted_numbers_path: str = Field(min_length=1)
    metric_results: list[MetricComparison] = Field(min_length=1)
    match_count: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)


class WriterNumericBenchmarkSummary(BaseModel):
    """Aggregate rollup for one writer numeric benchmark run."""

    model_config = ConfigDict(extra="forbid")

    case_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
    metric_count: int = Field(ge=0)
    match_count: int = Field(ge=0)
    mismatch_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)


class WriterNumericBenchmarkReport(BaseModel):
    """Top-level persisted report for the writer numeric benchmark."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    benchmark_file: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    requested_mode: RequestedMode
    executed_modes: list[PipelineMode] = Field(min_length=1)
    results: list[WriterNumericCaseResult] = Field(min_length=1)
    summary: WriterNumericBenchmarkSummary


__all__ = [
    "BaselineMetric",
    "ComparisonStatus",
    "MetricComparison",
    "MetricComponent",
    "PipelineMode",
    "RequestedMode",
    "WriterMetricExtraction",
    "WriterNumberExtraction",
    "WriterNumericBenchmarkCase",
    "WriterNumericBenchmarkDataset",
    "WriterNumericBenchmarkReport",
    "WriterNumericBenchmarkSummary",
]
