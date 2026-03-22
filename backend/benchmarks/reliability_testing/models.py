from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.utils.config import ReasoningEffort

PayloadCaptureMode = Literal["off", "failed_only", "all"]
BenchmarkRunStatus = Literal["running", "completed", "failed"]


def _clean_string_list(values: list[str]) -> list[str]:
    """Normalize string lists by stripping blanks and preserving order."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = str(value).strip()
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        cleaned.append(candidate)
    return cleaned


class ReliabilityMarkdownDefaults(BaseModel):
    """Benchmark-wide markdown settings kept constant across model runs."""

    max_turns: int = Field(ge=1)
    batch_max_chunks: int = Field(ge=1)
    max_workers: int = Field(ge=1)
    reasoning_effort: ReasoningEffort | None = None


class ReliabilityModelConfig(BaseModel):
    """One OpenRouter model entry from the reliability matrix."""

    id: str = Field(min_length=1)
    enabled: bool = True
    reasoning_effort: ReasoningEffort | None = None

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("Model id cannot be empty.")
        return candidate


class ReliabilityBenchmarkMatrix(BaseModel):
    """Matrix configuration for the markdown reliability benchmark."""

    question: str = Field(min_length=1)
    retrieval_queries: list[str] = Field(min_length=1)
    selected_cities: list[str] = Field(default_factory=list)
    payload_capture_mode: PayloadCaptureMode = "failed_only"
    markdown_defaults: ReliabilityMarkdownDefaults
    models: list[ReliabilityModelConfig] = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _normalize_question(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("Question cannot be empty.")
        return candidate

    @field_validator("retrieval_queries")
    @classmethod
    def _normalize_queries(cls, value: list[str]) -> list[str]:
        cleaned = _clean_string_list(value)
        if not cleaned:
            raise ValueError("retrieval_queries must include at least one query.")
        return cleaned

    @field_validator("selected_cities")
    @classmethod
    def _normalize_cities(cls, value: list[str]) -> list[str]:
        return _clean_string_list(value)

    @model_validator(mode="after")
    def _validate_unique_model_ids(self) -> "ReliabilityBenchmarkMatrix":
        ids = [model.id.casefold() for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("Model ids in the reliability matrix must be unique.")
        return self


class ReliabilityModelResult(BaseModel):
    """Per-model markdown reliability metrics and artifact paths."""

    model_id: str
    model_slug: str
    total_batches: int = Field(ge=0)
    successful_batches: int = Field(ge=0)
    failed_batches: int = Field(ge=0)
    cities_with_failed_batches: list[str] = Field(default_factory=list)
    retry_event_count: int = Field(ge=0)
    retry_exhausted_count: int = Field(ge=0)
    max_turns_count: int = Field(ge=0)
    bad_output_count: int = Field(ge=0)
    failure_reasons: dict[str, int] = Field(default_factory=dict)
    accepted_total: int = Field(ge=0)
    rejected_total: int = Field(ge=0)
    unresolved_total: int = Field(ge=0)
    excerpt_count: int = Field(ge=0)
    runtime_seconds: float = Field(ge=0.0)
    llm_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    failed_entirely: bool = False
    error_code: str = ""
    error_message: str = ""
    run_dir: str
    run_log_path: str
    error_log_path: str
    failed_batch_payloads_path: str | None = None


class ReliabilityBenchmarkProgress(BaseModel):
    """Live benchmark progress snapshot persisted while the benchmark is running."""

    benchmark_id: str
    status: BenchmarkRunStatus
    output_dir: str
    started_at: str
    updated_at: str
    current_model_id: str | None = None
    last_completed_model_id: str | None = None
    total_models: int = Field(default=0, ge=0)
    completed_model_count: int = Field(default=0, ge=0)
    completed_model_ids: list[str] = Field(default_factory=list)
    results_written: int = Field(default=0, ge=0)
    retrieval_written: bool = False
    retrieved_count: int = Field(default=0, ge=0)
    batch_count: int = Field(default=0, ge=0)
    error_type: str | None = None
    error_message: str | None = None


class ReliabilityBenchmarkReport(BaseModel):
    """Aggregate report written for one reliability benchmark execution."""

    benchmark_id: str
    status: BenchmarkRunStatus = "completed"
    started_at: str
    generated_at: str
    output_dir: str
    config_path: str
    matrix_config_path: str
    question: str
    retrieval_queries: list[str]
    selected_cities: list[str]
    payload_capture_mode: PayloadCaptureMode
    retrieved_count: int = Field(ge=0)
    batch_count: int = Field(ge=0)
    results: list[ReliabilityModelResult]
    summary: dict[str, object] = Field(default_factory=dict)


__all__ = [
    "BenchmarkRunStatus",
    "ReliabilityBenchmarkProgress",
    "PayloadCaptureMode",
    "ReliabilityBenchmarkMatrix",
    "ReliabilityBenchmarkReport",
    "ReliabilityMarkdownDefaults",
    "ReliabilityModelConfig",
    "ReliabilityModelResult",
]
