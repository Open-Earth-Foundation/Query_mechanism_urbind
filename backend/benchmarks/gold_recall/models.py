from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ChunkBucket = Literal["seed_hit", "neighbor_only_hit", "fallback_top_up_hit", "miss"]
FactJudgeVerdict = Literal["YES", "NO"]
FactJudgeStage = Literal["stage_b", "stage_c"]


def _normalize_string_list(value: object) -> list[str]:
    """Normalize one string-or-list input into a de-duplicated string list."""
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list):
        raw_values = [item for item in value if isinstance(item, str)]
    else:
        raw_values = []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        candidate = raw_value.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


class GoldBenchmarkCase(BaseModel):
    """One manually curated benchmark question and its gold answers."""

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_chunk_ids: list[str] = Field(min_length=1)
    gold_chunk_texts: list[str] | None = None
    gold_facts: list[str] = Field(min_length=1)
    gold_city: list[str] = Field(min_length=1)
    selected_cities: list[str] | None = None
    cached_run_dir: str | None = None

    @field_validator("gold_chunk_ids", "gold_facts", "gold_city", mode="before")
    @classmethod
    def _validate_required_lists(cls, value: object) -> list[str]:
        """Normalize required list fields before model validation."""
        normalized = _normalize_string_list(value)
        if not normalized:
            raise ValueError("Expected a non-empty string list.")
        return normalized

    @field_validator("selected_cities", mode="before")
    @classmethod
    def _validate_optional_city_list(cls, value: object) -> list[str] | None:
        """Normalize the optional selected-cities override."""
        if value == []:
            return []
        normalized = _normalize_string_list(value)
        return normalized or None

    @field_validator("gold_chunk_texts", mode="before")
    @classmethod
    def _validate_optional_chunk_texts(cls, value: object) -> list[str] | None:
        """Normalize optional canonical chunk texts."""
        if value is None:
            return None
        normalized = _normalize_string_list(value)
        return normalized or None

    @model_validator(mode="after")
    def _validate_case(self) -> "GoldBenchmarkCase":
        """Enforce per-case invariants."""
        if self.gold_chunk_texts is not None and len(self.gold_chunk_texts) != len(
            self.gold_chunk_ids
        ):
            raise ValueError(
                "gold_chunk_texts must align 1:1 with gold_chunk_ids when provided."
            )
        return self

    def resolved_selected_cities(self) -> list[str]:
        """Return the cities that should be passed to the live pipeline."""
        if self.selected_cities is not None:
            return list(self.selected_cities)
        return list(self.gold_city)


class GoldBenchmarkDataset(BaseModel):
    """Versioned gold benchmark input file."""

    version: int
    cases: list[GoldBenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dataset(self) -> "GoldBenchmarkDataset":
        """Enforce dataset invariants that span multiple cases."""
        if self.version != 1:
            raise ValueError("Only gold benchmark dataset version=1 is supported.")
        seen_case_ids: set[str] = set()
        for case in self.cases:
            if case.case_id in seen_case_ids:
                raise ValueError(f"Duplicate case_id found: {case.case_id}")
            seen_case_ids.add(case.case_id)
        return self


class RetrievalChunkDiagnostic(BaseModel):
    """Per-gold-chunk retrieval diagnosis used in benchmark output."""

    chunk_id: str
    bucket: ChunkBucket
    seed_rank: int | None = None
    selection_mode: str | None = None


class FactPresenceJudgement(BaseModel):
    """One LLM-as-judge fact presence decision."""

    stage: FactJudgeStage
    fact: str
    verdict: FactJudgeVerdict
    rationale: str = Field(min_length=1)

    @property
    def is_hit(self) -> bool:
        """Return True when the judged fact is present."""
        return self.verdict == "YES"


class StageARetrievalMetrics(BaseModel):
    """Stage A retrieval metrics for one benchmark case."""

    retrieval_recall: float = Field(ge=0.0, le=1.0)
    retrieval_precision: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    delivery_recall: float = Field(ge=0.0, le=1.0)
    delivery_precision: float = Field(ge=0.0, le=1.0)
    seed_hit_count: int = Field(ge=0)
    neighbor_only_hit_count: int = Field(ge=0)
    fallback_top_up_hit_count: int = Field(ge=0)
    miss_count: int = Field(ge=0)


class StageBExtractionMetrics(BaseModel):
    """Stage B markdown researcher metrics for one benchmark case."""

    extraction_recall: float = Field(ge=0.0, le=1.0)
    fact_extraction_rate: float = Field(ge=0.0, le=1.0)


class StageCWriterMetrics(BaseModel):
    """Stage C writer metrics for one benchmark case."""

    end_to_end_fact_recall: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)


class LossWaterfall(BaseModel):
    """Per-case loss waterfall that shows where information dropped."""

    gold_chunk_count: int = Field(ge=0)
    seed_hit_chunk_count: int = Field(ge=0)
    delivery_hit_chunk_count: int = Field(ge=0)
    stage_b_fact_hit_count: int = Field(ge=0)
    stage_c_fact_hit_count: int = Field(ge=0)


class RecallBenchmarkCaseResult(BaseModel):
    """Full benchmark output for one gold case."""

    case_id: str
    question: str
    gold_city: list[str]
    selected_cities: list[str]
    used_cached_run: bool
    run_dir: str
    retrieval_path: str
    excerpts_path: str
    references_path: str
    final_output_path: str
    stage_a: StageARetrievalMetrics
    stage_b: StageBExtractionMetrics
    stage_c: StageCWriterMetrics
    loss_waterfall: LossWaterfall
    chunk_diagnostics: list[RetrievalChunkDiagnostic]
    stage_b_judgements: list[FactPresenceJudgement]
    stage_c_judgements: list[FactPresenceJudgement]


class RecallBenchmarkSummary(BaseModel):
    """Aggregate rollup across all benchmark cases."""

    case_count: int = Field(ge=0)
    retrieval_recall_mean: float = Field(ge=0.0, le=1.0)
    retrieval_precision_mean: float = Field(ge=0.0, le=1.0)
    mrr_mean: float = Field(ge=0.0, le=1.0)
    delivery_recall_mean: float = Field(ge=0.0, le=1.0)
    delivery_precision_mean: float = Field(ge=0.0, le=1.0)
    extraction_recall_mean: float = Field(ge=0.0, le=1.0)
    fact_extraction_rate_mean: float = Field(ge=0.0, le=1.0)
    end_to_end_fact_recall_mean: float = Field(ge=0.0, le=1.0)
    citation_coverage_mean: float = Field(ge=0.0, le=1.0)


class RecallBenchmarkReport(BaseModel):
    """Persisted benchmark report covering every evaluated gold case."""

    benchmark_id: str
    generated_at: str
    output_dir: str
    gold_file: str
    judge_model: str
    results: list[RecallBenchmarkCaseResult]
    summary: RecallBenchmarkSummary


class FactJudgeDecision(BaseModel):
    """Structured output returned by the fact-presence judge."""

    verdict: FactJudgeVerdict
    rationale: str = Field(min_length=1)


__all__ = [
    "ChunkBucket",
    "FactJudgeDecision",
    "FactJudgeStage",
    "FactPresenceJudgement",
    "GoldBenchmarkCase",
    "GoldBenchmarkDataset",
    "LossWaterfall",
    "RecallBenchmarkCaseResult",
    "RecallBenchmarkReport",
    "RecallBenchmarkSummary",
    "RetrievalChunkDiagnostic",
    "StageARetrievalMetrics",
    "StageBExtractionMetrics",
    "StageCWriterMetrics",
]
