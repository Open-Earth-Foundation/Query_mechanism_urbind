from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkJudgeScore(BaseModel):
    """Rubric scores for one candidate output."""

    factual_coverage: int = Field(ge=0, le=5)
    numeric_specificity: int = Field(ge=0, le=5)
    faithfulness_to_text: int = Field(ge=0, le=5)
    structure_and_clarity: int = Field(ge=0, le=5)
    total_score: int = Field(ge=0, le=20)
    rationale: str = Field(min_length=1)


class BenchmarkJudgeEvaluation(BaseModel):
    """Pairwise comparison output for benchmark judging."""

    left_label: str = Field(min_length=1)
    right_label: str = Field(min_length=1)
    left: BenchmarkJudgeScore
    right: BenchmarkJudgeScore
    winner: str = Field(pattern="^(left|right|tie)$")
    confidence: float = Field(ge=0.0, le=1.0)
    comparative_rationale: str = Field(min_length=1)


__all__ = ["BenchmarkJudgeScore", "BenchmarkJudgeEvaluation"]
