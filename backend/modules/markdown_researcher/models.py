from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.models import ErrorInfo


class MarkdownExcerpt(BaseModel):
    quote: str
    city_name: str
    partial_answer: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class ThrownExcerpt(BaseModel):
    quote: str
    city_name: str
    partial_answer: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    rejection_stage: Literal["batch_validation", "dedupe"]
    reason_codes: list[str] = Field(default_factory=list)
    batch_index: int
    expected_city_name: str
    invalid_source_chunk_ids: list[str] = Field(default_factory=list)


class MarkdownResearchResult(BaseModel):
    status: Literal["success", "error"] = "success"
    excerpts: list[MarkdownExcerpt] = Field(default_factory=list)
    thrown_excerpts: list[ThrownExcerpt] = Field(default_factory=list)
    error: ErrorInfo | None = None


__all__ = ["MarkdownExcerpt", "MarkdownResearchResult", "ThrownExcerpt"]
