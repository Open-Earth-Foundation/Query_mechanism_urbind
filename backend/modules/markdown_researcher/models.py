from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.models import ErrorInfo


class MarkdownExcerpt(BaseModel):
    quote: str
    city_name: str
    partial_answer: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class MarkdownBatchFailure(BaseModel):
    """Structured failure payload for one markdown batch."""

    city_name: str
    batch_index: int
    reason: str
    unresolved_chunk_ids: list[str] = Field(default_factory=list)


class MarkdownResearchResult(BaseModel):
    """Structured markdown extraction result with explicit chunk decisions."""

    status: Literal["success", "error"] = "success"
    excerpts: list[MarkdownExcerpt] = Field(default_factory=list)
    accepted_chunk_ids: list[str] = Field(default_factory=list)
    rejected_chunk_ids: list[str] = Field(default_factory=list)
    unresolved_chunk_ids: list[str] = Field(default_factory=list)
    batch_failures: list[MarkdownBatchFailure] = Field(default_factory=list)
    error: ErrorInfo | None = None


__all__ = ["MarkdownBatchFailure", "MarkdownExcerpt", "MarkdownResearchResult"]
