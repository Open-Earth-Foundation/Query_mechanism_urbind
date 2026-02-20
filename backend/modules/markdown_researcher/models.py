from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.models import ErrorInfo


class MarkdownExcerpt(BaseModel):
    quote: str
    city_name: str
    partial_answer: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class MarkdownResearchResult(BaseModel):
    status: Literal["success", "error"] = "success"
    excerpts: list[MarkdownExcerpt] = []
    error: ErrorInfo | None = None


__all__ = ["MarkdownExcerpt", "MarkdownResearchResult"]
