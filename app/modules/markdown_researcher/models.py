from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.models import ErrorInfo


class MarkdownExcerpt(BaseModel):
    quote: str
    city_name: str
    partial_answer: str


class MarkdownResearchResult(BaseModel):
    status: Literal["success", "error"] = "success"
    excerpts: list[MarkdownExcerpt] = []
    error: ErrorInfo | None = None


__all__ = ["MarkdownExcerpt", "MarkdownResearchResult"]
