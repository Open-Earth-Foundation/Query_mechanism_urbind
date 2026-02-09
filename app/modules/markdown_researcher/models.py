from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.models import ErrorInfo


class MarkdownExcerpt(BaseModel):
    snippet: str
    city_name: str
    answer: str
    relevant: Literal["yes", "no"]


class MarkdownResearchResult(BaseModel):
    status: Literal["success", "error"] = "success"
    excerpts: list[MarkdownExcerpt] = []
    error: ErrorInfo | None = None


__all__ = ["MarkdownExcerpt", "MarkdownResearchResult"]
