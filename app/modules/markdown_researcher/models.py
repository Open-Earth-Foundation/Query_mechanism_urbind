from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.models import ErrorInfo


class MarkdownExcerpt(BaseModel):
    snippet: str
    city_name: str
    answer: str
    relevant: Literal["yes", "no"]


class MarkdownCityScope(BaseModel):
    status: Literal["success", "error"] = "success"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scope: Literal["all", "subset"]
    city_names: list[str] = []
    reason: str | None = None
    error: ErrorInfo | None = None


class MarkdownResearchResult(BaseModel):
    status: Literal["success", "error"] = "success"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    excerpts: list[MarkdownExcerpt] = []
    city_scope: MarkdownCityScope | None = None
    error: ErrorInfo | None = None


__all__ = ["MarkdownExcerpt", "MarkdownCityScope", "MarkdownResearchResult"]
