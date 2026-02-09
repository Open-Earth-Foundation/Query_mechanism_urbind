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


class MarkdownResearchResult(BaseModel):
    status: Literal["success", "error"] = "success"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    excerpts: list[MarkdownExcerpt] = []
    error: ErrorInfo | None = None


__all__ = ["MarkdownExcerpt", "MarkdownResearchResult"]
