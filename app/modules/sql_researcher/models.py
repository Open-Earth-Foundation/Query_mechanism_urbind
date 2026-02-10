from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.models import ErrorInfo


class SqlQuery(BaseModel):
    query_id: str
    query: str
    rationale: str | None = None


class SqlQueryPlan(BaseModel):
    status: Literal["success", "error"] = "success"
    queries: list[SqlQuery] = []
    error: ErrorInfo | None = None


class SqlQueryResult(BaseModel):
    query_id: str
    columns: list[str]
    rows: list[list[str | int | float | None]]
    row_count: int
    elapsed_ms: int
    token_count: int
    truncated: bool = False


class SqlResearchResult(BaseModel):
    status: Literal["success", "error"] = "success"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    queries: list[SqlQuery]
    results: list[SqlQueryResult]
    total_token_count: int
    truncation_applied: bool
    error: ErrorInfo | None = None


__all__ = ["SqlQuery", "SqlQueryPlan", "SqlQueryResult", "SqlResearchResult"]
