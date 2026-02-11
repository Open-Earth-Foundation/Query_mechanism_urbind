from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from backend.models import ErrorInfo


class OrchestratorDecision(BaseModel):
    status: Literal["success", "error"] = "success"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: Literal["write", "run_sql", "run_markdown", "stop"]
    reason: str
    confidence: float | None = None
    follow_up_question: str | None = None
    error: ErrorInfo | None = None


__all__ = ["OrchestratorDecision"]
