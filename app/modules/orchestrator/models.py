from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.models import ErrorInfo


class OrchestratorDecision(BaseModel):
    status: Literal["success", "error"] = "success"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: Literal[
        "write",
        "run_sql",
        # "run_markdown",  # Disabled in current architecture: markdown runs once pre-loop.
        "stop",
    ]
    reason: str
    confidence: float | None = None
    follow_up_question: str | None = None
    error: ErrorInfo | None = None

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_legacy_action(cls, value: object) -> object:
        # Backward compatibility: map legacy run_markdown requests to write.
        if value == "run_markdown":
            return "write"
        return value


__all__ = ["OrchestratorDecision"]
