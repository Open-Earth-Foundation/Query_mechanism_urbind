from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.models import ErrorInfo


class OrchestratorDecision(BaseModel):
    status: Literal["success", "error"] = "success"
    action: Literal["write", "stop"]
    reason: str
    confidence: float | None = None
    error: ErrorInfo | None = None


class ResearchQuestionRefinement(BaseModel):
    research_question: str


__all__ = ["OrchestratorDecision", "ResearchQuestionRefinement"]
