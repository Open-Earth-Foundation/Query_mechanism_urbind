from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.models import ErrorInfo


class OrchestratorDecision(BaseModel):
    status: Literal["success", "error"] = "success"
    action: Literal["write", "stop"]
    reason: str
    confidence: float | None = None
    error: ErrorInfo | None = None


class ResearchQuestionRefinement(BaseModel):
    research_question: str
    retrieval_queries: list[str] = Field(default_factory=list)


class ChatFollowupDecision(BaseModel):
    action: Literal[
        "answer_from_context",
        "search_single_city",
        "out_of_scope",
        "needs_city_clarification",
    ]
    reason: str
    target_city: str | None = None
    rewritten_question: str | None = None
    confidence: float | None = None


__all__ = [
    "ChatFollowupDecision",
    "OrchestratorDecision",
    "ResearchQuestionRefinement",
]
