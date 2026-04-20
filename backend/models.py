from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: str | list[str] | None = None


class BaseResult(BaseModel):
    status: Literal["success", "error"] = "success"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: ErrorInfo | None = None


class RunMetadata(BaseModel):
    run_id: str
    question: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    final_output_path: str | None = None


__all__ = [
    "ErrorInfo",
    "BaseResult",
    "RunMetadata",
]
