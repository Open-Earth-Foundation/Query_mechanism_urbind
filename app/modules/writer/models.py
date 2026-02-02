from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.models import ErrorInfo


class WriterOutput(BaseModel):
    status: Literal["success", "error"] = "success"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: str
    output_path: str | None = None
    draft_paths: list[str] = []
    summary: str | None = None
    error: ErrorInfo | None = None


__all__ = ["WriterOutput"]
