from __future__ import annotations
from pydantic import BaseModel


class WriterOutput(BaseModel):
    content: str


__all__ = ["WriterOutput"]
