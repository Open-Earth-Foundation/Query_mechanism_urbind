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


class SchemaForeignKey(BaseModel):
    column: str
    ref_table: str
    ref_column: str


class SchemaColumn(BaseModel):
    name: str
    type: str  # e.g., "String", "Text", "Numeric", "Integer", "Boolean", etc.


class SchemaTable(BaseModel):
    name: str
    columns: list[str]  # kept for backward compatibility
    columns_with_types: list[SchemaColumn] = []  # detailed column info
    foreign_keys: list[SchemaForeignKey]
    unique_constraints: list[list[str]]


class SchemaSummary(BaseModel):
    tables: list[SchemaTable]


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
    "SchemaForeignKey",
    "SchemaColumn",
    "SchemaTable",
    "SchemaSummary",
    "RunMetadata",
]
