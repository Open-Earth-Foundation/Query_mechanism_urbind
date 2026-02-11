from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class TefCategory(Base):
    __tablename__ = "TefCategory"

    tef_id: Mapped[UUID] = mapped_column(
        "tefId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    parent_id: Mapped[UUID | None] = mapped_column(
        "parentId",
        PG_UUID(as_uuid=True),
        ForeignKey("TefCategory.tefId"),
        nullable=True,
    )
    code: Mapped[str] = mapped_column("code", String, nullable=False)
    name: Mapped[str] = mapped_column("name", String, nullable=False)
    description: Mapped[str | None] = mapped_column("description", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )


class InitiativeTef(Base):
    __tablename__ = "InitiativeTef"
    __table_args__ = (UniqueConstraint("initiativeId", "tefId"),)

    initiative_tef_id: Mapped[UUID] = mapped_column(
        "initiativeTefId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    initiative_id: Mapped[UUID | None] = mapped_column(
        "initiativeId",
        PG_UUID(as_uuid=True),
        ForeignKey("Initiative.initiativeId"),
        nullable=True,
    )
    tef_id: Mapped[UUID | None] = mapped_column(
        "tefId", PG_UUID(as_uuid=True), ForeignKey("TefCategory.tefId"), nullable=True
    )
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )
