from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class Initiative(Base):
    __tablename__ = "Initiative"

    initiative_id: Mapped[UUID] = mapped_column(
        "initiativeId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    city_id: Mapped[UUID | None] = mapped_column(
        "cityId", PG_UUID(as_uuid=True), ForeignKey("City.cityId"), nullable=True
    )
    title: Mapped[str] = mapped_column("title", String, nullable=False)
    description: Mapped[str | None] = mapped_column("description", Text, nullable=True)
    start_year: Mapped[int | None] = mapped_column("startYear", Integer, nullable=True)
    end_year: Mapped[int | None] = mapped_column("endYear", Integer, nullable=True)
    total_estimated_cost: Mapped[int | None] = mapped_column(
        "totalEstimatedCost", BigInteger, nullable=True
    )
    currency: Mapped[str | None] = mapped_column("currency", String, nullable=True)
    status: Mapped[str | None] = mapped_column("status", String, nullable=True)
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )


class Stakeholder(Base):
    __tablename__ = "Stakeholder"

    stakeholder_id: Mapped[UUID] = mapped_column(
        "stakeholderId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column("name", String, nullable=False)
    type: Mapped[str | None] = mapped_column("type", String, nullable=True)
    description: Mapped[str | None] = mapped_column("description", Text, nullable=True)
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )


class InitiativeStakeholder(Base):
    __tablename__ = "InitiativeStakeholder"
    __table_args__ = (UniqueConstraint("initiativeId", "stakeholderId"),)

    initiative_stakeholder_id: Mapped[UUID] = mapped_column(
        "initiativeStakeholderId",
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    initiative_id: Mapped[UUID | None] = mapped_column(
        "initiativeId",
        PG_UUID(as_uuid=True),
        ForeignKey("Initiative.initiativeId"),
        nullable=True,
    )
    stakeholder_id: Mapped[UUID | None] = mapped_column(
        "stakeholderId",
        PG_UUID(as_uuid=True),
        ForeignKey("Stakeholder.stakeholderId"),
        nullable=True,
    )
    role: Mapped[str | None] = mapped_column("role", String, nullable=True)
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )


class InitiativeIndicator(Base):
    __tablename__ = "InitiativeIndicator"
    __table_args__ = (UniqueConstraint("initiativeId", "indicatorId"),)

    initiative_indicator_id: Mapped[UUID] = mapped_column(
        "initiativeIndicatorId",
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    initiative_id: Mapped[UUID | None] = mapped_column(
        "initiativeId",
        PG_UUID(as_uuid=True),
        ForeignKey("Initiative.initiativeId"),
        nullable=True,
    )
    indicator_id: Mapped[UUID | None] = mapped_column(
        "indicatorId",
        PG_UUID(as_uuid=True),
        ForeignKey("Indicator.indicatorId"),
        nullable=True,
    )
    contribution_type: Mapped[str] = mapped_column(
        "contributionType", String, nullable=False
    )
    expected_change: Mapped[Decimal | None] = mapped_column(
        "expectedChange", Numeric, nullable=True
    )
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )
