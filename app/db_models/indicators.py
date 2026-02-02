from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class Indicator(Base):
    __tablename__ = "Indicator"

    indicator_id: Mapped[UUID] = mapped_column(
        "indicatorId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    city_id: Mapped[UUID | None] = mapped_column(
        "cityId", PG_UUID(as_uuid=True), ForeignKey("City.cityId"), nullable=True
    )
    sector_id: Mapped[UUID | None] = mapped_column(
        "sectorId", PG_UUID(as_uuid=True), ForeignKey("Sector.sectorId"), nullable=True
    )
    name: Mapped[str] = mapped_column("name", String, nullable=False)
    description: Mapped[str | None] = mapped_column("description", Text, nullable=True)
    unit: Mapped[str] = mapped_column("unit", String, nullable=False)
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )


class IndicatorValue(Base):
    __tablename__ = "IndicatorValue"
    __table_args__ = (UniqueConstraint("indicatorId", "year"),)

    indicator_value_id: Mapped[UUID] = mapped_column(
        "indicatorValueId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    indicator_id: Mapped[UUID | None] = mapped_column(
        "indicatorId",
        PG_UUID(as_uuid=True),
        ForeignKey("Indicator.indicatorId"),
        nullable=True,
    )
    year: Mapped[int] = mapped_column("year", Integer, nullable=False)
    value: Mapped[Decimal] = mapped_column("value", Numeric, nullable=False)
    value_type: Mapped[str] = mapped_column("valueType", String, nullable=False)
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )


class CityTarget(Base):
    __tablename__ = "CityTarget"

    city_target_id: Mapped[UUID] = mapped_column(
        "cityTargetId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    city_id: Mapped[UUID | None] = mapped_column(
        "cityId", PG_UUID(as_uuid=True), ForeignKey("City.cityId"), nullable=True
    )
    indicator_id: Mapped[UUID | None] = mapped_column(
        "indicatorId",
        PG_UUID(as_uuid=True),
        ForeignKey("Indicator.indicatorId"),
        nullable=True,
    )
    description: Mapped[str] = mapped_column("description", Text, nullable=False)
    target_year: Mapped[date] = mapped_column("targetYear", Date, nullable=False)
    target_value: Mapped[Decimal] = mapped_column(
        "targetValue", Numeric, nullable=False
    )
    baseline_year: Mapped[date | None] = mapped_column(
        "baselineYear", Date, nullable=True
    )
    baseline_value: Mapped[Decimal | None] = mapped_column(
        "baselineValue", Numeric, nullable=True
    )
    status: Mapped[str] = mapped_column("status", String, nullable=False)
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )
