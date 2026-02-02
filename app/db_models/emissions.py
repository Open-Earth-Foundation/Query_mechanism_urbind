from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Date, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class EmissionRecord(Base):
    __tablename__ = "EmissionRecord"
    __table_args__ = (
        UniqueConstraint("cityId", "year", "sectorId", "scope", "ghgType"),
    )

    emission_record_id: Mapped[UUID] = mapped_column(
        "emissionRecordId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    city_id: Mapped[UUID | None] = mapped_column(
        "cityId", PG_UUID(as_uuid=True), ForeignKey("City.cityId"), nullable=True
    )
    year: Mapped[int] = mapped_column("year", Integer, nullable=False)
    sector_id: Mapped[UUID | None] = mapped_column(
        "sectorId", PG_UUID(as_uuid=True), ForeignKey("Sector.sectorId"), nullable=True
    )
    scope: Mapped[str] = mapped_column("scope", String, nullable=False)
    ghg_type: Mapped[str] = mapped_column("ghgType", String, nullable=False)
    value: Mapped[int] = mapped_column("value", Integer, nullable=False)
    unit: Mapped[str] = mapped_column("unit", String, nullable=False)
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )
