from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class City(Base):
    __tablename__ = "City"

    city_id: Mapped[UUID] = mapped_column(
        "cityId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    city_name: Mapped[str] = mapped_column("cityName", String, nullable=False)
    country: Mapped[str] = mapped_column("country", String, nullable=False)
    locode: Mapped[str | None] = mapped_column("locode", String, nullable=True)
    area_km2: Mapped[Decimal | None] = mapped_column("areaKm2", Numeric, nullable=True)
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )


class CityAnnualStats(Base):
    __tablename__ = "CityAnnualStats"
    __table_args__ = (
        UniqueConstraint("cityId", "year"),
    )

    stat_id: Mapped[UUID] = mapped_column(
        "statId", PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    city_id: Mapped[UUID | None] = mapped_column(
        "cityId", PG_UUID(as_uuid=True), ForeignKey("City.cityId"), nullable=True
    )
    year: Mapped[int] = mapped_column("year", Integer, nullable=False)
    population: Mapped[int | None] = mapped_column("population", Integer, nullable=True)
    population_density: Mapped[Decimal | None] = mapped_column(
        "populationDensity", Numeric, nullable=True
    )
    gdp_per_capita: Mapped[Decimal | None] = mapped_column(
        "gdpPerCapita", Numeric, nullable=True
    )
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )
