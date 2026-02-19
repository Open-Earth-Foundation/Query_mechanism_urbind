from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class ClimateCityContract(Base):
    __tablename__ = "ClimateCityContract"
    __table_args__ = (UniqueConstraint("cityId"),)

    climate_city_contract_id: Mapped[UUID] = mapped_column(
        "climateCityContractId",
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    city_id: Mapped[UUID | None] = mapped_column(
        "cityId", PG_UUID(as_uuid=True), ForeignKey("City.cityId"), nullable=True
    )
    contract_date: Mapped[datetime] = mapped_column("contractDate", DateTime, nullable=False)
    title: Mapped[str] = mapped_column("title", String, nullable=False)
    version: Mapped[str | None] = mapped_column("version", String, nullable=True)
    language: Mapped[str | None] = mapped_column("language", String, nullable=True)
    document_url: Mapped[str | None] = mapped_column("documentUrl", String, nullable=True)
    notes: Mapped[str | None] = mapped_column("notes", Text, nullable=True)
    misc: Mapped[dict[str, Any] | None] = mapped_column(
        "misc", JSONB, nullable=True
    )
