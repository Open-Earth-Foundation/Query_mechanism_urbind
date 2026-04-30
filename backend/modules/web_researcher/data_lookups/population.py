"""City population lookup, sourced from Eurostat URBAN AUDIT seed parquet.

Returns ``city-proper`` population for a given city key.  The parquet is
produced by ``backend.modules.sources.handlers.urban_audit_population_etl``;
this module is a thin reader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from backend.utils.city_normalization import normalize_city_key

logger = logging.getLogger(__name__)

DEFAULT_PARQUET_PATH = Path("data/lookups/urban_audit_population.parquet")
SOURCE_NAME = "Eurostat URBAN AUDIT"


@dataclass(frozen=True)
class CityPopulation:
    city_query: str
    matched_key: str | None
    population: int | None
    year: int | None
    source: str = SOURCE_NAME

    @property
    def is_empty(self) -> bool:
        return self.population is None


def _load_table(parquet_path: Path | None) -> pl.DataFrame | None:
    path = parquet_path or DEFAULT_PARQUET_PATH
    if not path.exists():
        logger.info("population lookup: parquet missing at %s", path)
        return None
    try:
        return pl.read_parquet(path)
    except Exception:  # noqa: BLE001
        logger.warning("population lookup: failed to read %s", path, exc_info=True)
        return None


def population_for_city(
    city: str,
    *,
    parquet_path: Path | None = None,
) -> CityPopulation:
    """Look up a city's population by normalized key.

    Returns an empty record when the city isn't seeded; callers decide
    whether that's a hard error or a graceful skip.
    """
    key = normalize_city_key(city) or ""
    df = _load_table(parquet_path)
    if df is None or df.is_empty():
        return CityPopulation(city_query=city, matched_key=None, population=None, year=None)

    match = df.filter(pl.col("city_key") == key)
    if match.is_empty():
        return CityPopulation(city_query=city, matched_key=None, population=None, year=None)

    row = match.row(0, named=True)
    pop = row.get("population")
    year = row.get("year")
    return CityPopulation(
        city_query=city,
        matched_key=key,
        population=int(pop) if pop is not None else None,
        year=int(year) if isinstance(year, (int, float)) and year is not None else None,
    )


__all__ = ["CityPopulation", "DEFAULT_PARQUET_PATH", "SOURCE_NAME", "population_for_city"]
