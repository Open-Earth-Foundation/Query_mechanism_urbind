"""Bundesnetzagentur Ladesäulenregister lookups.

Reads ``data/lookups/bnetza_chargers.parquet`` (built by the
``ingest.bnetza_etl`` handler) and exposes typed aggregate queries.

The parquet schema matches the canonical fields defined in
``backend.modules.sources.handlers.bnetza_etl``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

BNETZA_LOOKUP_ID = "bnetza_chargers"
DEFAULT_PARQUET_PATH = Path("data/lookups/bnetza_chargers.parquet")
SOURCE_NAME = "Bundesnetzagentur Ladesäulenregister"


# ---------------------------------------------------------------------------
# City normalization (German + English aliases)
# ---------------------------------------------------------------------------
_UMLAUT_MAP = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})

# English / international names mapped to the canonical German form.
_CITY_ALIASES: dict[str, str] = {
    "munich": "muenchen",
    "cologne": "koeln",
    "vienna": "wien",
    "nuremberg": "nuernberg",
    "hanover": "hannover",
    "brunswick": "braunschweig",
    "frankfort": "frankfurt",
    "treves": "trier",
}


def normalize_de_city(value: str) -> str:
    """Lowercase + transliterate umlauts + strip non-alphanumerics."""
    if not value:
        return ""
    cleaned = value.strip().casefold().translate(_UMLAUT_MAP)
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    return cleaned


def expand_city_query(value: str) -> set[str]:
    """Return all forms a Bnetza row's ``ort_normalized`` might take for `value`."""
    keys: set[str] = set()
    primary = normalize_de_city(value)
    if primary:
        keys.add(primary)
    if primary in _CITY_ALIASES:
        keys.add(normalize_de_city(_CITY_ALIASES[primary]))
    # Also try the inverse: if the user types the German name, no alias step needed.
    return keys


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChargerStats:
    city_query: str
    matched_normalized: list[str]
    station_count: int
    port_count: int
    kw_total: float
    ac_station_count: int
    dc_station_count: int
    asof: date | None
    top_operators: list[tuple[str, int]]
    source: str = SOURCE_NAME

    @property
    def is_empty(self) -> bool:
        return self.station_count == 0


# ---------------------------------------------------------------------------
# Parquet loader (cached per process)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_parquet(path_str: str) -> pl.LazyFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Bnetza parquet not found at {path}. Run "
            f"`uv run python -m backend.scripts.sources_ingest bnetza_chargers` first."
        )
    return pl.scan_parquet(path)


def _resolve_path(parquet_path: Path | None) -> Path:
    return parquet_path or DEFAULT_PARQUET_PATH


# ---------------------------------------------------------------------------
# Public lookups
# ---------------------------------------------------------------------------
def chargers_in_city(
    city: str,
    *,
    art: Literal["AC", "DC"] | None = None,
    asof: date | None = None,
    parquet_path: Path | None = None,
    top_operators_n: int = 5,
) -> ChargerStats:
    """Aggregate charger statistics for a city.

    Args:
        city: City name (German or English alias accepted).
        art: Optional filter on charger type. ``"AC"`` selects normal
            charging stations (Normalladeeinrichtung); ``"DC"`` selects
            fast charging stations (Schnellladeeinrichtung).
        asof: Optional cutoff: include only stations commissioned on or
            before this date.  When None, include all rows.
        parquet_path: Override the parquet location (mostly for tests).
        top_operators_n: Cap on returned ``top_operators`` length.
    """
    keys = expand_city_query(city)
    if not keys:
        return ChargerStats(
            city_query=city,
            matched_normalized=[],
            station_count=0,
            port_count=0,
            kw_total=0.0,
            ac_station_count=0,
            dc_station_count=0,
            asof=asof,
            top_operators=[],
        )

    lf = _load_parquet(str(_resolve_path(parquet_path).resolve()))

    lf = lf.filter(pl.col("ort_normalized").is_in(list(keys)))
    if art is not None:
        lf = lf.filter(pl.col("art") == art)
    if asof is not None:
        lf = lf.filter(
            pl.col("inbetriebnahme").is_null()
            | (pl.col("inbetriebnahme") <= asof)
        )

    df = lf.collect()

    if df.is_empty():
        return ChargerStats(
            city_query=city,
            matched_normalized=sorted(keys),
            station_count=0,
            port_count=0,
            kw_total=0.0,
            ac_station_count=0,
            dc_station_count=0,
            asof=asof,
            top_operators=[],
        )

    aggregates = df.select(
        pl.len().alias("station_count"),
        pl.col("count_ports").sum().alias("port_count"),
        pl.col("nennleistung_kw").sum().alias("kw_total"),
        (pl.col("art") == "AC").sum().alias("ac_station_count"),
        (pl.col("art") == "DC").sum().alias("dc_station_count"),
    ).row(0, named=True)

    operator_counts = (
        df.group_by("betreiber")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_operators_n)
    )
    top_operators = [
        (str(row["betreiber"] or ""), int(row["n"]))
        for row in operator_counts.iter_rows(named=True)
        if row["betreiber"]
    ]

    return ChargerStats(
        city_query=city,
        matched_normalized=sorted(keys),
        station_count=int(aggregates["station_count"]),
        port_count=int(aggregates["port_count"] or 0),
        kw_total=float(aggregates["kw_total"] or 0.0),
        ac_station_count=int(aggregates["ac_station_count"] or 0),
        dc_station_count=int(aggregates["dc_station_count"] or 0),
        asof=asof,
        top_operators=top_operators,
    )


__all__ = [
    "BNETZA_LOOKUP_ID",
    "ChargerStats",
    "DEFAULT_PARQUET_PATH",
    "SOURCE_NAME",
    "chargers_in_city",
    "expand_city_query",
    "normalize_de_city",
]
