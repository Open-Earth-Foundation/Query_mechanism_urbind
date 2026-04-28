"""Tests for the Bundesnetzagentur lookup module.

Uses a tiny synthetic parquet fixture so we don't depend on the real
27 MB upstream xlsx during unit tests.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from backend.modules.web_researcher.data_lookups import bnetza
from backend.modules.web_researcher.data_lookups.bnetza import (
    chargers_in_city,
    expand_city_query,
    normalize_de_city,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    bnetza._load_parquet.cache_clear()
    yield
    bnetza._load_parquet.cache_clear()


def _write_fixture(path: Path) -> Path:
    df = pl.DataFrame(
        {
            "ladeeinrichtungs_id": [1, 2, 3, 4, 5, 6, 7],
            "betreiber": ["A", "A", "B", "C", "C", "C", "B"],
            "status": ["In Betrieb"] * 7,
            "art": ["AC", "AC", "DC", "AC", "DC", "DC", "AC"],
            "art_raw": [
                "Normalladeeinrichtung",
                "Normalladeeinrichtung",
                "Schnellladeeinrichtung",
                "Normalladeeinrichtung",
                "Schnellladeeinrichtung",
                "Schnellladeeinrichtung",
                "Normalladeeinrichtung",
            ],
            "count_ports": [2, 2, 4, 2, 4, 4, 2],
            "nennleistung_kw": [22.0, 22.0, 150.0, 11.0, 300.0, 150.0, 22.0],
            "inbetriebnahme": [
                date(2020, 1, 1),
                date(2022, 6, 1),
                date(2024, 3, 15),
                date(2019, 5, 5),
                date(2025, 1, 10),
                date(2023, 9, 2),
                date(2018, 1, 1),
            ],
            "plz": ["01067", "01067", "01069", "80331", "80331", "80799", "10115"],
            "ort": [
                "Dresden",
                "Dresden",
                "Dresden",
                "München",
                "München",
                "München",
                "Berlin",
            ],
            "ort_normalized": [
                "dresden",
                "dresden",
                "dresden",
                "muenchen",
                "muenchen",
                "muenchen",
                "berlin",
            ],
            "kreis": ["Dresden"] * 3 + ["München"] * 3 + ["Berlin"],
            "bundesland": ["Sachsen"] * 3 + ["Bayern"] * 3 + ["Berlin"],
            "lat": [51.05, 51.05, 51.06, 48.13, 48.13, 48.16, 52.52],
            "lon": [13.74, 13.74, 13.75, 11.57, 11.57, 11.58, 13.40],
        }
    )
    df.write_parquet(path)
    return path


@pytest.fixture
def fixture_parquet(tmp_path: Path) -> Path:
    return _write_fixture(tmp_path / "b.parquet")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_de_city_strips_diacritics_and_punct() -> None:
    assert normalize_de_city("München") == "muenchen"
    assert normalize_de_city("Köln-Mitte") == "koelnmitte"
    assert normalize_de_city("  Dresden ") == "dresden"
    assert normalize_de_city("") == ""


def test_expand_city_query_includes_alias() -> None:
    assert "muenchen" in expand_city_query("munich")
    assert "muenchen" in expand_city_query("München")
    assert "muenchen" in expand_city_query("Munich")


# ---------------------------------------------------------------------------
# chargers_in_city
# ---------------------------------------------------------------------------


def test_chargers_in_city_dresden_total(fixture_parquet: Path) -> None:
    stats = chargers_in_city("Dresden", parquet_path=fixture_parquet)
    assert stats.station_count == 3
    assert stats.port_count == 8
    assert stats.kw_total == pytest.approx(22.0 + 22.0 + 150.0)
    assert stats.ac_station_count == 2
    assert stats.dc_station_count == 1
    assert stats.matched_normalized == ["dresden"]
    assert stats.source == bnetza.SOURCE_NAME


def test_chargers_in_city_munich_via_english_alias(fixture_parquet: Path) -> None:
    stats = chargers_in_city("Munich", parquet_path=fixture_parquet)
    assert stats.station_count == 3
    assert stats.dc_station_count == 2
    assert stats.ac_station_count == 1
    # Both forms in the matched set since the alias expands.
    assert "muenchen" in stats.matched_normalized


def test_chargers_in_city_filter_by_art(fixture_parquet: Path) -> None:
    ac = chargers_in_city("Dresden", art="AC", parquet_path=fixture_parquet)
    dc = chargers_in_city("Dresden", art="DC", parquet_path=fixture_parquet)
    assert ac.station_count == 2
    assert dc.station_count == 1
    assert ac.dc_station_count == 0
    assert dc.ac_station_count == 0


def test_chargers_in_city_asof_excludes_later(fixture_parquet: Path) -> None:
    # Dresden has commissionings in 2020, 2022, 2024.  asof=2023 keeps two.
    stats = chargers_in_city(
        "Dresden", asof=date(2023, 1, 1), parquet_path=fixture_parquet
    )
    assert stats.station_count == 2
    assert stats.asof == date(2023, 1, 1)


def test_chargers_in_city_unknown_returns_empty(fixture_parquet: Path) -> None:
    stats = chargers_in_city("Atlantis", parquet_path=fixture_parquet)
    assert stats.is_empty
    assert stats.station_count == 0
    assert stats.kw_total == 0.0
    assert stats.top_operators == []


def test_chargers_in_city_top_operators(fixture_parquet: Path) -> None:
    stats = chargers_in_city("Munich", parquet_path=fixture_parquet)
    operators = dict(stats.top_operators)
    assert operators.get("C") == 3


def test_chargers_in_city_missing_parquet_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        chargers_in_city("Dresden", parquet_path=tmp_path / "missing.parquet")


def test_handler_is_registered() -> None:
    from backend.modules.sources.handlers import REGISTRY

    assert "ingest.bnetza_etl" in REGISTRY
