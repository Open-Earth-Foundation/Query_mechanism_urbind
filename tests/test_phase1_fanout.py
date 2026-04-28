"""Tests for Phase 1 fan-out and the structured-lookup dispatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
import yaml

from backend.modules.sources.manifest import Manifest, load_manifest
from backend.modules.web_researcher.data_lookups import (
    find_matching_structured_lookups,
)
from backend.modules.web_researcher.models import (
    BenchmarkExcerptRecord,
    FieldClassification,
    FieldDecomposition,
)
from backend.modules.web_researcher.phase1_fanout import (
    merge_phase1_into_bundle,
    run_phase1_fanout,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _decomposition_with_charger_fields() -> FieldDecomposition:
    return FieldDecomposition(
        query_fields=[
            FieldClassification(
                field="public_dc_charger_count",
                classification="estimable_numerical",
                searchable=True,
                rationale="Reported by national registries.",
            ),
            FieldClassification(
                field="public_ac_charger_count",
                classification="estimable_numerical",
                searchable=True,
                rationale="Reported by national registries.",
            ),
            FieldClassification(
                field="residential_onstreet_charging",
                classification="non_estimable",
                searchable=False,
                rationale="local",
            ),
        ],
        non_estimable_fields=["residential_onstreet_charging"],
    )


def _bundle_with_cities(cities: list[str]) -> dict[str, Any]:
    return {"markdown": {"selected_city_names": cities, "excerpts": []}}


def _bnetza_manifest_yaml(parquet_path: Path) -> dict:
    return {
        "version": 1,
        "sources": [
            {
                "id": "urbind_additional",
                "name": "URBIND",
                "provider": "github",
                "repo": "Open-Earth-Foundation/urbind-additional-documentation",
                "pinned_commit": "abcdef",
                "ingestions": [
                    {
                        "id": "bnetza_chargers",
                        "kind": "structured_lookup",
                        "inputs": {
                            "paths": ["foo.xlsx"],
                            "formats": ["xlsx"],
                        },
                        "output": {"path": str(parquet_path)},
                        "handler": "ingest.bnetza_etl",
                        "consumed_by": "web_researcher.data_lookups.bnetza",
                        "coverage": {
                            "country": "DE",
                            "fields": [
                                "public_ac_charger_count",
                                "public_dc_charger_count",
                                "public_charger_count",
                                "charger_kw_total",
                            ],
                        },
                    }
                ],
            }
        ],
    }


def _write_bnetza_fixture_parquet(path: Path) -> None:
    from datetime import date

    df = pl.DataFrame(
        {
            "ladeeinrichtungs_id": [1, 2, 3, 4],
            "betreiber": ["A", "A", "B", "C"],
            "status": ["In Betrieb"] * 4,
            "art": ["AC", "AC", "DC", "DC"],
            "art_raw": ["x"] * 4,
            "count_ports": [2, 2, 4, 4],
            "nennleistung_kw": [22.0, 22.0, 150.0, 300.0],
            "inbetriebnahme": [date(2023, 1, 1)] * 4,
            "plz": ["01067"] * 4,
            "ort": ["Dresden", "Dresden", "Dresden", "München"],
            "ort_normalized": ["dresden", "dresden", "dresden", "muenchen"],
            "kreis": ["Dresden"] * 3 + ["München"],
            "bundesland": ["Sachsen"] * 3 + ["Bayern"],
            "lat": [51.0, 51.0, 51.0, 48.0],
            "lon": [13.7, 13.7, 13.7, 11.5],
        }
    )
    df.write_parquet(path)


@pytest.fixture
def manifest_with_bnetza(tmp_path: Path) -> tuple[Manifest, Path]:
    parquet_path = tmp_path / "bnetza.parquet"
    _write_bnetza_fixture_parquet(parquet_path)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_bnetza_manifest_yaml(parquet_path)), encoding="utf-8"
    )
    return load_manifest(manifest_path), parquet_path


# Patch the bnetza module so chargers_in_city reads our fixture parquet
# regardless of the lazy lru_cache.
@pytest.fixture(autouse=True)
def _bnetza_cache_clear() -> None:
    from backend.modules.web_researcher.data_lookups import bnetza

    bnetza._load_parquet.cache_clear()
    yield
    bnetza._load_parquet.cache_clear()


# ---------------------------------------------------------------------------
# find_matching_structured_lookups
# ---------------------------------------------------------------------------


def test_find_matching_structured_lookups_returns_per_field_records(
    manifest_with_bnetza: tuple[Manifest, Path],
) -> None:
    manifest, parquet_path = manifest_with_bnetza
    decomposition = _decomposition_with_charger_fields()

    with patch(
        "backend.modules.web_researcher.data_lookups.bnetza.DEFAULT_PARQUET_PATH",
        parquet_path,
    ):
        results = find_matching_structured_lookups(
            decomposition, ["Dresden", "Munich"], manifest
        )

    fields_by_city: dict[str, dict[str, Any]] = {}
    for r in results:
        fields_by_city.setdefault(r.city, {})[r.field] = r

    # Dresden has 2 AC + 1 DC = 3 stations.
    dresden = fields_by_city["Dresden"]
    assert dresden["public_ac_charger_count"].value == 2
    assert dresden["public_dc_charger_count"].value == 1
    assert dresden["public_dc_charger_count"].source_id == "urbind_additional"
    assert dresden["public_dc_charger_count"].ingestion_id == "bnetza_chargers"
    assert dresden["public_dc_charger_count"].unit == "stations"

    # Munich has 0 AC + 1 DC = 1 station total. AC is 0 (not None), so it's still emitted.
    munich = fields_by_city["Munich"]
    assert munich["public_dc_charger_count"].value == 1


def test_find_matching_structured_lookups_skips_when_no_field_match(
    manifest_with_bnetza: tuple[Manifest, Path],
) -> None:
    manifest, _ = manifest_with_bnetza
    decomposition = FieldDecomposition(
        query_fields=[
            FieldClassification(
                field="ev_share",  # not in bnetza coverage
                classification="estimable_numerical",
                searchable=True,
                rationale="x",
            )
        ],
        non_estimable_fields=[],
    )
    results = find_matching_structured_lookups(decomposition, ["Dresden"], manifest)
    assert results == []


def test_find_matching_structured_lookups_skips_non_estimable_only(
    manifest_with_bnetza: tuple[Manifest, Path],
) -> None:
    manifest, _ = manifest_with_bnetza
    decomposition = FieldDecomposition(
        query_fields=[
            FieldClassification(
                field="public_dc_charger_count",
                classification="non_estimable",
                searchable=False,
                rationale="x",
            )
        ],
        non_estimable_fields=["public_dc_charger_count"],
    )
    results = find_matching_structured_lookups(decomposition, ["Dresden"], manifest)
    assert results == []


# ---------------------------------------------------------------------------
# run_phase1_fanout
# ---------------------------------------------------------------------------


def test_phase1_fanout_runs_structured_and_benchmarks(
    manifest_with_bnetza: tuple[Manifest, Path],
) -> None:
    manifest, parquet_path = manifest_with_bnetza
    decomposition = _decomposition_with_charger_fields()
    bundle = _bundle_with_cities(["Dresden"])

    fake_excerpt = type(
        "FakeExcerpt",
        (),
        dict(
            chunk_id="c1",
            source_id="urbind_additional",
            ingestion_id="urbind_benchmarks",
            source_path="tier-2/x.pdf",
            doc_slug="x",
            tier="tier-2-national-datasets",
            heading_path="Section A",
            block_type="paragraph",
            raw_text="benchmark text",
            distance=0.4,
            chunk_index=2,
        ),
    )()

    def _fake_benchmark_retrieve(queries: list[str], **kwargs: Any) -> list:
        assert queries  # we should pass at least one
        return [fake_excerpt]

    with patch(
        "backend.modules.web_researcher.data_lookups.bnetza.DEFAULT_PARQUET_PATH",
        parquet_path,
    ):
        artefacts = run_phase1_fanout(
            decomposition,
            bundle,
            manifest=manifest,
            benchmark_persist_path=parquet_path.parent,  # exists; satisfies guard
            benchmark_retrieve=_fake_benchmark_retrieve,
        )

    assert artefacts.queried_cities == ["Dresden"]
    assert "public_dc_charger_count" in artefacts.queried_fields
    assert any(r.field == "public_dc_charger_count" for r in artefacts.structured_lookups)
    assert artefacts.benchmark_excerpts
    assert isinstance(artefacts.benchmark_excerpts[0], BenchmarkExcerptRecord)
    assert artefacts.benchmark_excerpts[0].source_id == "urbind_additional"


def test_phase1_fanout_without_manifest_skips_lookups(tmp_path: Path) -> None:
    decomposition = _decomposition_with_charger_fields()
    bundle = _bundle_with_cities(["Dresden"])
    artefacts = run_phase1_fanout(
        decomposition,
        bundle,
        manifest=None,
        manifest_path=tmp_path / "missing.yaml",
        benchmark_persist_path=tmp_path / "no-chroma",
    )
    assert artefacts.structured_lookups == []
    assert artefacts.benchmark_excerpts == []
    assert artefacts.queried_cities == ["Dresden"]


def test_phase1_fanout_uses_cities_override(
    manifest_with_bnetza: tuple[Manifest, Path],
) -> None:
    manifest, parquet_path = manifest_with_bnetza
    decomposition = _decomposition_with_charger_fields()
    bundle = _bundle_with_cities(["Aachen"])  # would yield zero rows

    with patch(
        "backend.modules.web_researcher.data_lookups.bnetza.DEFAULT_PARQUET_PATH",
        parquet_path,
    ):
        artefacts = run_phase1_fanout(
            decomposition,
            bundle,
            manifest=manifest,
            cities_override=["Dresden"],
            benchmark_persist_path=parquet_path.parent.parent / "no-chroma-here",
        )

    assert artefacts.queried_cities == ["Dresden"]
    assert artefacts.structured_lookups  # bnetza yields Dresden rows


def test_merge_phase1_into_bundle_attaches_phase1_payload() -> None:
    decomposition = _decomposition_with_charger_fields()
    bundle = _bundle_with_cities(["Dresden"])
    artefacts = run_phase1_fanout(
        decomposition,
        bundle,
        manifest=None,
        benchmark_persist_path=Path("/no/such/path"),
    )
    new_bundle = merge_phase1_into_bundle(bundle, artefacts)

    assert "phase1" in new_bundle
    assert new_bundle["phase1"]["queried_cities"] == ["Dresden"]
    # Original bundle is untouched.
    assert "phase1" not in bundle
