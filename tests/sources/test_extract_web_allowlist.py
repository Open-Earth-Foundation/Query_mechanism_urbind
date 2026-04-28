"""Tests for the tier-1 web allowlist extraction handler and loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from backend.modules.sources.handlers import IngestionContext
from backend.modules.sources.handlers.extract_web_allowlist import (
    _CURATED,
    run_extract_web_allowlist,
)
from backend.modules.sources.manifest import (
    IngestionConfig,
    IngestionInputs,
    IngestionOutput,
    SourceConfig,
)
from backend.modules.web_researcher.tier1_web import (
    Tier1WebAllowlist,
    load_tier1_web_allowlist,
)


def _make_upstream(tmp_path: Path) -> Path:
    root = tmp_path / "upstream"
    root.mkdir()
    (root / "README.md").write_text(
        "# README\n\n"
        "Sources Not Included:\n"
        "- **Bundesnetzagentur Ladesäulenkarte** - https://www.bundesnetzagentur.de/foo\n"
        "- **NetZeroCities App** - https://netzerocities.app/\n"
        "- **EAFO** - https://alternative-fuels-observatory.ec.europa.eu/\n",
        encoding="utf-8",
    )
    (root / "tier-5-eu-programmes").mkdir()
    (root / "tier-5-eu-programmes" / "web-platforms-index.md").write_text(
        "# Index\n\n"
        "- URL: https://netzerocities.eu/\n"
        "- URL: https://urban-mobility-observatory.transport.ec.europa.eu/...\n"
        "- URL: https://mobidata-bw.de/\n"
        "- URL: https://www.opendata.sachsen.de/\n"
        "- URL: https://www.govdata.de/\n",
        encoding="utf-8",
    )
    return root


def _make_context(tmp_path: Path) -> IngestionContext:
    upstream = _make_upstream(tmp_path)
    project_root = tmp_path / "project"
    source = SourceConfig(
        id="urbind_additional",
        name="URBIND",
        provider="github",
        repo="Open-Earth-Foundation/urbind-additional-documentation",
        pinned_commit="abcdef",
    )
    ingestion = IngestionConfig(
        id="urbind_web_allowlist",
        kind="web_allowlist",
        inputs=IngestionInputs(
            paths=["README.md", "tier-5-eu-programmes/web-platforms-index.md"],
            formats=["markdown"],
        ),
        output=IngestionOutput(path="backend/data/tier1_web_sources.yaml"),
        handler="ingest.extract_web_allowlist",
        consumed_by="web_researcher.tier1_probe",
    )
    return IngestionContext(
        source=source,
        ingestion=ingestion,
        upstream_root=upstream,
        project_root=project_root,
        resolved_commit="abcdef",
    )


def test_extract_writes_yaml_with_all_curated_sources(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    state = run_extract_web_allowlist(ctx)

    output = ctx.project_root / "backend" / "data" / "tier1_web_sources.yaml"
    assert output.exists()
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    ids = [src["id"] for src in payload["sources"]]
    assert len(ids) == len(_CURATED)
    assert "bnetza_ladekarte" in ids
    assert "eafo" in ids
    assert "netzerocities_public" in ids

    dump = state.model_dump()
    assert dump["extracted_entries"] == len(_CURATED)
    assert dump["output_path"].endswith("tier1_web_sources.yaml")


def test_loaded_allowlist_validates(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    run_extract_web_allowlist(ctx)
    output = ctx.project_root / "backend" / "data" / "tier1_web_sources.yaml"

    allowlist = load_tier1_web_allowlist(output)
    assert isinstance(allowlist, Tier1WebAllowlist)
    assert allowlist.version == 1
    assert allowlist.sources

    bnetza = next(s for s in allowlist.sources if s.id == "bnetza_ladekarte")
    assert bnetza.coverage.countries == ["DE"]
    assert "public_dc_charger_count" in bnetza.coverage.fields

    nzc_public = next(s for s in allowlist.sources if s.id == "netzerocities_public")
    assert "/app/" in nzc_public.excluded_paths


def test_drift_warning_when_curated_domain_missing_from_upstream(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ctx = _make_context(tmp_path)
    # Replace the upstream README with one that drops Bnetza references.
    (ctx.upstream_root / "README.md").write_text(
        "# README\n\nNo Bnetza, no EAFO, just text.\n", encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        run_extract_web_allowlist(ctx)
    warning_text = " ".join(record.message for record in caplog.records)
    assert "bundesnetzagentur.de" in warning_text
    assert "alternative-fuels-observatory.ec.europa.eu" in warning_text


def test_handler_is_registered() -> None:
    from backend.modules.sources.handlers import REGISTRY

    assert "ingest.extract_web_allowlist" in REGISTRY


def test_matching_filters_by_country(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    run_extract_web_allowlist(ctx)
    output = ctx.project_root / "backend" / "data" / "tier1_web_sources.yaml"
    allowlist = load_tier1_web_allowlist(output)

    de_only = allowlist.matching(country="DE")
    de_ids = {s.id for s in de_only}
    assert "bnetza_ladekarte" in de_ids
    assert "standorttool" in de_ids
    # Sources with no country restriction (e.g. NetZeroCities public) also pass through.
    assert "netzerocities_public" in de_ids
    # IEA covers global; should match too.
    assert "iea_ev_explorer" in de_ids


def test_matching_filters_by_field(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    run_extract_web_allowlist(ctx)
    allowlist = load_tier1_web_allowlist(
        ctx.project_root / "backend" / "data" / "tier1_web_sources.yaml"
    )
    chargers = allowlist.matching(fields=["public_dc_charger_count"])
    ids = {s.id for s in chargers}
    assert "bnetza_ladekarte" in ids
    assert "eafo" in ids
    # Mobilithek covers transport_data scope but doesn't list charger fields,
    # so it should still pass (broad — empty fields means "any").
    assert "mobilithek" in ids


def test_matching_excludes_when_field_does_not_match(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    run_extract_web_allowlist(ctx)
    allowlist = load_tier1_web_allowlist(
        ctx.project_root / "backend" / "data" / "tier1_web_sources.yaml"
    )
    # Looking for ev_share.  Bnetza has explicit fields, none of which match.
    results = allowlist.matching(fields=["ev_share"])
    ids = {s.id for s in results}
    assert "bnetza_ladekarte" not in ids
    assert "eafo" in ids  # listed
    assert "iea_ev_explorer" in ids  # listed


def test_matching_filters_by_city(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    run_extract_web_allowlist(ctx)
    allowlist = load_tier1_web_allowlist(
        ctx.project_root / "backend" / "data" / "tier1_web_sources.yaml"
    )

    munster = allowlist.matching(cities=["munster"])
    munster_ids = {s.id for s in munster}
    assert "klimadashboard_munster" in munster_ids
    # Sources without a city restriction (broad) are also included.
    assert "bnetza_ladekarte" in munster_ids


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tier1_web_allowlist(tmp_path / "nope.yaml")
