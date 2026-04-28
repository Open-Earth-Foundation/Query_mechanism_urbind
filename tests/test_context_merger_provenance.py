"""Tests for the provenance pass — source_id / source_tier / source_name flow."""

from __future__ import annotations

from unittest.mock import patch

from backend.modules.web_researcher.context_merger import compute_field_statuses
from backend.modules.web_researcher.models import (
    CityGap,
    EnrichedField,
    FieldClassification,
    FreshnessResult,
    GapManifest,
    WebFinding,
)


def _gap_manifest_one_blank() -> GapManifest:
    return GapManifest(
        query_fields=[
            FieldClassification(
                field="public_dc_charger_count",
                classification="estimable_numerical",
                searchable=True,
                rationale="x",
            ),
        ],
        city_gaps=[
            CityGap(
                city="Dresden",
                blank_fields=["public_dc_charger_count"],
                stale_flags=[],
                search_priority="high",
            )
        ],
        non_estimable_fields=[],
    )


def _tier1_finding() -> WebFinding:
    return WebFinding(
        city="Dresden",
        field="public_dc_charger_count",
        value=159,
        unit="stations",
        source_url="https://bundesnetzagentur.de/dresden",
        source_type="government_report",
        extraction_confidence=0.9,
        source_id="bnetza_ladekarte",
        source_tier="tier1",
    )


def _open_finding() -> WebFinding:
    return WebFinding(
        city="Dresden",
        field="public_dc_charger_count",
        value=159,
        unit="stations",
        source_url="https://random-blog.example.com/dresden",
        source_type="press_release",
        extraction_confidence=0.85,
        source_id=None,
        source_tier="open",
    )


def _patch_allowlist_with_bnetza_name():
    """Patch the allowlist loader inside context_merger to return a known source."""
    from backend.modules.web_researcher.tier1_web import (
        CoverageHint,
        Tier1WebAllowlist,
        Tier1WebSource,
    )

    allowlist = Tier1WebAllowlist(
        sources=[
            Tier1WebSource(
                id="bnetza_ladekarte",
                name="Bundesnetzagentur Ladesäulenkarte",
                domain="bundesnetzagentur.de",
                coverage=CoverageHint(),
            )
        ]
    )
    return patch(
        "backend.modules.web_researcher.tier1_web.load_tier1_web_allowlist",
        return_value=allowlist,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tier1_finding_carries_source_id_and_resolved_name() -> None:
    manifest = _gap_manifest_one_blank()
    web_findings = [_tier1_finding()]
    freshness_results: list[FreshnessResult] = []

    with _patch_allowlist_with_bnetza_name():
        enriched = compute_field_statuses(
            manifest, web_findings, freshness_results, context_bundle={}
        )

    assert len(enriched) == 1
    ef = enriched[0]
    assert ef.status == "resolved"
    assert ef.value == 159
    assert ef.source_id == "bnetza_ladekarte"
    assert ef.source_tier == "tier1"
    assert ef.provenance.get("source_name") == "Bundesnetzagentur Ladesäulenkarte"
    assert ef.provenance.get("source_url") == "https://bundesnetzagentur.de/dresden"


def test_open_finding_keeps_open_tier_and_no_source_name() -> None:
    manifest = _gap_manifest_one_blank()
    web_findings = [_open_finding()]
    freshness_results: list[FreshnessResult] = []

    with _patch_allowlist_with_bnetza_name():
        enriched = compute_field_statuses(
            manifest, web_findings, freshness_results, context_bundle={}
        )

    ef = enriched[0]
    assert ef.source_tier == "open"
    assert ef.source_id is None
    # Without an id, no source_name should be attached.
    assert "source_name" not in ef.provenance


def test_provenance_pass_tolerates_missing_allowlist() -> None:
    """If the allowlist file isn't built yet, the pass still works (no crash, no name)."""
    manifest = _gap_manifest_one_blank()
    web_findings = [_tier1_finding()]

    with patch(
        "backend.modules.web_researcher.tier1_web.load_tier1_web_allowlist",
        side_effect=FileNotFoundError("no allowlist yet"),
    ):
        enriched = compute_field_statuses(manifest, web_findings, [], context_bundle={})

    ef = enriched[0]
    assert ef.source_id == "bnetza_ladekarte"
    assert "source_name" not in ef.provenance


def test_freshness_superseded_carries_tier1_provenance() -> None:
    manifest = _gap_manifest_one_blank()
    wf = _tier1_finding()
    fr = FreshnessResult(
        city="Dresden",
        field="public_dc_charger_count",
        ccc_value="100",
        web_value="159",
        classification="superseded",
        reason="newer authoritative source",
    )

    with _patch_allowlist_with_bnetza_name():
        enriched = compute_field_statuses(
            manifest, [wf], [fr], context_bundle={}
        )

    ef = enriched[0]
    assert ef.freshness_flag == "superseded"
    assert ef.source_id == "bnetza_ladekarte"
    assert ef.source_tier == "tier1"
    assert ef.provenance.get("source_name") == "Bundesnetzagentur Ladesäulenkarte"
