"""Tests for context merger: cancelled handling and resolved-but-null invariant."""

from __future__ import annotations

from backend.modules.web_researcher.context_merger import (
    _ensure_resolved_has_value,
    compute_field_statuses,
)
from backend.modules.web_researcher.models import (
    CityGap,
    EnrichedField,
    FieldClassification,
    FreshnessResult,
    GapManifest,
    WebFinding,
)


def _make_gap_manifest(city: str = "Klagenfurt", fields: list[str] | None = None):
    fields = fields or ["ev_bus_count"]
    return GapManifest(
        query_fields=[
            FieldClassification(
                field=f, classification="estimable_numerical",
                searchable=True, rationale="test",
            )
            for f in fields
        ],
        city_gaps=[
            CityGap(
                city=city,
                blank_fields=fields,
                stale_flags=[],
                search_priority="high",
            )
        ],
        non_estimable_fields=[],
    )


def _make_web_finding(city: str = "Klagenfurt", field: str = "ev_bus_count", **kw):
    defaults = {
        "city": city,
        "field": field,
        "value": 50,
        "unit": "vehicles",
        "source_url": "https://example.com",
        "source_type": "government_report",
        "source_date": "2024",
        "extraction_confidence": 0.9,
    }
    defaults.update(kw)
    return WebFinding(**defaults)


def _make_freshness(city: str = "Klagenfurt", field: str = "ev_bus_count", **kw):
    defaults = {
        "city": city,
        "field": field,
        "ccc_value": "30",
        "web_value": "50",
        "classification": "consistent",
        "reason": "test",
        "web_source_url": "https://example.com",
    }
    defaults.update(kw)
    return FreshnessResult(**defaults)


class TestCancelledClassification:
    def test_cancelled_produces_still_missing(self):
        gm = _make_gap_manifest()
        wf = _make_web_finding(value=0, unit="cancelled_programme", source_type="cancellation_notice")
        fr = _make_freshness(classification="cancelled", web_value="0")

        result = compute_field_statuses(gm, [wf], [fr], {})

        assert len(result) == 1
        ef = result[0]
        assert ef.status == "still_missing"
        assert ef.freshness_flag == "cancelled"
        assert ef.source == "none"

    def test_cancelled_with_no_ccc_value(self):
        gm = _make_gap_manifest()
        wf = _make_web_finding(value=0)
        fr = _make_freshness(classification="cancelled", ccc_value=None, web_value="0")

        result = compute_field_statuses(gm, [wf], [fr], {})

        assert len(result) == 1
        assert result[0].status == "still_missing"
        assert result[0].freshness_flag == "cancelled"


class TestResolvedRequiresValue:
    def test_ensure_resolved_has_value_downgrades(self):
        ef = EnrichedField(
            city="Munich", field="taxi_count",
            status="resolved", value=None, source="web",
        )
        fixed = _ensure_resolved_has_value(ef)
        assert fixed.status == "partially_resolved"

    def test_ensure_resolved_has_value_keeps_valid(self):
        ef = EnrichedField(
            city="Munich", field="taxi_count",
            status="resolved", value=100, source="web",
        )
        fixed = _ensure_resolved_has_value(ef)
        assert fixed.status == "resolved"

    def test_ensure_non_resolved_untouched(self):
        ef = EnrichedField(
            city="Munich", field="taxi_count",
            status="still_missing", value=None, source="none",
        )
        fixed = _ensure_resolved_has_value(ef)
        assert fixed.status == "still_missing"

    def test_superseded_with_null_value_downgraded(self):
        """If web finding has null value but freshness says superseded, should downgrade."""
        gm = _make_gap_manifest(city="Munich", fields=["taxi_count"])
        wf = _make_web_finding(city="Munich", field="taxi_count", value=None)
        fr = _make_freshness(
            city="Munich", field="taxi_count",
            classification="superseded",
        )

        result = compute_field_statuses(gm, [wf], [fr], {})
        # The web finding has value=None, which gets discarded by null check
        # in validate_finding, but context merger still sees it if validator
        # wasn't run. The invariant guard catches it.
        assert len(result) == 1
        # value is None → resolved downgraded to partially_resolved
        assert result[0].status == "partially_resolved"

    def test_web_only_null_value_downgraded(self):
        """Web finding with null value and no freshness → resolved downgraded."""
        gm = _make_gap_manifest(city="Munich", fields=["taxi_count"])
        wf = _make_web_finding(city="Munich", field="taxi_count", value=None)

        result = compute_field_statuses(gm, [wf], [], {})
        assert len(result) == 1
        assert result[0].status == "partially_resolved"


class TestNormalFlowUnchanged:
    def test_superseded_with_value_stays_resolved(self):
        gm = _make_gap_manifest(city="Munich", fields=["taxi_count"])
        wf = _make_web_finding(city="Munich", field="taxi_count", value=200)
        fr = _make_freshness(
            city="Munich", field="taxi_count",
            classification="superseded",
        )

        result = compute_field_statuses(gm, [wf], [fr], {})
        assert len(result) == 1
        assert result[0].status == "resolved"
        assert result[0].value == 200
        assert result[0].source == "web"

    def test_consistent_keeps_ccc_value(self):
        gm = _make_gap_manifest(city="Munich", fields=["taxi_count"])
        wf = _make_web_finding(city="Munich", field="taxi_count", value=200)
        fr = _make_freshness(
            city="Munich", field="taxi_count",
            classification="consistent", ccc_value="195",
        )

        result = compute_field_statuses(gm, [wf], [fr], {})
        assert len(result) == 1
        assert result[0].status == "resolved"
        assert result[0].value == "195"
        assert result[0].source == "ccc"

    def test_no_findings_still_missing(self):
        gm = _make_gap_manifest(city="Munich", fields=["taxi_count"])
        result = compute_field_statuses(gm, [], [], {})
        assert len(result) == 1
        assert result[0].status == "still_missing"

    def test_bundled_field_routes_to_bundled_only_with_scope(self) -> None:
        """Bundled CCC values stay estimable while preserving field scope."""
        gm = GapManifest(
            query_fields=[
                FieldClassification(
                    field="fleet_capex",
                    classification="estimable_numerical",
                    searchable=True,
                    rationale="Requested as a disaggregated fleet capital cost.",
                    scope="public_transport",
                )
            ],
            city_gaps=[
                CityGap(
                    city="Dresden",
                    blank_fields=[],
                    stale_flags=[],
                    bundled_fields=["fleet_capex"],
                    search_priority="medium",
                )
            ],
            non_estimable_fields=[],
        )

        result = compute_field_statuses(gm, [], [], {})

        assert len(result) == 1
        ef = result[0]
        assert ef.status == "bundled_only"
        assert ef.source == "ccc"
        assert ef.freshness_flag == "bundled_only"
        assert ef.scope == "public_transport"
