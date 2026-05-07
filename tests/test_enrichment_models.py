"""Tests for enrichment Pydantic models and JSON helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.modules.web_researcher.models import (
    AssumptionRecord,
    CityGap,
    EnrichedField,
    EnrichmentBundle,
    EnrichmentMeta,
    EstimateRange,
    FieldClassification,
    FreshnessResult,
    GapManifest,
    NonEstimableRecord,
    WebFinding,
    _AssumptionsEnvelope,
    _GapManifestEnvelope,
)
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)


# ---------------------------------------------------------------------------
# FieldClassification
# ---------------------------------------------------------------------------


class TestFieldClassification:
    def test_valid_estimable(self) -> None:
        fc = FieldClassification(
            field="total_capex",
            classification="estimable_numerical",
            searchable=True,
            rationale="Concrete quantity",
            scope="municipal",
        )
        assert fc.classification == "estimable_numerical"
        assert fc.searchable is True
        assert fc.scope == "municipal"

    def test_valid_derivable(self) -> None:
        fc = FieldClassification(
            field="per_unit_cost",
            classification="derivable_from_ratio",
            searchable=True,
            rationale="Can derive from total/count",
        )
        assert fc.classification == "derivable_from_ratio"

    def test_valid_non_estimable(self) -> None:
        fc = FieldClassification(
            field="operator_name",
            classification="non_estimable",
            searchable=False,
            rationale="Unique to city",
        )
        assert fc.classification == "non_estimable"

    def test_invalid_classification_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FieldClassification(
                field="x",
                classification="unknown_type",  # type: ignore[arg-type]
                searchable=True,
                rationale="test",
            )

    def test_invalid_scope_rejected(self) -> None:
        """Field scopes are limited to writer-safe aggregation buckets."""
        with pytest.raises(ValidationError):
            FieldClassification(
                field="x",
                classification="estimable_numerical",
                searchable=True,
                rationale="test",
                scope="regional",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# CityGap
# ---------------------------------------------------------------------------


class TestCityGap:
    def test_valid_city_gap(self) -> None:
        cg = CityGap(
            city="Dresden",
            blank_fields=["total_capex", "vehicle_count"],
            stale_flags=["timeline"],
            bundled_fields=["fleet_capex"],
            search_priority="high",
        )
        assert cg.city == "Dresden"
        assert len(cg.blank_fields) == 2
        assert cg.bundled_fields == ["fleet_capex"]
        assert cg.search_priority == "high"

    def test_invalid_priority_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CityGap(
                city="X",
                blank_fields=[],
                stale_flags=[],
                search_priority="critical",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# GapManifest
# ---------------------------------------------------------------------------


class TestGapManifest:
    def test_empty_manifest(self) -> None:
        gm = GapManifest(query_fields=[], city_gaps=[], non_estimable_fields=[])
        assert len(gm.query_fields) == 0
        assert len(gm.city_gaps) == 0

    def test_manifest_roundtrip(self) -> None:
        gm = GapManifest(
            query_fields=[
                FieldClassification(
                    field="capex",
                    classification="estimable_numerical",
                    searchable=True,
                    rationale="R",
                )
            ],
            city_gaps=[
                CityGap(
                    city="Berlin",
                    blank_fields=["capex"],
                    stale_flags=[],
                    search_priority="high",
                )
            ],
            non_estimable_fields=["operator_name"],
        )
        dumped = gm.model_dump(mode="json")
        restored = GapManifest.model_validate(dumped)
        assert restored.query_fields[0].field == "capex"
        assert restored.city_gaps[0].city == "Berlin"


# ---------------------------------------------------------------------------
# WebFinding
# ---------------------------------------------------------------------------


class TestWebFinding:
    def test_valid_web_finding(self) -> None:
        wf = WebFinding(
            city="Dresden",
            field="total_capex",
            value=50_000_000,
            unit="EUR",
            source_url="https://example.com/report",
            source_type="government_report",
            source_date="2024-06",
            extraction_confidence=0.85,
        )
        assert wf.value == 50_000_000
        assert wf.extraction_confidence == 0.85

    def test_nullable_value(self) -> None:
        wf = WebFinding(
            city="X",
            field="f",
            value=None,
            source_url="https://example.com",
            source_type="blog",
            extraction_confidence=0.5,
        )
        assert wf.value is None


# ---------------------------------------------------------------------------
# FreshnessResult
# ---------------------------------------------------------------------------


class TestFreshnessResult:
    def test_valid_classifications(self) -> None:
        for cls in ("consistent", "superseded", "uncertain"):
            fr = FreshnessResult(
                city="X",
                field="f",
                ccc_value="100",
                web_value="120",
                classification=cls,  # type: ignore[arg-type]
                reason="test",
            )
            assert fr.classification == cls


# ---------------------------------------------------------------------------
# EnrichedField
# ---------------------------------------------------------------------------


class TestEnrichedField:
    def test_defaults(self) -> None:
        ef = EnrichedField(city="X", field="f", status="still_missing")
        assert ef.source == "none"
        assert ef.provenance == {}
        assert ef.freshness_flag is None

    def test_all_statuses(self) -> None:
        for status in (
            "resolved",
            "bundled_only",
            "partially_resolved",
            "still_missing",
        ):
            ef = EnrichedField(city="X", field="f", status=status)  # type: ignore[arg-type]
            assert ef.status == status


# ---------------------------------------------------------------------------
# AssumptionRecord + EstimateRange
# ---------------------------------------------------------------------------


class TestAssumptionRecord:
    def test_valid_assumption(self) -> None:
        ar = AssumptionRecord(
            city="Dresden",
            field_name="total_capex",
            gap_description="Missing total CAPEX for Dresden",
            method_used="peer_city_proxy",
            estimate=EstimateRange(low=40_000_000, mid=50_000_000, high=60_000_000),
            confidence="MEDIUM",
            reference_data="Munich: 55M, Hamburg: 48M",
            rationale="Based on similar German cities",
            basis="Peer city comparison",
        )
        assert ar.is_replaceable is True
        assert ar.confidence == "MEDIUM"
        assert ar.estimate.mid == 50_000_000

    def test_string_ranges_allowed(self) -> None:
        er = EstimateRange(low="40M", mid="50M", high="60M")
        assert er.mid == "50M"

    def test_removed_lookup_method_rejected(self) -> None:
        """Removed source-library lookup is not a valid estimation method."""
        removed_method = "structured" + "_lookup"
        with pytest.raises(ValidationError):
            AssumptionRecord(
                city="Dresden",
                field_name="total_capex",
                gap_description="Missing total CAPEX for Dresden",
                method_used=removed_method,  # type: ignore[arg-type]
                estimate=EstimateRange(low=40_000_000, mid=50_000_000, high=60_000_000),
                confidence="MEDIUM",
                reference_data="Structured city lookup",
                rationale="Removed source-library behavior",
                basis="Structured lookup",
            )


# ---------------------------------------------------------------------------
# NonEstimableRecord
# ---------------------------------------------------------------------------


class TestNonEstimableRecord:
    def test_valid_record(self) -> None:
        nr = NonEstimableRecord(
            city="Dresden",
            field_name="operator_name",
            gap_description="Missing operator name",
            explanation="Unique to city",
            recommendation="Contact city directly",
        )
        assert nr.status == "NON_ESTIMABLE"


# ---------------------------------------------------------------------------
# EnrichmentBundle + EnrichmentMeta
# ---------------------------------------------------------------------------


class TestEnrichmentBundle:
    def test_full_bundle_roundtrip(self) -> None:
        meta = EnrichmentMeta(
            created_at=datetime.now(timezone.utc),
            gap_analyst_model="test-model",
            assumptions_estimator_model="test-model",
            total_gaps=5,
            estimable_count=3,
            non_estimable_count=2,
            elapsed_seconds=12.5,
        )
        bundle = EnrichmentBundle(
            gap_manifest=GapManifest(
                query_fields=[], city_gaps=[], non_estimable_fields=[]
            ),
            meta=meta,
        )
        dumped = bundle.model_dump(mode="json")
        assert isinstance(dumped["meta"]["created_at"], str)
        restored = EnrichmentBundle.model_validate(dumped)
        assert restored.meta.total_gaps == 5
        assert restored.assumptions == []
        assert restored.web_findings == []


# ---------------------------------------------------------------------------
# Envelope models (parsing)
# ---------------------------------------------------------------------------


class TestEnvelopes:
    def test_gap_manifest_envelope_empty(self) -> None:
        env = _GapManifestEnvelope.model_validate({})
        assert env.query_fields == []
        assert env.city_gaps == []
        assert env.non_estimable_fields == []

    def test_gap_manifest_envelope_with_data(self) -> None:
        env = _GapManifestEnvelope.model_validate({
            "query_fields": [
                {
                    "field": "capex",
                    "classification": "estimable_numerical",
                    "searchable": True,
                    "rationale": "R",
                }
            ],
            "city_gaps": [
                {
                    "city": "Berlin",
                    "blank_fields": ["capex"],
                    "stale_flags": [],
                    "search_priority": "high",
                }
            ],
            "non_estimable_fields": ["operator"],
        })
        assert len(env.query_fields) == 1
        assert env.non_estimable_fields == ["operator"]

    def test_assumptions_envelope_empty(self) -> None:
        env = _AssumptionsEnvelope.model_validate({})
        assert env.assumptions == []
        assert env.non_estimable == []


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


class TestExtractMessageText:
    def test_string_input(self) -> None:
        assert extract_message_text("  hello  ") == "hello"

    def test_list_input_with_text_attrs(self) -> None:
        class FakePart:
            def __init__(self, text: str) -> None:
                self.text = text

        parts = [FakePart("hello "), FakePart("world")]
        assert extract_message_text(parts) == "hello world"

    def test_other_input(self) -> None:
        assert extract_message_text(42) == "42"


class TestExtractJsonCandidate:
    def test_empty_returns_empty_object(self) -> None:
        assert extract_json_candidate("") == "{}"
        assert extract_json_candidate("   ") == "{}"

    def test_fenced_json(self) -> None:
        raw = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert extract_json_candidate(raw) == '{"key": "value"}'

    def test_bare_json_object(self) -> None:
        raw = 'Here is the result: {"items": [1, 2, 3]} done.'
        assert extract_json_candidate(raw) == '{"items": [1, 2, 3]}'

    def test_bare_json_array(self) -> None:
        raw = "Result: [1, 2, 3] end"
        assert extract_json_candidate(raw) == "[1, 2, 3]"

    def test_raw_fallback(self) -> None:
        raw = "just plain text"
        assert extract_json_candidate(raw) == "just plain text"

    def test_fenced_takes_priority_over_braces(self) -> None:
        raw = '{"outer": 1}\n```json\n{"inner": 2}\n```'
        assert extract_json_candidate(raw) == '{"inner": 2}'
