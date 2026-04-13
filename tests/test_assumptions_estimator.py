"""Unit tests for assumptions estimator: peer reference table and prompt inclusion."""

from __future__ import annotations

from backend.modules.web_researcher.assumptions_estimator import (
    _MIN_AUTHORITATIVE_CONFIDENCE,
    _MIN_PEER_ANCHORS_METHOD_B,
    _MIN_PEER_ANCHORS_METHOD_C,
    _MIN_PEER_CONFIDENCE,
    _build_peer_reference_table,
    _build_system_prompt,
    _build_user_prompt,
    _check_anchor_sufficiency,
    _format_comparative_data,
    _format_national_benchmarks,
)
from backend.modules.web_researcher.models import (
    EnrichedField,
    FieldClassification,
    GapManifest,
    WebFinding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ef(
    city: str, field: str, status: str, value=None,
    source: str = "none", provenance: dict | None = None,
) -> EnrichedField:
    return EnrichedField(
        city=city, field=field, status=status, value=value,
        source=source, provenance=provenance or {},
    )


def _minimal_gap_manifest(fields: list[str]) -> GapManifest:
    return GapManifest(
        query_fields=[
            FieldClassification(
                field=f, classification="estimable_numerical",
                searchable=True, rationale="test",
            )
            for f in fields
        ],
        city_gaps=[],
        non_estimable_fields=[],
    )


# ---------------------------------------------------------------------------
# _build_peer_reference_table
# ---------------------------------------------------------------------------

class TestBuildPeerReferenceTable:
    def test_extracts_resolved_peers_for_estimable_fields(self):
        all_enriched = [
            _ef("Munich", "vehicle_count", "resolved", 280, "ccc"),
            _ef("Mannheim", "vehicle_count", "resolved", 749, "ccc"),
            _ef("Aachen", "vehicle_count", "still_missing"),
            _ef("Munich", "bus_count", "resolved", 90, "web",
                 provenance={"extraction_confidence": 0.9}),
            _ef("Aachen", "bus_count", "still_missing"),
        ]
        estimable = [
            _ef("Aachen", "vehicle_count", "still_missing"),
            _ef("Aachen", "bus_count", "still_missing"),
        ]
        table = _build_peer_reference_table(all_enriched, estimable)

        assert "vehicle_count" in table
        assert len(table["vehicle_count"]) == 2
        assert table["vehicle_count"][0]["city"] == "Munich"
        assert table["vehicle_count"][0]["value"] == 280
        assert table["vehicle_count"][0]["confidence"] == 1.0  # CCC default
        assert table["vehicle_count"][1]["city"] == "Mannheim"

        assert "bus_count" in table
        assert len(table["bus_count"]) == 1
        assert table["bus_count"][0]["city"] == "Munich"
        assert table["bus_count"][0]["confidence"] == 0.9

    def test_ignores_fields_not_needing_estimation(self):
        all_enriched = [
            _ef("Munich", "vehicle_count", "resolved", 280, "ccc"),
            _ef("Munich", "population", "resolved", 1500000, "ccc"),
        ]
        estimable = [_ef("Aachen", "vehicle_count", "still_missing")]
        table = _build_peer_reference_table(all_enriched, estimable)

        assert "vehicle_count" in table
        assert "population" not in table

    def test_ignores_none_values(self):
        all_enriched = [
            _ef("Munich", "vehicle_count", "resolved", None, "ccc"),
        ]
        estimable = [_ef("Aachen", "vehicle_count", "still_missing")]
        table = _build_peer_reference_table(all_enriched, estimable)

        assert table == {}

    def test_ignores_non_resolved_status(self):
        all_enriched = [
            _ef("Munich", "vehicle_count", "partially_resolved", 100, "web",
                 provenance={"extraction_confidence": 0.9}),
            _ef("Mannheim", "vehicle_count", "still_missing"),
        ]
        estimable = [_ef("Aachen", "vehicle_count", "still_missing")]
        table = _build_peer_reference_table(all_enriched, estimable)

        assert table == {}

    def test_empty_inputs(self):
        assert _build_peer_reference_table([], []) == {}
        assert _build_peer_reference_table(
            [_ef("Munich", "x", "resolved", 1, "ccc")], []
        ) == {}

    def test_preserves_source_and_confidence(self):
        all_enriched = [
            _ef("Klagenfurt", "vehicle_count", "resolved", 35, "web",
                 provenance={"extraction_confidence": 0.85}),
        ]
        estimable = [_ef("Aachen", "vehicle_count", "still_missing")]
        table = _build_peer_reference_table(all_enriched, estimable)

        assert table["vehicle_count"][0]["source"] == "web"
        assert table["vehicle_count"][0]["confidence"] == 0.85

    def test_filters_low_confidence_peers(self):
        all_enriched = [
            _ef("Munich", "vehicle_count", "resolved", 280, "ccc"),  # conf 1.0
            _ef("Klagenfurt", "vehicle_count", "resolved", 35, "web"),  # conf 0.5 (default)
            _ef("Mannheim", "vehicle_count", "resolved", 100, "web",
                 provenance={"extraction_confidence": 0.3}),  # below threshold
        ]
        estimable = [_ef("Aachen", "vehicle_count", "still_missing")]
        table = _build_peer_reference_table(all_enriched, estimable)

        # Only Munich (ccc, conf 1.0) should pass; web defaults to 0.5 < 0.7
        assert len(table["vehicle_count"]) == 1
        assert table["vehicle_count"][0]["city"] == "Munich"


# ---------------------------------------------------------------------------
# _build_user_prompt — peer reference section
# ---------------------------------------------------------------------------

class TestUserPromptPeerReference:
    def test_peer_reference_appears_in_prompt(self):
        peer_ref = {
            "vehicle_count": [
                {"city": "Munich", "value": 280, "source": "ccc"},
                {"city": "Mannheim", "value": 749, "source": "ccc"},
            ],
        }
        prompt = _build_user_prompt(
            question="How many vehicles?",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["vehicle_count"]),
            estimable_fields=[_ef("Aachen", "vehicle_count", "still_missing")],
            pass_name="generate",
            peer_reference=peer_ref,
        )

        assert "Peer reference data" in prompt
        assert "vehicle_count" in prompt
        assert "Munich: 280 (ccc)" in prompt
        assert "Mannheim: 749 (ccc)" in prompt
        assert "Method B (peer_city_proxy)" in prompt

    def test_no_peer_reference_when_empty(self):
        prompt = _build_user_prompt(
            question="How many vehicles?",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["vehicle_count"]),
            estimable_fields=[_ef("Aachen", "vehicle_count", "still_missing")],
            pass_name="generate",
            peer_reference={},
        )

        assert "Peer reference data" not in prompt

    def test_no_peer_reference_when_none(self):
        prompt = _build_user_prompt(
            question="How many vehicles?",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["vehicle_count"]),
            estimable_fields=[_ef("Aachen", "vehicle_count", "still_missing")],
            pass_name="generate",
            peer_reference=None,
        )

        assert "Peer reference data" not in prompt

    def test_peer_reference_before_context_bundle(self):
        peer_ref = {
            "vehicle_count": [
                {"city": "Munich", "value": 280, "source": "ccc"},
            ],
        }
        prompt = _build_user_prompt(
            question="Q",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["vehicle_count"]),
            estimable_fields=[_ef("Aachen", "vehicle_count", "still_missing")],
            pass_name="generate",
            peer_reference=peer_ref,
        )

        peer_pos = prompt.index("Peer reference data")
        context_pos = prompt.index("Data summary")
        assert peer_pos < context_pos


# ---------------------------------------------------------------------------
# _build_system_prompt — Method B guidance
# ---------------------------------------------------------------------------

class TestSystemPromptMethodB:
    def test_method_b_has_ratio_scaling_guidance(self):
        prompt = _build_system_prompt("generate")
        assert "Ratio-scale from peers" in prompt
        assert "PREFER this over Method C" in prompt

    def test_method_b_has_example(self):
        prompt = _build_system_prompt("generate")
        assert "Munich (pop 1.5M)" in prompt
        assert "280 × (250K/1.5M)" in prompt

    def test_must_attempt_method_b_rule(self):
        prompt = _build_system_prompt("generate")
        assert "MUST attempt Method B before falling" in prompt
        assert "Cite specific peer city values" in prompt

    def test_critique_pass_includes_method_b_guidance(self):
        prompt = _build_system_prompt("critique")
        assert "PREFER this over Method C" in prompt
        assert "CRITIQUE PASS" in prompt


# ---------------------------------------------------------------------------
# _check_anchor_sufficiency
# ---------------------------------------------------------------------------

class TestCheckAnchorSufficiency:
    def test_sufficient_with_enough_peers(self):
        peer_ref = {
            "vehicle_count": [
                {"city": "Munich", "value": 280, "source": "ccc", "confidence": 1.0},
                {"city": "Hamburg", "value": 350, "source": "ccc", "confidence": 1.0},
            ],
        }
        fields = [_ef("Aachen", "vehicle_count", "still_missing")]
        sufficient, insufficient = _check_anchor_sufficiency(peer_ref, fields)

        assert len(sufficient) == 1
        assert len(insufficient) == 0

    def test_insufficient_with_no_peers(self):
        fields = [_ef("Aachen", "vehicle_count", "still_missing")]
        sufficient, insufficient = _check_anchor_sufficiency({}, fields)

        assert len(sufficient) == 0
        assert len(insufficient) == 1
        assert "Insufficient peer anchors" in insufficient[0].gap_description

    def test_single_authoritative_peer_with_high_confidence_passes(self):
        peer_ref = {
            "vehicle_count": [
                {"city": "Munich", "value": 280, "source": "ccc", "confidence": 0.9},
            ],
        }
        fields = [_ef("Aachen", "vehicle_count", "still_missing")]
        sufficient, insufficient = _check_anchor_sufficiency(peer_ref, fields)

        assert len(sufficient) == 1
        assert len(insufficient) == 0

    def test_single_authoritative_peer_with_low_confidence_fails(self):
        peer_ref = {
            "vehicle_count": [
                {"city": "Munich", "value": 280, "source": "ccc", "confidence": 0.5},
            ],
        }
        fields = [_ef("Aachen", "vehicle_count", "still_missing")]
        sufficient, insufficient = _check_anchor_sufficiency(peer_ref, fields)

        assert len(sufficient) == 0
        assert len(insufficient) == 1

    def test_single_web_peer_fails(self):
        peer_ref = {
            "vehicle_count": [
                {"city": "Munich", "value": 280, "source": "web", "confidence": 0.9},
            ],
        }
        fields = [_ef("Aachen", "vehicle_count", "still_missing")]
        sufficient, insufficient = _check_anchor_sufficiency(peer_ref, fields)

        assert len(sufficient) == 0
        assert len(insufficient) == 1

    def test_explanation_mentions_method_thresholds(self):
        fields = [_ef("Aachen", "vehicle_count", "still_missing")]
        _, insufficient = _check_anchor_sufficiency({}, fields)

        explanation = insufficient[0].explanation
        assert str(_MIN_PEER_ANCHORS_METHOD_B) in explanation
        assert str(_MIN_PEER_ANCHORS_METHOD_C) in explanation
        assert str(_MIN_AUTHORITATIVE_CONFIDENCE) in explanation


# ---------------------------------------------------------------------------
# _format_national_benchmarks
# ---------------------------------------------------------------------------

def _wf(
    city: str, field: str, value, unit: str | None = None,
    source_type: str = "government_report", source_date: str | None = None,
    extraction_confidence: float = 0.9,
) -> WebFinding:
    return WebFinding(
        city=city, field=field, value=value, unit=unit,
        source_url="https://example.com", source_type=source_type,
        source_date=source_date, extraction_confidence=extraction_confidence,
    )


class TestFormatNationalBenchmarks:
    def test_returns_empty_string_for_none(self):
        assert _format_national_benchmarks(None) == ""

    def test_returns_empty_string_for_empty_list(self):
        assert _format_national_benchmarks([]) == ""

    def test_formats_single_benchmark(self):
        findings = [
            _wf("Germany", "bus_procurement_unit_cost", 450000, "EUR/unit",
                 source_type="government_report", source_date="2024"),
        ]
        result = _format_national_benchmarks(findings)

        assert "National/regional benchmarks" in result
        assert "bus_procurement_unit_cost" in result
        assert "Germany avg = 450000 EUR/unit" in result
        assert "government_report" in result
        assert "2024" in result
        assert "conf: 0.9" in result

    def test_formats_multiple_benchmarks(self):
        findings = [
            _wf("Germany", "bus_cost", 450000, "EUR/unit"),
            _wf("Germany", "charger_cost", 35000, "EUR/unit",
                 source_type="industry_report", source_date="2023",
                 extraction_confidence=0.8),
        ]
        result = _format_national_benchmarks(findings)

        assert "bus_cost" in result
        assert "charger_cost" in result
        assert "conf: 0.8" in result

    def test_includes_usage_guidance(self):
        findings = [_wf("Germany", "bus_cost", 450000, "EUR/unit")]
        result = _format_national_benchmarks(findings)

        assert "USE for Method A when" in result
        assert "DO NOT use when" in result
        assert "qualitative or city-specific" in result
        assert "Peer reference data" in result

    def test_handles_finding_without_unit(self):
        findings = [_wf("Germany", "fleet_size_avg", 200)]
        result = _format_national_benchmarks(findings)

        assert "Germany avg = 200" in result

    def test_handles_finding_without_date(self):
        findings = [_wf("Germany", "bus_cost", 450000, "EUR/unit", source_date=None)]
        result = _format_national_benchmarks(findings)

        assert "government_report)" in result
        # No date should mean no comma-date suffix
        assert "government_report, " not in result


# ---------------------------------------------------------------------------
# _build_user_prompt — national benchmarks section
# ---------------------------------------------------------------------------

class TestUserPromptNationalBenchmarks:
    def test_national_benchmarks_appear_in_prompt(self):
        benchmarks = [
            _wf("Germany", "bus_cost", 450000, "EUR/unit",
                 source_date="2024"),
        ]
        prompt = _build_user_prompt(
            question="How much do buses cost?",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["bus_cost"]),
            estimable_fields=[_ef("Dresden", "bus_cost", "still_missing")],
            pass_name="generate",
            national_benchmarks=benchmarks,
        )

        assert "National/regional benchmarks" in prompt
        assert "bus_cost" in prompt
        assert "Germany avg" in prompt

    def test_no_section_when_benchmarks_none(self):
        prompt = _build_user_prompt(
            question="Q",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["bus_cost"]),
            estimable_fields=[_ef("Dresden", "bus_cost", "still_missing")],
            pass_name="generate",
            national_benchmarks=None,
        )

        assert "National/regional benchmarks" not in prompt

    def test_no_section_when_benchmarks_empty(self):
        prompt = _build_user_prompt(
            question="Q",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["bus_cost"]),
            estimable_fields=[_ef("Dresden", "bus_cost", "still_missing")],
            pass_name="generate",
            national_benchmarks=[],
        )

        assert "National/regional benchmarks" not in prompt

    def test_benchmarks_between_peer_reference_and_data_summary(self):
        peer_ref = {
            "bus_cost": [
                {"city": "Munich", "value": 460000, "source": "ccc"},
                {"city": "Hamburg", "value": 440000, "source": "ccc"},
            ],
        }
        benchmarks = [
            _wf("Germany", "bus_cost", 450000, "EUR/unit"),
        ]
        prompt = _build_user_prompt(
            question="Q",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["bus_cost"]),
            estimable_fields=[_ef("Dresden", "bus_cost", "still_missing")],
            pass_name="generate",
            peer_reference=peer_ref,
            national_benchmarks=benchmarks,
        )

        peer_pos = prompt.index("Peer reference data")
        national_pos = prompt.index("National/regional benchmarks")
        data_pos = prompt.index("Data summary")
        assert peer_pos < national_pos < data_pos


# ---------------------------------------------------------------------------
# _build_system_prompt — Method A references benchmarks section
# ---------------------------------------------------------------------------

class TestSystemPromptMethodA:
    def test_method_a_references_benchmarks_section(self):
        prompt = _build_system_prompt("generate")
        assert "National/regional benchmarks" in prompt
        assert "retrieved from real web searches" in prompt


# ---------------------------------------------------------------------------
# _format_comparative_data
# ---------------------------------------------------------------------------

class TestFormatComparativeData:
    def test_returns_empty_string_for_none(self):
        assert _format_comparative_data(None) == ""

    def test_returns_empty_string_for_empty_list(self):
        assert _format_comparative_data([]) == ""

    def test_formats_single_finding(self):
        findings = [
            _wf("cross_country", "ev_bus_count", 12.5, "per 100k inhabitants",
                 source_type="industry_report", source_date="2023"),
        ]
        result = _format_comparative_data(findings)

        assert "Cross-country comparative data" in result
        assert "ev_bus_count" in result
        assert "cross_country = 12.5 per 100k inhabitants" in result
        assert "industry_report" in result
        assert "2023" in result
        assert "conf: 0.9" in result

    def test_formats_multiple_findings(self):
        findings = [
            _wf("cross_country", "ev_bus_count", 12.5, "per 100k",
                 source_type="industry_report", source_date="2023"),
            _wf("cross_country", "charging_capacity_kw", 45, "kW/charger avg",
                 source_type="industry_report", source_date="2024",
                 extraction_confidence=0.8),
        ]
        result = _format_comparative_data(findings)

        assert "ev_bus_count" in result
        assert "charging_capacity_kw" in result
        assert "conf: 0.8" in result

    def test_includes_usage_guidance(self):
        findings = [_wf("cross_country", "ev_bus_count", 12.5, "per 100k")]
        result = _format_comparative_data(findings)

        assert "USE for Method C when" in result
        assert "DO NOT use when" in result
        assert "cross-border comparison" in result
        assert "prefer Method B" in result
        assert "prefer Method A" in result

    def test_handles_finding_without_unit(self):
        findings = [_wf("cross_country", "fleet_ratio", 0.15)]
        result = _format_comparative_data(findings)

        assert "cross_country = 0.15" in result

    def test_handles_finding_without_date(self):
        findings = [_wf("cross_country", "ev_bus_count", 12.5, "per 100k", source_date=None)]
        result = _format_comparative_data(findings)

        assert "government_report)" in result
        assert "government_report, " not in result


# ---------------------------------------------------------------------------
# _build_user_prompt — comparative data section
# ---------------------------------------------------------------------------

class TestUserPromptComparativeData:
    def test_comparative_data_appears_in_prompt(self):
        comparative = [
            _wf("cross_country", "ev_bus_count", 12.5, "per 100k inhabitants",
                 source_type="industry_report", source_date="2023"),
        ]
        prompt = _build_user_prompt(
            question="How many electric buses?",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["ev_bus_count"]),
            estimable_fields=[_ef("Dresden", "ev_bus_count", "still_missing")],
            pass_name="generate",
            comparative_data=comparative,
        )

        assert "Cross-country comparative data" in prompt
        assert "ev_bus_count" in prompt
        assert "cross_country" in prompt

    def test_no_section_when_comparative_none(self):
        prompt = _build_user_prompt(
            question="Q",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["ev_bus_count"]),
            estimable_fields=[_ef("Dresden", "ev_bus_count", "still_missing")],
            pass_name="generate",
            comparative_data=None,
        )

        assert "Cross-country comparative data" not in prompt

    def test_no_section_when_comparative_empty(self):
        prompt = _build_user_prompt(
            question="Q",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["ev_bus_count"]),
            estimable_fields=[_ef("Dresden", "ev_bus_count", "still_missing")],
            pass_name="generate",
            comparative_data=[],
        )

        assert "Cross-country comparative data" not in prompt

    def test_comparative_between_national_and_data_summary(self):
        peer_ref = {
            "ev_bus_count": [
                {"city": "Munich", "value": 100, "source": "ccc"},
                {"city": "Hamburg", "value": 150, "source": "ccc"},
            ],
        }
        national = [
            _wf("Germany", "ev_bus_count", 120, "units"),
        ]
        comparative = [
            _wf("cross_country", "ev_bus_count", 12.5, "per 100k inhabitants"),
        ]
        prompt = _build_user_prompt(
            question="Q",
            context_bundle={"key": "val"},
            gap_manifest=_minimal_gap_manifest(["ev_bus_count"]),
            estimable_fields=[_ef("Dresden", "ev_bus_count", "still_missing")],
            pass_name="generate",
            peer_reference=peer_ref,
            national_benchmarks=national,
            comparative_data=comparative,
        )

        peer_pos = prompt.index("Peer reference data")
        national_pos = prompt.index("National/regional benchmarks")
        comparative_pos = prompt.index("Cross-country comparative data")
        data_pos = prompt.index("Data summary")
        assert peer_pos < national_pos < comparative_pos < data_pos


# ---------------------------------------------------------------------------
# _build_system_prompt — Method C references comparative section
# ---------------------------------------------------------------------------

class TestSystemPromptMethodC:
    def test_method_c_references_comparative_section(self):
        prompt = _build_system_prompt("generate")
        assert "Cross-country comparative data" in prompt
        assert "retrieved from real web searches" in prompt

    def test_method_c_in_critique_pass(self):
        prompt = _build_system_prompt("critique")
        assert "Cross-country comparative data" in prompt
