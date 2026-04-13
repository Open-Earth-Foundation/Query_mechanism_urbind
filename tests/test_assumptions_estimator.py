"""Unit tests for assumptions estimator: peer reference table and prompt inclusion."""

from __future__ import annotations

from backend.modules.web_researcher.assumptions_estimator import (
    _build_peer_reference_table,
    _build_system_prompt,
    _build_user_prompt,
)
from backend.modules.web_researcher.models import (
    EnrichedField,
    FieldClassification,
    GapManifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ef(city: str, field: str, status: str, value=None, source: str = "none") -> EnrichedField:
    return EnrichedField(city=city, field=field, status=status, value=value, source=source)


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
            _ef("Munich", "bus_count", "resolved", 90, "web"),
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
        assert table["vehicle_count"][1]["city"] == "Mannheim"

        assert "bus_count" in table
        assert len(table["bus_count"]) == 1
        assert table["bus_count"][0]["city"] == "Munich"

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
            _ef("Munich", "vehicle_count", "partially_resolved", 100, "web"),
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

    def test_preserves_source_info(self):
        all_enriched = [
            _ef("Klagenfurt", "vehicle_count", "resolved", 35, "web"),
        ]
        estimable = [_ef("Aachen", "vehicle_count", "still_missing")]
        table = _build_peer_reference_table(all_enriched, estimable)

        assert table["vehicle_count"][0]["source"] == "web"


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
