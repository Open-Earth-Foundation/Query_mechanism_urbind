"""Regression tests for the freshness markdown-based CCC extraction.

Covers the bug where _extract_ccc_values() stored keys as (city, "_has_data")
while check_freshness() looked them up as (city, field), effectively disabling
the freshness gate. The current extractor collects per-city markdown excerpts
as prose evidence, and the LLM step is responsible for pulling structured
values out of that evidence.
"""

from __future__ import annotations

from backend.modules.web_researcher.freshness import (
    _extract_ccc_evidence,
    check_freshness,
)
from backend.modules.web_researcher.models import WebFinding


def _make_markdown_bundle(excerpts: list[dict]) -> dict:
    return {"markdown": {"excerpts": excerpts}}


def _wf(city: str = "Dresden", field: str = "capex", **kw) -> WebFinding:
    defaults = {
        "city": city,
        "field": field,
        "value": 50,
        "unit": "EUR_millions",
        "source_url": "https://example.com",
        "source_type": "government_report",
        "source_date": "2025",
        "extraction_confidence": 0.9,
    }
    defaults.update(kw)
    return WebFinding(**defaults)


class TestExtractCccEvidence:
    def test_collects_partial_answer_per_city(self):
        bundle = _make_markdown_bundle([
            {"city_key": "Dresden", "partial_answer": "Dresden allocated 45M EUR to climate capex."},
            {"city_key": "Berlin", "partial_answer": "Berlin reports 120M EUR in capex."},
        ])
        evidence = _extract_ccc_evidence(bundle)
        assert evidence["dresden"] == ["Dresden allocated 45M EUR to climate capex."]
        assert evidence["berlin"] == ["Berlin reports 120M EUR in capex."]

    def test_lowercases_city_key(self):
        bundle = _make_markdown_bundle([
            {"city_key": "DRESDEN", "partial_answer": "hello"},
        ])
        assert list(_extract_ccc_evidence(bundle)) == ["dresden"]

    def test_falls_back_to_quote_when_partial_answer_missing(self):
        bundle = _make_markdown_bundle([
            {"city_key": "Berlin", "quote": "Berlin allocated 120M EUR."},
        ])
        evidence = _extract_ccc_evidence(bundle)
        assert evidence["berlin"] == ["Berlin allocated 120M EUR."]

    def test_skips_empty_city_or_text(self):
        bundle = _make_markdown_bundle([
            {"city_key": "", "partial_answer": "orphan"},
            {"city_key": "Oslo", "partial_answer": "   "},
            {"city_key": "Oslo", "partial_answer": "Oslo evidence"},
        ])
        evidence = _extract_ccc_evidence(bundle)
        assert evidence == {"oslo": ["Oslo evidence"]}

    def test_caps_excerpts_per_city(self):
        excerpts = [
            {"city_key": "Oslo", "partial_answer": f"snippet {i}"}
            for i in range(20)
        ]
        evidence = _extract_ccc_evidence(_make_markdown_bundle(excerpts))
        assert len(evidence["oslo"]) == 6  # _MAX_EXCERPTS_PER_CITY

    def test_truncates_long_excerpt(self):
        long_text = "x" * 5000
        bundle = _make_markdown_bundle([
            {"city_key": "Oslo", "partial_answer": long_text},
        ])
        evidence = _extract_ccc_evidence(bundle)
        assert len(evidence["oslo"][0]) == 800  # _MAX_EXCERPT_CHARS

    def test_handles_missing_markdown_section(self):
        assert _extract_ccc_evidence({}) == {}
        assert _extract_ccc_evidence({"markdown": None}) == {}
        assert _extract_ccc_evidence({"markdown": {"excerpts": None}}) == {}


class TestCheckFreshnessShortCircuits:
    """When there are no overlaps, check_freshness() must skip the LLM call."""

    def test_no_findings_returns_empty(self):
        assert check_freshness([], {}, config=None, api_key="") == []  # type: ignore[arg-type]

    def test_no_markdown_evidence_returns_empty_without_llm(self):
        bundle = _make_markdown_bundle([])
        assert check_freshness([_wf()], bundle, config=None, api_key="") == []  # type: ignore[arg-type]

    def test_other_city_evidence_returns_empty(self):
        """Web finding for Dresden, but markdown only has Berlin → no overlap."""
        bundle = _make_markdown_bundle([
            {"city_key": "Berlin", "partial_answer": "Berlin has 120M EUR."},
        ])
        assert check_freshness([_wf(city="Dresden")], bundle, config=None, api_key="") == []  # type: ignore[arg-type]

    def test_dresden_capex_repro_has_overlap(self):
        """The reviewer's repro: web finding for Dresden/capex + markdown excerpt
        for Dresden must be reachable as overlapping evidence so classification
        can run (short-circuits pre-LLM if no evidence; here evidence exists)."""
        bundle = _make_markdown_bundle([
            {"city_key": "Dresden", "partial_answer": "Dresden has allocated 45M EUR to climate capex."},
        ])
        evidence = _extract_ccc_evidence(bundle)
        assert "dresden" in evidence, "Dresden evidence must be reachable by lowercased city key"
        assert evidence["dresden"], "Must contain at least one non-empty excerpt"
