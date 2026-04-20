"""Regression tests for the freshness CCC extraction.

Covers the bug where _extract_ccc_values() stored keys as (city, "_has_data")
while check_freshness() looked them up as (city, field), effectively disabling
the freshness gate.
"""

from __future__ import annotations

from backend.modules.web_researcher.freshness import (
    _extract_ccc_values,
    _find_city_column,
    check_freshness,
)
from backend.modules.web_researcher.models import WebFinding


def _make_sql_bundle(columns: list[str], rows: list[list]) -> dict:
    return {
        "sql": {
            "results": [
                {
                    "query_id": "q1",
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                    "elapsed_ms": 1,
                    "token_count": 0,
                }
            ]
        }
    }


class TestFindCityColumn:
    def test_prefers_city_over_name(self):
        assert _find_city_column(["city", "name", "capex"]) == 0

    def test_falls_back_to_city_key(self):
        assert _find_city_column(["capex", "city_key", "year"]) == 1

    def test_case_insensitive(self):
        assert _find_city_column(["CITY", "capex"]) == 0

    def test_returns_none_when_absent(self):
        assert _find_city_column(["capex", "year"]) is None


class TestExtractCccValues:
    def test_extracts_city_field_value_from_sql(self):
        bundle = _make_sql_bundle(
            ["city", "capex", "ev_bus_count"],
            [["Dresden", 50, 10], ["Berlin", 100, 20]],
        )
        values = _extract_ccc_values(bundle)
        assert values[("dresden", "capex")] == "50"
        assert values[("dresden", "ev_bus_count")] == "10"
        assert values[("berlin", "capex")] == "100"

    def test_skips_null_cells(self):
        bundle = _make_sql_bundle(
            ["city", "capex", "ev_bus_count"],
            [["Dresden", None, 10]],
        )
        values = _extract_ccc_values(bundle)
        assert ("dresden", "capex") not in values
        assert values[("dresden", "ev_bus_count")] == "10"

    def test_skips_empty_city(self):
        bundle = _make_sql_bundle(
            ["city", "capex"],
            [["", 50], ["   ", 60], ["Berlin", 70]],
        )
        values = _extract_ccc_values(bundle)
        assert list(values) == [("berlin", "capex")]

    def test_skips_result_without_city_column(self):
        bundle = _make_sql_bundle(["year", "capex"], [[2024, 50]])
        assert _extract_ccc_values(bundle) == {}

    def test_handles_missing_sql_section(self):
        assert _extract_ccc_values({}) == {}
        assert _extract_ccc_values({"sql": None}) == {}

    def test_lowercases_city_and_field(self):
        bundle = _make_sql_bundle(["City", "CAPEX"], [["DRESDEN", 50]])
        values = _extract_ccc_values(bundle)
        assert values == {("dresden", "capex"): "50"}

    def test_last_write_wins_on_duplicate_key(self):
        bundle = {
            "sql": {
                "results": [
                    {"columns": ["city", "capex"], "rows": [["Berlin", 100]]},
                    {"columns": ["city", "capex"], "rows": [["Berlin", 200]]},
                ]
            }
        }
        values = _extract_ccc_values(bundle)
        assert values[("berlin", "capex")] == "200"


class TestCheckFreshnessShortCircuits:
    """When there are no overlaps, check_freshness() must skip the LLM call."""

    def test_no_findings_returns_empty(self):
        assert check_freshness([], {}, config=None, api_key="") == []  # type: ignore[arg-type]

    def test_no_sql_overlap_returns_empty_without_llm(self):
        # Config/api_key are unused because the short-circuit fires before
        # any LLM client is constructed.
        wf = WebFinding(
            city="Dresden",
            field="capex",
            value=50,
            unit="EUR_millions",
            source_url="https://example.com",
            source_type="government_report",
            source_date="2025",
            extraction_confidence=0.9,
        )
        bundle = _make_sql_bundle(["city", "taxi_count"], [["Dresden", 200]])
        assert check_freshness([wf], bundle, config=None, api_key="") == []  # type: ignore[arg-type]

    def test_dresden_capex_repro_now_has_overlap(self):
        """The reviewer's repro: Dresden/capex web finding + CCC Dresden/capex row
        must now produce an overlap so classification runs."""
        wf = WebFinding(
            city="Dresden",
            field="capex",
            value=50,
            unit="EUR_millions",
            source_url="https://example.com",
            source_type="government_report",
            source_date="2025",
            extraction_confidence=0.9,
        )
        bundle = _make_sql_bundle(["city", "capex"], [["Dresden", 30]])
        values = _extract_ccc_values(bundle)
        key = (wf.city.lower(), wf.field.lower())
        assert values.get(key) == "30", "CCC value must be reachable by (city, field)"
