"""Tests for the post-extraction validator."""

from __future__ import annotations

from backend.modules.web_researcher.models import WebFinding
from backend.modules.web_researcher.post_extraction_validator import (
    validate_finding,
    validate_findings,
)


def _make_finding(**overrides) -> WebFinding:
    defaults = {
        "city": "Munich",
        "field": "taxi_count",
        "value": 100,
        "unit": "vehicles",
        "source_url": "https://example.com",
        "source_type": "government_report",
        "source_date": "2024",
        "extraction_confidence": 0.9,
    }
    defaults.update(overrides)
    return WebFinding(**defaults)


_POP = {"munich": 1_500_000, "munster": 315_000, "klagenfurt": 100_000}


class TestNullEmpty:
    def test_discard_none_value(self):
        f = _make_finding(value=None)
        assert validate_finding(f, _POP) is None

    def test_discard_empty_string_value(self):
        f = _make_finding(value="")
        assert validate_finding(f, _POP) is None

    def test_discard_whitespace_string_value(self):
        f = _make_finding(value="   ")
        assert validate_finding(f, _POP) is None

    def test_keep_zero_value(self):
        f = _make_finding(value=0)
        assert validate_finding(f, _POP) is not None


class TestCancellationPassthrough:
    def test_cancellation_passes_all_checks(self):
        f = _make_finding(
            value=0,
            unit="cancelled_programme",
            source_type="cancellation_notice",
        )
        result = validate_finding(f, _POP)
        assert result is not None
        assert result.unit == "cancelled_programme"

    def test_cancellation_bypasses_integer_check(self):
        """Even if value were weird, cancellation passthrough wins."""
        f = _make_finding(
            value=0,
            unit="cancelled_programme",
            field="ev_bus_count",
        )
        assert validate_finding(f, _POP) is not None


class TestIntegerFieldCheck:
    def test_reject_fractional_bus_count(self):
        f = _make_finding(field="ev_bus_count", value=2.4)
        assert validate_finding(f, _POP) is None

    def test_reject_fractional_taxi_count(self):
        f = _make_finding(field="taxi_count", value=15.7)
        assert validate_finding(f, _POP) is None

    def test_accept_integer_bus_count(self):
        f = _make_finding(field="ev_bus_count", value=50)
        assert validate_finding(f, _POP) is not None

    def test_accept_float_whole_number(self):
        f = _make_finding(field="ev_bus_count", value=50.0)
        assert validate_finding(f, _POP) is not None

    def test_non_integer_field_allows_fractional(self):
        f = _make_finding(field="modal_share_cycling", value=12.5)
        assert validate_finding(f, _POP) is not None


class TestAbsoluteCaps:
    def test_reject_over_cap_ev_bus(self):
        f = _make_finding(field="ev_bus_count", value=100_000)
        assert validate_finding(f, _POP) is None

    def test_accept_under_cap_ev_bus(self):
        f = _make_finding(field="ev_bus_count", value=200)
        assert validate_finding(f, _POP) is not None

    def test_reject_over_cap_taxi(self):
        f = _make_finding(field="taxi_count", value=60_000)
        assert validate_finding(f, _POP) is None

    def test_accept_under_cap_taxi(self):
        f = _make_finding(field="taxi_count", value=3_000)
        assert validate_finding(f, _POP) is not None

    def test_no_cap_for_unknown_field(self):
        f = _make_finding(field="co2_emissions_total", value=999_999)
        assert validate_finding(f, _POP) is not None


class TestPerCapitaPlausibility:
    def test_reject_implausible_taxis(self):
        """1000 taxis for Munster (~315k pop) → ~317 per 100k, cap is 500 → passes.
        But 2000 taxis → ~635 per 100k → rejected."""
        f = _make_finding(city="Munster", field="taxi_count", value=2000)
        assert validate_finding(f, _POP) is None

    def test_accept_plausible_taxis(self):
        f = _make_finding(city="Munich", field="taxi_count", value=3000)
        # 3000 / 1_500_000 * 100_000 = 200, under cap of 500
        assert validate_finding(f, _POP) is not None

    def test_skip_when_city_not_in_lookup(self):
        f = _make_finding(city="UnknownCity", field="taxi_count", value=99999)
        # No population data → skip per-capita check, but absolute cap catches it
        pop = {}
        assert validate_finding(f, pop) is None  # caught by absolute cap

    def test_skip_per_capita_unknown_city_under_abs_cap(self):
        f = _make_finding(city="UnknownCity", field="taxi_count", value=1000)
        pop = {}
        # Under absolute cap, no population → passes
        assert validate_finding(f, pop) is not None


class TestValidateFindings:
    def test_filters_bad_findings(self):
        findings = [
            _make_finding(value=100),           # good
            _make_finding(value=None),           # bad: null
            _make_finding(field="ev_bus_count", value=2.4),  # bad: fractional
        ]
        result = validate_findings(findings, ["Munich"], _POP)
        assert len(result) == 1
        assert result[0].value == 100

    def test_uses_default_populations(self):
        """When no population_lookup is passed, uses _CITY_POPULATIONS."""
        f = _make_finding(city="Munster", field="taxi_count", value=2000)
        result = validate_findings([f], ["Munster"])
        # 2000 / 315_000 * 100_000 ≈ 635 > 500 cap → rejected
        assert len(result) == 0

    def test_empty_list(self):
        assert validate_findings([], ["Munich"]) == []

    def test_all_pass(self):
        findings = [
            _make_finding(value=100),
            _make_finding(value=200),
        ]
        result = validate_findings(findings, ["Munich"], _POP)
        assert len(result) == 2
