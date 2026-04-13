"""Unit tests for search planner improvements: budget scaling, field categorization, source hints."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.modules.web_researcher.models import (
    CityGap,
    FieldClassification,
    GapManifest,
    SearchBatch,
)
from backend.modules.web_researcher.search_planner import (
    _categorize_fields,
    _compute_budget,
    _formulate_national_benchmark_queries,
    _get_source_hints,
    _plan_national_benchmark_batches,
    plan_searches,
)
from tests.support import build_test_app_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> Any:
    return build_test_app_config(
        enrichment_overrides={"enabled": True, "model": "test-model", **overrides}
    )


def _mock_openai_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# _get_source_hints
# ---------------------------------------------------------------------------

class TestGetSourceHints:
    def test_fleet_fields_return_fleet_hints(self):
        hints = _get_source_hints(["ev_fleet_count", "bus_count_electric"])
        assert "fleet register" in hints
        assert "tender award notice" in hints

    def test_capex_fields_return_financial_hints(self):
        hints = _get_source_hints(["total_capex_eur", "investment_programme"])
        assert "investment programme" in hints
        assert "budget allocation" in hints

    def test_emissions_fields_return_emissions_hints(self):
        hints = _get_source_hints(["co2_emissions_tons", "ghg_reduction_pct"])
        assert "national GHG inventory" in hints

    def test_charging_fields_return_infra_hints(self):
        hints = _get_source_hints(["charging_capacity_kw", "charger_count"])
        assert "technical specification" in hints

    def test_unrecognized_fields_return_empty(self):
        hints = _get_source_hints(["population", "area_km2"])
        assert hints == []

    def test_mixed_fields_deduplicate(self):
        hints = _get_source_hints(["fleet_count", "bus_count_total"])
        # Both match the same pattern, so hints should not duplicate
        assert len(hints) == len(set(hints))

    def test_multi_category_fields(self):
        hints = _get_source_hints(["fleet_count", "capex_total_eur"])
        assert "fleet register" in hints
        assert "investment programme" in hints


# ---------------------------------------------------------------------------
# _categorize_fields
# ---------------------------------------------------------------------------

class TestCategorizeFields:
    def test_fleet_fields_grouped(self):
        result = _categorize_fields(["ev_bus_count", "tram_fleet_total", "ev_ferry_count"])
        assert "fleet_vehicle" in result
        assert len(result["fleet_vehicle"]) == 3

    def test_financial_fields_grouped(self):
        result = _categorize_fields(["total_capex_eur", "opex_per_km", "investment_volume"])
        assert "financial" in result
        assert len(result["financial"]) == 3

    def test_unrecognized_go_to_general(self):
        result = _categorize_fields(["population", "area_km2", "avg_temperature"])
        assert "general" in result
        assert len(result["general"]) == 3

    def test_mixed_categories(self):
        result = _categorize_fields([
            "ev_bus_count", "tram_fleet_total",
            "total_capex_eur", "opex_per_km",
        ])
        assert "fleet_vehicle" in result
        assert "financial" in result
        assert len(result["fleet_vehicle"]) == 2
        assert len(result["financial"]) == 2

    def test_single_field_category_merges_into_general(self):
        result = _categorize_fields([
            "ev_bus_count", "tram_fleet_total",
            "co2_emissions_tons",  # single emissions field
        ])
        assert "fleet_vehicle" in result
        # Single-field emissions should merge into general
        assert "emissions" not in result
        assert "general" in result
        assert "co2_emissions_tons" in result["general"]

    def test_empty_input(self):
        result = _categorize_fields([])
        assert result == {}


# ---------------------------------------------------------------------------
# _compute_budget — gap surface scaling
# ---------------------------------------------------------------------------

class TestComputeBudget:
    def test_high_priority_small_surface(self):
        config = _make_config(max_queries_per_batch=25)
        budget = _compute_budget(
            blank_count=2, stale_count=0, city_count=2,
            priority="high", config=config,
        )
        # gap_surface = 2*2 = 4, max(15, 4*2//3)=max(15,2)=15
        assert budget["max_queries"] == 15
        assert budget["deep_dive_allowed"] is True

    def test_high_priority_large_surface(self):
        config = _make_config(max_queries_per_batch=25)
        budget = _compute_budget(
            blank_count=6, stale_count=0, city_count=8,
            priority="high", config=config,
        )
        # gap_surface = 6*8 = 48, max(15, 48*2//3)=max(15,32)=32 -> capped at 25
        assert budget["max_queries"] == 25
        assert budget["deep_dive_allowed"] is True

    def test_high_priority_capped_by_max_per_batch(self):
        config = _make_config(max_queries_per_batch=10)
        budget = _compute_budget(
            blank_count=6, stale_count=0, city_count=8,
            priority="high", config=config,
        )
        # capped at max_per_batch=10
        assert budget["max_queries"] == 10

    def test_stale_only_budget(self):
        config = _make_config(max_queries_per_batch=25)
        budget = _compute_budget(
            blank_count=0, stale_count=5, city_count=3,
            priority="medium", config=config,
        )
        assert budget["max_queries"] == 10
        assert budget["deep_dive_allowed"] is False

    def test_few_partial_gaps(self):
        config = _make_config(max_queries_per_batch=25)
        budget = _compute_budget(
            blank_count=0, stale_count=1, city_count=1,
            priority="low", config=config,
        )
        assert budget["max_queries"] == 5
        assert budget["deep_dive_allowed"] is False


# ---------------------------------------------------------------------------
# plan_searches — integration-level with mocked LLM
# ---------------------------------------------------------------------------

class TestPlanSearches:
    def _make_gap_manifest(self, *, cities: list[str], fields: list[str]) -> GapManifest:
        return GapManifest(
            query_fields=[
                FieldClassification(
                    field=f,
                    classification="estimable_numerical",
                    searchable=True,
                    rationale="test",
                )
                for f in fields
            ],
            city_gaps=[
                CityGap(
                    city=c,
                    blank_fields=fields,
                    stale_flags=[],
                    search_priority="high",
                )
                for c in cities
            ],
            non_estimable_fields=[],
        )

    @patch("backend.modules.web_researcher.search_planner._load_city_groups")
    @patch("backend.modules.web_researcher.search_planner.OpenAI")
    def test_question_passed_to_prompt(self, mock_openai_cls, mock_city_groups):
        mock_city_groups.return_value = []
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '["Dresden fleet register 2025"]'
        )

        config = _make_config(max_queries_per_batch=25, max_total_queries_per_run=100)
        manifest = self._make_gap_manifest(
            cities=["Dresden"],
            fields=["ev_fleet_count"],
        )

        plan_searches(manifest, config, "test-key", question="How many electric buses in Dresden?")

        # First LLM call is city-specific queries; use call_args_list[0]
        call_args = mock_client.chat.completions.create.call_args_list[0]
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        assert "How many electric buses in Dresden?" in system_msg
        assert "How many electric buses in Dresden?" in user_msg

    @patch("backend.modules.web_researcher.search_planner._load_city_groups")
    @patch("backend.modules.web_researcher.search_planner.OpenAI")
    def test_field_category_split_creates_separate_batches(self, mock_openai_cls, mock_city_groups):
        mock_city_groups.return_value = []
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '["query 1", "query 2"]'
        )

        config = _make_config(max_queries_per_batch=25, max_total_queries_per_run=100)
        manifest = self._make_gap_manifest(
            cities=["Dresden"],
            fields=["ev_bus_count", "tram_fleet_total", "total_capex_eur", "opex_per_km"],
        )

        batches = plan_searches(manifest, config, "test-key")

        # Should have separate batches for fleet_vehicle and financial
        assert len(batches) >= 2
        batch_ids = [b.batch_id for b in batches]
        categories_in_ids = [bid.split("_")[2] for bid in batch_ids]
        assert "fleet" in categories_in_ids or any("fleet" in c for c in categories_in_ids)

    @patch("backend.modules.web_researcher.search_planner._load_city_groups")
    @patch("backend.modules.web_researcher.search_planner.OpenAI")
    def test_source_hints_in_prompt_for_fleet_fields(self, mock_openai_cls, mock_city_groups):
        mock_city_groups.return_value = []
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '["Dresden fleet register 2025"]'
        )

        config = _make_config(max_queries_per_batch=25, max_total_queries_per_run=100)
        manifest = self._make_gap_manifest(
            cities=["Dresden"],
            fields=["ev_fleet_count", "bus_count_electric"],
        )

        plan_searches(manifest, config, "test-key")

        # First LLM call is city-specific queries
        call_args = mock_client.chat.completions.create.call_args_list[0]
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]
        assert "fleet register" in system_msg
        assert "Recommended source types" in system_msg

    @patch("backend.modules.web_researcher.search_planner._load_city_groups")
    @patch("backend.modules.web_researcher.search_planner.OpenAI")
    def test_high_priority_variant_instructions(self, mock_openai_cls, mock_city_groups):
        mock_city_groups.return_value = []
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '["query 1"]'
        )

        config = _make_config(max_queries_per_batch=25, max_total_queries_per_run=100)
        manifest = self._make_gap_manifest(
            cities=["Dresden"],
            fields=["ev_fleet_count", "bus_count_electric"],
        )

        plan_searches(manifest, config, "test-key")

        # First LLM call is city-specific queries
        call_args = mock_client.chat.completions.create.call_args_list[0]
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]
        assert "Variant A" in system_msg
        assert "Variant B" in system_msg

    @patch("backend.modules.web_researcher.search_planner._load_city_groups")
    @patch("backend.modules.web_researcher.search_planner.OpenAI")
    def test_batch_id_encodes_region_and_category(self, mock_openai_cls, mock_city_groups):
        mock_city_groups.return_value = []
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '["query 1"]'
        )

        config = _make_config(max_queries_per_batch=25, max_total_queries_per_run=100)
        manifest = self._make_gap_manifest(
            cities=["Dresden"],
            fields=["ev_fleet_count", "bus_count_electric"],
        )

        batches = plan_searches(manifest, config, "test-key")

        # Filter to city-specific batches (exclude national benchmark batches)
        city_batches = [b for b in batches if b.search_type != "national_benchmark"]
        assert len(city_batches) == 1
        # batch_id should be: batch_{region}_{category}_{hex}
        parts = city_batches[0].batch_id.split("_")
        assert parts[0] == "batch"
        # region is "other" since no city_groups loaded
        assert parts[1] == "other"

    @patch("backend.modules.web_researcher.search_planner._load_city_groups")
    @patch("backend.modules.web_researcher.search_planner.OpenAI")
    def test_plan_searches_includes_national_benchmark_batches(
        self, mock_openai_cls, mock_city_groups,
    ):
        mock_city_groups.return_value = []
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # First call: city queries; Second call: national benchmark queries
        mock_client.chat.completions.create.side_effect = [
            _mock_openai_response('["Dresden fleet register 2025"]'),
            _mock_openai_response(
                '["Germany electric bus unit cost 2024", "Germany charging infra benchmark"]'
            ),
        ]

        config = _make_config(max_queries_per_batch=25, max_total_queries_per_run=100)
        manifest = self._make_gap_manifest(
            cities=["Dresden"],
            fields=["ev_fleet_count"],
        )

        batches = plan_searches(manifest, config, "test-key")

        national = [b for b in batches if b.search_type == "national_benchmark"]
        city = [b for b in batches if b.search_type != "national_benchmark"]
        assert len(national) >= 1
        assert len(city) >= 1
        assert "national" in national[0].batch_id
        assert national[0].budget["deep_dive_allowed"] is False


# ---------------------------------------------------------------------------
# _plan_national_benchmark_batches
# ---------------------------------------------------------------------------

class TestPlanNationalBenchmarkBatches:
    @patch("backend.modules.web_researcher.search_planner.OpenAI")
    def test_generates_batches_for_regions(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '["Germany bus unit cost 2024", "Germany charging benchmark"]'
        )

        config = _make_config()
        gap_manifest = GapManifest(
            query_fields=[
                FieldClassification(
                    field="bus_procurement_unit_cost",
                    classification="estimable_numerical",
                    searchable=True,
                    rationale="test",
                ),
            ],
            city_gaps=[],
            non_estimable_fields=[],
        )
        region_gaps = {"dach": [{"city": "Dresden"}]}
        city_groups = [{"id": "dach", "cities": ["Dresden", "Munich"]}]

        batches, count = _plan_national_benchmark_batches(
            gap_manifest=gap_manifest,
            region_gaps=region_gaps,
            city_groups=city_groups,
            config=config,
            api_key="test-key",
            remaining_budget=10,
        )

        assert len(batches) == 1
        assert batches[0].search_type == "national_benchmark"
        assert count == 2
        assert batches[0].cities == ["dach"]
        assert batches[0].budget["deep_dive_allowed"] is False

    @patch("backend.modules.web_researcher.search_planner.OpenAI")
    def test_respects_remaining_budget(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '["q1", "q2", "q3"]'
        )

        config = _make_config()
        gap_manifest = GapManifest(
            query_fields=[
                FieldClassification(
                    field="cost_field",
                    classification="estimable_numerical",
                    searchable=True,
                    rationale="test",
                ),
            ],
            city_gaps=[],
            non_estimable_fields=[],
        )

        batches, count = _plan_national_benchmark_batches(
            gap_manifest=gap_manifest,
            region_gaps={"region_a": [{"city": "CityA"}]},
            city_groups=[],
            config=config,
            api_key="test-key",
            remaining_budget=2,
        )

        assert count <= 2

    def test_returns_empty_when_no_budget(self):
        gap_manifest = GapManifest(
            query_fields=[
                FieldClassification(
                    field="cost_field",
                    classification="estimable_numerical",
                    searchable=True,
                    rationale="test",
                ),
            ],
            city_gaps=[],
            non_estimable_fields=[],
        )

        batches, count = _plan_national_benchmark_batches(
            gap_manifest=gap_manifest,
            region_gaps={"r": [{"city": "C"}]},
            city_groups=[],
            config=_make_config(),
            api_key="test-key",
            remaining_budget=0,
        )

        assert batches == []
        assert count == 0

    def test_returns_empty_when_all_non_estimable(self):
        gap_manifest = GapManifest(
            query_fields=[
                FieldClassification(
                    field="operator_name",
                    classification="non_estimable",
                    searchable=False,
                    rationale="qualitative",
                ),
            ],
            city_gaps=[],
            non_estimable_fields=["operator_name"],
        )

        batches, count = _plan_national_benchmark_batches(
            gap_manifest=gap_manifest,
            region_gaps={"r": [{"city": "C"}]},
            city_groups=[],
            config=_make_config(),
            api_key="test-key",
            remaining_budget=10,
        )

        assert batches == []
        assert count == 0
