import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from backend.modules.initiative_extractor.models import (
    InitiativeExtraction,
    InitiativeExtractionRecord,
    InitiativeNumbers,
    InitiativeSourceRef,
)
from backend.modules.tef_mapper import agent as tef_agent
from backend.modules.tef_mapper.catalog import TefCatalog
from backend.modules.tef_mapper.models import (
    TefFinalMappingRecord,
    TefSectorRoute,
    TefSubsectorRoute,
    TefTransitionMapping,
)
from backend.modules.tef_mapper.numeric_rollup import build_numeric_facts
from tests.support import build_test_app_config

KRAKOW_SOURCE_TRUTH_PATH = Path("assets/tef_mapping/all_correct_initiatives.json")
KRAKOW_TEF_SOURCE_TRUTH_PATH = Path(
    "assets/tef_mapping/all_correct_initiatives_mapped_to_tef.json"
)


class _FakeAgent:
    """Minimal stage-agent object for mapper tests."""

    def __init__(self, stage: str) -> None:
        self.stage = stage


class _FakeRunResult:
    """Minimal fake Agents result for TEF mapper tests."""

    def __init__(self, final_output: object) -> None:
        self.final_output = final_output


def _record(
    *,
    record_id: str = "krakow:krakow:bic_7",
    initiative_name: str = "Local energy programme based on heat pumps",
) -> InitiativeExtractionRecord:
    """Build a valid initiative extraction record for TEF mapper tests."""
    return InitiativeExtractionRecord(
        record_id=record_id,
        source_document="Krakow.md",
        document_local_code="BIC-7",
        initiative=InitiativeExtraction(
            city="Krakow",
            initiative_name=initiative_name,
            general_description="Krakow plans heat-pump capacity for district heating.",
            objective_text="Improve and decarbonise the district heating system.",
            implementation_text="Implement heat pumps in the city energy system.",
            planned_outputs_text="Approximately 1 MW heat-pump-based capacity.",
            delivery_text="Municipal Heating Company delivery.",
            funding_text="Estimated investment outlay PLN 7,000,000.",
            timeline_text="2024 to 2028.",
            numbers=InitiativeNumbers(
                current={},
                planned={"capacity_mw": 1, "start_year": 2024, "end_year": 2028},
            ),
        ),
        source_refs=[],
    )


def _write_initiatives(path: Path, records: list[InitiativeExtractionRecord]) -> None:
    """Write initiative records as JSONL for pipeline tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record.model_dump(mode="json")) for record in records) + "\n",
        encoding="utf-8",
    )


def _source_truth_record(row: dict[str, Any]) -> InitiativeExtractionRecord:
    """Convert one curated Krakow CCC source-truth row into mapper input shape."""
    return InitiativeExtractionRecord(
        record_id=row["record_id"],
        source_document=row["source_document"],
        document_local_code=row.get("local_code"),
        initiative=InitiativeExtraction(
            city=row["city"],
            initiative_name=row["initiative_name"],
            general_description=row.get("general_description"),
            objective_text=row.get("objective_text"),
            implementation_text=row.get("implementation_text"),
            planned_outputs_text=row.get("planned_outputs_text"),
            delivery_text=row.get("delivery_text"),
            funding_text=row.get("funding_text"),
            timeline_text=row.get("timeline_text"),
            numbers=InitiativeNumbers.model_validate(row.get("numbers", {})),
        ),
        source_refs=[
            InitiativeSourceRef(
                source_document=ref.get("source_document", row["source_document"]),
                segment_id=f"{row['record_id']}:source:{index}",
                start_line=ref["start_line"],
                end_line=ref["end_line"],
            )
            for index, ref in enumerate(row.get("source_refs", []), start=1)
        ],
    )


def _next_source_truth_child_path(target_path: str, candidate_paths: list[str]) -> str:
    """Return the direct child path that leads to a curated source-truth target."""
    for path in sorted(candidate_paths, key=len, reverse=True):
        if target_path == path or target_path.startswith(f"{path}/"):
            return path
    raise AssertionError(f"No direct candidate leads to source-truth target {target_path!r}")


def _patch_source_truth_agents(
    monkeypatch: pytest.MonkeyPatch,
    catalog: TefCatalog,
    expected_by_record: dict[str, dict[str, Any]],
) -> None:
    """Patch mapper LLM stages with decisions from the curated Krakow TEF fixture."""
    tef_agent._thread_local.tef_mapper_agents = {}
    monkeypatch.setattr(
        tef_agent,
        "build_sector_router_agent",
        lambda *_args: _FakeAgent("sector"),
    )
    monkeypatch.setattr(
        tef_agent,
        "build_subsector_router_agent",
        lambda *_args: _FakeAgent("subsector"),
    )
    monkeypatch.setattr(
        tef_agent,
        "build_transition_mapper_agent",
        lambda *_args: _FakeAgent("transition"),
    )

    def fake_run_agent_sync(
        agent: _FakeAgent,
        input_data: str,
        **_kwargs: object,
    ) -> _FakeRunResult:
        payload = json.loads(input_data)
        record_id = payload["initiative"]["record_id"]
        expected = expected_by_record[record_id]
        primary_mapping = expected["primary_tef_mapping"]
        target_path = primary_mapping["target_path"]

        if agent.stage == "sector":
            sector_path = target_path.split("/", maxsplit=1)[0]
            sector = catalog.sectors_by_path[sector_path].sector
            return _FakeRunResult(
                TefSectorRoute(
                    sector=sector,
                    selected_path=sector_path,
                    confidence=0.99,
                    needs_review=False,
                    rationale="Curated Krakow CCC source-truth sector route.",
                    alternatives=[],
                )
            )

        if agent.stage == "subsector":
            candidate_paths = [
                candidate["path"] for candidate in payload["candidate_subcategories"]
            ]
            selected_path = _next_source_truth_child_path(target_path, candidate_paths)
            return _FakeRunResult(
                TefSubsectorRoute(
                    selected_path=selected_path,
                    confidence=0.99,
                    needs_review=False,
                    rationale="Curated Krakow CCC source-truth category route.",
                    alternatives=[],
                )
            )

        selected_path = payload["selected_category"]["path"]
        candidate_tef_ids = {
            candidate["tef_id"] for candidate in payload["candidate_transition_elements"]
        }
        transition_mappings = [
            mapping
            for mapping in expected["tef_mappings"]
            if mapping["target_type"] == "transition_element"
            and mapping["target_path"] == selected_path
        ]
        missing_tef_ids = {
            mapping["target_id"]
            for mapping in transition_mappings
            if mapping["target_id"] not in candidate_tef_ids
        }
        assert not missing_tef_ids
        return _FakeRunResult(
            TefTransitionMapping(
                needs_review=not transition_mappings
                or any(mapping["needs_review"] for mapping in transition_mappings),
                matches=[
                    {
                        "tef_id": mapping["target_id"],
                        "confidence": mapping["confidence"],
                        "is_primary": mapping["is_primary"],
                        "rationale": mapping["rationale"],
                    }
                    for mapping in transition_mappings
                ],
            )
        )

    monkeypatch.setattr(tef_agent, "run_agent_sync", fake_run_agent_sync)


def _patch_fake_agents(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, dict[str, object]]],
    *,
    no_transition_leaf_branch: bool = False,
    single_family_hvac_branch: bool = False,
    descendant_first_subsector_selection: bool = False,
    no_transition_matches: bool = False,
) -> None:
    """Patch mapper LLM calls with deterministic staged outputs."""
    tef_agent._thread_local.tef_mapper_agents = {}
    monkeypatch.setattr(
        tef_agent,
        "build_sector_router_agent",
        lambda *_args: _FakeAgent("sector"),
    )
    monkeypatch.setattr(
        tef_agent,
        "build_subsector_router_agent",
        lambda *_args: _FakeAgent("subsector"),
    )
    monkeypatch.setattr(
        tef_agent,
        "build_transition_mapper_agent",
        lambda *_args: _FakeAgent("transition"),
    )

    def fake_run_agent_sync(agent: _FakeAgent, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        calls.append((agent.stage, payload))
        if agent.stage == "sector":
            if single_family_hvac_branch:
                return _FakeRunResult(
                    TefSectorRoute(
                        sector="buildings",
                        selected_path="4-buildings",
                        confidence=0.98,
                        needs_review=False,
                        rationale="Single-family thermal retrofits belong under buildings.",
                        alternatives=[],
                    )
                )
            if no_transition_leaf_branch:
                return _FakeRunResult(
                    TefSectorRoute(
                        sector="industry",
                        selected_path="2-industry",
                        confidence=0.89,
                        needs_review=False,
                        rationale="The initiative concerns industrial mineral production.",
                        alternatives=[],
                    )
                )
            return _FakeRunResult(
                TefSectorRoute(
                    sector="energy",
                    selected_path="5-energy",
                    confidence=0.91,
                    needs_review=False,
                    rationale="District heating supply belongs under energy.",
                    alternatives=[],
                )
            )
        if agent.stage == "subsector":
            parent_path = payload["selected_category"]["path"]
            if single_family_hvac_branch:
                selected_path_by_parent = {
                    "4-buildings": "4-buildings/4a-residential",
                    "4-buildings/4a-residential": "4-buildings/4a-residential/4a1-hvac",
                    "4-buildings/4a-residential/4a1-hvac": (
                        "4-buildings/4a-residential/4a1-hvac/4a1b-single-family-hvac"
                    ),
                }
                return _FakeRunResult(
                    TefSubsectorRoute(
                        selected_path=selected_path_by_parent[parent_path],
                        confidence=0.94,
                        needs_review=False,
                        rationale="The initiative is a single-family heating retrofit.",
                        alternatives=[],
                    )
                )
            if no_transition_leaf_branch:
                selected_path = (
                    "2-industry/2a-minerals"
                    if parent_path == "2-industry"
                    else "2-industry/2a-minerals/2a5-soda-ash"
                )
                return _FakeRunResult(
                    TefSubsectorRoute(
                        selected_path=selected_path,
                        confidence=0.77,
                        needs_review=True,
                        rationale="The initiative concerns mineral production without a TE leaf.",
                        alternatives=[],
                    )
                )
            if descendant_first_subsector_selection and parent_path == "5-energy":
                return _FakeRunResult(
                    TefSubsectorRoute(
                        selected_path="5-energy/5a-energy-supply/5a2-heat",
                        confidence=0.83,
                        needs_review=False,
                        rationale="The initiative concerns district heat supply.",
                        alternatives=[
                            {
                                "path": "5-energy/5a-energy-supply/5a1-electricity",
                                "confidence": 0.41,
                            }
                        ],
                    )
                )
            selected_path = (
                "5-energy/5a-energy-supply"
                if parent_path == "5-energy"
                else "5-energy/5a-energy-supply/5a2-heat"
            )
            return _FakeRunResult(
                TefSubsectorRoute(
                    selected_path=selected_path,
                    confidence=0.88,
                    needs_review=False,
                    rationale="The initiative concerns district heat supply.",
                    alternatives=[],
                )
            )
        if no_transition_matches:
            return _FakeRunResult(
                TefTransitionMapping(
                    needs_review=True,
                    matches=[],
                )
            )
        if single_family_hvac_branch:
            return _FakeRunResult(
                TefTransitionMapping(
                    needs_review=False,
                    matches=[
                        {
                            "tef_id": (
                                "energy_efficient_single_family_residential_buildings_"
                                "retrofitting"
                            ),
                            "confidence": 0.93,
                            "is_primary": True,
                            "rationale": "Insulation and window replacement reduce heating demand.",
                        }
                    ],
                )
            )
        return _FakeRunResult(
            TefTransitionMapping(
                needs_review=False,
                matches=[
                    {
                        "tef_id": "district_heating_heat_pumps",
                        "confidence": 0.86,
                        "is_primary": True,
                        "rationale": "The initiative adds district-heating heat pumps.",
                    }
                ],
            )
        )

    monkeypatch.setattr(tef_agent, "run_agent_sync", fake_run_agent_sync)


def test_catalog_loader_finds_sector_and_heat_transitions() -> None:
    """Catalog loader should expose four JSON indexes and direct TE lookups."""
    catalog = TefCatalog(Path("tef_mapping"))

    catalog_json_files = {path.name for path in Path("tef_mapping/catalog").glob("*.json")}
    assert catalog_json_files == {
        "sectors.json",
        "subcategories.json",
        "subsubcategories.json",
        "transition_elements.json",
    }
    assert not Path("tef_mapping/categories").exists()
    assert not Path("tef_mapping/catalog/all_transition_elements.json").exists()
    assert len(catalog.sectors) == 6
    assert len(catalog.subcategories) == 19
    assert len(catalog.subsubcategories) == 95
    assert catalog.sector_path("energy") == "5-energy"

    industry_children = catalog.child_subsectors("2-industry")
    assert {child.path for child in industry_children} >= {"2-industry/2a-minerals"}
    mineral_children = catalog.child_subsectors("2-industry/2a-minerals")
    assert {child.path for child in mineral_children} >= {
        "2-industry/2a-minerals/2a1-cement",
        "2-industry/2a-minerals/2a5-soda-ash",
    }
    cement_candidates = catalog.transition_elements("2-industry/2a-minerals/2a1-cement")
    assert {item.tef_id for item in cement_candidates} >= {
        "biofuels_for_cement_and_mineral_industry",
        "electrification_of_production_in_cement_and_concrete_industry",
    }
    assert not catalog.transition_elements("2-industry/2a-minerals/2a5-soda-ash")
    assert not catalog.child_subsectors("2-industry/2a-minerals/2a5-soda-ash")

    road_children = catalog.child_subsectors("1-transport/1a-mobility/1a1-road")
    assert {child.path for child in road_children} >= {
        "1-transport/1a-mobility/1a1-road/1a1a-light-duty-vehicles"
    }
    heat_candidates = catalog.transition_elements("5-energy/5a-energy-supply/5a2-heat")
    assert {item.tef_id for item in heat_candidates} >= {"district_heating_heat_pumps"}


def test_waste_chp_transition_descriptions_distinguish_purpose() -> None:
    """Waste CHP candidates should separate diversion purpose from energy output."""
    catalog = TefCatalog(Path("tef_mapping"))
    candidates = {
        item.tef_id: item
        for item in catalog.transition_elements(
            "5-energy/5a-energy-supply/5a3-combined-heat-power"
        )
    }

    chp_waste = candidates["chp_waste_incineration"]
    waste_recovery = candidates["increased_energy_recovery_from_waste"]

    assert chp_waste.type == "supplyAlteration"
    assert chp_waste.unit_of_measure == "kwh"
    assert "energy-production transition" in chp_waste.description.casefold()
    assert "generate useful heat and power from waste" in chp_waste.description
    assert "not primarily to divert waste away from landfill" in chp_waste.description

    assert waste_recovery.type == "shift"
    assert waste_recovery.unit_of_measure == "tonne"
    assert waste_recovery.shift_to == ["chp_waste_incineration"]
    assert "waste-diversion transition" in waste_recovery.description.casefold()
    assert "measured as tonnes of waste shifted" in waste_recovery.description
    assert "not primarily as heat or electricity output" in waste_recovery.description


def test_grid_electricity_description_prioritizes_procurement_emission_factor() -> None:
    """Green electricity procurement should point to the emission-factor TE."""
    catalog = TefCatalog(Path("tef_mapping"))
    candidates = {
        item.tef_id: item
        for item in catalog.transition_elements(
            "5-energy/5a-energy-supply/5a1-electricity"
        )
    }

    emission_factor = candidates["imported_grid_electricity_emission_factor"]
    improved_grid = candidates["improved_grid_electricity"]

    assert emission_factor.type == "supplyUpdate"
    assert emission_factor.unit_of_measure == "g_co2e_kwh"
    assert "emission-factor update" in emission_factor.description.casefold()
    assert "green electricity purchase" in emission_factor.description
    assert "green power procurement" in emission_factor.description
    assert "importing lower-emission-intensity electricity" in (
        emission_factor.description
    )
    assert "prefer this over improved_grid_electricity" in (
        emission_factor.description.casefold()
    )

    assert improved_grid.type == "shift"
    assert improved_grid.unit_of_measure == "kwh"
    assert improved_grid.shift_from == ["grid_current"]
    assert improved_grid.shift_to == ["grid_future"]


def test_category_cards_have_sector_style_routing_guidance() -> None:
    """Every non-sector category card should give the router explicit guidance."""
    catalog = TefCatalog(Path("tef_mapping"))
    category_cards = [*catalog.subcategories, *catalog.subsubcategories]

    assert len(category_cards) == 114
    assert all(card.description.strip() for card in category_cards)
    assert all("## Routing Definition" in card.card_text for card in category_cards)
    assert all("## Use This Category When" in card.card_text for card in category_cards)
    assert all("## Avoid This Category When" in card.card_text for card in category_cards)
    assert not any("## Routing Guidance" in card.card_text for card in category_cards)
    assert not any(
        "Use this category when the initiative's extracted description" in card.card_text
        for card in category_cards
    )

    heat_card = catalog.subcategories_by_path["5-energy/5a-energy-supply/5a2-heat"]
    assert "district heating" in heat_card.card_text.casefold()
    residential_hvac = catalog.subcategories_by_path["4-buildings/4a-residential/4a1-hvac"]
    residential_hvac_text = residential_hvac.card_text.casefold()
    assert "thermal modernisation" in residential_hvac_text
    assert "heating-demand reduction" in residential_hvac_text
    assert "even when the source does not explicitly say hvac equipment replacement" in (
        residential_hvac_text
    )
    passenger_road = catalog.subcategories_by_path["1-transport/1a-mobility/1a1-road"]
    freight_road = catalog.subcategories_by_path["1-transport/1b-freight/1b1-road"]
    assert "Transport > Mobility > Road" in passenger_road.card_text
    passenger_road_text = passenger_road.card_text.casefold()
    assert "city tram, light rail, and subway modal-shift initiatives" in (
        passenger_road_text
    )
    assert "do not avoid road merely because a city project mentions tram track" in (
        passenger_road_text
    )
    assert "Transport > Freight > Road" in freight_road.card_text

    passenger_rail = catalog.subcategories_by_path["1-transport/1a-mobility/1a2-rail"]
    freight_rail = catalog.subcategories_by_path["1-transport/1b-freight/1b2-rail"]
    assert "Mobility > Rail" in passenger_rail.card_text
    assert "use road, not rail, for urban tram, light rail, or subway projects" in (
        passenger_rail.card_text.casefold()
    )
    assert "Freight > Rail" in freight_rail.card_text

    light_duty = catalog.subcategories_by_path[
        "1-transport/1a-mobility/1a1-road/1a1a-light-duty-vehicles"
    ]
    assert "trams, light rail and subway" in light_duty.card_text.casefold()
    assert "shifts trips from private cars" in light_duty.card_text.casefold()

    soda_ash = catalog.subcategories_by_path["2-industry/2a-minerals/2a5-soda-ash"]
    assert "no-transition TEF category" in soda_ash.description
    assert "- None directly under this category" in soda_ash.card_text


def test_transition_mapping_rejects_multiple_primary_matches() -> None:
    """Transition mapping schema should reject multiple primary matches."""
    with pytest.raises(ValidationError):
        TefTransitionMapping.model_validate(
            {
                "needs_review": False,
                "matches": [
                    {
                        "tef_id": "a",
                        "confidence": 0.8,
                        "is_primary": True,
                        "rationale": "A",
                    },
                    {
                        "tef_id": "b",
                        "confidence": 0.7,
                        "is_primary": True,
                        "rationale": "B",
                    },
                ],
            }
        )


def test_mapper_loads_only_stage_scoped_payloads_and_maps_heat_pump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline should use scoped pass payloads and map Krakow heat pumps to a TE."""
    calls: list[tuple[str, dict[str, object]]] = []
    _patch_fake_agents(monkeypatch, calls)
    initiatives_path = tmp_path / "initiatives.jsonl"
    _write_initiatives(initiatives_path, [_record()])
    config = build_test_app_config(tef_mapper_overrides={"max_workers": 1})

    result = tef_agent.map_initiatives_to_tef(
        config=config,
        api_key="test",
        tef_catalog_dir=Path("tef_mapping"),
        output_root=tmp_path / "output",
        initiatives_jsonl=initiatives_path,
        run_id="tef_test",
    )

    final_rows = [
        json.loads(line)
        for line in (Path(result.output_dir) / "05_final_mappings" / "final_mappings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert final_rows[0]["target_type"] == "transition_element"
    assert final_rows[0]["target_id"] == "district_heating_heat_pumps"
    assert final_rows[0]["target_path"] == "5-energy/5a-energy-supply/5a2-heat"
    assert (Path(result.output_dir) / "07_numeric_facts" / "initiative_numeric_facts.jsonl").exists()
    assert (Path(result.output_dir) / "08_tef_groups" / "tef_grouped_initiatives.jsonl").exists()
    assert (Path(result.output_dir) / "08_tef_groups" / "tef_metric_rollups.json").exists()

    sector_payload = next(payload for stage, payload in calls if stage == "sector")
    assert set(sector_payload) == {"initiative", "sectors"}
    assert "candidate_subcategories" not in sector_payload
    assert "candidate_transition_elements" not in sector_payload

    subsector_payloads = [payload for stage, payload in calls if stage == "subsector"]
    assert len(subsector_payloads) == 2
    assert all(set(payload) == {"initiative", "selected_category", "candidate_subcategories"} for payload in subsector_payloads)
    first_subsector_candidates = subsector_payloads[0]["candidate_subcategories"]
    energy_supply = next(
        candidate
        for candidate in first_subsector_candidates
        if candidate["path"] == "5-energy/5a-energy-supply"
    )
    assert energy_supply["description"]
    assert "## Routing Definition" in energy_supply["card_text"]
    assert "## Use This Category When" in energy_supply["card_text"]
    assert "## Avoid This Category When" in energy_supply["card_text"]

    transition_payload = next(payload for stage, payload in calls if stage == "transition")
    assert set(transition_payload) == {
        "initiative",
        "selected_category",
        "candidate_transition_elements",
    }
    assert {item["tef_id"] for item in transition_payload["candidate_transition_elements"]} >= {
        "district_heating_heat_pumps"
    }


def test_krakow_ccc_source_truth_initiatives_map_to_tef_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Krakow CCC source-truth initiatives should round-trip to curated TEF targets."""
    source_payload = json.loads(KRAKOW_SOURCE_TRUTH_PATH.read_text(encoding="utf-8"))
    mapped_payload = json.loads(KRAKOW_TEF_SOURCE_TRUTH_PATH.read_text(encoding="utf-8"))
    source_rows = source_payload["initiatives"]
    mapped_rows = mapped_payload["initiatives"]
    source_ids = {row["record_id"] for row in source_rows}
    mapped_ids = {row["record_id"] for row in mapped_rows}

    assert source_payload["metadata"]["counts"]["initiatives"] == 58
    assert mapped_payload["metadata"]["counts"]["initiatives_with_tef_mapping"] == 58
    assert all(row["city"] == "Krakow" for row in source_rows)
    assert source_ids == mapped_ids

    initiatives_path = tmp_path / "krakow_ccc_source_truth.jsonl"
    _write_initiatives(
        initiatives_path,
        [_source_truth_record(row) for row in source_rows],
    )
    catalog = TefCatalog(Path("tef_mapping"))
    _patch_source_truth_agents(
        monkeypatch,
        catalog,
        {row["record_id"]: row for row in mapped_rows},
    )
    config = build_test_app_config(tef_mapper_overrides={"max_workers": 1})

    result = tef_agent.map_initiatives_to_tef(
        config=config,
        api_key="test",
        tef_catalog_dir=Path("tef_mapping"),
        output_root=tmp_path / "output",
        initiatives_jsonl=initiatives_path,
        run_id="krakow_ccc_source_truth_tef_test",
    )

    final_rows = [
        json.loads(line)
        for line in (Path(result.output_dir) / "05_final_mappings" / "final_mappings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected_counts = mapped_payload["metadata"]["counts"]
    expected_targets = sorted(
        (
            row["record_id"],
            mapping["target_type"],
            mapping["target_id"],
            mapping["target_path"],
            mapping["is_primary"],
        )
        for row in mapped_rows
        for mapping in row["tef_mappings"]
    )
    actual_targets = sorted(
        (
            row["initiative_record_id"],
            row["target_type"],
            row["target_id"],
            row["target_path"],
            row["is_primary"],
        )
        for row in final_rows
    )

    assert result.initiatives_count == expected_counts["initiatives"]
    assert (
        result.mapped_initiatives_count
        == expected_counts["initiatives_with_tef_mapping"]
    )
    assert result.final_mappings_count == expected_counts["final_mapping_rows"]
    assert (
        sum(1 for row in final_rows if row["is_primary"])
        == expected_counts["primary_mapping_rows"]
    )
    assert all(row["target_id"] for row in final_rows)
    assert actual_targets == expected_targets


def test_mapper_stops_at_subcategory_when_branch_has_no_transitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-transition leaf category should become a subcategory final mapping."""
    calls: list[tuple[str, dict[str, object]]] = []
    _patch_fake_agents(monkeypatch, calls, no_transition_leaf_branch=True)
    initiatives_path = tmp_path / "initiatives.jsonl"
    _write_initiatives(initiatives_path, [_record(record_id="krakow:soda_ash")])
    config = build_test_app_config(tef_mapper_overrides={"max_workers": 1})

    result = tef_agent.map_initiatives_to_tef(
        config=config,
        api_key="test",
        tef_catalog_dir=Path("tef_mapping"),
        output_root=tmp_path / "output",
        initiatives_jsonl=initiatives_path,
        run_id="subcategory_test",
    )

    final_rows = [
        json.loads(line)
        for line in (Path(result.output_dir) / "05_final_mappings" / "final_mappings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert final_rows[0]["target_type"] == "subcategory"
    assert final_rows[0]["target_id"] == "2-industry/2a-minerals/2a5-soda-ash"
    assert final_rows[0]["target_path"] == "2-industry/2a-minerals/2a5-soda-ash"
    assert final_rows[0]["needs_review"] is True
    assert "transition" not in [stage for stage, _payload in calls]
    subsector_payloads = [payload for stage, payload in calls if stage == "subsector"]
    assert len(subsector_payloads) == 2


def test_mapper_falls_back_to_subcategory_when_transition_mapper_finds_no_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transition pass with no exact TE match should still emit a TEF target."""
    calls: list[tuple[str, dict[str, object]]] = []
    _patch_fake_agents(monkeypatch, calls, no_transition_matches=True)
    initiatives_path = tmp_path / "initiatives.jsonl"
    _write_initiatives(initiatives_path, [_record(record_id="krakow:no_match")])
    config = build_test_app_config(tef_mapper_overrides={"max_workers": 1})

    result = tef_agent.map_initiatives_to_tef(
        config=config,
        api_key="test",
        tef_catalog_dir=Path("tef_mapping"),
        output_root=tmp_path / "output",
        initiatives_jsonl=initiatives_path,
        run_id="no_transition_match_test",
    )

    output_dir = Path(result.output_dir)
    final_rows = [
        json.loads(line)
        for line in (output_dir / "05_final_mappings" / "final_mappings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    review_rows = [
        json.loads(line)
        for line in (output_dir / "06_review" / "review_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert result.mapped_initiatives_count == 1
    assert final_rows[0]["target_type"] == "subcategory"
    assert final_rows[0]["target_id"] == "5-energy/5a-energy-supply/5a2-heat"
    assert final_rows[0]["target_path"] == "5-energy/5a-energy-supply/5a2-heat"
    assert final_rows[0]["needs_review"] is True
    assert "transition" in [stage for stage, _payload in calls]
    assert any(row["review_type"] == "no_transition_match" for row in review_rows)


def test_mapper_routes_to_child_category_before_transition_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Categories with children should route deeper before TE candidate selection."""
    calls: list[tuple[str, dict[str, object]]] = []
    _patch_fake_agents(monkeypatch, calls, single_family_hvac_branch=True)
    initiatives_path = tmp_path / "initiatives.jsonl"
    _write_initiatives(
        initiatives_path,
        [
            _record(
                record_id="krakow:bic_1",
                initiative_name="Single-family thermal modernisation programme",
            )
        ],
    )
    config = build_test_app_config(tef_mapper_overrides={"max_workers": 1})

    result = tef_agent.map_initiatives_to_tef(
        config=config,
        api_key="test",
        tef_catalog_dir=Path("tef_mapping"),
        output_root=tmp_path / "output",
        initiatives_jsonl=initiatives_path,
        run_id="child_before_transition_test",
    )

    output_dir = Path(result.output_dir)
    final_rows = [
        json.loads(line)
        for line in (output_dir / "05_final_mappings" / "final_mappings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    subsector_payloads = [payload for stage, payload in calls if stage == "subsector"]
    transition_payload = next(payload for stage, payload in calls if stage == "transition")
    candidate_tef_ids = {
        item["tef_id"] for item in transition_payload["candidate_transition_elements"]
    }

    assert len(subsector_payloads) == 3
    assert transition_payload["selected_category"]["path"] == (
        "4-buildings/4a-residential/4a1-hvac/4a1b-single-family-hvac"
    )
    assert "energy_efficient_new_housing" not in candidate_tef_ids
    assert (
        "energy_efficient_single_family_residential_buildings_retrofitting"
        in candidate_tef_ids
    )
    assert final_rows[0]["target_type"] == "transition_element"
    assert (
        final_rows[0]["target_id"]
        == "energy_efficient_single_family_residential_buildings_retrofitting"
    )


def test_mapper_normalizes_descendant_subsector_path_to_direct_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned descendant subsector paths should not break the staged mapper."""
    calls: list[tuple[str, dict[str, object]]] = []
    _patch_fake_agents(monkeypatch, calls, descendant_first_subsector_selection=True)
    initiatives_path = tmp_path / "initiatives.jsonl"
    _write_initiatives(initiatives_path, [_record(record_id="krakow:descendant")])
    config = build_test_app_config(tef_mapper_overrides={"max_workers": 1})

    result = tef_agent.map_initiatives_to_tef(
        config=config,
        api_key="test",
        tef_catalog_dir=Path("tef_mapping"),
        output_root=tmp_path / "output",
        initiatives_jsonl=initiatives_path,
        run_id="descendant_test",
    )

    output_dir = Path(result.output_dir)
    subsector_rows = [
        json.loads(line)
        for line in (output_dir / "03_subsector_routes" / "subsector_routes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    review_rows = [
        json.loads(line)
        for line in (output_dir / "06_review" / "review_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    final_rows = [
        json.loads(line)
        for line in (output_dir / "05_final_mappings" / "final_mappings.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert subsector_rows[0]["route"]["selected_path"] == "5-energy/5a-energy-supply"
    assert subsector_rows[0]["route"]["alternatives"][0]["path"] == "5-energy/5a-energy-supply"
    assert final_rows[0]["target_id"] == "district_heating_heat_pumps"
    assert any(row["review_type"] == "subsector_route_path_normalized" for row in review_rows)


def test_mapper_has_no_backend_ambiguity_prompt() -> None:
    """The backend mapper should not include a Pass 4 ambiguity-verifier prompt."""
    assert tef_agent.SECTOR_PROMPT.name == "tef_mapper_sector_router_system.md"
    assert tef_agent.SUBSECTOR_PROMPT.name == "tef_mapper_subsector_router_system.md"
    assert tef_agent.TRANSITION_PROMPT.name == "tef_mapper_transition_mapper_system.md"
    assert not Path("backend/prompts/tef_mapper_ambiguity_verifier_system.md").exists()


def test_mapper_prefers_extraction_record_sidecar(tmp_path: Path) -> None:
    """Extraction-run discovery should use pipeline records, not canonical v1 rows."""
    deduped_dir = tmp_path / "03_deduped"
    deduped_dir.mkdir(parents=True)
    canonical_path = deduped_dir / "initiatives.jsonl"
    records_path = deduped_dir / "initiative_records.jsonl"
    canonical_path.write_text("{}\n", encoding="utf-8")
    records_path.write_text("{}\n", encoding="utf-8")

    assert tef_agent._resolve_initiatives_path(tmp_path, None) == records_path


def test_numeric_rollup_uses_clean_initiative_numbers_only() -> None:
    """Numeric facts should come from the clean v1 initiative object, not sidecar fields."""
    record = _record()
    record.number_context["capacity_mw"] = 999
    mapping = TefFinalMappingRecord(
        initiative_record_id=record.record_id,
        city=record.initiative.city,
        source_document=record.source_document,
        document_local_code=record.document_local_code,
        initiative_name=record.initiative.initiative_name,
        target_type="transition_element",
        target_id="district_heating_heat_pumps",
        target_path="5-energy/5a-energy-supply/5a2-heat",
        confidence=0.91,
        is_primary=True,
        needs_review=False,
        rationale="Heat pump capacity maps to district heating heat pumps.",
        sector_route={},
        subsector_routes=[],
        mapper_version="test",
        tef_source_version="test",
        extraction_run_id="extract_test",
    )

    facts = build_numeric_facts(
        run_id="tef_test",
        extraction_run_id="extract_test",
        initiative_records=[record],
        final_mappings=[mapping],
    )

    values_by_key = {fact.number_key_raw: fact.value_number for fact in facts}
    assert values_by_key["capacity_mw"] == 1
    assert 999 not in values_by_key.values()
    capacity_fact = next(fact for fact in facts if fact.number_key_raw == "capacity_mw")
    assert capacity_fact.fact_id == (
        "krakow:krakow:bic_7:planned:capacity_mw:"
        "transition_element:district_heating_heat_pumps"
    )
    assert capacity_fact.include_in_default_rollup is True


def test_prompt_contracts_match_stage_models() -> None:
    """Prompt tool names and output fields should match runtime stage schemas."""
    sector_prompt = tef_agent.SECTOR_PROMPT.read_text(encoding="utf-8")
    subsector_prompt = tef_agent.SUBSECTOR_PROMPT.read_text(encoding="utf-8")
    transition_prompt = tef_agent.TRANSITION_PROMPT.read_text(encoding="utf-8")

    assert "submit_tef_sector_route" in sector_prompt
    for field_name in TefSectorRoute.model_fields:
        assert f"`{field_name}`" in sector_prompt

    assert "submit_tef_subsector_route" in subsector_prompt
    for field_name in TefSubsectorRoute.model_fields:
        assert f"`{field_name}`" in subsector_prompt
    assert "main causal shift" in subsector_prompt
    assert "supporting component" in subsector_prompt
    assert "overall intervention and expected emissions impact" in subsector_prompt

    assert "submit_tef_transition_mapping" in transition_prompt
    for field_name in TefTransitionMapping.model_fields:
        assert f"`{field_name}`" in transition_prompt
    assert "main causal shift" in transition_prompt
    assert "supporting component" in transition_prompt
    assert "overall intervention and expected emissions impact" in transition_prompt
