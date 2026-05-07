"""
Brief: Apply the 2026-04-23 Poland TEF audit fixes to a completed TEF mapping run.

Inputs:
- CLI args:
  - `--source-tef-run-dir`: TEF mapping run directory to repair. It must contain
    `summary.json`, `00_source/source_manifest.json`, `01_inputs/initiatives.jsonl`,
    and `05_final_mappings/final_mappings.jsonl`.
  - `--extraction-run-dir`: Corrected extraction run directory containing
    `03_deduped/initiative_records.jsonl`.
  - `--output-tef-run-dir`: Destination directory for the corrected TEF run copy.
  - `--overwrite`: Remove the destination directory first if it already exists.
- Files/paths:
  - Reads the corrected extraction initiative records and the source TEF final mappings.
  - Reads the local TEF catalog under `tef_mapping/catalog` to validate audit-fix targets.
  - Copies the full source TEF run directory before overwriting corrected artifacts.
- Env vars: none required.

Outputs:
- A corrected copy of the TEF run directory at `--output-tef-run-dir`
- Updated `00_source/source_manifest.json`
- Updated `01_inputs/initiatives.jsonl`
- Updated `05_final_mappings/final_mappings.jsonl`
- Updated `06_review/review_items.jsonl`
- Regenerated `07_numeric_facts/initiative_numeric_facts.jsonl`
- Regenerated `08_tef_groups/{tef_grouped_initiatives.jsonl,tef_metric_rollups.json}`
- Updated `summary.json`
- `audit_fix_manifest.json` describing the applied fixes

Usage (from project root):
- python -m backend.scripts.apply_polish_tef_audit_fixes --source-tef-run-dir output/tef_mapping/polish_cities_flow_test_20260423_001_tef_retry --extraction-run-dir output/initiative_extraction/polish_cities_flow_test_20260423_001_audit_fixed --output-tef-run-dir output/tef_mapping/polish_cities_flow_test_20260423_001_tef_retry_audit_fixed --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.modules.tef_mapper.numeric_rollup import rollup_existing_tef_run
from backend.utils.logging_config import setup_logger


logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Apply the Poland TEF audit fixes to a completed TEF mapping run."
    )
    parser.add_argument(
        "--source-tef-run-dir",
        type=Path,
        required=True,
        help="Source TEF mapping run directory containing summary.json and staged artifacts.",
    )
    parser.add_argument(
        "--extraction-run-dir",
        type=Path,
        required=True,
        help="Corrected extraction run directory containing 03_deduped/initiative_records.jsonl.",
    )
    parser.add_argument(
        "--output-tef-run-dir",
        type=Path,
        required=True,
        help="Destination directory for the corrected TEF run copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the destination directory first if it already exists.",
    )
    return parser.parse_args()


def load_json(path: Path) -> JsonDict:
    """Load one JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: JsonDict) -> None:
    """Write one JSON object with stable formatting."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load JSONL rows."""
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    """Write JSONL rows with UTF-8 encoding and LF newlines."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_catalog_paths(catalog_root: Path) -> tuple[dict[str, str], set[str], dict[str, list[str]]]:
    """Load TEF transition ids, category paths, and category children."""
    transition_rows = load_json(catalog_root / "transition_elements.json")
    transition_paths = {row["tef_id"]: row["path"] for row in transition_rows}

    category_paths: set[str] = set()
    for filename in ("sectors.json", "subcategories.json", "subsubcategories.json"):
        for row in load_json(catalog_root / filename):
            category_paths.add(row["path"])

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for path in sorted(category_paths):
        parts = path.split("/")
        if len(parts) == 1:
            continue
        parent = "/".join(parts[:-1])
        children_by_parent[parent].append(path)
    for parent, children in children_by_parent.items():
        children.sort()
    return transition_paths, category_paths, dict(children_by_parent)


def build_city_fixes() -> dict[str, JsonDict]:
    """Return the city-scoped TEF repair manifest derived from the audit."""
    return {
        "Krakow": {
            "remove_all": {
                "krakow:krakow:title_cfafa67436d1",
                "krakow:krakow:title_03eef5cf7e2b",
                "krakow:krakow:title_2dd032e7f95e",
                "krakow:krakow:title_87636680a70e",
                "krakow:krakow:title_a5c831c856d0",
                "krakow:krakow:title_100b4c550241",
                "krakow:krakow:title_e5a603156e28",
                "krakow:krakow:title_1b41cceb7dc7",
                "krakow:krakow:title_14270522aca5",
                "krakow:krakow:title_52ed6fe81e9c",
                "krakow:krakow:title_a13be8294337",
                "krakow:krakow:title_c4fda0bd7e0f",
                "krakow:krakow:title_8762d9078d15",
                "krakow:krakow:title_e6dc25e3753c",
                "krakow:krakow:title_2a0abaa24ba3",
                "krakow:krakow:title_ba5e90275ed7",
                "krakow:krakow:title_b23a43dbef5d",
                "krakow:krakow:title_e47094bb5f69",
                "krakow:krakow:title_53ea3c650706",
            },
            "remove_rows": [],
            "rewrite_rows": [
                {
                    "initiative_record_id": "krakow:krakow:title_018a8764a4a8",
                    "match_target_id": "improved_load_factor_heavy_trucks",
                    "target_type": "subcategory",
                    "target_id": "1-transport/1b-freight/1b1-road/1b1c-other-road-freight",
                    "target_path": "1-transport/1b-freight/1b1-road/1b1c-other-road-freight",
                    "confidence": 0.74,
                    "needs_review": True,
                    "rationale": "Audit-fixed to broader road-freight subcategory because the source covers sustainable logistics across freight operations rather than a single heavy-truck transition.",
                },
                {
                    "initiative_record_id": "krakow:krakow:title_5c5262635b48",
                    "match_target_id": "increased_proportion_of_public_transport_by_trams_and_light_rail",
                    "target_type": "transition_element",
                    "target_id": "shift_to_electric_passenger_rail",
                    "target_path": "1-transport/1a-mobility/1a2-rail",
                    "confidence": 0.82,
                    "needs_review": True,
                    "rationale": "Audit-fixed to rail because the action builds tram infrastructure and is better represented as electric passenger rail than a road-light-duty proxy.",
                },
                {
                    "initiative_record_id": "krakow:krakow:title_0fa8b2619de2",
                    "match_target_id": "6-waste/6a-solids/6a1-solid-waste-disposal",
                    "target_type": "transition_element",
                    "target_id": "increased_recycling",
                    "target_path": "6-waste/6a-solids/6a1-solid-waste-disposal",
                    "confidence": 0.71,
                    "needs_review": True,
                    "rationale": "Audit-fixed to increased recycling because the circular-economy action is broader than disposal and is more defensibly anchored in material recovery.",
                },
            ],
            "add_rows": [
                {
                    "initiative_record_id": "krakow:krakow:title_e5681f7176f8",
                    "target_type": "transition_element",
                    "target_id": "energy_efficient_street_lighting",
                    "target_path": "4-buildings/4b-non-residential/4b2-other-energy-usage/4b2a-lighting",
                    "confidence": 0.9,
                    "needs_review": False,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for LED road-lighting modernization with control systems, which directly matches energy-efficient street lighting.",
                },
                {
                    "initiative_record_id": "krakow:krakow:title_d264d40a457b",
                    "target_type": "transition_element",
                    "target_id": "shift_to_electric_passenger_rail",
                    "target_path": "1-transport/1a-mobility/1a2-rail",
                    "confidence": 0.81,
                    "needs_review": True,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for the Krakow premetro because the initiative creates fast urban rail capacity and is best represented as electric passenger rail.",
                },
            ],
            "abstain": {
                "krakow:krakow:title_cfafa67436d1": "Governance / participation, not a TEF intervention.",
                "krakow:krakow:title_03eef5cf7e2b": "Education / outreach, not a TEF intervention.",
                "krakow:krakow:title_2dd032e7f95e": "Education / outreach, not a TEF intervention.",
                "krakow:krakow:title_87636680a70e": "Advisory governance, not a TEF intervention.",
                "krakow:krakow:title_a5c831c856d0": "Voluntary partnership and reporting framework.",
                "krakow:krakow:title_100b4c550241": "Cooperation framework, not a TEF intervention.",
                "krakow:krakow:title_e5a603156e28": "Cooperation programme, not a TEF intervention.",
                "krakow:krakow:title_1b41cceb7dc7": "Umbrella strategy record.",
                "krakow:krakow:title_14270522aca5": "Cross-sector umbrella plan.",
                "krakow:krakow:title_52ed6fe81e9c": "Cross-sector umbrella plan.",
                "krakow:krakow:title_a13be8294337": "General city strategy, not a TEF intervention.",
                "krakow:krakow:title_c4fda0bd7e0f": "Adaptation programme, not a TEF mitigation intervention.",
                "krakow:krakow:title_8762d9078d15": "Umbrella environmental programme.",
                "krakow:krakow:title_e6dc25e3753c": "Regional umbrella plan, not a single Krakow intervention.",
                "krakow:krakow:title_2a0abaa24ba3": "False positive legislative recommendation.",
                "krakow:krakow:title_ba5e90275ed7": "Umbrella circular-economy strategy, not a single TEF intervention.",
                "krakow:krakow:title_b23a43dbef5d": "Innovation / replication project without a clean TEF fit.",
                "krakow:krakow:title_e47094bb5f69": "Under-specified extraction split; current TEF fit is unreliable.",
                "krakow:krakow:title_53ea3c650706": "Under-specified extraction split; current TEF fit is unreliable.",
            },
        },
        "Lodz": {
            "remove_all": {
                "lodz:lodz:title_06abfd9a0aef",
                "lodz:lodz:title_1dbd6d0cedb5",
                "lodz:lodz:title_c1685b0de1ac",
                "lodz:lodz:title_f2ab52f1bec3",
                "lodz:lodz:title_fe32241dd1ea",
                "lodz:lodz:title_746b1d9536c6",
                "lodz:lodz:title_b93044ca1cdb",
                "lodz:lodz:title_9a490ecf07a7",
                "lodz:lodz:title_d1f54552f9e5",
                "lodz:lodz:title_d6ba2b140b4d",
                "lodz:lodz:title_38b62754cb43",
                "lodz:lodz:title_c95a6d8b0d39",
                "lodz:lodz:title_0f6fb8583884",
                "lodz:lodz:title_113aadec4cfa",
                "lodz:lodz:title_5375bd1b68bf",
                "lodz:lodz:title_58534ebc2312",
                "lodz:lodz:title_7424e17d051f",
                "lodz:lodz:title_79167b36f827",
                "lodz:lodz:title_92ca4b799477",
            },
            "remove_rows": [
                {
                    "initiative_record_id": "lodz:lodz:title_23487b92b78d",
                    "target_path": "6-waste/6a-solids/6a1-solid-waste-disposal",
                    "target_id": "shift_to_composting_of_organic_waste",
                },
                {
                    "initiative_record_id": "lodz:lodz:title_33f5907c40b2",
                    "target_path": "5-energy/5a-energy-supply/5a1-electricity",
                    "target_id": "improved_grid_electricity",
                },
                {
                    "initiative_record_id": "lodz:lodz:title_6f7c10be90c5",
                    "target_path": "5-energy/5a-energy-supply/5a1-electricity",
                    "target_id": "solar_parks",
                },
                {
                    "initiative_record_id": "lodz:lodz:title_da61b3b253fd",
                    "target_path": "4-buildings/4b-non-residential/4b1-hvac/4b1a-commercial-hvac",
                    "target_id": "heat_pumps_in_non_residential_commercial_buildings",
                },
            ],
            "rewrite_rows": [
                {
                    "initiative_record_id": "lodz:lodz:title_c3e4c09ffaec",
                    "match_target_path": "4-buildings/4a-residential/4a2-other-energy-usage/4a2d-unregulated",
                    "target_type": "transition_element",
                    "target_id": "solar_rooftops",
                    "target_path": "5-energy/5a-energy-supply/5a1-electricity",
                    "confidence": 0.88,
                    "needs_review": False,
                    "rationale": "Audit-fixed to rooftop solar because the action installs photovoltaic modules on residential buildings and is better represented as electricity generation.",
                },
                {
                    "initiative_record_id": "lodz:lodz:title_12a5445d8ffb",
                    "match_target_path": "5-energy/5d-other/5d1-non-specified-energy-use",
                    "target_type": "transition_element",
                    "target_id": "energy_efficient_street_lighting",
                    "target_path": "4-buildings/4b-non-residential/4b2-other-energy-usage/4b2a-lighting",
                    "confidence": 0.9,
                    "needs_review": False,
                    "rationale": "Audit-fixed to energy-efficient street lighting because the initiative explicitly modernizes city lighting to save electricity.",
                },
            ],
            "add_rows": [
                {
                    "initiative_record_id": "lodz:lodz:title_c6bbb0a654bc",
                    "target_type": "subcategory",
                    "target_id": "5-energy/5a-energy-supply/5a2-heat",
                    "target_path": "5-energy/5a-energy-supply/5a2-heat",
                    "confidence": 0.62,
                    "needs_review": True,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for the geothermal district-heating project at the heat-supply subcategory because the source supports decarbonized heat but not a narrower named TE.",
                },
            ],
            "abstain": {
                "lodz:lodz:title_06abfd9a0aef": "Community meeting / workshop spaces, not a TEF intervention.",
                "lodz:lodz:title_1dbd6d0cedb5": "Public consultation platform, not a TEF intervention.",
                "lodz:lodz:title_c1685b0de1ac": "Participatory budgeting, not a TEF intervention.",
                "lodz:lodz:title_f2ab52f1bec3": "Deliberative democracy mechanism, not a TEF intervention.",
                "lodz:lodz:title_fe32241dd1ea": "Consultation / NGO support office, not a TEF intervention.",
                "lodz:lodz:title_746b1d9536c6": "Information website, not a TEF intervention.",
                "lodz:lodz:title_b93044ca1cdb": "Consultation meetings, not a TEF intervention.",
                "lodz:lodz:title_9a490ecf07a7": "Advisory support point, not the retrofit intervention itself.",
                "lodz:lodz:title_d1f54552f9e5": "Workshop-note bundle, not a stable TEF initiative.",
                "lodz:lodz:title_d6ba2b140b4d": "Workshop-note bundle, not a stable TEF initiative.",
                "lodz:lodz:title_38b62754cb43": "Communication principle, not a TEF initiative.",
                "lodz:lodz:title_c95a6d8b0d39": "Conference format, not a TEF initiative.",
                "lodz:lodz:title_0f6fb8583884": "Communications tool, not a TEF initiative.",
                "lodz:lodz:title_113aadec4cfa": "Calculator / support tool, not a TEF initiative.",
                "lodz:lodz:title_5375bd1b68bf": "Workshop-note item, not a TEF initiative.",
                "lodz:lodz:title_58534ebc2312": "Workshop-note item, not a TEF initiative.",
                "lodz:lodz:title_7424e17d051f": "Workshop-note item, not a TEF initiative.",
                "lodz:lodz:title_79167b36f827": "Workshop-note item, not a TEF initiative.",
                "lodz:lodz:title_92ca4b799477": "Workshop-note item, not a TEF initiative.",
                "lodz:lodz:title_94d44623a39b": "Governance / planning initiative, not a direct TEF intervention.",
                "lodz:lodz:title_da7b71d5834a": "Monitoring / inventory system, not a direct TEF intervention.",
                "lodz:lodz:title_28a6109a7785": "Climate-budget tagging, not a direct TEF intervention.",
                "lodz:lodz:title_f8312bf86ec3": "Investment concept and audits, not a direct TEF intervention.",
            },
        },
        "Rzeszow": {
            "remove_all": {
                "rzeszow:rzeszow:title_17e350b60697",
                "rzeszow:rzeszow:title_3d8de539b280",
                "rzeszow:rzeszow:title_6e97c368c11a",
                "rzeszow:rzeszow:title_80ba03f69d2e",
                "rzeszow:rzeszow:title_816ad29d1753",
                "rzeszow:rzeszow:title_a71c6d461bdb",
                "rzeszow:rzeszow:title_d184313985fe",
                "rzeszow:rzeszow:title_dcee1fbb6c02",
                "rzeszow:rzeszow:title_e92f44f4741c",
                "rzeszow:rzeszow:title_c9f675409da9",
                "rzeszow:rzeszow:title_22d37aeb4cb3",
                "rzeszow:rzeszow:title_3c61d0c6c90e",
                "rzeszow:rzeszow:title_7bed32c087f7",
                "rzeszow:rzeszow:title_9bfcfe3bdab6",
                "rzeszow:rzeszow:title_9d0e64bcde1e",
                "rzeszow:rzeszow:title_acac80a33c35",
                "rzeszow:rzeszow:title_ff315201e140",
                "rzeszow:rzeszow:title_0a88398b7329",
                "rzeszow:rzeszow:title_62425ea1394f",
                "rzeszow:rzeszow:title_a3d9a70168b1",
                "rzeszow:rzeszow:title_1427914835e9",
                "rzeszow:rzeszow:title_cd80cd614adc",
            },
            "remove_rows": [
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_049ed4fbcfaa",
                    "target_path": "1-transport/1a-mobility/1a1-road/1a1a-light-duty-vehicles",
                    "target_id": "increased_proportion_of_public_transport_by_trams_and_light_rail",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_f68699e7bde9",
                    "target_path": "4-buildings/4a-residential/4a1-hvac/4a1a-multi-family-hvac",
                    "target_id": "shift_to_solar_thermal_in_multi_family_buildings",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_4a1ac4597c8b",
                    "target_path": "6-waste/6a-solids/6a1-solid-waste-disposal",
                    "target_id": "shift_to_composting_of_organic_waste",
                },
            ],
            "rewrite_rows": [
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_0e4224cd1034",
                    "match_target_id": "heat_pumps_in_multi_family_residential_buildings",
                    "target_type": "transition_element",
                    "target_id": "heat_pumps_in_single_family_residential_buildings",
                    "target_path": "4-buildings/4a-residential/4a1-hvac/4a1b-single-family-hvac",
                    "confidence": 0.9,
                    "needs_review": True,
                    "rationale": "Audit-fixed to single-family heat pumps because the corrected extraction record is explicitly about single-family residential retrofits.",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_0e4224cd1034",
                    "match_target_id": "energy_efficient_multi_family_residential_buildings_retrofitting",
                    "target_type": "transition_element",
                    "target_id": "energy_efficient_single_family_residential_buildings_retrofitting",
                    "target_path": "4-buildings/4a-residential/4a1-hvac/4a1b-single-family-hvac",
                    "confidence": 0.78,
                    "needs_review": True,
                    "rationale": "Audit-fixed to single-family retrofitting because the corrected extraction record is explicitly about single-family residential retrofits.",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_2c50e5154e22",
                    "match_target_path": "1-transport/1a-mobility/1a1-road/1a1a-light-duty-vehicles",
                    "target_type": "subcategory",
                    "target_id": "1-transport/1a-mobility/1a1-road/1a1c-buses",
                    "target_path": "1-transport/1a-mobility/1a1-road/1a1c-buses",
                    "confidence": 0.68,
                    "needs_review": True,
                    "rationale": "Audit-fixed to the buses subcategory because the action expands public transport service and interchange infrastructure rather than a light-duty modal proxy.",
                },
            ],
            "add_rows": [
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_34d776da19d3",
                    "target_type": "subcategory",
                    "target_id": "3-afolu/3b-land/3b5-settlements",
                    "target_path": "3-afolu/3b-land/3b5-settlements",
                    "confidence": 0.63,
                    "needs_review": True,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for stormwater retention on development sites at the settlements land-use subcategory because the intervention changes site-level urban land design.",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_93a16cf73ba6",
                    "target_type": "subcategory",
                    "target_id": "3-afolu/3b-land/3b5-settlements",
                    "target_path": "3-afolu/3b-land/3b5-settlements",
                    "confidence": 0.64,
                    "needs_review": True,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for depaving at the settlements land-use subcategory because the action redesigns sealed urban land surfaces.",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_ad529f49cf75",
                    "target_type": "transition_element",
                    "target_id": "energy_efficient_non_residential_public_buildings_retrofitting",
                    "target_path": "4-buildings/4b-non-residential/4b1-hvac/4b1b-institutional-hvac",
                    "confidence": 0.83,
                    "needs_review": True,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for eco-friendly municipal-unit upgrades because the source supports public-building retrofits, thermal modernization, and smart energy management.",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_ad529f49cf75",
                    "target_type": "transition_element",
                    "target_id": "heat_pumps_in_non_residential_public_buildings",
                    "target_path": "4-buildings/4b-non-residential/4b1-hvac/4b1b-institutional-hvac",
                    "confidence": 0.71,
                    "needs_review": True,
                    "is_primary": False,
                    "rationale": "Audit-added secondary TEF mapping because the source explicitly includes heat pumps in municipal units and companies.",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_330e862ef94f",
                    "target_type": "transition_element",
                    "target_id": "increased_recycling",
                    "target_path": "6-waste/6a-solids/6a1-solid-waste-disposal",
                    "confidence": 0.9,
                    "needs_review": False,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for selective collection and recycling infrastructure because the initiative directly targets higher recycling and recovery rates.",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_f67022e9febb",
                    "target_type": "transition_element",
                    "target_id": "shift_to_composting_of_organic_waste",
                    "target_path": "6-waste/6a-solids/6a1-solid-waste-disposal",
                    "confidence": 0.77,
                    "needs_review": True,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for waste-disposal and management investments because the source includes organic-waste treatment and associated waste-system modernization.",
                },
                {
                    "initiative_record_id": "rzeszow:rzeszow:title_f67022e9febb",
                    "target_type": "transition_element",
                    "target_id": "increased_recycling",
                    "target_path": "6-waste/6a-solids/6a1-solid-waste-disposal",
                    "confidence": 0.69,
                    "needs_review": True,
                    "is_primary": False,
                    "rationale": "Audit-added secondary TEF mapping because the waste-management investments also support recovery and recycling improvements.",
                },
            ],
            "abstain": {
                "rzeszow:rzeszow:title_c9f675409da9": "Participatory budgeting, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_22d37aeb4cb3": "Advisory body, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_3c61d0c6c90e": "Digital reporting / information platform, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_7bed32c087f7": "Deliberative panel, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_9bfcfe3bdab6": "Civic collaboration space, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_9d0e64bcde1e": "Partnership framework, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_acac80a33c35": "Education activity, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_ff315201e140": "Organizational setup, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_0a88398b7329": "Advisory council, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_62425ea1394f": "Advisory councils, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_a3d9a70168b1": "Cooperation / advisory programme, not a TEF sector intervention.",
                "rzeszow:rzeszow:title_1427914835e9": "Cross-cutting digital / governance umbrella, not a concrete TEF intervention.",
                "rzeszow:rzeszow:title_cd80cd614adc": "Support / outreach programme, not a direct TEF intervention.",
            },
        },
        "Warszawa": {
            "remove_all": {
                "warszawa:warszawa:title_003a7a4082d0",
                "warszawa:warszawa:title_19c2f36cc621",
                "warszawa:warszawa:title_1b2b3bf81306",
                "warszawa:warszawa:title_35570c6efd29",
                "warszawa:warszawa:title_43e492ea22c9",
                "warszawa:warszawa:title_492e9808a1c2",
                "warszawa:warszawa:title_4c156d945682",
                "warszawa:warszawa:title_541a31ca3ccf",
                "warszawa:warszawa:title_605e277bff4c",
                "warszawa:warszawa:title_708ba04573da",
                "warszawa:warszawa:title_75a474bea622",
                "warszawa:warszawa:title_7b300cf89f3e",
                "warszawa:warszawa:title_844ca5a30d18",
                "warszawa:warszawa:title_905dd8374711",
                "warszawa:warszawa:title_92a8482e8a3a",
                "warszawa:warszawa:title_9b9d8e4a79e9",
                "warszawa:warszawa:title_9bf80178644e",
                "warszawa:warszawa:title_9ce68c9aa373",
                "warszawa:warszawa:title_9ff3e7c18ebd",
                "warszawa:warszawa:title_a4047ba0e5d4",
                "warszawa:warszawa:title_bdcf8f713b5e",
                "warszawa:warszawa:title_bec43be989a2",
                "warszawa:warszawa:title_c3536f972074",
                "warszawa:warszawa:title_ca2d15a0de6d",
                "warszawa:warszawa:title_f472182e2752",
                "warszawa:warszawa:title_f594319c7d5e",
                "warszawa:warszawa:title_1ce38da0517b",
            },
            "remove_rows": [
                {
                    "initiative_record_id": "warszawa:warszawa:title_01481fbdf5ac",
                    "target_path": "1-transport/1a-mobility/1a1-road/1a1a-light-duty-vehicles",
                    "target_id": "improved_urban_planning",
                },
                {
                    "initiative_record_id": "warszawa:warszawa:title_a1d86b7b2266",
                    "target_path": "5-energy/5a-energy-supply/5a2-heat",
                    "target_id": "district_heating_heat_pumps",
                },
                {
                    "initiative_record_id": "warszawa:warszawa:title_a1e6ceb1edb2",
                    "target_path": "5-energy/5a-energy-supply/5a2-heat",
                    "target_id": "shift_to_residual_heat_in_district_heating",
                },
            ],
            "rewrite_rows": [],
            "add_rows": [],
            "abstain": {
                "warszawa:warszawa:title_003a7a4082d0": "Appendix adaptation / crisis-management training.",
                "warszawa:warszawa:title_19c2f36cc621": "Appendix resident protection / public safety.",
                "warszawa:warszawa:title_1b2b3bf81306": "Appendix education / awareness.",
                "warszawa:warszawa:title_1ce38da0517b": "Part C social-innovation / networking intervention.",
                "warszawa:warszawa:title_35570c6efd29": "Appendix generic adaptation bullet, not a TEF-ready standalone initiative.",
                "warszawa:warszawa:title_43e492ea22c9": "Appendix education / awareness.",
                "warszawa:warszawa:title_492e9808a1c2": "Appendix adaptation governance.",
                "warszawa:warszawa:title_4c156d945682": "Appendix governance / process enabler.",
                "warszawa:warszawa:title_541a31ca3ccf": "Appendix communication / warning-system enabler.",
                "warszawa:warszawa:title_605e277bff4c": "Appendix communication / awareness.",
                "warszawa:warszawa:title_708ba04573da": "Part C management / coordination intervention.",
                "warszawa:warszawa:title_75a474bea622": "Appendix adaptation governance.",
                "warszawa:warszawa:title_7b300cf89f3e": "Appendix monitoring / coordination.",
                "warszawa:warszawa:title_844ca5a30d18": "Part C stakeholder-coordination intervention.",
                "warszawa:warszawa:title_905dd8374711": "Appendix education / awareness.",
                "warszawa:warszawa:title_92a8482e8a3a": "Appendix education / coordination.",
                "warszawa:warszawa:title_9b9d8e4a79e9": "Appendix risk-management / public safety.",
                "warszawa:warszawa:title_9bf80178644e": "Appendix training / awareness.",
                "warszawa:warszawa:title_9ce68c9aa373": "Appendix warning / information.",
                "warszawa:warszawa:title_9ff3e7c18ebd": "Appendix warning-system / coordination.",
                "warszawa:warszawa:title_a4047ba0e5d4": "Part C intergovernmental coordination intervention.",
                "warszawa:warszawa:title_bdcf8f713b5e": "Part C collaborative-method intervention.",
                "warszawa:warszawa:title_bec43be989a2": "Appendix adaptation-governance / planning.",
                "warszawa:warszawa:title_c3536f972074": "Appendix social-infrastructure adaptation / public safety.",
                "warszawa:warszawa:title_ca2d15a0de6d": "Appendix coordination / knowledge exchange.",
                "warszawa:warszawa:title_f472182e2752": "Appendix broad resilience / utility continuity.",
                "warszawa:warszawa:title_f594319c7d5e": "Part C start-up acceleration intervention.",
            },
        },
        "Wroclaw": {
            "remove_all": {
                "wroclaw:wroclaw:title_2f3057a3b89d",
                "wroclaw:wroclaw:title_08f7937dcc18",
                "wroclaw:wroclaw:title_ed1bc058f436",
                "wroclaw:wroclaw:title_40f0e3e9e3b2",
                "wroclaw:wroclaw:title_0d208aff63b5",
            },
            "remove_rows": [
                {
                    "initiative_record_id": "wroclaw:wroclaw:title_192513aaf97e",
                    "target_path": "5-energy/5a-energy-supply/5a2-heat",
                    "target_id": "shift_to_heat_pumps_in_district_heating",
                },
                {
                    "initiative_record_id": "wroclaw:wroclaw:title_ddeab5c7c60c",
                    "target_path": "1-transport/1a-mobility/1a1-road/1a1a-light-duty-vehicles",
                    "target_id": "improved_urban_planning",
                },
                {
                    "initiative_record_id": "wroclaw:wroclaw:title_d2ca93edb6f3",
                    "target_path": "6-waste/6a-solids/6a1-solid-waste-disposal",
                    "target_id": "shift_to_composting_of_organic_waste",
                },
            ],
            "rewrite_rows": [
                {
                    "initiative_record_id": "wroclaw:wroclaw:title_f0fa092bad1e",
                    "match_target_id": "energy_efficient_multi_family_residential_buildings_retrofitting",
                    "target_type": "subcategory",
                    "target_id": "4-buildings",
                    "target_path": "4-buildings",
                    "confidence": 0.63,
                    "needs_review": True,
                    "rationale": "Audit-fixed to the broader buildings sector because the initiative spans residential, commercial, educational, cultural, social, administrative, and new buildings.",
                },
                {
                    "initiative_record_id": "wroclaw:wroclaw:title_da2d58eba552",
                    "match_target_id": "energy_efficient_multi_family_residential_buildings_retrofitting",
                    "target_type": "subcategory",
                    "target_id": "4-buildings/4a-residential/4a1-hvac",
                    "target_path": "4-buildings/4a-residential/4a1-hvac",
                    "confidence": 0.64,
                    "needs_review": True,
                    "rationale": "Audit-fixed to the broader residential HVAC branch because the initiative is enabling advice and promotion rather than a specific multi-family retrofit action.",
                },
                {
                    "initiative_record_id": "wroclaw:wroclaw:title_4bd29fca17ee",
                    "match_target_path": "5-energy/5a-energy-supply/5a1-electricity",
                    "target_type": "subcategory",
                    "target_id": "5-energy/5a-energy-supply",
                    "target_path": "5-energy/5a-energy-supply",
                    "confidence": 0.61,
                    "needs_review": True,
                    "rationale": "Audit-fixed to the broader energy-supply subcategory because the initiative jointly covers electricity and heat decarbonization through education, planning, and promotion.",
                },
                {
                    "initiative_record_id": "wroclaw:wroclaw:title_d2ca93edb6f3",
                    "match_target_id": "increased_recycling",
                    "target_type": "subcategory",
                    "target_id": "6-waste",
                    "target_path": "6-waste",
                    "confidence": 0.58,
                    "needs_review": True,
                    "rationale": "Audit-fixed to the broader waste sector because the initiative mixes waste, wastewater, energy-efficiency, and recovery measures that are too broad for a single waste leaf.",
                },
            ],
            "add_rows": [
                {
                    "initiative_record_id": "wroclaw:wroclaw:title_80e208cf64fe",
                    "target_type": "subcategory",
                    "target_id": "4-buildings/4a-residential/4a1-hvac",
                    "target_path": "4-buildings/4a-residential/4a1-hvac",
                    "confidence": 0.6,
                    "needs_review": True,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for the stove-replacement programme at the residential HVAC branch because the source supports individual heat-source replacement but not a narrower named TE with confidence.",
                },
                {
                    "initiative_record_id": "wroclaw:wroclaw:title_45b63b5175c5",
                    "target_type": "subcategory",
                    "target_id": "1-transport/1a-mobility/1a1-road/1a1a-light-duty-vehicles",
                    "target_path": "1-transport/1a-mobility/1a1-road/1a1a-light-duty-vehicles",
                    "confidence": 0.62,
                    "needs_review": True,
                    "is_primary": True,
                    "rationale": "Audit-added TEF mapping for the low-emission zone at the light-duty road subcategory because the initiative regulates car access and urban road mobility but does not cleanly match a named transition element.",
                },
            ],
            "abstain": {
                "wroclaw:wroclaw:title_2f3057a3b89d": "Education / information / organizational climate action, not a TEF intervention.",
                "wroclaw:wroclaw:title_08f7937dcc18": "Transport education / promotion / information, not a TEF intervention.",
                "wroclaw:wroclaw:title_ed1bc058f436": "Civic panel, not a TEF sector intervention.",
                "wroclaw:wroclaw:title_40f0e3e9e3b2": "NEEST governance / planning intervention, not a direct TEF intervention.",
                "wroclaw:wroclaw:title_0d208aff63b5": "Participatory budgeting, not a TEF sector intervention.",
            },
        },
    }


def build_remove_rules(city_fixes: dict[str, JsonDict]) -> tuple[set[str], set[tuple[str, str, str | None]], dict[str, str]]:
    """Build fast lookup sets for remove-all, remove-row, and abstain reasons."""
    remove_all: set[str] = set()
    remove_rows: set[tuple[str, str, str | None]] = set()
    abstain: dict[str, str] = {}
    for fix in city_fixes.values():
        remove_all.update(fix["remove_all"])
        for row in fix["remove_rows"]:
            remove_rows.add(
                (
                    row["initiative_record_id"],
                    row["target_path"],
                    row.get("target_id"),
                )
            )
        abstain.update(fix["abstain"])
    return remove_all, remove_rows, abstain


def build_rewrite_specs(city_fixes: dict[str, JsonDict]) -> dict[str, list[JsonDict]]:
    """Group rewrite specs by initiative id."""
    specs: dict[str, list[JsonDict]] = defaultdict(list)
    for fix in city_fixes.values():
        for row in fix["rewrite_rows"]:
            specs[row["initiative_record_id"]].append(row)
    return dict(specs)


def build_add_specs(city_fixes: dict[str, JsonDict]) -> dict[str, list[JsonDict]]:
    """Group add specs by initiative id."""
    specs: dict[str, list[JsonDict]] = defaultdict(list)
    for fix in city_fixes.values():
        for row in fix["add_rows"]:
            specs[row["initiative_record_id"]].append(row)
    return dict(specs)


def validate_fix_specs(
    *,
    city_fixes: dict[str, JsonDict],
    transition_paths: dict[str, str],
    category_paths: set[str],
) -> None:
    """Validate that every audit-fix target exists in the TEF catalog."""
    for fix in city_fixes.values():
        for spec in [*fix["rewrite_rows"], *fix["add_rows"]]:
            target_type = spec["target_type"]
            target_id = spec["target_id"]
            target_path = spec["target_path"]
            if target_type == "transition_element":
                expected_path = transition_paths.get(target_id)
                if expected_path != target_path:
                    raise ValueError(
                        f"Transition target mismatch for {spec['initiative_record_id']}: "
                        f"{target_id} -> {target_path}, catalog has {expected_path}."
                    )
            else:
                if target_id != target_path:
                    raise ValueError(
                        f"Subcategory target id must equal target path for {spec['initiative_record_id']}: "
                        f"{target_id} != {target_path}."
                    )
                if target_path not in category_paths:
                    raise ValueError(
                        f"Unknown TEF category path for {spec['initiative_record_id']}: {target_path}"
                    )


def refresh_mapping_record(
    mapping: JsonDict,
    record: JsonDict,
    extraction_run_id: str,
) -> JsonDict:
    """Refresh a TEF mapping row with corrected extraction-side metadata."""
    mapping["city"] = record["initiative"]["city"]
    mapping["source_document"] = record["source_document"]
    mapping["document_local_code"] = record.get("document_local_code")
    mapping["initiative_name"] = record["initiative"]["initiative_name"]
    mapping["source_quote"] = record.get("source_quote")
    mapping["extraction_run_id"] = extraction_run_id
    return mapping


def build_route_metadata(
    *,
    initiative_record_id: str,
    source_document: str,
    target_path: str,
    category_children: dict[str, list[str]],
    confidence: float,
    needs_review: bool,
    rationale: str,
) -> tuple[JsonDict, list[JsonDict]]:
    """Build lightweight route metadata consistent with the corrected target path."""
    parts = target_path.split("/")
    root = parts[0]
    sector_route = {
        "sector": root.split("-", maxsplit=1)[1] if "-" in root else root,
        "selected_path": root,
        "confidence": confidence,
        "needs_review": needs_review,
        "rationale": rationale,
        "alternatives": [],
    }

    subsector_routes: list[JsonDict] = []
    current = root
    for child in parts[1:]:
        selected_path = f"{current}/{child}"
        subsector_routes.append(
            {
                "initiative_record_id": initiative_record_id,
                "source_document": source_document,
                "parent_path": current,
                "candidate_paths": category_children.get(current, [selected_path]),
                "status": "success",
                "route": {
                    "selected_path": selected_path,
                    "confidence": confidence,
                    "needs_review": needs_review,
                    "rationale": rationale,
                    "alternatives": [],
                },
                "error": None,
            }
        )
        current = selected_path
    return sector_route, subsector_routes


def replace_target(
    *,
    row: JsonDict,
    record: JsonDict,
    spec: JsonDict,
    category_children: dict[str, list[str]],
    extraction_run_id: str,
) -> JsonDict:
    """Return a rewritten TEF mapping row using the audit-fixed target."""
    rewritten = refresh_mapping_record(copy.deepcopy(row), record, extraction_run_id)
    rewritten["target_type"] = spec["target_type"]
    rewritten["target_id"] = spec["target_id"]
    rewritten["target_path"] = spec["target_path"]
    rewritten["confidence"] = spec["confidence"]
    rewritten["needs_review"] = spec["needs_review"]
    rewritten["rationale"] = spec["rationale"]
    sector_route, subsector_routes = build_route_metadata(
        initiative_record_id=rewritten["initiative_record_id"],
        source_document=rewritten["source_document"],
        target_path=rewritten["target_path"],
        category_children=category_children,
        confidence=rewritten["confidence"],
        needs_review=rewritten["needs_review"],
        rationale=rewritten["rationale"],
    )
    rewritten["sector_route"] = sector_route
    rewritten["subsector_routes"] = subsector_routes
    return rewritten


def make_added_mapping(
    *,
    record: JsonDict,
    spec: JsonDict,
    mapper_version: str,
    tef_source_version: str,
    extraction_run_id: str,
    category_children: dict[str, list[str]],
) -> JsonDict:
    """Build a durable TEF final-mapping row for an audit-added initiative."""
    sector_route, subsector_routes = build_route_metadata(
        initiative_record_id=record["record_id"],
        source_document=record["source_document"],
        target_path=spec["target_path"],
        category_children=category_children,
        confidence=spec["confidence"],
        needs_review=spec["needs_review"],
        rationale=spec["rationale"],
    )
    return {
        "initiative_record_id": record["record_id"],
        "city": record["initiative"]["city"],
        "source_document": record["source_document"],
        "document_local_code": record.get("document_local_code"),
        "initiative_name": record["initiative"]["initiative_name"],
        "source_quote": record.get("source_quote"),
        "target_type": spec["target_type"],
        "target_id": spec["target_id"],
        "target_path": spec["target_path"],
        "confidence": spec["confidence"],
        "is_primary": spec["is_primary"],
        "needs_review": spec["needs_review"],
        "rationale": spec["rationale"],
        "sector_route": sector_route,
        "subsector_routes": subsector_routes,
        "mapper_version": mapper_version,
        "tef_source_version": tef_source_version,
        "extraction_run_id": extraction_run_id,
    }


def matches_rewrite_spec(row: JsonDict, spec: JsonDict) -> bool:
    """Return whether one mapping row matches one rewrite spec."""
    if row["initiative_record_id"] != spec["initiative_record_id"]:
        return False
    if spec.get("match_target_id") and row["target_id"] != spec["match_target_id"]:
        return False
    if spec.get("match_target_path") and row["target_path"] != spec["match_target_path"]:
        return False
    return True


def should_remove_row(
    row: JsonDict,
    *,
    valid_record_ids: set[str],
    remove_all: set[str],
    remove_rows: set[tuple[str, str, str | None]],
) -> bool:
    """Return whether the original row should be removed before rewrites/additions."""
    if row["initiative_record_id"] not in valid_record_ids:
        return True
    if row["initiative_record_id"] in remove_all:
        return True
    key = (row["initiative_record_id"], row["target_path"], row["target_id"])
    wildcard_key = (row["initiative_record_id"], row["target_path"], None)
    return key in remove_rows or wildcard_key in remove_rows


def build_review_items(
    *,
    corrected_records_by_id: dict[str, JsonDict],
    final_mappings: list[JsonDict],
    abstain_reasons: dict[str, str],
) -> list[JsonDict]:
    """Build compact manual-review rows for the corrected TEF run."""
    review_items: list[JsonDict] = []
    mapped_ids = {row["initiative_record_id"] for row in final_mappings}

    for record_id in sorted(mapped_ids):
        record = corrected_records_by_id[record_id]
        for flag in record.get("data_quality_flags", []):
            review_items.append(
                {
                    "review_type": "source_quality_flag",
                    "severity": "info",
                    "message": f"Initiative extraction has source quality flag: {flag}",
                    "initiative_record_id": record_id,
                    "source_document": record["source_document"],
                    "target_id": None,
                    "details": {"flag": flag},
                }
            )

    for row in final_mappings:
        if not row["needs_review"]:
            continue
        review_items.append(
            {
                "review_type": "audit_fixed_mapping_needs_review",
                "severity": "warning",
                "message": "Audit-fixed TEF mapping still needs manual review.",
                "initiative_record_id": row["initiative_record_id"],
                "source_document": row["source_document"],
                "target_id": row["target_id"],
                "details": {
                    "target_path": row["target_path"],
                    "is_primary": row["is_primary"],
                    "confidence": row["confidence"],
                },
            }
        )

    for record_id, reason in sorted(abstain_reasons.items()):
        record = corrected_records_by_id.get(record_id)
        if record is None:
            continue
        review_items.append(
            {
                "review_type": "audit_abstained_from_tef",
                "severity": "warning",
                "message": "Audit removed this initiative from TEF because no reliable TEF mapping was supported.",
                "initiative_record_id": record_id,
                "source_document": record["source_document"],
                "target_id": None,
                "details": {"reason": reason},
            }
        )

    return review_items


def count_by_city(
    rows: list[JsonDict],
    *,
    key: str,
    record_lookup: dict[str, JsonDict] | None = None,
) -> dict[str, int]:
    """Count rows by city from mapping rows or initiative ids."""
    counts: dict[str, int] = {}
    for row in rows:
        if key == "mapping":
            city = row["city"]
        else:
            assert record_lookup is not None
            city = record_lookup[row["initiative_record_id"]]["initiative"]["city"]
        counts[city] = counts.get(city, 0) + 1
    return counts


def count_unique_initiatives_by_city(
    rows: list[JsonDict],
    record_lookup: dict[str, JsonDict],
) -> dict[str, int]:
    """Count unique mapped initiative ids by city."""
    ids_by_city: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        record = record_lookup[row["initiative_record_id"]]
        ids_by_city[record["initiative"]["city"]].add(row["initiative_record_id"])
    return {city: len(ids) for city, ids in ids_by_city.items()}


def apply_tef_fixes(
    *,
    source_final_mappings: list[JsonDict],
    corrected_records_by_id: dict[str, JsonDict],
    city_fixes: dict[str, JsonDict],
    category_children: dict[str, list[str]],
    mapper_version: str,
    tef_source_version: str,
    extraction_run_id: str,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    """Apply TEF repair rules and return corrected mappings, review items, and manifest data."""
    valid_record_ids = set(corrected_records_by_id)
    remove_all, remove_rows, abstain_reasons = build_remove_rules(city_fixes)
    rewrite_specs = build_rewrite_specs(city_fixes)
    add_specs = build_add_specs(city_fixes)

    kept_rows: list[JsonDict] = []
    removed_initiatives: set[str] = set()
    removed_rows: list[JsonDict] = []
    rewritten_rows: list[JsonDict] = []

    for row in source_final_mappings:
        if should_remove_row(
            row,
            valid_record_ids=valid_record_ids,
            remove_all=remove_all,
            remove_rows=remove_rows,
        ):
            removed_rows.append(
                {
                    "initiative_record_id": row["initiative_record_id"],
                    "target_path": row["target_path"],
                    "target_id": row["target_id"],
                }
            )
            removed_initiatives.add(row["initiative_record_id"])
            continue

        record = corrected_records_by_id[row["initiative_record_id"]]
        matched_spec = next(
            (spec for spec in rewrite_specs.get(row["initiative_record_id"], []) if matches_rewrite_spec(row, spec)),
            None,
        )
        if matched_spec is not None:
            kept_rows.append(
                replace_target(
                    row=row,
                    record=record,
                    spec=matched_spec,
                    category_children=category_children,
                    extraction_run_id=extraction_run_id,
                )
            )
            rewritten_rows.append(
                {
                    "initiative_record_id": row["initiative_record_id"],
                    "from_target_id": row["target_id"],
                    "from_target_path": row["target_path"],
                    "to_target_id": matched_spec["target_id"],
                    "to_target_path": matched_spec["target_path"],
                }
            )
            continue

        kept_rows.append(refresh_mapping_record(copy.deepcopy(row), record, extraction_run_id))

    added_rows: list[JsonDict] = []
    for initiative_record_id, specs in add_specs.items():
        if initiative_record_id not in valid_record_ids:
            continue
        existing_keys = {
            (row["initiative_record_id"], row["target_path"], row["target_id"])
            for row in kept_rows
            if row["initiative_record_id"] == initiative_record_id
        }
        for spec in specs:
            key = (initiative_record_id, spec["target_path"], spec["target_id"])
            if key in existing_keys:
                continue
            record = corrected_records_by_id[initiative_record_id]
            added_rows.append(
                make_added_mapping(
                    record=record,
                    spec=spec,
                    mapper_version=mapper_version,
                    tef_source_version=tef_source_version,
                    extraction_run_id=extraction_run_id,
                    category_children=category_children,
                )
            )

    corrected_rows = sorted(
        [*kept_rows, *added_rows],
        key=lambda row: (
            row["city"],
            row["initiative_record_id"],
            not row["is_primary"],
            row["target_path"],
            row["target_id"],
        ),
    )

    unique_keys = set()
    for row in corrected_rows:
        key = (row["initiative_record_id"], row["target_type"], row["target_path"], row["target_id"])
        if key in unique_keys:
            raise ValueError(f"Duplicate corrected TEF mapping row: {key}")
        unique_keys.add(key)

    mapped_ids = {row["initiative_record_id"] for row in corrected_rows}
    abstained_ids = sorted(valid_record_ids - mapped_ids)
    missing_without_reason = [record_id for record_id in abstained_ids if record_id not in abstain_reasons]
    if missing_without_reason:
        raise ValueError(
            "Some initiatives lost all TEF mappings without an abstain reason: "
            + ", ".join(sorted(missing_without_reason))
        )

    review_items = build_review_items(
        corrected_records_by_id=corrected_records_by_id,
        final_mappings=corrected_rows,
        abstain_reasons={record_id: abstain_reasons[record_id] for record_id in abstained_ids},
    )

    manifest = {
        "removed_initiatives": sorted(removed_initiatives),
        "removed_rows": removed_rows,
        "rewritten_rows": rewritten_rows,
        "added_rows": [
            {
                "initiative_record_id": row["initiative_record_id"],
                "target_path": row["target_path"],
                "target_id": row["target_id"],
            }
            for row in added_rows
        ],
        "abstained_initiatives": [
            {"initiative_record_id": record_id, "reason": abstain_reasons[record_id]}
            for record_id in abstained_ids
        ],
    }
    return corrected_rows, review_items, manifest


def write_corrected_run(
    *,
    source_tef_run_dir: Path,
    extraction_run_dir: Path,
    output_tef_run_dir: Path,
    corrected_records: list[JsonDict],
    corrected_final_mappings: list[JsonDict],
    corrected_review_items: list[JsonDict],
    manifest: JsonDict,
    overwrite: bool,
) -> None:
    """Copy the source TEF run and overwrite the corrected audit-fixed artifacts."""
    if output_tef_run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_tef_run_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_tef_run_dir)

    shutil.copytree(source_tef_run_dir, output_tef_run_dir)

    source_manifest_path = output_tef_run_dir / "00_source" / "source_manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_extraction_run_id = source_manifest.get("extraction_run_id")
    source_manifest["run_id"] = output_tef_run_dir.name
    source_manifest["source_run_id"] = source_tef_run_dir.name
    source_manifest["initiatives_path"] = str(
        (extraction_run_dir / "03_deduped" / "initiative_records.jsonl").as_posix()
    )
    source_manifest["extraction_run_id"] = extraction_run_dir.name
    write_json(source_manifest_path, source_manifest)

    write_jsonl(output_tef_run_dir / "01_inputs" / "initiatives.jsonl", corrected_records)
    write_jsonl(
        output_tef_run_dir / "05_final_mappings" / "final_mappings.jsonl",
        corrected_final_mappings,
    )
    write_jsonl(output_tef_run_dir / "06_review" / "review_items.jsonl", corrected_review_items)

    summary_path = output_tef_run_dir / "summary.json"
    summary = load_json(summary_path)
    summary["run_id"] = output_tef_run_dir.name
    summary["source_run_id"] = source_tef_run_dir.name
    summary["extraction_run_id"] = extraction_run_dir.name
    summary["initiatives_count"] = len(corrected_records)
    summary["mapped_initiatives_count"] = len(
        {row["initiative_record_id"] for row in corrected_final_mappings}
    )
    summary["final_mappings_count"] = len(corrected_final_mappings)
    summary["review_items_count"] = len(corrected_review_items)
    summary["error_count"] = 0
    summary["audit_fix_counts"] = {
        "removed_initiatives": len(manifest["removed_initiatives"]),
        "removed_rows": len(manifest["removed_rows"]),
        "rewritten_rows": len(manifest["rewritten_rows"]),
        "added_rows": len(manifest["added_rows"]),
        "abstained_initiatives": len(manifest["abstained_initiatives"]),
    }
    record_lookup = {record["record_id"]: record for record in corrected_records}
    summary["city_mapped_initiatives_count"] = count_unique_initiatives_by_city(
        corrected_final_mappings,
        record_lookup,
    )
    summary["city_final_mappings_count"] = count_by_city(corrected_final_mappings, key="mapping")
    summary["city_abstained_initiatives_count"] = count_by_city(
        manifest["abstained_initiatives"],
        key="initiative",
        record_lookup=record_lookup,
    )
    write_json(summary_path, summary)

    manifest_payload = {
        "source_run_id": source_tef_run_dir.name,
        "fixed_run_id": output_tef_run_dir.name,
        "source_extraction_run_id": source_extraction_run_id,
        "fixed_extraction_run_id": extraction_run_dir.name,
        **manifest,
    }
    write_json(output_tef_run_dir / "audit_fix_manifest.json", manifest_payload)

    readme_path = output_tef_run_dir / "README.md"
    original_readme = readme_path.read_text(encoding="utf-8")
    note = (
        "AUDIT-FIX NOTE\n"
        f"This run is a corrected copy of `{source_tef_run_dir.name}` based on "
        "`polish_cities_extraction_tef_audit_report_2026-04-23.md` and the corrected extraction run "
        f"`{extraction_run_dir.name}`. The copied source-stage routing artifacts remain for reference, "
        "while `00_source/source_manifest.json`, `01_inputs/initiatives.jsonl`, "
        "`05_final_mappings/final_mappings.jsonl`, `06_review/review_items.jsonl`, "
        "`07_numeric_facts/*`, `08_tef_groups/*`, `summary.json`, and `audit_fix_manifest.json` were regenerated "
        "or overwritten by this audit-fix pass.\n\n"
    )
    readme_path.write_text(note + original_readme, encoding="utf-8")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    source_tef_run_dir = args.source_tef_run_dir.resolve()
    extraction_run_dir = args.extraction_run_dir.resolve()
    output_tef_run_dir = args.output_tef_run_dir.resolve()

    logger.info(
        "Applying TEF audit fixes from source_tef_run_dir=%s using extraction_run_dir=%s",
        source_tef_run_dir,
        extraction_run_dir,
    )

    source_manifest = load_json(source_tef_run_dir / "00_source" / "source_manifest.json")
    corrected_records = load_jsonl(extraction_run_dir / "03_deduped" / "initiative_records.jsonl")
    corrected_records_by_id = {record["record_id"]: record for record in corrected_records}
    source_final_mappings = load_jsonl(
        source_tef_run_dir / "05_final_mappings" / "final_mappings.jsonl"
    )
    transition_paths, category_paths, category_children = load_catalog_paths(
        Path(source_manifest["tef_catalog_root"]) / "catalog"
    )

    city_fixes = build_city_fixes()
    validate_fix_specs(
        city_fixes=city_fixes,
        transition_paths=transition_paths,
        category_paths=category_paths,
    )

    corrected_final_mappings, corrected_review_items, manifest = apply_tef_fixes(
        source_final_mappings=source_final_mappings,
        corrected_records_by_id=corrected_records_by_id,
        city_fixes=city_fixes,
        category_children=category_children,
        mapper_version=source_manifest["mapper_version"],
        tef_source_version=source_manifest["tef_source_version"],
        extraction_run_id=extraction_run_dir.name,
    )

    write_corrected_run(
        source_tef_run_dir=source_tef_run_dir,
        extraction_run_dir=extraction_run_dir,
        output_tef_run_dir=output_tef_run_dir,
        corrected_records=corrected_records,
        corrected_final_mappings=corrected_final_mappings,
        corrected_review_items=corrected_review_items,
        manifest=manifest,
        overwrite=args.overwrite,
    )

    rollup_summary = rollup_existing_tef_run(
        tef_run_dir=output_tef_run_dir,
        extraction_run_dir=extraction_run_dir,
    )
    logger.info("TEF numeric rollup finished: %s", rollup_summary)
    logger.info(
        "Wrote corrected TEF run to %s with %s mapped initiatives and %s final mappings.",
        output_tef_run_dir,
        len({row["initiative_record_id"] for row in corrected_final_mappings}),
        len(corrected_final_mappings),
    )


if __name__ == "__main__":
    setup_logger()
    main()
