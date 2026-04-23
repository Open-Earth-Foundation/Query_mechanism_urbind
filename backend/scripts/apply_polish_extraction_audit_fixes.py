"""
Brief: Apply the 2026-04-23 Poland extraction audit fixes to a completed extraction run.

Inputs:
- `--source-run-dir`: Extraction run directory to repair. Expected to contain `summary.json`,
  `README.md`, and `03_deduped/{initiative_records.jsonl,initiatives.jsonl}`.
- `--output-run-dir`: Destination directory for the corrected copy of the run.
- `--overwrite`: Remove the output directory first if it already exists.
- Files/paths: the script reads only the final deduped extraction artifacts from the source run and
  copies the full source run directory for reference.
- Env vars: none required.

Outputs:
- A corrected copy of the run directory at `--output-run-dir`
- Updated `summary.json`
- Updated `03_deduped/initiative_records.jsonl`
- Updated `03_deduped/initiatives.jsonl`
- `audit_fix_manifest.json` describing the applied fixes

Usage (from project root):
- python -m backend.scripts.apply_polish_extraction_audit_fixes --source-run-dir output/initiative_extraction/polish_cities_flow_test_20260423_001 --output-run-dir output/initiative_extraction/polish_cities_flow_test_20260423_001_audit_fixed --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
from hashlib import sha1
from pathlib import Path
from typing import Any

from backend.utils.logging_config import setup_logger


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Apply the Poland extraction audit fixes to a completed extraction run."
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        help="Source extraction run directory containing summary.json and 03_deduped outputs.",
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        required=True,
        help="Destination directory for the corrected run copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the destination directory first if it already exists.",
    )
    return parser.parse_args()


def normalize_key(value: str) -> str:
    """Normalize a token for record ids."""
    return re.sub(r"[\W_]+", "_", value.strip().casefold()).strip("_")


def normalize_title(value: str) -> str:
    """Normalize a title for the extractor-style title hash."""
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def make_record_id(city: str, source_document: str, title: str) -> str:
    """Build a deterministic title-hash record id matching the extractor pattern."""
    city_key = normalize_key(city) or "unknown_city"
    doc_slug = normalize_key(Path(source_document).stem) or "document"
    title_hash = sha1(  # noqa: S324
        f"{city_key}|{doc_slug}|{normalize_title(title)}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{city_key}:{doc_slug}:title_{title_hash}"


def make_record(
    *,
    city: str,
    title: str,
    general_description: str,
    objective_text: str,
    implementation_text: str,
    planned_outputs_text: str,
    delivery_text: str,
    source_document: str,
    source_quote: str,
    funding_text: str | None = None,
    timeline_text: str | None = None,
    numbers_current: dict[str, Any] | None = None,
    numbers_planned: dict[str, Any] | None = None,
    document_local_code: str | None = None,
    record_id: str | None = None,
    data_quality_flags: list[str] | None = None,
) -> dict[str, Any]:
    """Build an initiative-record artifact row."""
    return {
        "initiative": {
            "city": city,
            "initiative_name": title,
            "general_description": general_description,
            "objective_text": objective_text,
            "implementation_text": implementation_text,
            "planned_outputs_text": planned_outputs_text,
            "delivery_text": delivery_text,
            "funding_text": funding_text,
            "timeline_text": timeline_text,
            "numbers": {
                "current": numbers_current or {},
                "planned": numbers_planned or {},
            },
        },
        "document_local_code": document_local_code,
        "source_quote": source_quote,
        "data_quality_flags": data_quality_flags or ["page_artifacts_inside_section"],
        "number_context": {},
        "number_deferred": {},
        "number_uncertain": {},
        "extraction_notes": [],
        "record_id": record_id or make_record_id(city, source_document, title),
        "source_document": source_document,
    }


def load_records(records_path: Path) -> list[dict[str, Any]]:
    """Load initiative-record rows from JSONL."""
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def count_by_city(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count records by city."""
    counts: dict[str, int] = {}
    for record in records:
        city = record["initiative"]["city"]
        counts[city] = counts.get(city, 0) + 1
    return counts


def build_removals() -> set[str]:
    """Return record ids to remove entirely."""
    return {
        "krakow:krakow:title_2a0abaa24ba3",
        "lodz:lodz:title_38b62754cb43",
        "lodz:lodz:title_79167b36f827",
        "lodz:lodz:title_5375bd1b68bf",
        "lodz:lodz:title_7424e17d051f",
        "lodz:lodz:title_92ca4b799477",
        "lodz:lodz:title_58534ebc2312",
        "lodz:lodz:title_0f6fb8583884",
        "lodz:lodz:title_113aadec4cfa",
        "rzeszow:rzeszow:title_80ba03f69d2e",
        "rzeszow:rzeszow:title_3d8de539b280",
        "rzeszow:rzeszow:title_d184313985fe",
        "rzeszow:rzeszow:title_6e97c368c11a",
        "rzeszow:rzeszow:title_e92f44f4741c",
        "warszawa:warszawa:title_7b300cf89f3e",
        "warszawa:warszawa:title_492e9808a1c2",
        "warszawa:warszawa:title_541a31ca3ccf",
        "warszawa:warszawa:title_905dd8374711",
        "warszawa:warszawa:title_35570c6efd29",
        "warszawa:warszawa:title_c3536f972074",
        "rzeszow:rzeszow:title_17e350b60697",
        "rzeszow:rzeszow:title_a71c6d461bdb",
        "rzeszow:rzeszow:title_dcee1fbb6c02",
        "rzeszow:rzeszow:title_816ad29d1753",
    }


def build_additions() -> list[dict[str, Any]]:
    """Return fully materialized record additions."""
    return [
        make_record(
            city="Krakow",
            title="Modernisation of road lighting in Krakow",
            general_description=(
                "Road lighting modernisation using LED luminaires, lighting management, "
                "and related control systems."
            ),
            objective_text=(
                "Reduce electricity consumption through the use of modern LED luminaires "
                "and a lighting management system."
            ),
            implementation_text=(
                "Replace lamps, extension arms and controllers; design reactive power reduction "
                "subsystems; implement a master lighting management system; adapt existing "
                "cabinets for energy-efficient LED luminaires."
            ),
            planned_outputs_text="Dynamic remote management of LED luminaires and better lighting control.",
            delivery_text=(
                "Krakow City Roads Board; operation within the administrative boundaries of the "
                "City of Krakow."
            ),
            funding_text="Estimated investment costs - PLN 85 712 361 (approx. EUR 19 047 000).",
            timeline_text="The planned timeframe for the task is 2024-2028.",
            numbers_planned={
                "estimated_emissions_reduction_electricity_tco2e": 4408,
                "investment_cost_pln": 85712361,
                "investment_cost_eur_approx": 19047000,
                "start_year": 2024,
                "end_year": 2028,
            },
            source_document="Krakow.md",
            source_quote="Modernisation of road lighting in Krakow.",
            data_quality_flags=["page_artifacts_inside_section", "source_contains_deferred_values"],
            document_local_code="E-11",
        ),
        make_record(
            city="Krakow",
            title="Project for fast, collision-free rail transport in Krakow (Premetro)",
            general_description=(
                "A pre-metro project to implement one of seven analysed variants of fast, "
                "collision-free rail transport in Krakow."
            ),
            objective_text=(
                "Implement one of the Krakow pre-metro variants analysed in the feasibility "
                "study to expand public transport infrastructure and improve transport accessibility."
            ),
            implementation_text=(
                "Construct a route beginning at Jasnogórska Street and ending at Mogila housing "
                "estate or Kocmyrzowska Street; one likely version is about 25 km and about 20 "
                "interchange stops."
            ),
            planned_outputs_text="About 25 km of route and about 20 interchange stops.",
            delivery_text="Municipality of Krakow; city-wide action to increase transport accessibility.",
            funding_text=(
                "Preliminary investment costs vary depending on the variant. The cost of one "
                "likely version is approximately PLN 4 753 050 000 (approx. EUR 1 056 233 000)."
            ),
            timeline_text="Action only in the design phase, currently no timetable or milestones.",
            numbers_planned={
                "route_length_km_approx": 25,
                "interchange_stops_approx": 20,
                "estimated_emissions_reduction_tco2e": 6230,
                "investment_cost_pln_approx": 4753050000,
                "investment_cost_eur_approx": 1056233000,
            },
            source_document="Krakow.md",
            source_quote="Project for fast, collision-free rail transport in Krakow (Premetro).",
            data_quality_flags=["page_artifacts_inside_section", "source_contains_deferred_values"],
            document_local_code="TR-2",
        ),
        make_record(
            city="Lodz",
            title="NEEST - NetZero Emission and Environmentally Sustainable Territories",
            general_description=(
                "A governance and planning initiative that develops scalable solutions for "
                "energy efficiency projects in representative Lodz building areas."
            ),
            objective_text=(
                "Prepare innovative solutions ready for implementation, scaling, and replication, "
                "covering technical, financial, environmental, and social aspects."
            ),
            implementation_text=(
                "Study the Radiostacja district in Lodz and develop pilot solutions, a Monitoring "
                "Evaluation and Learning framework, and a guidebook that can be replicated by Lodz "
                "and other cities for similar building types and quarters."
            ),
            planned_outputs_text=(
                "Pilot solutions for representative building quarters or groups of buildings, plus "
                "a guidebook for testing, scaling, and improving model solutions."
            ),
            delivery_text=(
                "Local government and internal departments, with the Mission platform and other "
                "Polish cities participating in the Mission."
            ),
            source_document="Lodz.md",
            source_quote="NEEST-NetZero Emission and Environmentally Sustainable Territories",
        ),
        make_record(
            city="Lodz",
            title="GHG emissions monitoring and accounting system for city-wide inventory",
            general_description=(
                "A city-wide system for maintaining activity-data collection and regularly updating "
                "the greenhouse gas inventory."
            ),
            objective_text=(
                "Maintain procedures to collect required activity data and update emission factors "
                "so the city-wide GHG inventory can be regularly updated."
            ),
            implementation_text=(
                "Use local-government and utility cooperation to maintain data collection "
                "procedures and update the city-wide GHG emissions inventory covering buildings, "
                "transport, and other sectors."
            ),
            planned_outputs_text=(
                "Regularly updated city-wide GHG inventory and stronger data-driven planning and "
                "monitoring of emission-reduction projects."
            ),
            delivery_text="Local government and internal departments, with utilities as external stakeholders.",
            source_document="Lodz.md",
            source_quote="GHG emissions monitoring and accounting system for city-wide inventory",
        ),
        make_record(
            city="Lodz",
            title="City's budget climate tagging",
            general_description=(
                "An initiative to integrate climate tagging into the city's financial management "
                "system for planning, reporting, and climate-finance tracking."
            ),
            objective_text=(
                "Incorporate climate tagging into the existing Integrated City Financial Management "
                "System using suitable methodology, project types, implementing rules, and reporting."
            ),
            implementation_text=(
                "Integrate climate-tagging rules and reporting into the city's existing financial "
                "management system based on city experience and best practices on sub-national "
                "climate budget tagging."
            ),
            planned_outputs_text=(
                "Climate tagging within the city budget system, stronger cost-efficiency "
                "assessment of projects, and clearer reporting on climate-relevant spending."
            ),
            delivery_text="Local government and internal departments.",
            source_document="Lodz.md",
            source_quote="City's budget climate tagging",
        ),
        make_record(
            city="Lodz",
            title="Veolia + Innargi geothermal project",
            general_description=(
                "A planned cooperation with Veolia and Innargi to explore and use geothermal "
                "resources as part of decarbonizing Lodz district heating."
            ),
            objective_text=(
                "Launch a joint project for the exploration and use of geothermal resources in "
                "Lodz to support district-heating decarbonization and renewable-energy use."
            ),
            implementation_text=(
                "Start cooperation between the city, Veolia, and Innargi on a joint geothermal "
                "project related to exploration and use of geothermal resources in Lodz."
            ),
            planned_outputs_text=(
                "Exploration and use of geothermal resources as a significant source of heating "
                "and potentially electricity, supporting faster emissions reductions."
            ),
            delivery_text="City of Lodz, Veolia, and Innargi.",
            timeline_text="Plans for the end of 2024 include the start of cooperation.",
            source_document="Lodz.md",
            source_quote=(
                "Plans for the end of 2024 include the start of cooperation with Veolia and "
                "Innargi to launch a joint project related to the exploration and use of "
                "geothermal resources in Lodz."
            ),
        ),
        make_record(
            city="Lodz",
            title=(
                "European City Facility concept for Emission reduction in 100 municipal "
                "buildings in the historic center of Lodz"
            ),
            general_description=(
                "An EUCF-supported investment-concept initiative for reducing emissions in 100 "
                "municipal buildings in the historic center of Lodz."
            ),
            objective_text=(
                "Develop an investment concept and mobilize sustainable-energy investment for "
                "emission reduction in 100 municipal buildings in the historic center of Lodz."
            ),
            implementation_text=(
                "Use the European City Facility grant to develop the investment concept and carry "
                "out energy audits and feasibility assessment of GHG-reduction options in 100 "
                "municipal buildings, including renovation, comprehensive thermal modernization, "
                "heat-source replacement and connection to the heating network, and PV installations."
            ),
            planned_outputs_text=(
                "An investment concept plus energy audits and feasibility assessment of "
                "GHG-emission reduction options for 100 municipal buildings in the historic center."
            ),
            delivery_text="City of Lodz through the European City Facility initiative.",
            funding_text="EUCF grant.",
            source_document="Lodz.md",
            source_quote=(
                "Within the EUCF Lodz received a grant to develop an Investment Concept entitled: "
                "\"Emission reduction in 100 municipal buildings in the historic center of Lodz\"."
            ),
        ),
        make_record(
            city="Rzeszow",
            title=(
                "Increasing the quality of selective waste collection at the source and increasing "
                "the quantity of waste collected selectively and sent for recycling. Investments "
                "in solutions that enable achieving higher levels of recovery and recycling"
            ),
            general_description=(
                "Waste-collection and recycling infrastructure action centered on PSZOK expansion, "
                "repair and reuse, sorting upgrades, and environmental education."
            ),
            objective_text=(
                "Increase in the amount of selectively collected waste. Significant increase in "
                "the level of waste recycling, including bio-waste."
            ),
            implementation_text=(
                "Construct a modern PSZOK with an office building and storage building, create a "
                "repair and reuse point for products, build an Ecological Education Center, "
                "implement an Ecological Education Path for children and youth, add the necessary "
                "selective-waste infrastructure on the site, and reconstruct the installation for "
                "mechanical processing of municipal waste so it can sort selectively collected waste."
            ),
            planned_outputs_text=(
                "A modern PSZOK, repair and reuse point, Ecological Education Center, Ecological "
                "Education Path, selective-waste infrastructure, and upgraded sorting capacity "
                "for selectively collected waste."
            ),
            delivery_text=(
                "City Municipality of Rzeszow - Rzeszow City Office/Department of Municipal "
                "Services and Department of Investments, Miejskie Przedsiebiorstwo Gospodarki "
                "Komunalnej-Rzeszow Sp. z o. o. (City Public Utility Company), residents, and entrepreneurs."
            ),
            funding_text="Total costs: PLN 30,000,000, EURO 6,666,667. Costs by CO2e unit: 56,285.18 EUR 12,507.82.",
            timeline_text=(
                "Continuous and systematic action; by 2030 the plan is to use only companies with "
                "low- and zero-emission transportation for the task."
            ),
            numbers_planned={
                "ghg_emissions_reduction_tco2e": 533,
                "total_cost_pln": 30000000,
                "total_cost_eur": 6666667,
                "end_year": 2030,
            },
            source_document="Rzeszow.md",
            source_quote=(
                "Action name | Increasing the quality of selective waste collection at the source "
                "and increasing the quantity of waste collected selectively and sent for recycling. "
                "Investments in solutions that enable achieving higher levels of recovery and recycling."
            ),
        ),
        make_record(
            city="Rzeszow",
            title="Investments in solutions aimed at waste disposal and management",
            general_description=(
                "Waste-management infrastructure and fleet-modernization action focused on "
                "bio-waste processing and operational modernization."
            ),
            objective_text="Significant increase in the level of waste recycling, including bio-waste.",
            implementation_text=(
                "Construct a green waste biocomposting facility, implement a waste processing site "
                "with waste bins until 2027, and systematically modernize the vehicle fleet."
            ),
            planned_outputs_text=(
                "A green-waste biocomposting facility, a waste processing site with waste bins, "
                "and a modernized municipal waste fleet."
            ),
            delivery_text=(
                "Miejskie Przedsiebiorstwo Gospodarki Komunalnej-Rzeszow Sp. z o. o. "
                "(City Public Utility Company), the City Municipality of Rzeszow, and residents."
            ),
            funding_text="Total costs PLN 45,000,000, EURO 10,000,000. Costs by CO2e unit: PLN 36,976.17 EURO 8,216.93.",
            timeline_text=(
                "The waste processing site with waste bins is planned until 2027; fleet "
                "modernization is systematic and ongoing."
            ),
            numbers_planned={
                "ghg_emissions_reduction_tco2e": 1581,
                "total_cost_pln": 45000000,
                "total_cost_eur": 10000000,
                "end_year": 2027,
            },
            source_document="Rzeszow.md",
            source_quote="Action name | Investments in solutions aimed at waste disposal and management.",
        ),
        make_record(
            city="Rzeszow",
            title="De-paving of concrete surfaces, including parking lots",
            general_description=(
                "Green-infrastructure action to remove sealed concrete surfaces and replace them "
                "with biologically active and cooling urban space."
            ),
            objective_text=(
                "Increasing green areas. Reducing the urban heat island effect. Increasing "
                "retention of rainfall."
            ),
            implementation_text=(
                "Develop green infrastructure and deconcrete sealed surfaces in the city area of "
                "over 1 hectare. In the places where concrete has been removed, create biologically "
                "active surfaces such as sports fields with grass surface, rain gardens in road "
                "strips, pocket parks, and new green arrangements in the spaces between buildings."
            ),
            planned_outputs_text=(
                "More than 1 hectare of de-paved surface, new biologically active areas, rain "
                "gardens, pocket parks, and other green arrangements."
            ),
            delivery_text=(
                "City Municipality of Rzeszow - City Greenery Management Authority, ZZM, "
                "Rzeszow City Office/WI, MZBM, housing cooperatives, developers, and entrepreneurs."
            ),
            funding_text="Total costs: PLN 5,338,580, EUR 1,186,351. Costs by CO2e unit: PLN 39,545, EUR 8,787.8.",
            timeline_text="Actions related to deconcreting will be systematic and intensified.",
            numbers_planned={
                "de_paved_area_ha_min": 1,
                "ghg_emissions_compensated_tco2": 135,
                "total_cost_pln": 5338580,
                "total_cost_eur": 1186351,
            },
            source_document="Rzeszow.md",
            source_quote="Action name | De-paving of concrete surfaces, including parking lots.",
        ),
        make_record(
            city="Rzeszow",
            title="Application of solutions for retaining a portion of stormwater on new development sites",
            general_description=(
                "Stormwater-retention action for new development sites to reduce runoff, improve "
                "soil moisture, and lower flood risk."
            ),
            objective_text="Increasing retention of rainfall.",
            implementation_text=(
                "Implement and plan new tasks to reduce surface runoff from heavy rains, prolong "
                "soil moisture retention, support local greenery, limit the load on the wastewater "
                "system, and mitigate the risk of flooding for lower-lying areas."
            ),
            planned_outputs_text=(
                "New development sites with on-site stormwater-retention requirements and reduced "
                "runoff and flood risk."
            ),
            delivery_text=(
                "City Municipality of Rzeszow - Rzeszow City Office/Department of Investments, WI, "
                "KS, Department of Architecture, BRMR, residents, entrepreneurs, developers, and "
                "housing cooperatives."
            ),
            funding_text="Total costs: PLN 78,248,974, EUR 17,388,661.",
            timeline_text=(
                "Retention requirements are to be reflected in prepared documentation and imposed "
                "during environmental-decision or development-conditions processes for new investments."
            ),
            numbers_planned={
                "total_cost_pln": 78248974,
                "total_cost_eur": 17388661,
            },
            source_document="Rzeszow.md",
            source_quote=(
                "Action name | Application of solutions for retaining a portion of stormwater on "
                "new development sites."
            ),
        ),
        make_record(
            city="Rzeszow",
            title="Implementing eco-friendly solutions in Municipal Units and Companies",
            general_description=(
                "Municipal-buildings decarbonization action for municipal units and companies "
                "through photovoltaic installations, efficient lighting, and smart energy management."
            ),
            objective_text=(
                "All buildings of Municipal Units and Companies with energy-efficient management "
                "systems. Increase in the number of RES."
            ),
            implementation_text=(
                "Develop photovoltaic installations, replace lighting with energy-efficient "
                "alternatives, and implement smart energy management systems. Reduce non-renewable "
                "energy demand through photovoltaic installations, heat pumps, other renewable "
                "energy sources, deep thermal modernization, optimization of energy consumption, "
                "and smart energy management systems."
            ),
            planned_outputs_text=(
                "Photovoltaic installations, energy-efficient lighting, and smart energy-management "
                "systems across municipal units and companies."
            ),
            delivery_text=(
                "City Municipality of Rzeszow - Rzeszow City Office/WI, municipal companies, "
                "municipal units, MPEC, NFOŚiGW, WFOŚiGW, the Marshal of the Podkarpackie "
                "Voivodeship, and financing institutions."
            ),
            funding_text="Total costs: PLN 1,630,000, EUR 362,222. Costs by CO2e unit: PLN 4,630.68 EUR 1,029.04.",
            timeline_text=(
                "Municipal companies and units have been taking actions for years; in the near "
                "future PV installations with a total capacity of about 550 kWp will be installed "
                "by ROSiR, MZMB, and MPGK."
            ),
            numbers_planned={
                "generated_renewable_energy_mwh": 513.94,
                "ghg_emissions_reduction_tco2e": 352,
                "pv_capacity_kwp": 550,
                "total_cost_pln": 1630000,
                "total_cost_eur": 362222,
            },
            source_document="Rzeszow.md",
            source_quote="Action name | Implementing eco-friendly solutions in Municipal Units and Companies.",
        ),
        make_record(
            city="Warszawa",
            title="Interdisciplinary structures at the City of Warsaw",
            general_description=(
                "Warsaw will appoint interdisciplinary structures within the city administration, "
                "including the Internal Transition Team and Climate Team, to coordinate and manage "
                "cross-cutting climate-neutrality processes."
            ),
            objective_text=(
                "Appointing structures with an interdisciplinary character in the City of Warsaw, "
                "dealing with the coordination or management of processes that go beyond the narrow "
                "specialisation of the office's units."
            ),
            implementation_text=(
                "Create and operate interdisciplinary city structures to coordinate actions, "
                "communication, and expertise across units, addressing siloed work, dispersed "
                "activities, blue-green infrastructure constraints, data-use difficulties, rainwater-"
                "use problems, energy-efficiency uncertainty, fuel-poverty uncertainty, and "
                "financial barriers."
            ),
            planned_outputs_text=(
                "Better allocation of resources, improved coordination of actions and communication, "
                "and stronger ability to use diverse sources of expertise across the City of Warsaw."
            ),
            delivery_text="Self-government: cells and units of the City of Warsaw, municipal companies.",
            source_document="Warszawa.md",
            source_quote="Interdisciplinary structures at the City of Warsaw",
        ),
        make_record(
            city="Warszawa",
            title="Cooperation with representatives of the central administration within the framework",
            general_description=(
                "Warsaw will hold systematic meetings with central government representatives as "
                "part of the National Cooperation Platform to address needs, barriers, and systemic "
                "problems relevant to climate neutrality."
            ),
            objective_text=(
                "Systematic meetings with central government representatives as part of the National "
                "Cooperation Platform to address needs and problems."
            ),
            implementation_text=(
                "Run systematic cooperation between the City of Warsaw and central administration to "
                "address dependence on the National Grid, barriers to renewable-energy development, "
                "financial barriers, and building-efficiency needs, and to develop systemic solutions."
            ),
            planned_outputs_text="Direct feedback to central administration and cooperation concerning developing systemic solutions.",
            delivery_text=(
                "Local Government: Air Protection and Climate Policy Department; Central "
                "administration: Ministry of Climate and Environment."
            ),
            source_document="Warszawa.md",
            source_quote="Cooperation with representatives of the central administration within the framework.",
        ),
        make_record(
            city="Warszawa",
            title="Cooperation with urban stakeholders",
            general_description=(
                "Warsaw will run systematic meetings with urban stakeholders within the External "
                "Transition Team and the Climate Partnership platform to co-develop solutions and "
                "support climate-contract implementation."
            ),
            objective_text=(
                "Systematic meetings with urban stakeholders within the External Transition Team and "
                "within the Climate Partnership platform."
            ),
            implementation_text=(
                "Coordinate regular cooperation with city units, municipal companies, state-owned "
                "energy companies, energy distributors, funding-programme operators, and the science "
                "sector to address financial barriers, rainwater-use problems, data-use difficulties, "
                "and blue-green infrastructure constraints, while working out sector solutions, "
                "co-funding, legal and organisational initiatives, and new technology and funding options."
            ),
            planned_outputs_text=(
                "Cooperation in terms of developing systemic solutions, working out solutions for "
                "sectors in sector groups, co-funding or funding activities, stakeholders' own "
                "actions, seeking new sources of funding, seeking technology and organisational "
                "solutions, and legal and organisational initiatives."
            ),
            delivery_text=(
                "Self-government: cells and units of the City of Warsaw, municipal companies; "
                "state-owned energy companies and energy distributors; National Fund for "
                "Environmental Protection and Water Management and other operators of funding "
                "programmes; science sector."
            ),
            source_document="Warszawa.md",
            source_quote="Cooperation with urban stakeholders.",
        ),
        make_record(
            city="Warszawa",
            title="Warsaw Booster acceleration programme",
            general_description=(
                "The city's acceleration programme for innovative projects and start-ups supports "
                "young technology companies in improving their business competences and accelerating "
                "projects with market-implementation potential."
            ),
            objective_text=(
                "The city's acceleration programme for innovative projects and start-ups, which "
                "supports young technology companies in improving their business competences and "
                "accelerates the development of business projects, increasing their chances of being "
                "implemented in the market."
            ),
            implementation_text=(
                "Use the city's acceleration programme to support projects that develop analytical "
                "tools and new technologies relevant to the Climate City Contract and to address "
                "barriers related to data use, local and district centres, blue-green infrastructure, "
                "building-efficiency needs, and energy poverty."
            ),
            planned_outputs_text=(
                "City support for projects that will facilitate developing analytical tools and "
                "using new technologies for supporting the activities of the Climate City Contract "
                "and overcoming barriers."
            ),
            delivery_text=(
                "Local Government: Economic Development Department; the cells and units of the "
                "City of Warsaw, municipal companies; service companies (new technologies)."
            ),
            source_document="Warszawa.md",
            source_quote="Warsaw Booster acceleration programme",
        ),
        make_record(
            city="Warszawa",
            title="Research Issues Exchange",
            general_description=(
                "A networking project aimed at increasing cooperation between the local government "
                "and the scientific community by publishing research needs and matching them with "
                "scientific partners."
            ),
            objective_text=(
                "A networking project aimed at increasing cooperation between the local government "
                "and the scientific community."
            ),
            implementation_text=(
                "Employees of the local government publish topics concerning which they require "
                "knowledge and research-based solutions, and representatives of scientific "
                "institutions make proposals to cooperate and carry out research projects addressing "
                "barriers related to finance, local and district centres, blue-green infrastructure, "
                "building efficiency, fuel poverty, data use, and rainwater use."
            ),
            planned_outputs_text=(
                "Networking and the search for appropriate partners from the scientific world to "
                "address barriers and challenges to the research, analysis and solutions needed to "
                "implement the Climate City Contract."
            ),
            delivery_text=(
                "Local Government: Strategy & Analysis Department; the cells and units of the "
                "City of Warsaw, municipal companies; science sector: universities and scientific institutions."
            ),
            source_document="Warszawa.md",
            source_quote="Research Issues Exchange",
        ),
        make_record(
            city="Warszawa",
            title="Urban Living Lab",
            general_description=(
                "A collaborative method for generating innovative solutions by bringing together "
                "city authorities, citizens, businesses, and research actors to solve problems and "
                "improve quality of life."
            ),
            objective_text="A working method for generating innovative solutions and supporting innovation.",
            implementation_text=(
                "Use a collaborative instrument between city authorities and citizens, businesses "
                "and research actors to bring together city resources and innovative solutions to "
                "solve problems and improve the quality of life of citizens, while addressing "
                "barriers related to finance, local and district centres, blue-green infrastructure, "
                "building-efficiency needs, fuel poverty, and data use."
            ),
            planned_outputs_text=(
                "Actions in terms of the Living Lab involving a large number of diverse urban "
                "stakeholders will increase the chance of developing innovative solutions and taking "
                "diverse needs into account concerning the planned activities."
            ),
            delivery_text=(
                "Local Government: Economic Development Department; the cells and units of the City "
                "of Warsaw, municipal companies; science sector: universities and scientific "
                "institutions; manufacturing, trading, service companies; residents."
            ),
            source_document="Warszawa.md",
            source_quote="Urban Living Lab",
        ),
        make_record(
            city="Wroclaw",
            title="Change the stove",
            general_description=(
                "A municipal support and engagement programme for replacing stoves and reducing "
                "emissions from individual heat sources."
            ),
            objective_text="Support system consisting of municipal Energy Advisors, dedicated funds and cooperation of many groups of stakeholders.",
            implementation_text=(
                "Coordinated by the Sustainable Development Department with municipal Energy "
                "Advisors, dedicated funds, and cooperation across multiple stakeholders; since "
                "2020 related actions have been gathered under the umbrella \"Change the stove\" "
                "action, including outreach such as the \"Change the stove\" local hero competition."
            ),
            planned_outputs_text=(
                "Improved air quality, reduced carbon footprint of individual heat sources, and "
                "higher resident awareness and participation in stove replacement."
            ),
            delivery_text=(
                "Led by the municipality and coordinated by the Sustainable Development Department "
                "in cooperation with various entities; beneficiaries are city residents, especially "
                "from poorer neighbourhoods who needed the most support."
            ),
            source_document="Wroclaw.md",
            source_quote="Change the stove",
        ),
        make_record(
            city="Wroclaw",
            title="Wroclaw Participatory Budget",
            general_description=(
                "A participatory budgeting tool through which residents directly decide how part of "
                "the city budget is allocated, including climate-related and green projects."
            ),
            objective_text=(
                "It is a part of the city's budget, the allocation of which is decided by residents "
                "themselves through direct voting."
            ),
            implementation_text=(
                "Residents submit proposals and vote on projects; the programme includes many green "
                "investments such as parks, squares, bicycle and pedestrian paths, and other nature-"
                "based solutions, and has included a special green edition."
            ),
            planned_outputs_text=(
                "Trust-building, stronger cross-sector partnerships, deeper climate-transition "
                "awareness, shared ownership of climate action, and support for environmental "
                "projects including NBS, clean transportation, education, and light-pollution reduction."
            ),
            delivery_text=(
                "Led by the Public Participatory Division in cooperation with city divisions "
                "reviewing projects and municipal companies implementing them; each project also "
                "depends on a local proposal leader and resident voting."
            ),
            source_document="Wroclaw.md",
            source_quote="Wroclaw Participatory Budget",
        ),
        make_record(
            city="Wroclaw",
            title="Civic panel",
            general_description=(
                "A deliberative public-engagement process using a demographically selected panel to "
                "shape mobility and climate-related city policy."
            ),
            objective_text=(
                "A civic panel representing the demographic structure of the city was chosen to "
                "analyze how to improve movement around Wroclaw with a view to improving quality of "
                "life and protecting the climate."
            ),
            implementation_text=(
                "Coordinated by the Public Participatory Office in cooperation with the Department "
                "of Infrastructure and Transportation and other divisions; 75 participants were "
                "selected to reflect the city's demographic structure, supported by experts, and "
                "produced recommendations reflected in transport and planning documents."
            ),
            planned_outputs_text=(
                "Trust-building, stronger partnerships, deeper awareness of the climate transition, "
                "shared ownership of climate action, and support for solutions improving air quality "
                "and biodiversity."
            ),
            delivery_text=(
                "Led by the public sector at city level and coordinated by the Public Participatory "
                "Office with the Department of Infrastructure and Transportation; intended for a "
                "broad stakeholder group represented by lottery-selected participants."
            ),
            source_document="Wroclaw.md",
            source_quote="Civic panel",
        ),
        make_record(
            city="Wroclaw",
            title="NEEST (on-going)",
            general_description=(
                "An ongoing pilot project to develop comprehensive and locally acceptable retrofit "
                "models for buildings and neighbourhoods to improve energy efficiency and reduce "
                "greenhouse gas emissions."
            ),
            objective_text=(
                "Project focuses on solutions that will enable comprehensive retrofitting of "
                "buildings and neighbourhoods to improve their energy efficiency."
            ),
            implementation_text=(
                "Led by the Sustainable Development Department in cooperation with the Transition "
                "Team; includes systems-thinking and systems-innovation workshops, participatory "
                "systems mapping, and co-creation with building managers, housing communities, "
                "residents, energy suppliers, and researchers."
            ),
            planned_outputs_text=(
                "Recommendations on effective, economically efficient, and socially acceptable "
                "modernisation of residential and non-residential buildings, plus deeper knowledge "
                "and awareness of the climate transition and systemic dependencies."
            ),
            delivery_text=(
                "Led by the public sector at city level through the Sustainable Development "
                "Department in cooperation with the Transition Team; dedicated to stakeholders "
                "involved in building energy efficiency, including building managers, housing "
                "communities, residents, energy suppliers, and researchers."
            ),
            source_document="Wroclaw.md",
            source_quote="NEEST (on-going)",
        ),
        make_record(
            city="Wroclaw",
            title="Low-emission Zone (participatory process on-going)",
            general_description=(
                "A participatory process to introduce a clean transportation zone in the city "
                "centre and parts of the downtown area."
            ),
            objective_text="An introduction of a clean transportation zone for the city centre and parts of the downtown area.",
            implementation_text=(
                "Led by the Department of Infrastructure and Transportation through a three-stage "
                "consultation process covering zone variants, a draft zone for further consultation, "
                "a civic meeting stage, and a final public-opinion stage on the draft resolution."
            ),
            planned_outputs_text=(
                "Deeper awareness of the climate transition and stronger incentives to choose "
                "public transportation through a clean transportation zone."
            ),
            delivery_text=(
                "Led by the public sector at city level through the Department of Infrastructure "
                "and Transportation; stakeholders include residents, the private sector, academia, "
                "and the third sector."
            ),
            source_document="Wroclaw.md",
            source_quote="Low-emission Zone (participatory process on-going)",
        ),
    ]


def apply_updates(records_by_id: dict[str, dict[str, Any]]) -> None:
    """Apply field updates and record rewrites in place."""
    records_by_id["krakow:krakow:title_b23a43dbef5d"] = make_record(
        city="Krakow",
        title="ATELIER project - AmsTErdam and BiLbao citizen drivEn smaRt cities",
        general_description=(
            "An innovation and technology project to create Positive Energy Districts and "
            "replicate smart city solutions."
        ),
        objective_text=(
            "Reduce CO2 emissions through local smart city solutions; support sustainable, safe "
            "and accessible energy systems; promote cooperation and knowledge sharing; replicate "
            "solutions in partner cities."
        ),
        implementation_text=(
            "Gain knowledge and skills in integrated energy districts, organise a local "
            "Innovation Atelier, integrate renewable energy into district heating and cooling, "
            "develop a replication plan for selected areas, increase resident and stakeholder "
            "involvement, and develop a City Vision 2050."
        ),
        planned_outputs_text="Positive Energy Districts, a replication plan, and a City Vision 2050 pathway.",
        delivery_text=(
            "Project partners; on the Krakow side - the Department of Municipal Economy and "
            "Climate; international project with partner cities."
        ),
        funding_text=(
            "The project is funded by the European Union under the Horizon 2020 programme. "
            "Estimated budget on the Krakow side - PLN 1 259 765 (approx. EUR 279 000)."
        ),
        timeline_text="The planned timeline for the project is 2019-2024.",
        numbers_planned={
            "estimated_emissions_reduction_electricity_tco2e": 48,
            "project_budget_pln": 1259765,
            "project_budget_eur_approx": 279000,
            "start_year": 2019,
            "end_year": 2024,
        },
        source_document="Krakow.md",
        source_quote="ATELIER project - AmsTErdam and BiLbao citizen drivEn smaRt cities.",
        data_quality_flags=["page_artifacts_inside_section", "source_contains_deferred_values"],
        record_id="krakow:krakow:title_b23a43dbef5d",
        document_local_code="E-12",
    )
    records_by_id["krakow:krakow:title_a5c831c856d0"]["initiative"]["numbers"]["current"][
        "signatories_count"
    ] = 11

    records_by_id["rzeszow:rzeszow:title_0e4224cd1034"] = make_record(
        city="Rzeszow",
        title="Replacement of heating systems and comprehensive thermal modernization of single-family residential buildings",
        general_description=(
            "Single-family home decarbonization and retrofit action focused on replacing heating "
            "systems, eliminating solid-fuel sources, and comprehensive thermal modernization."
        ),
        objective_text=(
            "Elimination of all sources of fossil fuels in the municipal sector. Increase in the "
            "number of buildings undergoing deep thermal modernization. Increase in the number of "
            "buildings using energy management systems. Increase in the number of RES."
        ),
        implementation_text=(
            "Implement the provisions of the anti-smog resolution for the Podkarpackie "
            "Voivodeship and the Air Protection Program for the urban zone. Completely eliminate "
            "individual solid-fuel heating sources in residential buildings, carry out thermal "
            "modernization, install photovoltaic installations, heat pumps, and other renewable "
            "energy sources, optimize energy consumption, and help households access financing "
            "through programs such as Clean Air, STOP SMOG, and Warm Apartment."
        ),
        planned_outputs_text=(
            "Replaced heating systems in single-family residential buildings, comprehensive "
            "thermal modernization, broader use of photovoltaic installations and heat pumps, and "
            "higher uptake of retrofit-support programs."
        ),
        delivery_text=(
            "City Municipality of Rzeszow - Rzeszow City Office/WI and KS, residents, MPEC, "
            "entrepreneurs, housing cooperatives, housing communities, NFOŚiGW, WFOŚiGW, the "
            "Marshal of the Podkarpackie Voivodeship, and financing institutions."
        ),
        source_document="Rzeszow.md",
        source_quote="Action name | Replacement of heating systems and comprehensive thermal modernization of single-family residential buildings.",
        record_id="rzeszow:rzeszow:title_0e4224cd1034",
    )
    records_by_id["rzeszow:rzeszow:title_7a596c4cf28a"] = make_record(
        city="Rzeszow",
        title="Developing or modifying spatial development plans and other planning documents to increase public spaces with new green areas, blue-green infrastructure solutions, and de-paving of high-density urban areas",
        general_description=(
            "Spatial-planning action to revise development plans and other planning documents so "
            "they create more green public space, blue-green infrastructure, and de-paving."
        ),
        objective_text="Adaptation of urban planning and regulatory tools related to increasing the area of green spaces.",
        implementation_text=(
            "Include environmental and climate-protection requirements in planning documents and "
            "diagnoses. Introduce provisions on permissible heating sources for buildings, green "
            "areas with various functions, minimum biologically active areas for plots or land, "
            "and stormwater and meltwater management. Identify and protect land for future green "
            "spaces and improve local law to increase land retention, reduce surface runoff, lower "
            "air temperature, and improve air quality."
        ),
        planned_outputs_text=(
            "Planning documents and diagnoses with provisions on heating sources, green areas, "
            "biologically active land, stormwater and meltwater management, and protection of land "
            "for future green spaces and blue-green infrastructure."
        ),
        delivery_text=(
            "City Municipality of Rzeszow - Office for the Development of the City of Rzeszow, "
            "BRMR, BGM, ZZM, Rzeszow City Office/WI, KS, residents, entrepreneurs, developers, "
            "and housing cooperatives."
        ),
        timeline_text=(
            "From January 2022 to June 2024, 15 spatial development plans were adopted covering "
            "about 310 ha. Another 58 plans are in preparation, and work on the general plan is underway."
        ),
        source_document="Rzeszow.md",
        source_quote="Action name | Developing or modifying spatial development plans and other planning documents to increase public spaces with new green areas, blue-green infrastructure solutions, and de-paving of high-density urban areas.",
        record_id="rzeszow:rzeszow:title_7a596c4cf28a",
    )
    records_by_id["rzeszow:rzeszow:title_c9f675409da9"] = make_record(
        city="Rzeszow",
        title="Rzeszów Citizens' Budget",
        general_description=(
            "A participatory budgeting mechanism allowing residents to propose and select projects "
            "financed from the city budget, including green projects."
        ),
        objective_text=(
            "Enable residents to decide how part of the city budget is spent and increase the "
            "number of green projects selected for implementation."
        ),
        implementation_text=(
            "Residents submit projects and vote on which will be implemented through the citizens' budget."
        ),
        planned_outputs_text=(
            "Resident-selected projects financed through the city budget, including more green projects."
        ),
        delivery_text="Rzeszów City Office, residents, and non-governmental organizations.",
        funding_text="Projects are financed from the city budget.",
        timeline_text="Ongoing participatory budget process since 2014.",
        source_document="Rzeszow.md",
        source_quote="Possibility of submitting and selecting projects financed from the city budget.",
        record_id="rzeszow:rzeszow:title_c9f675409da9",
    )
    records_by_id["rzeszow:rzeszow:title_3a30fdb4ac8d"] = make_record(
        city="Rzeszow",
        title="Development of charging infrastructure throughout the city",
        general_description="Expansion of electric vehicle charging infrastructure citywide to support vehicle electrification.",
        objective_text="Make urban electric vehicle charging networks more attractive and ensure their high availability.",
        implementation_text=(
            "Develop charging infrastructure throughout the city, with the goal of providing "
            "access to charging stations at every parking lot. Plan at least 100 publicly "
            "accessible EV charging stations and improve infrastructure needed for electrification "
            "of road transport."
        ),
        planned_outputs_text="Charging access across parking lots and at least 100 publicly accessible charging stations.",
        delivery_text=(
            "City Municipality of Rzeszów, City Transport Management Authority, Department of "
            "Climate and Environment, Department of Energy, Department of Investments, network "
            "owners, fleet owners, property owners, providers, and neighboring municipalities."
        ),
        timeline_text=(
            "New charging locations and gradual replacement of the municipal fleet are planned "
            "until 2027; a significant increase in EVs is expected by 2030."
        ),
        numbers_planned={
            "charging_access_parking_lots_percent": 30,
            "publicly_accessible_charging_stations_min": 100,
            "ghg_emissions_reduction_tco2e": 199869,
        },
        source_document="Rzeszow.md",
        source_quote="Development of charging infrastructure throughout the city (with the goal of providing access to charging stations at every parking lot).",
        record_id="rzeszow:rzeszow:title_3a30fdb4ac8d",
    )
    records_by_id["rzeszow:rzeszow:title_66c25ef734e4"] = make_record(
        city="Rzeszow",
        title="Decarbonization and development of local heating networks",
        general_description="Expansion and modernization of local district heating networks with smart-network solutions.",
        objective_text=(
            "Expand the district heating network, replace heat sources from solid fuels, oil, and "
            "gas with heat pumps or connection to an efficient district heating system, and "
            "develop smart district heating networks."
        ),
        implementation_text=(
            "Replace heat sources in local heating networks; ban solid fuels and enforce it; "
            "change heat sources; provide subsidies for heat pumps; develop smart district heating "
            "networks integrating monitoring, control, and communication technologies; construct "
            "and modernize district heating networks and connections; eliminate group heat "
            "exchangers; and add telemetry and remote monitoring systems."
        ),
        planned_outputs_text=(
            "Expanded local heating networks, heat source replacement, smart district heating "
            "networks, telemetry and telemechanics systems, and PV-supported local heat source infrastructure."
        ),
        delivery_text="MPEC Rzeszów with heat recipients and financial institutions.",
        funding_text="Total costs: PLN 258,826,786, EUR 57,517,064; costs by CO2e unit: PLN 6,026.38, EUR 1,339.20.",
        timeline_text="The company's development plans are planned until 2029.",
        numbers_planned={
            "ghg_emissions_reduction_tco2e": 53676,
            "total_cost_pln": 258826786,
            "total_cost_eur": 57517064,
            "cost_per_co2e_pln": 6026.38,
            "cost_per_co2e_eur": 1339.2,
            "planned_until_year": 2029,
        },
        source_document="Rzeszow.md",
        source_quote="Decarbonization and development of local heating networks. Smart network solutions.",
        record_id="rzeszow:rzeszow:title_66c25ef734e4",
    )
    records_by_id["warszawa:warszawa:title_7735a997af6b"] = make_record(
        city="Warszawa",
        title="Implementation of sustainable rainwater management systems",
        general_description=(
            "Warsaw will implement sustainable rainwater and snowmelt management measures to "
            "reduce urban flooding and waterlogging, improve blue-green infrastructure monitoring, "
            "and strengthen adaptive water management across the city."
        ),
        objective_text=(
            "Developing blue-green infrastructure in the city\n"
            "Increasing the quality of life of residents, the level of health, as well as the "
            "tourist attractiveness of Warsaw and the surrounding area by expanding and diversifying "
            "the urban ecosystem and creating friendly places for recreation and relaxation."
        ),
        implementation_text=(
            "Examples of actions to be implemented: using rainwater and snowmelt on site and to "
            "counteract urban flooding and waterlogging; implementing measures and monitoring "
            "rainwater and snowmelt management concepts within the city (followed by the "
            "metropolitan area); and using smart solutions for urban green inventory and ecosystem "
            "service assessment. Actions should focus on implementing the concept of rainwater and "
            "snowmelt management by all urban units, introducing a uniform reporting system for the "
            "entire city and building incentives for the private sector, including financial "
            "support and educational and informational activities related to adaptive water management."
        ),
        planned_outputs_text=(
            "Site-level rainwater and snowmelt use, citywide rainwater-management implementation "
            "and monitoring, a uniform reporting system, incentives for private adaptive water "
            "management, and strengthened protection, restoration, and renaturalisation of Warsaw's hydrographic system."
        ),
        delivery_text=(
            "City Hall, District Offices, and municipal entities; district authorities, city "
            "companies, residents, national and regional authorities, private sector."
        ),
        funding_text=(
            "1 rainwater retention tank (design documentation, purchase of tank, installation of "
            "pumps and modules) PLN 80,000 / EUR 17,777; 1 rain garden (design documentation, "
            "implementation, maintenance - price depends on garden size and number of individual "
            "elements) PLN 200,000 / EUR 44,444."
        ),
        source_document="Warszawa.md",
        source_quote="Action G-4\n| Implementation of sustainable rainwater management systems",
        data_quality_flags=["page_artifacts_inside_section"],
        record_id="warszawa:warszawa:title_7735a997af6b",
        document_local_code="G-4",
    )


def normalize_wroclaw_units(records_by_id: dict[str, dict[str, Any]]) -> list[str]:
    """Rename wrong Wroclaw `kt` emissions fields to tonnes-based fields."""
    fixed_ids = [
        "wroclaw:wroclaw:title_192513aaf97e",
        "wroclaw:wroclaw:title_7dd8a50acc71",
        "wroclaw:wroclaw:title_d6a4a4b884db",
        "wroclaw:wroclaw:title_f0fa092bad1e",
        "wroclaw:wroclaw:title_c4036654c9b6",
        "wroclaw:wroclaw:title_a2ccfe7078f6",
        "wroclaw:wroclaw:title_ddeab5c7c60c",
        "wroclaw:wroclaw:title_d363160f998c",
        "wroclaw:wroclaw:title_66331778250e",
        "wroclaw:wroclaw:title_d2ca93edb6f3",
        "wroclaw:wroclaw:title_95ccc7c798ec",
    ]
    for record_id in fixed_ids:
        record = records_by_id.get(record_id)
        if not record:
            continue
        planned = record["initiative"]["numbers"].setdefault("planned", {})
        if "emissions_reduction_kt_co2" in planned:
            planned["emissions_reduction_t_co2e"] = planned.pop("emissions_reduction_kt_co2")

    pedestrian = records_by_id["wroclaw:wroclaw:title_ddeab5c7c60c"]
    pedestrian["source_quote"] = (
        "Action T-3 Implementation of a pedestrian programme (in accordance with "
        "Wroclaw standards for shaping pedestrian-friendly urban spaces)"
    )
    pedestrian["data_quality_flags"] = [
        flag for flag in pedestrian["data_quality_flags"] if flag != "source_quote_not_found"
    ]
    return fixed_ids


def rewrite_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the full fix set and return corrected records plus manifest data."""
    records_by_id = {record["record_id"]: copy.deepcopy(record) for record in records}
    removals = build_removals()
    for record_id in removals:
        records_by_id.pop(record_id, None)

    apply_updates(records_by_id)
    unit_fix_ids = normalize_wroclaw_units(records_by_id)
    additions = build_additions()
    for record in additions:
        if record["record_id"] in records_by_id:
            raise ValueError(f"Added record id already exists: {record['record_id']}")
        records_by_id[record["record_id"]] = record

    corrected = sorted(
        records_by_id.values(),
        key=lambda item: (
            item["initiative"]["city"],
            item["initiative"]["initiative_name"].casefold(),
            item["record_id"],
        ),
    )
    record_ids = [record["record_id"] for record in corrected]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Record id collision detected after audit fixes.")

    manifest = {
        "removed_record_ids": sorted(removals),
        "rewritten_record_ids": [
            "krakow:krakow:title_b23a43dbef5d",
            "krakow:krakow:title_a5c831c856d0",
            "rzeszow:rzeszow:title_0e4224cd1034",
            "rzeszow:rzeszow:title_7a596c4cf28a",
            "rzeszow:rzeszow:title_c9f675409da9",
            "rzeszow:rzeszow:title_3a30fdb4ac8d",
            "rzeszow:rzeszow:title_66c25ef734e4",
            "warszawa:warszawa:title_7735a997af6b",
            "wroclaw:wroclaw:title_ddeab5c7c60c",
        ],
        "unit_normalized_record_ids": unit_fix_ids,
        "added_record_ids": [record["record_id"] for record in additions],
    }
    return corrected, manifest


def write_corrected_run(
    *,
    source_run_dir: Path,
    output_run_dir: Path,
    corrected_records: list[dict[str, Any]],
    manifest: dict[str, Any],
    city_counts_before: dict[str, int],
    city_counts_after: dict[str, int],
    overwrite: bool,
) -> None:
    """Copy the source run and overwrite the corrected final artifacts."""
    if output_run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_run_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_run_dir)

    shutil.copytree(source_run_dir, output_run_dir)

    initiative_records_path = output_run_dir / "03_deduped" / "initiative_records.jsonl"
    initiatives_path = output_run_dir / "03_deduped" / "initiatives.jsonl"
    with initiative_records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in corrected_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with initiatives_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in corrected_records:
            handle.write(json.dumps(record["initiative"], ensure_ascii=False) + "\n")

    summary_path = output_run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_id"] = output_run_dir.name
    summary["source_run_id"] = source_run_dir.name
    summary["deduped_initiatives_count"] = len(corrected_records)
    summary["audit_fix_counts"] = {
        "removed_records": len(manifest["removed_record_ids"]),
        "added_records": len(manifest["added_record_ids"]),
        "updated_or_rewritten_records": len(manifest["rewritten_record_ids"]),
        "unit_normalized_records": len(manifest["unit_normalized_record_ids"]),
    }
    summary["city_counts_before"] = city_counts_before
    summary["city_counts_after"] = city_counts_after
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_payload = {
        "source_run_id": source_run_dir.name,
        "fixed_run_id": output_run_dir.name,
        **manifest,
        "city_counts_before": summary["city_counts_before"],
        "city_counts_after": summary["city_counts_after"],
    }
    (output_run_dir / "audit_fix_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme_path = output_run_dir / "README.md"
    original_readme = readme_path.read_text(encoding="utf-8")
    note = (
        "AUDIT-FIX NOTE\n"
        f"This run is a corrected copy of `{source_run_dir.name}` based on "
        "`polish_cities_extraction_tef_audit_report_2026-04-23.md`. "
        "Only `summary.json`, `audit_fix_manifest.json`, and "
        "`03_deduped/{initiative_records.jsonl,initiatives.jsonl}` were changed here; "
        "upstream stage artifacts were copied from the source run for reference.\n\n"
    )
    readme_path.write_text(note + original_readme, encoding="utf-8")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    output_run_dir = args.output_run_dir.resolve()

    logger.info("Applying extraction audit fixes from source_run_dir=%s", source_run_dir)
    source_records = load_records(source_run_dir / "03_deduped" / "initiative_records.jsonl")
    corrected_records, manifest = rewrite_records(source_records)
    city_counts_before = count_by_city(source_records)
    city_counts_after = count_by_city(corrected_records)
    write_corrected_run(
        source_run_dir=source_run_dir,
        output_run_dir=output_run_dir,
        corrected_records=corrected_records,
        manifest=manifest,
        city_counts_before=city_counts_before,
        city_counts_after=city_counts_after,
        overwrite=args.overwrite,
    )

    logger.info(
        "Wrote corrected run to %s with %s deduped initiatives.",
        output_run_dir,
        len(corrected_records),
    )


if __name__ == "__main__":
    setup_logger()
    main()
