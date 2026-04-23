"""
Brief: Apply the second-pass 2026-04-23 Poland extraction fixes to the corrected Polish-city extraction run.

Inputs:
- `--source-run-dir`: Corrected extraction run directory to extend. Expected to contain
  `summary.json`, `README.md`, and `03_deduped/{initiative_records.jsonl,initiatives.jsonl}`.
- `--output-run-dir`: Destination directory for the second-pass corrected copy of the run.
- `--overwrite`: Remove the output directory first if it already exists.
- Files/paths: the script reads only the final deduped extraction artifacts from the source run and
  copies the full source run directory for reference.
- Env vars: none required.

Outputs:
- A second-pass corrected copy of the run directory at `--output-run-dir`
- Updated `summary.json`
- Updated `03_deduped/initiative_records.jsonl`
- Updated `03_deduped/initiatives.jsonl`
- `audit_fix_manifest.json` describing the second-pass additions

Usage (from project root):
- python -m backend.scripts.apply_polish_remaining_extraction_fixes --source-run-dir output/initiative_extraction/polish_cities_flow_test_20260423_001_audit_fixed --output-run-dir output/initiative_extraction/polish_cities_flow_test_20260423_001_audit_fixed_v2 --overwrite
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from backend.scripts.apply_polish_extraction_audit_fixes import (
    count_by_city,
    load_records,
    make_record,
    write_corrected_run,
)
from backend.utils.logging_config import setup_logger


logger = logging.getLogger(__name__)

CITY_DOCUMENTS = {
    "Krakow": "Krakow.md",
    "Lodz": "Lodz.md",
    "Rzeszow": "Rzeszow.md",
    "Warszawa": "Warszawa.md",
    "Wroclaw": "Wroclaw.md",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Apply the second-pass Poland extraction fixes to a corrected extraction run."
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
        help="Destination directory for the second-pass corrected run copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the destination directory first if it already exists.",
    )
    return parser.parse_args()


def make_city_record(
    *,
    city: str,
    title: str,
    general_description: str,
    objective_text: str,
    implementation_text: str,
    planned_outputs_text: str,
    delivery_text: str,
    source_quote: str,
    funding_text: str | None = None,
    timeline_text: str | None = None,
    numbers_current: dict[str, Any] | None = None,
    numbers_planned: dict[str, Any] | None = None,
    document_local_code: str | None = None,
    record_id: str | None = None,
) -> dict[str, Any]:
    """Build one manual extraction record anchored to a Polish city markdown file."""
    return make_record(
        city=city,
        title=title,
        general_description=general_description,
        objective_text=objective_text,
        implementation_text=implementation_text,
        planned_outputs_text=planned_outputs_text,
        delivery_text=delivery_text,
        funding_text=funding_text,
        timeline_text=timeline_text,
        numbers_current=numbers_current,
        numbers_planned=numbers_planned,
        source_document=CITY_DOCUMENTS[city],
        source_quote=source_quote,
        document_local_code=document_local_code,
        record_id=record_id,
    )


def build_remaining_additions() -> list[dict[str, Any]]:
    """Return the verified second-pass additions still missing after the first repair pass."""
    return [
        make_city_record(
            city="Krakow",
            title="Project NEEST - NetZero Emission and Environmentally Sustainable Territories",
            general_description=(
                "A NetZeroCities pilot project in Krakow focused on scalable deep-retrofit models "
                "for urban quarters and the creation of local energy communities."
            ),
            objective_text=(
                "Prepare and test innovative solutions for deep thermal modernisation, smart urban "
                "resource management, and citizen-centred energy communities."
            ),
            implementation_text=(
                "Krakow groups NEEST with ATELIER and COMANAGE as part of its smart-city and "
                "resource-management work, with a focus on large-scale thermal retrofit projects in "
                "urban quarters and on creating local energy communities."
            ),
            planned_outputs_text=(
                "Scalable retrofit and energy-community solutions that can support Krakow's climate "
                "neutrality pathway."
            ),
            delivery_text="Municipality of Krakow and project partners working on climate-neutral city pilots.",
            source_quote="BIC-8 - Project NEEST - NetZero Emission and Environmentally Sustainable Territories.",
            document_local_code="BIC-8",
            record_id="krakow:krakow:bic_8_neest",
        ),
        make_city_record(
            city="Krakow",
            title="Programme for the development of renewable energy sources in the Municipality of Krakow",
            general_description=(
                "A city subsidy programme for residents and housing entities to install renewable "
                "energy systems, energy storage, EV charging, and energy-management systems."
            ),
            objective_text=(
                "Expand renewable energy uptake in residential and housing-association buildings and "
                "support cleaner energy carriers in Krakow."
            ),
            implementation_text=(
                "Since 2020, PROZE has offered targeted grants for air-source and ground-source heat "
                "pumps, solar collectors, PV systems, electricity storage, EV charging stations, and "
                "energy-management systems; from 2022 the programme was extended to housing "
                "associations and cooperatives."
            ),
            planned_outputs_text=(
                "Subsidised installations of heat pumps, solar collectors, PV systems, energy "
                "storage, EV charging stations, and energy-management systems across Krakow."
            ),
            delivery_text="Municipality of Krakow - Department of Air Quality; residents, housing entities, and contractors.",
            funding_text=(
                "Costs for 2020-2024 - approx. PLN 33,000,000; estimated costs for 2025-2030 - "
                "approx. PLN 55,000,000; total estimated costs - PLN 88,000,000 (approx. EUR 19,555,000)."
            ),
            timeline_text="The planned timeframe for the task is 2020-2030.",
            numbers_planned={
                "estimated_emissions_reduction_buildings_heating_tco2e": 483,
                "estimated_emissions_reduction_electricity_tco2e": 1931,
                "estimated_cost_pln": 88000000,
                "estimated_cost_eur_approx": 19555000,
                "start_year": 2020,
                "end_year": 2030,
            },
            source_quote="Programme for the development of renewable energy sources in the Municipality of Krakow.",
            document_local_code="E-2",
        ),
        make_city_record(
            city="Krakow",
            title="Construction of new CHP systems",
            general_description=(
                "A Municipal Heating Company action to build new H2-ready cogeneration systems for "
                "Krakow's energy system."
            ),
            objective_text=(
                "Transform the city's energy system to a less carbon-intensive one through new CHP "
                "capacity."
            ),
            implementation_text=(
                "The action covers construction of H2Ready cogeneration systems by the Municipal "
                "Heating Company to support district-heating decarbonisation and cleaner energy supply."
            ),
            planned_outputs_text="New H2-ready CHP capacity serving Krakow's heat and electricity system.",
            delivery_text="Municipal Heating Company S.A.",
            funding_text="Estimated investment costs - PLN 92,000,000 (approx. EUR 20,444,000).",
            timeline_text="The planned timeframe for the task is 2025-2028.",
            numbers_planned={
                "capacity_mw_approx": 13,
                "estimated_emissions_reduction_buildings_heating_tco2e": 6803,
                "estimated_emissions_reduction_electricity_tco2e": 27212,
                "investment_cost_pln": 92000000,
                "investment_cost_eur_approx": 20444000,
                "start_year": 2025,
                "end_year": 2028,
            },
            source_quote="Construction of new CHP systems.",
            document_local_code="E-7",
        ),
        make_city_record(
            city="Krakow",
            title="Construction of photovoltaic farms and local installations by MPEC",
            general_description=(
                "A Municipal Heating Company action to build photovoltaic farms and local PV "
                "installations that support a less carbon-intensive urban energy system."
            ),
            objective_text="Increase local renewable electricity generation within Krakow's energy system.",
            implementation_text=(
                "Municipal Heating Company S.A. plans farms and local photovoltaic installations to "
                "supply the city's energy system and support the shift away from carbon-intensive generation."
            ),
            planned_outputs_text="Photovoltaic farms and local PV installations operated by MPEC.",
            delivery_text="Municipal Heating Company S.A.",
            funding_text="Estimated investment costs - PLN 134,000,000 (approx. EUR 29,777,000).",
            timeline_text="The planned timeframe for the task is 2025-2028.",
            numbers_planned={
                "pv_capacity_mw_approx": 32,
                "estimated_emissions_reduction_electricity_tco2e": 23520,
                "investment_cost_pln": 134000000,
                "investment_cost_eur_approx": 29777000,
                "start_year": 2025,
                "end_year": 2028,
            },
            source_quote="Construction of photovoltaic farms and local installations by MPEC.",
            document_local_code="E-8",
        ),
        make_city_record(
            city="Krakow",
            title="ZIT - Construction of energy storages",
            general_description=(
                "A ZIT project to build a scalable energy storage facility at the Thermal Waste "
                "Conversion Plant."
            ),
            objective_text=(
                "Support renewable-energy use, improve energy efficiency, and reduce air pollution "
                "through scalable storage infrastructure."
            ),
            implementation_text=(
                "The project covers construction of an energy-storage facility at the Thermal Waste "
                "Conversion Plant and is linked to other metropolitan strategy projects addressing "
                "renewable energy, air pollution, and public-building energy efficiency."
            ),
            planned_outputs_text="A scalable energy storage facility for the Krakow metropolitan energy system.",
            delivery_text=(
                "The Krakow Metropolia Association together with the Municipality of Krakow and "
                "Krakow Municipal Holding S.A."
            ),
            funding_text=(
                "The measure has secured funding from the European Fund for Małopolska 2021-2027. "
                "Estimated investment costs - PLN 2,000,000 (approx. EUR 444,000)."
            ),
            timeline_text="Action only in the design phase, currently no timetable or milestones.",
            numbers_planned={
                "investment_cost_pln": 2000000,
                "investment_cost_eur_approx": 444000,
            },
            source_quote="ZIT - Construction of energy storages.",
            document_local_code="E-9",
        ),
        make_city_record(
            city="Krakow",
            title="Installation of two cogeneration units at the Płaszów Treatment Plant",
            general_description=(
                "A Waterworks action to install two cogeneration units and supporting electrical "
                "infrastructure at the Płaszów Treatment Plant."
            ),
            objective_text=(
                "Increase local cogeneration capacity and support cleaner electricity and heat supply "
                "at the treatment plant."
            ),
            implementation_text=(
                "The task covers installation of two cogeneration units with transformer-station and "
                "switchgear works; the action was in the final implementation phase with planned "
                "completion in 2024."
            ),
            planned_outputs_text=(
                "Two CHP units with electrical output of 800 kW and thermal output of 790 kW each."
            ),
            delivery_text="Waterworks of the City of Krakow S.A.",
            funding_text="Estimated investment costs - PLN 8,000,000 (approx. EUR 1,777,000).",
            timeline_text="Planned completion date for the task - 2024.",
            numbers_planned={
                "electrical_output_kw_per_unit": 800,
                "thermal_output_kw_per_unit": 790,
                "units_count": 2,
                "estimated_emissions_reduction_buildings_heating_tco2e": 592,
                "estimated_emissions_reduction_electricity_tco2e": 2366,
                "investment_cost_pln": 8000000,
                "investment_cost_eur_approx": 1777000,
                "end_year": 2024,
            },
            source_quote="Installation of two cogeneration units at the Płaszów Treatment Plant.",
            document_local_code="E-10",
        ),
        make_city_record(
            city="Krakow",
            title="COMANAGE project",
            general_description=(
                "A project to develop a toolkit, support centres, and learning resources for the "
                "creation and management of energy communities."
            ),
            objective_text=(
                "Enable existing energy communities to operate and grow and stimulate the creation "
                "of new citizen-led energy projects."
            ),
            implementation_text=(
                "The project develops a methodological and operational framework for energy "
                "communities, mobilises a transnational network of competence providers, and pilots "
                "three Operational Integrated Service Centres."
            ),
            planned_outputs_text=(
                "An energy-community management toolkit, e-learning resources, pilot HUBs, and "
                "implementation plans for energy-community support."
            ),
            delivery_text="Krakow Metropolis Association with the international COMANAGE consortium.",
            funding_text="Estimated total project budget - PLN 7,300,000 (approx. EUR 1,622,000).",
            timeline_text="The planned timeframe for the task is 2022-2025.",
            numbers_planned={
                "estimated_emissions_reduction_electricity_tco2e": 269,
                "project_budget_pln": 7300000,
                "project_budget_eur_approx": 1622000,
                "start_year": 2022,
                "end_year": 2025,
            },
            source_quote="COMANAGE project.",
            document_local_code="E-13",
        ),
        make_city_record(
            city="Krakow",
            title="Development of distributed renewable energy generation",
            general_description=(
                "A city-wide action to accelerate distributed renewable-energy generation on "
                "buildings, brownfields, industrial sites, and through heat pumps."
            ),
            objective_text=(
                "Expand individual electricity and heat generation from distributed renewable "
                "energy sources across Krakow."
            ),
            implementation_text=(
                "The measure supports rooftop PV on private buildings, heat pumps, plan changes that "
                "facilitate PV and brownfield farms, support for historic-building documentation, and "
                "broader advisory and subsidy support for distributed generation."
            ),
            planned_outputs_text=(
                "Broader distributed PV and heat-pump deployment across residential, commercial, "
                "industrial, and brownfield locations in Krakow."
            ),
            delivery_text="Municipality of Krakow as monitoring entity, with municipal companies, private companies, and residents.",
            funding_text="Estimated costs - PLN 5,186,553,899 (approx. EUR 1,152,567,000).",
            timeline_text="The planned timeframe for the task is 2024-2030.",
            numbers_planned={
                "residential_and_commercial_rooftop_pv_mwh_per_year_2030": 345564,
                "industrial_site_res_mwh_per_year_2030": 129865,
                "heat_pump_generation_mwh_per_year_2030": 103270,
                "estimated_emissions_reduction_buildings_heating_tco2e": 46553,
                "estimated_emissions_reduction_electricity_tco2e": 330004,
                "estimated_cost_pln": 5186553899,
                "estimated_cost_eur_approx": 1152567000,
                "start_year": 2024,
                "end_year": 2030,
            },
            source_quote="Development of distributed renewable energy generation.",
            document_local_code="E-14",
        ),
        make_city_record(
            city="Krakow",
            title="Participation in the SmartEPC project",
            general_description=(
                "A Horizon 2020 smart-lighting project in which Krakow tests smart-city elements "
                "through its street-lighting infrastructure."
            ),
            objective_text=(
                "Use intelligent street-lighting infrastructure to reduce electricity consumption and "
                "test smart-city applications."
            ),
            implementation_text=(
                "Krakow participates as a project partner and pilot city, using SmartEPC standards "
                "for energy-efficient street-lighting replacement and testing LED lanterns for vehicle charging."
            ),
            planned_outputs_text="Smart-city lighting standards and pilot charging solutions using street-lighting assets.",
            delivery_text="Krakow City Roads Board with the SmartEPC project consortium.",
            timeline_text="The planned timeframe for the task is 2022-2024.",
            source_quote="Participation in the SmartEPC project.",
            document_local_code="TR-1",
        ),
        make_city_record(
            city="Krakow",
            title=(
                "ZIT - Reconstruction of tram tracks together with turnouts and associated "
                "infrastructure at Straszewskiego, Karmelicka Street, and Starowislna Street"
            ),
            general_description=(
                "A ZIT transport project to reconstruct central Krakow tram tracks, turnouts, road "
                "systems, overhead lines, lighting, drainage, pavements, and cycle paths."
            ),
            objective_text=(
                "Improve tram infrastructure and the functioning of the public-transport system in "
                "central Krakow."
            ),
            implementation_text=(
                "The measure covers reconstruction works on Straszewskiego and Karmelicka Streets, "
                "track and road works up to Powstańców Śląskich Bridge, and modernisation of related "
                "tram, catenary, lighting, drainage, pavement, and cycling infrastructure."
            ),
            planned_outputs_text="Reconstructed tram tracks, turnouts, and associated transport infrastructure in central Krakow.",
            delivery_text=(
                "Krakow Metropolis Association together with the Municipality of Krakow as a member "
                "of the association."
            ),
            timeline_text="Action only in the design phase, currently no timetable or milestones.",
            numbers_planned={"estimated_emissions_reduction_transport_tco2e": 2807},
            source_quote=(
                "ZIT - Reconstruction of tram tracks together with turnouts and associated "
                "infrastructure at Straszewskiego, Karmelicka Street, and Starowislna Street."
            ),
            document_local_code="TR-7",
        ),
        make_city_record(
            city="Krakow",
            title="Construction of new tram lines with associated infrastructure",
            general_description=(
                "A tram-expansion action covering several new lines and related infrastructure "
                "modernisation projects in Krakow."
            ),
            objective_text=(
                "Expand public transport infrastructure and improve the functioning of Krakow's tram system."
            ),
            implementation_text=(
                "The action includes modernisation of tracks and associated infrastructure and the "
                "creation of new tram lines such as Cichy Kącik - Azory, Osiedle Krowodrza Górka - "
                "Azory, part of the KST line Meissnera - Mistrzejowice, and Krowodrza Górka - Górka Narodowa."
            ),
            planned_outputs_text="New tram lines and modernised associated infrastructure across Krakow.",
            delivery_text=(
                "Municipal Investment Board, Department of City Treasury, and Krakow City Roads Board."
            ),
            funding_text="Approx. PLN 615,000,000 by 2030 (approx. EUR 136,666,000).",
            timeline_text="Planned completion dates depend on the specific section of infrastructure.",
            numbers_planned={
                "estimated_emissions_reduction_transport_tco2e": 11107,
                "estimated_cost_pln_approx": 615000000,
                "estimated_cost_eur_approx": 136666000,
                "end_year": 2030,
            },
            source_quote="Construction of new tram lines with associated infrastructure.",
            document_local_code="TR-10",
        ),
        make_city_record(
            city="Krakow",
            title="Increasing the area of urban forests",
            general_description=(
                "A forest-expansion action implementing Krakow's district programme for increasing "
                "forest cover through land acquisition, afforestation, and public planting campaigns."
            ),
            objective_text="Increase urban forest cover and expand green infrastructure within Krakow.",
            implementation_text=(
                "The action implements the District Programme for Increasing the Forest Cover of the "
                "City of Krakow 2018-2040, including land purchase, afforestation, and forest "
                "planting campaigns together with residents."
            ),
            planned_outputs_text="Expanded urban forests and progress toward higher municipal forest cover.",
            delivery_text="Krakow Municipal Greenery Board.",
            funding_text="Estimated costs of action - PLN 37,000,000 (approx. EUR 8,222,000).",
            numbers_planned={
                "forest_cover_percent_min": 8,
                "estimated_emissions_reduction_other_tco2e": 15259,
                "estimated_compensated_emissions_tco2e": 15259,
                "estimated_cost_pln": 37000000,
                "estimated_cost_eur_approx": 8222000,
            },
            source_quote="Increasing the area of urban forests.",
            document_local_code="I-2",
        ),
        make_city_record(
            city="Krakow",
            title="Maintain a high ratio of biologically active area in the city",
            general_description=(
                "A land-use and de-sealing action to protect and increase biologically active areas "
                "through planning rules, land acquisition, and local unsealing measures."
            ),
            objective_text="Increase biologically active urban space and strengthen green infrastructure.",
            implementation_text=(
                "The action introduces provisions in spatial plans and the General Plan, acquires "
                "land for green areas, protects valuable natural areas, and unseals impermeable "
                "surfaces in roads, squares, and car parks."
            ),
            planned_outputs_text="Protected and expanded biologically active urban areas across Krakow.",
            delivery_text=(
                "Municipality of Krakow - Department of Spatial Planning, Department of City Treasury, "
                "Department of Environmental Management, Krakow City Roads Board, and Krakow Municipal Greenery Board."
            ),
            funding_text="Estimated costs - PLN 325,000,000 (approx. EUR 72,222,000).",
            numbers_planned={
                "estimated_emissions_reduction_other_tco2e": 142,
                "estimated_compensated_emissions_tco2e": 142,
                "estimated_cost_pln": 325000000,
                "estimated_cost_eur_approx": 72222000,
            },
            source_quote="Maintain a high ratio of biologically active area in the city.",
            document_local_code="I-3",
        ),
        make_city_record(
            city="Krakow",
            title="Introduce smaller forms of green infrastructure in heavily build-up parts of the city",
            general_description=(
                "A green-infrastructure action for dense urban areas using green roofs, green walls, "
                "pocket parks, flower meadows, urban groves, and similar small-scale interventions."
            ),
            objective_text="Introduce additional green infrastructure in heavily built-up parts of Krakow.",
            implementation_text=(
                "The measure includes green roofs and walls, pocket parks, community gardens, flower "
                "meadows, new square and roadside tree planting, urban groves, and potential incentive "
                "schemes for private and external green-roof and wall investments."
            ),
            planned_outputs_text="New small-scale green infrastructure across dense urban locations in Krakow.",
            delivery_text=(
                "Krakow Municipal Greenery Board, Krakow City Roads Board, Climate-Energy-Water "
                "Management, and other municipal units and companies managing investments or buildings."
            ),
            funding_text="Estimated costs of action - PLN 28,000,000 (approx. EUR 6,222,000).",
            numbers_planned={
                "estimated_emissions_reduction_other_tco2e": 12,
                "estimated_compensated_emissions_tco2e": 12,
                "estimated_cost_pln": 28000000,
                "estimated_cost_eur_approx": 6222000,
            },
            source_quote="Introduce smaller forms of green infrastructure in heavily build-up parts of the city.",
            document_local_code="I-4",
        ),
        make_city_record(
            city="Krakow",
            title="Planting of trees, shrubs and protection of existing trees",
            general_description=(
                "A city-wide planting and tree-protection action covering new plantings, natural "
                "compensation requirements, and detailed rules for protecting trees during investments."
            ),
            objective_text="Increase urban greenery and strengthen tree protection across Krakow.",
            implementation_text=(
                "The action includes tree and shrub planting on municipal land, higher natural "
                "compensation requirements for external removals, and implementation of detailed "
                "municipal rules for tree protection and replacement plantings."
            ),
            planned_outputs_text="More trees and shrubs planted and stronger protection of existing greenery.",
            delivery_text=(
                "Department of Environmental Management, the Municipal Conservator of Monuments, and "
                "entities managing property on behalf of the Municipality of Krakow."
            ),
            funding_text="Estimated costs of action - PLN 64,000,000 (approx. EUR 14,222,000).",
            numbers_planned={
                "estimated_emissions_reduction_other_tco2e": 28,
                "estimated_compensated_emissions_tco2e": 28,
                "estimated_cost_pln": 64000000,
                "estimated_cost_eur_approx": 14222000,
            },
            source_quote="Planting of trees, shrubs and protection of existing trees.",
            document_local_code="I-5",
        ),
        make_city_record(
            city="Krakow",
            title="The CoFarm4Cities project",
            general_description=(
                "An international project to identify and sustainably use peri-urban and abandoned "
                "land and transform it into urban farms."
            ),
            objective_text=(
                "Promote urban agriculture, biodiversity, better use of undeveloped land, and "
                "nature-based solutions for sustainable urban development."
            ),
            implementation_text=(
                "The project works with nine actors to develop a stakeholder engagement model and "
                "ready-to-implement solutions that transform peri-urban agricultural, mixed, or "
                "abandoned areas into urban farms."
            ),
            planned_outputs_text=(
                "Urban agriculture models, stakeholder-engagement methods, and nature-based solutions "
                "for peri-urban land in Krakow."
            ),
            delivery_text="Krakow Municipal Greenery Board with project partner cities and local stakeholders.",
            funding_text="Estimated total costs - PLN 1,300,935 (approx. EUR 289,000).",
            timeline_text="The planned timeframe for the task runs from 2023 to 2026.",
            numbers_planned={
                "estimated_total_cost_pln": 1300935,
                "estimated_total_cost_eur_approx": 289000,
                "start_year": 2023,
                "end_year": 2026,
            },
            source_quote="The CoFarm4Cities project.",
            document_local_code="I-6",
        ),
        make_city_record(
            city="Krakow",
            title="Life Pact Project - The human factor: Adapting the City for Tomorrow",
            general_description=(
                "A LIFE project that develops and tests an integrated urban adaptation approach "
                "through nature-based solutions."
            ),
            objective_text="Develop and test integrated city adaptation methods using nature-based solutions.",
            implementation_text=(
                "The project focuses on climate adaptation through two rain gardens and three green "
                "public spaces and is implemented with partners from Belgium, Spain, and Poland."
            ),
            planned_outputs_text="Nature-based adaptation pilots and improved urban heat-island resilience.",
            delivery_text="Climate - Energy - Water Management with partner cities Madrid and Leuven.",
            funding_text="Estimated total costs - approx. PLN 3,709,123 (approx. EUR 824,000).",
            timeline_text="The planned timeframe for the task is 2022-2025.",
            numbers_planned={
                "rain_gardens_count": 2,
                "green_public_spaces_count": 3,
                "estimated_total_cost_pln": 3709123,
                "estimated_total_cost_eur_approx": 824000,
                "start_year": 2022,
                "end_year": 2025,
            },
            source_quote="Life Pact Project - The human factor: Adapting the City for Tomorrow.",
            document_local_code="I-7",
        ),
        make_city_record(
            city="Krakow",
            title="Greene 4.0 project",
            general_description=(
                "An international project helping manufacturing SMEs implement green and digital "
                "innovations and build new supply chains."
            ),
            objective_text="Support green and digital transformation in manufacturing companies in the Małopolska region.",
            implementation_text=(
                "The consortium conducts surveys, reports, workshops, exchange platforms, and start-up "
                "support to help manufacturing companies adopt green production practices and new supply chains."
            ),
            planned_outputs_text="Knowledge, tools, networks, and support instruments for green industrial transformation.",
            delivery_text="Kraków Technology Park with the Greene 4.0 consortium and manufacturing companies.",
            funding_text="Total funding - approx. EUR 270,000 (approx. PLN 1,210,000).",
            timeline_text="Planned completion date: 2023-2026.",
            numbers_planned={
                "funding_eur_approx": 270000,
                "funding_pln_approx": 1210000,
                "start_year": 2023,
                "end_year": 2026,
            },
            source_quote="Greene 4.0 project.",
            document_local_code="I-8",
        ),
        make_city_record(
            city="Krakow",
            title="Usage of CCS, CCU technology and carbon capture",
            general_description=(
                "A carbon-capture and utilisation action built around know-how transfer, reports, "
                "and future CCS deployment options for Krakow's waste-conversion facilities."
            ),
            objective_text=(
                "Prepare Krakow to implement CCS and CCU technologies that reduce carbon-intensive "
                "emissions from urban infrastructure."
            ),
            implementation_text=(
                "The action includes strategic cooperation with Norwegian partners, business and "
                "technical analyses for CCS/CCU at the Thermal Waste Conversion Plant, and work on "
                "capture, transport, and storage options."
            ),
            planned_outputs_text="Technical and organisational readiness for future CCS and CCU deployment in Krakow.",
            delivery_text="Municipality of Krakow as monitoring entity with municipal companies and technical partners.",
            timeline_text="The planned timeframe for the task is 2024-2030.",
            numbers_planned={
                "estimated_emissions_reduction_other_tco2e": 180000,
                "estimated_compensated_emissions_tco2e": 180000,
                "target_operational_year": 2030,
                "start_year": 2024,
                "end_year": 2030,
            },
            source_quote="Usage of CCS, CCU technology and carbon capture.",
            document_local_code="I-9",
        ),
        make_city_record(
            city="Krakow",
            title="Reclamation of post-industrial areas in Nowa Huta district",
            general_description=(
                "A brownfield-reclamation and redevelopment action for post-industrial areas in "
                "Nowa Huta, combining social, economic, and green-space objectives."
            ),
            objective_text=(
                "Reclaim brownfield land, improve quality of life for residents, and create space "
                "for new technologies, green infrastructure, and inclusive redevelopment."
            ),
            implementation_text=(
                "The city plans to acquire 300 hectares of industrial land, reclaim it, consult "
                "residents on redevelopment, support technology parks and R&D centres, create "
                "recreational and green areas, and improve social and labour-market reintegration."
            ),
            planned_outputs_text="Reclaimed post-industrial land with new green areas, innovation uses, and public access.",
            delivery_text="Municipality of Krakow with residents, universities, research institutes, and business partners.",
            numbers_planned={"planned_land_acquisition_ha": 300},
            source_quote="Reclamation of post-industrial areas in Nowa Huta district.",
            document_local_code="I-10",
        ),
        make_city_record(
            city="Krakow",
            title="Kraków Transport Assembly",
            general_description=(
                "A citizen assembly on sustainable transport in Krakow that developed recommendations "
                "on transport optimisation, parking, traffic management, and emissions reduction."
            ),
            objective_text=(
                "Engage residents directly in shaping sustainable transport measures for Krakow."
            ),
            implementation_text=(
                "After the climate assembly, Krakow organised a transport citizen assembly in 2022; "
                "a two-stage draw selected 80 participants, followed by educational and deliberative "
                "meetings that produced transport recommendations."
            ),
            planned_outputs_text="Binding citizen-backed recommendations on sustainable transport in Krakow.",
            delivery_text="City of Krakow with citizen participants and the monitoring panel.",
            timeline_text="Held in 2022.",
            numbers_current={"participants_count": 80, "recommendations_count": 43},
            source_quote="Krakow Transport Citizen Assembly (KTCA).",
        ),
        make_city_record(
            city="Lodz",
            title="Sustainable Urban Mobility Plan for the Łódź Metropolitan Area (SUMP)",
            general_description=(
                "A metropolitan mobility-planning document being developed for the Łódź Metropolitan Area."
            ),
            objective_text=(
                "Guide the development of public transportation systems in a sustainable and "
                "low-carbon manner."
            ),
            implementation_text=(
                "Local government units within the Łódź Metropolitan Area Association are developing "
                "the SUMP to improve quality of life and reduce the environmental and climate impact "
                "of road transport."
            ),
            planned_outputs_text="A regional mobility plan for low-carbon transport development in the Łódź metropolitan area.",
            delivery_text="Łódź Metropolitan Area Association and participating local government units.",
            source_quote="Sustainable Urban Mobility Plan for the Łódź Metropolitan Area (SUMP).",
        ),
        make_city_record(
            city="Lodz",
            title="Anti-smog Resolution of the Łódź Voivodeship",
            general_description=(
                "A regional anti-smog regulation that phases out inefficient boilers, stoves, and fireplaces."
            ),
            objective_text="Reduce air pollution and accelerate the phase-out of inefficient heating sources.",
            implementation_text=(
                "The resolution adopted by the Łódź Voivodeship Assembly in 2017 and updated in 2022 "
                "sets replacement deadlines for old boilers, stoves, and fireplaces that do not meet "
                "required classes."
            ),
            planned_outputs_text="A staged replacement schedule for high-emission heating sources in the region.",
            delivery_text="Assembly of the Łódź Voivodeship and local authorities implementing the rules.",
            timeline_text="Updated in 2022; key replacement deadlines run through 2028.",
            numbers_planned={
                "boiler_replacement_deadline_year": 2025,
                "stove_and_fireplace_replacement_deadline_year": 2026,
                "all_non_compliant_heating_replacement_deadline_year": 2028,
            },
            source_quote="Anti-smog Resolution adopted by the Assembly of the Łódź Voivodeship in 2017 and updated in 2022.",
        ),
        make_city_record(
            city="Lodz",
            title="Warm Housing Program (Ciepłe Mieszkanie)",
            general_description=(
                "A complex housing-support programme for cleaner heating and building efficiency improvements."
            ),
            objective_text="Support cleaner heating, district-heating connections, and energy-efficiency upgrades in housing.",
            implementation_text=(
                "The programme supports replacement of inefficient heating sources, connection of "
                "residential buildings to district heating, installation of clean heating sources, "
                "window replacements, and mechanical ventilation."
            ),
            planned_outputs_text="Residential buildings with cleaner heating sources and better energy performance.",
            delivery_text="Public programme serving residential-building owners and occupants in Łódź.",
            source_quote="Warm Housing Program (Ciepłe Mieszkanie).",
        ),
        make_city_record(
            city="Lodz",
            title="Clean Air Program (Czyste Powietrze)",
            general_description=(
                "A national clean-air programme used in Łódź to cut pollution and improve building energy performance."
            ),
            objective_text="Reduce dust and other pollutants from individual residential buildings.",
            implementation_text=(
                "Although focused on air quality, the programme supports replacement of coal-fired "
                "heating sources and energy-efficiency improvements that also contribute to climate mitigation."
            ),
            planned_outputs_text="Cleaner residential heating systems and more energy-efficient buildings.",
            delivery_text="Public programme used by residents and local actors in Łódź.",
            source_quote="Clean Air Program (Czyste Powietrze).",
        ),
        make_city_record(
            city="Lodz",
            title="Targeted subsidies from the city budget of Łódź for investment projects aimed at air protection",
            general_description=(
                "A city-financed subsidy scheme for permanently decommissioning solid-fuel heat sources."
            ),
            objective_text="Reduce low emissions by switching buildings away from solid-fuel heat sources.",
            implementation_text=(
                "In 2024, Łódź awarded targeted subsidies for permanent solid-fuel decommissioning "
                "and changing the heating system to district heating, gas, electric heating, or heat pumps."
            ),
            planned_outputs_text="Subsidised heating-system replacements that cut low-emission pollution in Łódź.",
            delivery_text=(
                "City of Łódź for natural persons, housing communities, legal entities, entrepreneurs, "
                "and public finance sector units."
            ),
            timeline_text="Implemented in 2024.",
            source_quote="Targeted subsidies from the city budget of Łódź for investment projects aimed at air protection.",
        ),
        make_city_record(
            city="Lodz",
            title="Łódź Citizen Card",
            general_description=(
                "A city card and mobile application that provides discounts and promotions for Łódź residents."
            ),
            objective_text="Encourage public-transport use and support resident-focused mobility incentives.",
            implementation_text=(
                "The Łódź Citizen Card works as a mobile app or plastic card and offers city-wide "
                "discounts, including lower prices for season tickets."
            ),
            planned_outputs_text="Resident incentives that make public transport more attractive in Łódź.",
            delivery_text="City of Łódź and participating city services and partners.",
            source_quote="The Łódź Citizen Card.",
        ),
        make_city_record(
            city="Lodz",
            title="Łódzki Rower Miejski",
            general_description="Łódź's municipal bike-rental system.",
            objective_text="Support low-carbon mobility by making bicycle transport easier to use across the city.",
            implementation_text=(
                "The city bike-rental system operates alongside expanding cycling infrastructure and "
                "helps residents shift to bicycle transport."
            ),
            planned_outputs_text="A citywide public bike system supporting everyday low-carbon travel.",
            delivery_text="City of Łódź and operators of the municipal bike-rental system.",
            numbers_current={
                "bicycles_count": 1500,
                "stations_count": 150,
                "bicycle_racks_count": 2250,
            },
            source_quote="Łódzki Rower Miejski.",
        ),
        make_city_record(
            city="Lodz",
            title="Rationalisation of energy consumption - Thermal modernisation of educational facilities in the City of Łódź",
            general_description=(
                "A large-scale thermal-modernisation project for educational facilities in Łódź."
            ),
            objective_text="Reduce energy consumption, operating costs, and greenhouse gas emissions in educational facilities.",
            implementation_text=(
                "The city implemented the project across more than 80 educational facilities as part "
                "of five energy-efficiency projects."
            ),
            planned_outputs_text="Thermally modernised educational facilities with lower energy use and emissions.",
            delivery_text="City of Łódź and educational-facility managers.",
            numbers_current={"educational_facilities_modernised_count_min": 80, "projects_count": 5},
            source_quote="Rationalisation of energy consumption - Thermal modernisation of educational facilities in the City of Łódź.",
        ),
        make_city_record(
            city="Lodz",
            title="We illuminate Poland",
            general_description=(
                "A governmental programme used in Łódź to accelerate LED replacement of street lighting."
            ),
            objective_text="Reduce electricity use through LED street-light modernisation.",
            implementation_text=(
                "Under the programme, Łódź replaced about 1,000 street lights with LED in 2023 and "
                "obtained further funding for continued replacement."
            ),
            planned_outputs_text="A broader citywide shift to LED street lighting in Łódź.",
            delivery_text="City of Łódź using the governmental programme's funding and support.",
            timeline_text="Implemented in 2023 with further continuation planned.",
            numbers_current={"street_lights_replaced_led_2023_approx": 1000},
            source_quote="As part of the governmental programme \"We illuminate Poland\".",
        ),
        make_city_record(
            city="Lodz",
            title="Łódź oddychaMy anti-smog campaign",
            general_description=(
                "An anti-smog information and environmental-education campaign for Łódź residents."
            ),
            objective_text="Inform residents about heating-source replacement and available financial support programmes.",
            implementation_text=(
                "The campaign prepared a series of articles for residents, with particular emphasis "
                "on senior citizens, covering environmental education and replacement of heating sources."
            ),
            planned_outputs_text="Better-informed residents and stronger uptake of heating-source replacement support.",
            delivery_text="City of Łódź and its environmental and communication teams.",
            source_quote="As part of the anti-smog campaign \"Łódź oddychaMy\".",
        ),
        make_city_record(
            city="Lodz",
            title="Łódź - an Environmentally Friendly City",
            general_description=(
                "A city environmental-awareness project combining anti-smog publications, workshops, and campaigns."
            ),
            objective_text="Increase environmental awareness among Łódź residents and support cleaner urban behaviour.",
            implementation_text=(
                "The project included anti-smog publications, advertising spots about subsidies for "
                "replacing ovens, 25 environmental workshops, co-organised education events, and an "
                "urban information campaign on energy saving."
            ),
            planned_outputs_text="Broader environmental awareness and more engagement in low-emission behaviour.",
            delivery_text="City of Łódź and partners delivering environmental education and campaigns.",
            numbers_current={"environmental_workshops_count": 25},
            source_quote="As part of the project \"Łódź - an Environmentally Friendly City\".",
        ),
        make_city_record(
            city="Lodz",
            title="Local Government Climate Congress",
            general_description=(
                "A recurring city-organised climate congress bringing together local government, science, business, institutions, and residents."
            ),
            objective_text="Build a network for climate knowledge exchange, cooperation, and climate education.",
            implementation_text=(
                "The first editions took place in 2022 and 2023, and the event includes panels, "
                "expert meetings, and resident engagement focused on climate action and everyday choices."
            ),
            planned_outputs_text="A recurring climate forum and cooperation network centred on Łódź.",
            delivery_text="City of Łódź with scientific, business, institutional, and resident communities.",
            timeline_text="Held in 2022 and 2023; the next edition was planned for May 2025 in the source document.",
            numbers_current={"completed_editions_count": 2},
            source_quote="Local Government Climate Congress organised by the City.",
        ),
        make_city_record(
            city="Lodz",
            title="EKOinnowacje conference",
            general_description=(
                "A city-organised conference presenting future energy solutions and supporting climate knowledge exchange."
            ),
            objective_text="Share knowledge on future-oriented energy solutions with residents and stakeholders.",
            implementation_text=(
                "The conference was organised in 2022 at the Łódź Orientarium as one of the events "
                "through which the city supports learning and exchange on climate and energy topics."
            ),
            planned_outputs_text="Knowledge exchange on energy solutions and climate-related innovation in Łódź.",
            delivery_text="City of Łódź and event partners.",
            timeline_text="Held in 2022.",
            source_quote="One of them was the \"EKOinnowacje\" conference (2022).",
        ),
        make_city_record(
            city="Rzeszow",
            title=(
                "Broad education in the field of protection of nature and the natural environment; "
                "promotion/information about the undertaken and planned actions in the city regarding "
                "urban greenery, informing residents about air quality in the city"
            ),
            general_description=(
                "An education and information action combining biodiversity education, rainwater "
                "retention support, and air-quality information."
            ),
            objective_text=(
                "Increase green-space awareness, rainwater retention, and air-quality knowledge among residents."
            ),
            implementation_text=(
                "The action includes construction of a biodiversity-oriented playground, the Small "
                "Retention Program for rainwater measures, and educational air-quality sensors at schools and in the city."
            ),
            planned_outputs_text="Biodiversity education, rainwater-retention uptake, and stronger resident awareness of air quality.",
            delivery_text=(
                "City Municipality of Rzeszów - City Greenery Management Authority and the Department "
                "of Climate and Environment, with schools and local stakeholders."
            ),
            funding_text="Total costs PLN 4,167,000, EUR 926,000.",
            numbers_planned={"total_cost_pln": 4167000, "total_cost_eur": 926000},
            source_quote=(
                "Broad education in the field of protection of nature and the natural environment; "
                "promotion/information about the undertaken and planned actions in the city regarding "
                "urban greenery, informing residents about air quality in the city."
            ),
        ),
        make_city_record(
            city="Rzeszow",
            title="Energy Management Database - ROBOT",
            general_description=(
                "An automation tool for managing energy sales documents and enabling smarter city energy management."
            ),
            objective_text="Improve monitoring, control, reporting, and smart management of city energy use.",
            implementation_text=(
                "ROBOT automates the acquisition, distribution, and archiving of electricity-related "
                "documents for the city and supports monitoring, analysis, reporting, and integration with renewable energy."
            ),
            planned_outputs_text="A city energy-management database that supports savings, reporting, and better control.",
            delivery_text="Rzeszów City Office, city units and municipal companies, and service providers.",
            source_quote="Energy Management Database - ROBOT.",
        ),
        make_city_record(
            city="Rzeszow",
            title="RzeszówToMy application",
            general_description=(
                "A resident-facing mobile application and online account for city information and remote handling of selected official matters."
            ),
            objective_text="Improve communication between residents and the City Office and strengthen participation.",
            implementation_text=(
                "The application combines news, events, education, consultations, taxes, warnings, "
                "road disruptions, intervention functions, and an integrated resident account for selected services."
            ),
            planned_outputs_text="A single resident application for city information, service access, and civic contact.",
            delivery_text="City Office and residents of Rzeszów.",
            source_quote="RzeszówToMy application.",
        ),
        make_city_record(
            city="Rzeszow",
            title="NEEST - Neutral and Environmentally Sustainable Areas",
            general_description=(
                "A NetZeroCities Pilot Cities project in Rzeszów used to prepare and scale building-modernisation solutions."
            ),
            objective_text=(
                "Prepare pilot solutions, analyses, and organisational capacity for neutral and environmentally sustainable urban areas."
            ),
            implementation_text=(
                "The city established a dedicated project team under the NetZeroCities Pilot Cities "
                "Programme, carried out site-visit and preparatory work, and used NEEST in building "
                "selection, analysis, and thermal-modernisation model preparation."
            ),
            planned_outputs_text="Pilot-ready building-modernisation solutions and stronger city capacity for climate-neutral urban areas.",
            delivery_text="Rzeszów City Office with businesses, universities, residents, NGOs, and project partners.",
            source_quote="NEEST - Neutral and Environmentally Sustainable Areas.",
        ),
        make_city_record(
            city="Rzeszow",
            title="Small Retention Program",
            general_description=(
                "A cyclic subsidy programme for rainwater retention investments by residents and institutions."
            ),
            objective_text="Increase local rainwater retention and support small-scale water-management measures.",
            implementation_text=(
                "The programme finances above-ground and underground tanks, septic or overflow-dispersal "
                "systems, rain gardens, and small ponds, and is implemented cyclically every year."
            ),
            planned_outputs_text="More rainwater-retention systems installed by residents, schools, and housing entities.",
            delivery_text=(
                "City Municipality of Rzeszów for residents, businesses, housing cooperatives and communities, and schools."
            ),
            timeline_text="Implemented cyclically every year.",
            numbers_planned={
                "above_ground_tank_capacity_l_min": 500,
                "underground_tank_capacity_l_min": 2000,
                "water_pond_capacity_l_min": 500,
            },
            source_quote="Small Retention Program.",
        ),
        make_city_record(
            city="Warszawa",
            title="NEEST (NetZero Emission and Environmentally Sustainable Territories)",
            general_description=(
                "A Pilot Cities Programme project in Warsaw's Praga-Południe district focused on "
                "deep energy modernisation and neighbourhood revitalisation."
            ),
            objective_text=(
                "Prepare innovative solutions that are ready to be implemented, scaled, and replicated "
                "for deep energy modernisation of buildings and their surroundings."
            ),
            implementation_text=(
                "The NEEST research project works on technical, financial, environmental, social, "
                "and MEL aspects of deep energy modernisation and neighbourhood revitalisation."
            ),
            planned_outputs_text="Model solutions for deep energy modernisation and scalable neighbourhood revitalisation in Warsaw.",
            delivery_text="City of Warsaw, district actors, and the NEEST project consortium.",
            source_quote="NEEST (NetZero Emission and Environmentally Sustainable Territories).",
        ),
        make_city_record(
            city="Warszawa",
            title="Participatory Budgeting",
            general_description=(
                "Warsaw's annual participatory budgeting mechanism through which residents decide how part of the city budget is spent."
            ),
            objective_text="Enable residents to choose local and citywide projects that support quality of life and climate goals.",
            implementation_text=(
                "Residents submit projects and vote each year on district and citywide investments, "
                "including climate-relevant measures such as biodiversity, water, microclimate, and clean transport."
            ),
            planned_outputs_text="Resident-selected investments financed from Warsaw's budget each year.",
            delivery_text="Centre for Public Communication and residents of Warsaw.",
            source_quote="Participatory Budgeting.",
        ),
        make_city_record(
            city="Warszawa",
            title="Partnership for Climate",
            general_description=(
                "A multi-stakeholder climate platform for exchanging experience, education, and joint projects."
            ),
            objective_text="Support climate protection in Warsaw through coordinated cooperation among public, private, civic, and scientific actors.",
            implementation_text=(
                "The platform brings together the Air Protection and Climate Policy Department, NGOs, "
                "companies, institutions, scientific bodies, and diplomatic institutions for education, "
                "knowledge exchange, networking, and joint climate projects."
            ),
            planned_outputs_text="A cooperative climate platform that supports project delivery, competence development, and private-sector involvement.",
            delivery_text="Air Protection and Climate Policy Department with NGOs, companies, institutions, and scientific partners.",
            source_quote="Partnership for Climate.",
        ),
        make_city_record(
            city="Warszawa",
            title="Warsaw Climate Panel",
            general_description=(
                "A citizens' panel that involved residents directly in shaping Warsaw's energy-efficiency and renewable-energy policy."
            ),
            objective_text="Bring citizens into city climate decision-making and produce actionable pro-climate recommendations.",
            implementation_text=(
                "In 2020-2021 Warsaw organised the panel with citizens, community organisations, and "
                "experts; 90 representative residents produced 49 recommendations, and those receiving "
                "over 80% support are being implemented by the city."
            ),
            planned_outputs_text="Citizen-backed climate recommendations implemented by the City of Warsaw.",
            delivery_text="City of Warsaw, panellists, community organisations, and experts.",
            timeline_text="Held in 2020-2021.",
            numbers_current={"participants_count": 90, "recommendations_count": 49},
            source_quote="Warsaw Climate Panel.",
        ),
        make_city_record(
            city="Warszawa",
            title="Green Vision for Warsaw - Green City and Climate Action Plan",
            general_description=(
                "Warsaw's comprehensive climate and green-city roadmap adopted in 2023."
            ),
            objective_text="Guide Warsaw's development with specific measures that move the city toward climate neutrality.",
            implementation_text=(
                "The action plan combines Green City Action Plan and Climate Action Plan methodologies, "
                "uses verified city data and 122 indicators, and sets reduction scenarios and actions through 2030 and 2050."
            ),
            planned_outputs_text="A citywide roadmap, reduction scenarios, and implementation measures for Warsaw's green transition.",
            delivery_text="City of Warsaw under Resolution No. LXXX/2648/2023 of the Warsaw City Council.",
            timeline_text="Adopted on 20 April 2023; targets extend to 2030 and 2050.",
            numbers_planned={
                "indicators_count": 122,
                "ghg_reduction_target_percent_2030": 40,
                "climate_neutrality_target_year": 2050,
            },
            source_quote="Green Vision for Warsaw - Green City and Climate Action Plan.",
        ),
        make_city_record(
            city="Warszawa",
            title=(
                "Climate Change Adaptation Strategy for Warsaw up to 2030 with a perspective for 2050 "
                "(Municipal Adaptation Plan)"
            ),
            general_description=(
                "Warsaw's urban policy document for preparing the city and its residents for climate-change impacts."
            ),
            objective_text="Set the directions of operation, priority areas, and adaptation options for climate resilience in Warsaw.",
            implementation_text=(
                "The Municipal Adaptation Plan defines Warsaw's adaptation policy, describes climate "
                "threats and risk areas, and points to actions and a future monitoring and implementation system."
            ),
            planned_outputs_text="A monitored urban adaptation policy and implementation framework for Warsaw.",
            delivery_text="City of Warsaw and the institutions involved in the Municipal Adaptation Plan update and implementation.",
            timeline_text="Policy horizon to 2030 with a perspective to 2050.",
            source_quote="Climate Change Adaptation Strategy for Warsaw up to 2030 with a perspective for 2050 Municipal Adaptation Plan.",
        ),
        make_city_record(
            city="Warszawa",
            title="Clean Transport Zone",
            general_description=(
                "A transport-regulation measure to reduce the role of high-emission individual car transport in Warsaw."
            ),
            objective_text="Reduce transport emissions by creating a clean transport zone and shifting mobility toward cleaner modes.",
            implementation_text=(
                "The clean transport zone is listed among measures accompanying transport-system "
                "improvements together with paid parking zones, pedestrian infrastructure, bicycle "
                "expansion, and public-transport improvements."
            ),
            planned_outputs_text="A clean transport zone that supports lower transport emissions in Warsaw.",
            delivery_text="City of Warsaw with relevant transport and regulatory stakeholders.",
            source_quote="Creating the Clean Transport Zone.",
        ),
        make_city_record(
            city="Warszawa",
            title="ADAPTCITY",
            general_description=(
                "The LIFE_ADAPTCITY_PL project that supported preparation of Warsaw's urban adaptation plan."
            ),
            objective_text="Strengthen Warsaw's resilience to climate change through an urban adaptation plan and related actions.",
            implementation_text=(
                "The project, co-funded by the LIFE instrument and the National Fund for "
                "Environmental Protection and Water Management, supported the adaptation strategy "
                "developed with Warsaw residents, entrepreneurs, organisations, and city authorities."
            ),
            planned_outputs_text="An urban adaptation plan and stronger climate-change preparedness for Warsaw.",
            delivery_text="City of Warsaw with residents, entrepreneurs, organisations, and national and EU funding partners.",
            timeline_text="The published adaptation plan is dated 4 July 2019.",
            source_quote="ADAPTCITY.",
        ),
        make_city_record(
            city="Wroclaw",
            title="Sustainable Energy and Climate Action Plan (SECAP)",
            general_description=(
                "Wroclaw's main energy and climate action plan prepared under the Covenant of Mayors."
            ),
            objective_text="Reduce greenhouse gas emissions and promote renewable energy use in Wroclaw.",
            implementation_text=(
                "Prepared in 2019, the SECAP maps climate risks and vulnerabilities, sets emission "
                "reduction scenarios and impact estimates, and identifies key measures for the city."
            ),
            planned_outputs_text="A citywide framework for climate mitigation, renewable energy, and adaptation in Wroclaw.",
            delivery_text="Municipality of Wroclaw under the Covenant of Mayors framework.",
            timeline_text="Prepared in 2019 with targets to 2030 and 2050.",
            numbers_planned={
                "ghg_reduction_target_percent_2030": 40,
                "ghg_reduction_target_percent_2050": 80,
            },
            source_quote="Sustainable Energy and Climate Action Plan (2019).",
        ),
        make_city_record(
            city="Wroclaw",
            title="Climate Change Adaptation Plan (2019)",
            general_description=(
                "Wroclaw's climate-adaptation plan focused on resilience and sustainable city development."
            ),
            objective_text="Adapt Wroclaw to climate-change impacts and strengthen urban resilience.",
            implementation_text=(
                "The plan sets out adaptation strategies for Wroclaw's development and is one of the "
                "council-adopted climate documents that the city intends to keep aligned with the CCC."
            ),
            planned_outputs_text="A climate-adaptation framework for Wroclaw's urban development and resilience.",
            delivery_text="Wroclaw City Council and municipal authorities.",
            timeline_text="Adopted in 2019.",
            source_quote="Climate Change Adaptation Plan (2019).",
        ),
        make_city_record(
            city="Wroclaw",
            title="Sustainable Mobility Plan for the Wroclaw Functional Urban Area (2022)",
            general_description=(
                "A functional-area mobility plan emphasising public transport, cycling, walking, and lower emissions."
            ),
            objective_text="Improve transportation efficiency, reduce congestion, and lower emissions in the Wroclaw Functional Urban Area.",
            implementation_text=(
                "The plan emphasises public transit, cycling, and walking as core sustainable "
                "mobility modes for the broader functional urban area."
            ),
            planned_outputs_text="A mobility framework supporting sustainable urban transport across the Wroclaw functional area.",
            delivery_text="City of Wroclaw and partners in the Wroclaw Functional Urban Area.",
            timeline_text="Adopted in 2022.",
            source_quote="Sustainable Mobility Plan for the Wroclaw Functional Urban Area (2022).",
        ),
        make_city_record(
            city="Wroclaw",
            title="Wroclaw Strategy for the Development of Electromobility (2020)",
            general_description=(
                "A city strategy for electric-vehicle uptake and charging infrastructure development."
            ),
            objective_text="Promote electromobility and reduce transport emissions in Wroclaw.",
            implementation_text=(
                "The strategy frames city action on EV adoption and charging infrastructure as part "
                "of transport-sector decarbonisation."
            ),
            planned_outputs_text="A city strategy supporting electric vehicles and associated charging infrastructure.",
            delivery_text="City of Wroclaw and transport-sector stakeholders.",
            timeline_text="Adopted in 2020.",
            source_quote="Wroclaw Strategy for the Development of Electromobility (2020).",
        ),
        make_city_record(
            city="Wroclaw",
            title="The Strategy for Managing Stormwater and Meltwater in Wroclaw (2023)",
            general_description=(
                "A city strategy for stormwater and meltwater management in support of climate resilience."
            ),
            objective_text="Reduce flood risks and improve sustainable water management in Wroclaw.",
            implementation_text=(
                "The strategy addresses stormwater and meltwater in the city as part of climate "
                "resilience and sustainable urban-water management."
            ),
            planned_outputs_text="A city strategy guiding stormwater and meltwater management in Wroclaw.",
            delivery_text="City of Wroclaw and urban water-management stakeholders.",
            timeline_text="Adopted in 2023.",
            source_quote="The Strategy for Managing Stormwater and Meltwater in Wroclaw (2023).",
        ),
        make_city_record(
            city="Wroclaw",
            title="Assumptions to the Plan for Supply of Heat, Electricity and Gaseous Fuels in the Area of City of Wroclaw (2023)",
            general_description=(
                "A strategic energy-supply planning document for heat, electricity, and gaseous fuels in Wroclaw."
            ),
            objective_text="Align energy-supply planning with Wroclaw's climate goals and lower-emission pathways.",
            implementation_text=(
                "The document sets assumptions for sustainable heat, electricity, and fuel supply "
                "and is one of the city-level strategic planning instruments supporting the climate transition."
            ),
            planned_outputs_text="An updated energy-supply planning framework for Wroclaw.",
            delivery_text="City of Wroclaw and energy-system stakeholders.",
            timeline_text="Adopted in 2023.",
            source_quote="Assumptions to the Plan for Supply of Heat, Electricity and Gaseous Fuels in the Area of City of Wroclaw (2023).",
        ),
        make_city_record(
            city="Wroclaw",
            title="The Wroclaw Tram Program for the years 2024-2032 (2023)",
            general_description=(
                "A strategic programme for tram expansion and modernisation in Wroclaw."
            ),
            objective_text="Reduce congestion and emissions by expanding and modernising tram infrastructure.",
            implementation_text=(
                "The programme focuses on extending and upgrading tram infrastructure as part of "
                "Wroclaw's wider low-emission public-transport strategy."
            ),
            planned_outputs_text="A long-term tram investment programme for Wroclaw.",
            delivery_text="City of Wroclaw and public-transport infrastructure stakeholders.",
            timeline_text="Programme horizon 2024-2032; adopted in 2023.",
            numbers_planned={"start_year": 2024, "end_year": 2032},
            source_quote="The Wroclaw Tram Program for the years 2024-2032 (2023).",
        ),
        make_city_record(
            city="Wroclaw",
            title="Small RES Programme",
            general_description=(
                "A municipal renewable-energy programme equipping public facilities and municipal entities with RES installations."
            ),
            objective_text="Increase renewable-energy generation on municipal facilities and support municipal climate neutrality.",
            implementation_text=(
                "The programme equips municipal institutions and public utilities with renewable-energy "
                "installations; newly constructed municipal buildings are fitted with photovoltaics, "
                "green roofs and walls, and rainwater-management systems."
            ),
            planned_outputs_text="More renewable-energy installations on municipal buildings and facilities in Wroclaw.",
            delivery_text="City of Wroclaw, municipal institutions, and municipal companies.",
            timeline_text="Implemented before the CCC and continued from 2022 onward in the source document.",
            numbers_current={"stadium_energy_demand_covered_percent": 50},
            source_quote="Small RES Programme.",
        ),
        make_city_record(
            city="Wroclaw",
            title="Energy cluster between the City of Wroclaw and Wroclaw universities",
            general_description=(
                "A planned energy-cluster agreement between the city and Wroclaw's universities."
            ),
            objective_text="Create a city-university energy cluster that supports innovative local energy cooperation.",
            implementation_text=(
                "The city planned to sign an agreement with the rectors of all Wroclaw universities "
                "to establish an energy cluster as a direct result of participation in the Cities Mission."
            ),
            planned_outputs_text="A formal city-university energy cluster for local low-emission energy cooperation.",
            delivery_text="City of Wroclaw and the rectors of Wroclaw universities.",
            timeline_text="Planned in 2024 in the source document.",
            source_quote="An agreement to establish an energy cluster between the City of Wroclaw and the rectors of all Wroclaw universities.",
        ),
        make_city_record(
            city="Wroclaw",
            title="Decarbonisation Table",
            general_description=(
                "A district-heating decarbonisation partnership bringing city, utility, and infrastructure actors into one coordination body."
            ),
            objective_text="Coordinate the decarbonisation of Wroclaw's energy and district-heating systems.",
            implementation_text=(
                "The partnership connects energy companies, district-heating companies, gas companies, "
                "the municipal water and sewerage company, urban investment and environmental services, "
                "and planners to align heat-system decarbonisation."
            ),
            planned_outputs_text="A coordinated city-utility platform supporting energy and district-heating decarbonisation.",
            delivery_text="City of Wroclaw with energy, heating, gas, water, and planning stakeholders.",
            timeline_text="Established in June 2023.",
            numbers_planned={"kogeneracja_modernisation_target_year": 2028},
            source_quote="Decarbonisation Table.",
        ),
        make_city_record(
            city="Wroclaw",
            title="WROMPA project",
            general_description=(
                "An innovative wastewater-heat recovery project implemented by district-heating and water utilities in Wroclaw."
            ),
            objective_text="Recover heat from wastewater and use it to support district heating in Wroclaw.",
            implementation_text=(
                "FORTUM Power and Heat Polska and the Municipal Water and Sewerage Company installed "
                "one of Europe's largest heat pumps on a sewage collector to recover wastewater heat."
            ),
            planned_outputs_text="Recovered wastewater heat supplying part of Wroclaw's district-heating demand.",
            delivery_text="FORTUM Power and Heat Polska Sp. z o.o. and the Municipal Water and Sewerage Company.",
            numbers_planned={"district_heating_demand_share_percent": 5},
            source_quote="WROMPA project.",
        ),
        make_city_record(
            city="Wroclaw",
            title="Tenement returns",
            general_description=(
                "A programme for renovating historic tenements and courtyard interiors in Wroclaw."
            ),
            objective_text="Accelerate renovation of historic residential stock and improve urban living conditions.",
            implementation_text=(
                "The city launched the programme to renovate historic tenements, using lessons from "
                "NEEST and broader building-modernisation work to improve the housing stock."
            ),
            planned_outputs_text="Renovated historic tenements and courtyard interiors across Wroclaw.",
            delivery_text="City of Wroclaw and partners involved in historic-building renovation.",
            timeline_text="The source document sets targets through 2029.",
            numbers_planned={"buildings_to_renovate_count": 100, "courtyard_interiors_to_renovate_count": 70, "end_year": 2029},
            source_quote="Tenement returns.",
        ),
        make_city_record(
            city="Wroclaw",
            title="Wroclaw Transition Team / Municipal Mission Team",
            general_description=(
                "Wroclaw's cross-sector municipal coordination structure for climate-neutrality work."
            ),
            objective_text="Make the city's climate-neutrality work systemic and coordinated across departments, units, and companies.",
            implementation_text=(
                "The Mayor established the Municipal Mission Team in October 2022; it includes "
                "representatives of all city departments, entities, and companies and meets cyclically "
                "to ensure information flow and cross-sector cooperation."
            ),
            planned_outputs_text="A durable municipal coordination platform for Wroclaw's climate transition.",
            delivery_text="Mayor of Wroclaw, city departments, municipal entities, and municipal companies.",
            timeline_text="Established in October 2022.",
            source_quote="Wroclaw Transition Team.",
        ),
        make_city_record(
            city="Wroclaw",
            title="Green Lungs of Wroclaw",
            general_description=(
                "A planned programme for additional parks, urban forests, squares, and recreational green areas."
            ),
            objective_text="Increase urban greenery and create more places for rest and recreation among greenery.",
            implementation_text=(
                "The programme is planned as part of Wroclaw's blue-green infrastructure and climate-adaptation work."
            ),
            planned_outputs_text="Additional parks, urban forests, squares, and green recreational areas in Wroclaw.",
            delivery_text="City of Wroclaw and blue-green infrastructure stakeholders.",
            source_quote="Green Lungs of Wroclaw.",
        ),
        make_city_record(
            city="Wroclaw",
            title="Grey to Green",
            general_description=(
                "A programme for deconcretising courtyards and squares and replacing sealed surfaces with greenery."
            ),
            objective_text="Reduce impervious surfaces and expand climate-resilient urban greenery in Wroclaw.",
            implementation_text=(
                "The programme continues and expands courtyard and square deconcretisation; a flagship "
                "example replaced a concrete city square with nearly 200 trees and 60,000 other plants."
            ),
            planned_outputs_text="Deconcretised public spaces with more trees, planting, and climate-adaptive greenery.",
            delivery_text="City of Wroclaw and entities involved in public-space greening.",
            numbers_current={"trees_planted_approx": 200, "other_plants_approx": 60000},
            source_quote="Grey to Green.",
        ),
    ]


def rewrite_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add the verified second-pass records to the corrected Polish extraction output."""
    records_by_id = {record["record_id"]: record for record in records}
    existing_titles = {
        (record["initiative"]["city"], record["initiative"]["initiative_name"]): record["record_id"]
        for record in records
    }
    additions = build_remaining_additions()
    for record in additions:
        record_id = record["record_id"]
        title_key = (record["initiative"]["city"], record["initiative"]["initiative_name"])
        if record_id in records_by_id:
            raise ValueError(f"Added record id already exists: {record_id}")
        if title_key in existing_titles:
            raise ValueError(
                f"Added title already exists for city={title_key[0]}: {title_key[1]!r}"
            )
        records_by_id[record_id] = record

    corrected = sorted(
        records_by_id.values(),
        key=lambda item: (
            item["initiative"]["city"],
            item["initiative"]["initiative_name"].casefold(),
            item["record_id"],
        ),
    )
    manifest = {
        "removed_record_ids": [],
        "rewritten_record_ids": [],
        "unit_normalized_record_ids": [],
        "added_record_ids": [record["record_id"] for record in additions],
    }
    return corrected, manifest


def main() -> None:
    """Script entry point."""
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    output_run_dir = args.output_run_dir.resolve()

    logger.info(
        "Applying second-pass extraction fixes from source_run_dir=%s",
        source_run_dir,
    )
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
        "Wrote second-pass corrected run to %s with %s deduped initiatives.",
        output_run_dir,
        len(corrected_records),
    )


if __name__ == "__main__":
    setup_logger()
    main()
