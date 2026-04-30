"""Deterministic data lookups for the enrichment pipeline.

These modules expose typed functions over structured data sources
(Bundesnetzagentur registry, ICCT EV city stats, …) so that the
estimator and search worker can resolve a (city, field) request
locally without an LLM or a web call.

Each lookup module declares its coverage so the manifest can match
ingestions to the runtime adapters that consume them.
"""

from collections.abc import Iterable

from backend.modules.sources.manifest import IngestionConfig, Manifest
from backend.modules.web_researcher.data_lookups.bnetza import (
    BNETZA_LOOKUP_ID,
    ChargerStats,
    SOURCE_NAME as BNETZA_SOURCE_NAME,
    chargers_in_city,
)
from backend.modules.web_researcher.data_lookups.population import (
    CityPopulation,
    SOURCE_NAME as POPULATION_SOURCE_NAME,
    population_for_city,
)
from backend.modules.web_researcher.models import (
    FieldDecomposition,
    StructuredLookupResult,
)

POPULATION_LOOKUP_ID = "urban_audit_population"


# Mapping bnetza fields → ChargerStats accessor functions.
_BNETZA_FIELD_HANDLERS: dict[str, callable] = {
    "public_charger_count": lambda s: ("stations", s.station_count),
    "public_ac_charger_count": lambda s: ("stations", s.ac_station_count),
    "public_dc_charger_count": lambda s: ("stations", s.dc_station_count),
    "charger_kw_total": lambda s: ("kW", s.kw_total),
    "charger_port_count": lambda s: ("ports", s.port_count),
}


def _coverage_matches(
    ingestion: IngestionConfig,
    *,
    fields: list[str],
    cities: list[str],
) -> bool:
    """Return True when this ingestion's coverage overlaps the requested fields/cities."""
    coverage = ingestion.coverage
    if coverage is None:
        return True

    if coverage.fields:
        coverage_set = {f.casefold() for f in coverage.fields}
        if not any(f.casefold() in coverage_set for f in fields):
            return False

    if coverage.cities:
        coverage_cities = {c.casefold() for c in coverage.cities}
        if not any(c.casefold() in coverage_cities for c in cities):
            return False

    return True


def find_matching_structured_lookups(
    decomposition: FieldDecomposition,
    cities: list[str],
    manifest: Manifest,
) -> list[StructuredLookupResult]:
    """Run every structured_lookup ingestion whose coverage matches the request.

    Hardcoded dispatch by ``ingestion.id`` for now.  Add new lookup
    handlers here when introducing additional structured sources.
    """
    if not cities:
        return []

    field_names = [
        f.field
        for f in decomposition.query_fields
        if f.classification != "non_estimable"
    ]
    if not field_names:
        return []

    out: list[StructuredLookupResult] = []
    for source, ingestion in manifest.iter_ingestions():
        if ingestion.kind != "structured_lookup":
            continue
        if not _coverage_matches(ingestion, fields=field_names, cities=cities):
            continue

        if ingestion.id == BNETZA_LOOKUP_ID:
            out.extend(
                _bnetza_results(
                    source_id=source.id,
                    ingestion_id=ingestion.id,
                    cities=cities,
                    fields=field_names,
                )
            )
            continue

        if ingestion.id == POPULATION_LOOKUP_ID:
            out.extend(
                _population_results(
                    source_id=source.id,
                    ingestion_id=ingestion.id,
                    cities=cities,
                    fields=field_names,
                )
            )
            continue

        # Unknown structured-lookup ingestion — skip rather than fail the run.
        # When a new ingestion is added, register a dispatch branch here.

    return out


def _bnetza_results(
    *,
    source_id: str,
    ingestion_id: str,
    cities: Iterable[str],
    fields: Iterable[str],
) -> list[StructuredLookupResult]:
    fields_to_emit = [f for f in fields if f in _BNETZA_FIELD_HANDLERS]
    if not fields_to_emit:
        return []

    out: list[StructuredLookupResult] = []
    for city in cities:
        try:
            stats = chargers_in_city(city)
        except FileNotFoundError:
            return []
        if stats.is_empty:
            continue
        for field in fields_to_emit:
            unit, value = _BNETZA_FIELD_HANDLERS[field](stats)
            out.append(
                StructuredLookupResult(
                    source_id=source_id,
                    ingestion_id=ingestion_id,
                    city=city,
                    field=field,
                    value=value,
                    unit=unit,
                    asof=None,
                    extra={
                        "matched_normalized": list(stats.matched_normalized),
                        "top_operators": [
                            {"name": name, "stations": count}
                            for name, count in stats.top_operators
                        ],
                        "source_name": BNETZA_SOURCE_NAME,
                    },
                )
            )
    return out


_POPULATION_FIELDS = {"city_population", "population"}


def _population_results(
    *,
    source_id: str,
    ingestion_id: str,
    cities: Iterable[str],
    fields: Iterable[str],
) -> list[StructuredLookupResult]:
    fields_to_emit = [f for f in fields if f.casefold() in _POPULATION_FIELDS]
    if not fields_to_emit:
        return []

    out: list[StructuredLookupResult] = []
    for city in cities:
        result = population_for_city(city)
        if result.is_empty:
            continue
        for field in fields_to_emit:
            out.append(
                StructuredLookupResult(
                    source_id=source_id,
                    ingestion_id=ingestion_id,
                    city=city,
                    field=field,
                    value=result.population,
                    unit="inhabitants",
                    asof=str(result.year) if result.year else None,
                    extra={
                        "matched_key": result.matched_key,
                        "source_name": POPULATION_SOURCE_NAME,
                    },
                )
            )
    return out


__all__ = [
    "BNETZA_LOOKUP_ID",
    "POPULATION_LOOKUP_ID",
    "ChargerStats",
    "CityPopulation",
    "chargers_in_city",
    "find_matching_structured_lookups",
    "population_for_city",
]
