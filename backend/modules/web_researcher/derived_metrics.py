"""Derived metrics step — computes per-capita and per-unit ratios once.

Sits after ``compute_field_statuses`` and before ``merge_enrichment_into_context``.
Reads resolved values from ``enriched_fields`` plus the population lookup,
and emits ``DerivedMetric`` records for the writer to render.

Scope safety: every derived metric carries the parent field's ``scope`` so
the writer can keep per-scope subtotals intact.  Per-unit ratios only fire
within a single (city, scope) pair.
"""

from __future__ import annotations

import logging
from typing import Iterable

from backend.modules.web_researcher.data_lookups.population import (
    population_for_city,
)
from backend.modules.web_researcher.models import (
    DerivedMetric,
    EnrichedField,
    Scope,
)

logger = logging.getLogger(__name__)

# Numerator → denominator pairs for per-unit ratios.  Both sides must be
# present and resolved on the same (city, scope) for the ratio to fire.
# Add new pairs here as the field vocabulary grows; intentionally explicit
# rather than inferred so we don't create misleading ratios by accident.
_PER_UNIT_PAIRS: list[tuple[str, str, str, str]] = [
    # (numerator_field, denominator_field, output_metric_label, output_unit)
    ("total_capex", "vehicle_count", "per_vehicle_capex", "currency/vehicle"),
    ("fleet_capex", "vehicle_count", "per_vehicle_capex", "currency/vehicle"),
    ("public_charger_count", "vehicle_count", "chargers_per_vehicle", "chargers/vehicle"),
]

# Fields eligible for per-capita derivation.  Anything monetary, count, or
# capacity that scales naturally with population.  Anything ratio-like or
# already per-capita is excluded.
_PER_CAPITA_ELIGIBLE_SUFFIXES = (
    "_capex",
    "_opex",
    "_cost",
    "_count",
    "_total",
    "_kw",
    "_mw",
    "_emissions",
)
_PER_CAPITA_EXCLUDE = {
    "city_population",
    "population",
}


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _is_per_capita_eligible(field: str) -> bool:
    if field.casefold() in _PER_CAPITA_EXCLUDE:
        return False
    return any(field.casefold().endswith(suffix) for suffix in _PER_CAPITA_ELIGIBLE_SUFFIXES)


def _resolved_value(ef: EnrichedField) -> float | None:
    """Return a numeric value when the field has a usable, non-missing one."""
    if ef.status == "still_missing":
        return None
    if ef.freshness_flag == "cancelled":
        return None  # cancelled values are excluded from derived metrics
    return _coerce_float(ef.value)


def _per_capita_metric(
    *, ef: EnrichedField, value: float, population: int, year: int | None
) -> DerivedMetric:
    return DerivedMetric(
        city=ef.city,
        metric=f"per_capita_{ef.field}",
        kind="per_capita",
        value=value / population,
        unit=None,
        numerator_field=ef.field,
        numerator_value=value,
        denominator_field="city_population",
        denominator_value=float(population),
        scope=ef.scope,
        notes=(
            f"Population from URBAN AUDIT seed ({year})" if year else "Population from URBAN AUDIT seed"
        ),
    )


def _per_unit_metric(
    *,
    city: str,
    scope: Scope,
    label: str,
    unit: str,
    numerator_field: str,
    numerator_value: float,
    denominator_field: str,
    denominator_value: float,
) -> DerivedMetric | None:
    if denominator_value == 0:
        return None
    return DerivedMetric(
        city=city,
        metric=label,
        kind="per_unit",
        value=numerator_value / denominator_value,
        unit=unit,
        numerator_field=numerator_field,
        numerator_value=numerator_value,
        denominator_field=denominator_field,
        denominator_value=denominator_value,
        scope=scope,
    )


def compute_derived_metrics(
    enriched_fields: Iterable[EnrichedField],
) -> list[DerivedMetric]:
    """Compute per-capita and per-unit derived metrics from resolved fields.

    The function is read-only on its input; callers are responsible for
    attaching the result to the enrichment bundle.
    """
    metrics: list[DerivedMetric] = []

    # Index by (city_lower, field_lower) → enriched record (with usable value).
    by_city_field: dict[tuple[str, str], EnrichedField] = {}
    for ef in enriched_fields:
        if _resolved_value(ef) is None:
            continue
        by_city_field[(ef.city.lower(), ef.field.lower())] = ef

    seen_cities: set[str] = {ef.city.lower() for ef in by_city_field.values()}

    # Per-capita pass: cache populations per city to avoid re-reading the parquet.
    population_cache: dict[str, tuple[int | None, int | None]] = {}
    for city_lower in seen_cities:
        # Use the canonical city name from any field record we have.
        canonical = next(
            ef.city for (c, _f), ef in by_city_field.items() if c == city_lower
        )
        result = population_for_city(canonical)
        population_cache[city_lower] = (
            result.population if result.population else None,
            result.year,
        )

    for (city_lower, field_lower), ef in by_city_field.items():
        if not _is_per_capita_eligible(field_lower):
            continue
        value = _resolved_value(ef)
        if value is None:
            continue
        pop, year = population_cache.get(city_lower, (None, None))
        if not pop:
            continue
        metrics.append(_per_capita_metric(ef=ef, value=value, population=pop, year=year))

    # Per-unit pass: numerator + denominator must share city AND scope.
    for num_field, den_field, label, unit in _PER_UNIT_PAIRS:
        for city_lower in seen_cities:
            num = by_city_field.get((city_lower, num_field.lower()))
            den = by_city_field.get((city_lower, den_field.lower()))
            if num is None or den is None:
                continue
            if num.scope != den.scope:
                logger.debug(
                    "derived_metrics: skipping %s for %s — scope mismatch (%s vs %s)",
                    label, num.city, num.scope, den.scope,
                )
                continue
            num_value = _resolved_value(num)
            den_value = _resolved_value(den)
            if num_value is None or den_value is None:
                continue
            metric = _per_unit_metric(
                city=num.city,
                scope=num.scope,
                label=label,
                unit=unit,
                numerator_field=num.field,
                numerator_value=num_value,
                denominator_field=den.field,
                denominator_value=den_value,
            )
            if metric is not None:
                metrics.append(metric)

    return metrics


__all__ = ["compute_derived_metrics"]
