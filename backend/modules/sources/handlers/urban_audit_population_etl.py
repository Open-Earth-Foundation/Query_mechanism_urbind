"""Ingestion handler for the Eurostat URBAN AUDIT city-population seed.

Reads ``backend/data/sources/urban_audit/city_population.json`` (committed)
and converts it into a parquet at ``data/lookups/urban_audit_population.parquet``
that ``backend.modules.web_researcher.data_lookups.population`` consumes.

The seed JSON is small + diffable; refresh by re-pulling Eurostat ``urb_cpop1``
and extending the JSON, then re-running this ingestion.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import polars as pl

from backend.modules.sources.handlers import IngestionContext, register
from backend.modules.sources.state import IngestionState

logger = logging.getLogger(__name__)


@register("ingest.urban_audit_population_etl")
def urban_audit_population_etl(ctx: IngestionContext) -> IngestionState:
    """Convert the curated city-population JSON to a parquet lookup file."""
    inputs = ctx.ingestion.inputs.paths
    if not inputs:
        raise ValueError("urban_audit_population_etl: no input paths declared")

    # Local provider — paths are repo-relative, not upstream-relative.
    src_path = ctx.project_root / inputs[0]
    if not src_path.exists():
        raise FileNotFoundError(f"Population seed not found: {src_path}")

    payload = json.loads(src_path.read_text(encoding="utf-8"))
    cities = payload.get("cities", {})
    if not isinstance(cities, dict) or not cities:
        raise ValueError(f"Population seed has no cities: {src_path}")

    rows: list[dict[str, object]] = []
    for city_key, entry in cities.items():
        if not isinstance(entry, dict):
            continue
        pop = entry.get("population")
        if not isinstance(pop, (int, float)):
            continue
        rows.append(
            {
                "city_key": str(city_key).lower(),
                "population": int(pop),
                "year": entry.get("year"),
                "source": entry.get("source"),
            }
        )

    if not rows:
        raise ValueError(f"Population seed yielded zero usable rows: {src_path}")

    out_path_str = ctx.ingestion.output.path
    if not out_path_str:
        raise ValueError("urban_audit_population_etl: ingestion.output.path missing")
    out_path = ctx.project_root / out_path_str
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame(rows)
    df.write_parquet(out_path)

    logger.info(
        "urban_audit_population_etl: wrote %d rows to %s", len(rows), out_path
    )

    return IngestionState(
        ingestion_id=ctx.ingestion.id,
        source_id=ctx.source.id,
        last_ingested_at=datetime.now(timezone.utc).isoformat(),
        source_commit=ctx.resolved_commit,
        output=str(out_path.relative_to(ctx.project_root)),
        row_count=len(rows),
    )


__all__ = ["urban_audit_population_etl"]
