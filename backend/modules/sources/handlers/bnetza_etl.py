"""Handler: convert the Bundesnetzagentur Ladesäulenregister XLSX to parquet.

The XLSX has 47 columns and ~107k rows.  We project the columns we
actually use, normalise types, derive a charger-type taxonomy
(``art ∈ {AC, DC}``), and write parquet with snappy compression.

Schema written to parquet:
- ladeeinrichtungs_id   int
- betreiber             str
- status                str
- art                   str  ("AC" | "DC")
- art_raw               str  (original "Normalladeeinrichtung" / …)
- count_ports           int
- nennleistung_kw       float
- inbetriebnahme        date | null
- plz                   str
- ort                   str           # original, with case + diacritics
- ort_normalized        str           # lookup key (lowercase, no umlauts)
- kreis                 str
- bundesland            str
- lat                   float | null
- lon                   float | null
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from backend.modules.sources.handlers import IngestionContext, register
from backend.modules.sources.state import IngestionState
from backend.modules.web_researcher.data_lookups.bnetza import normalize_de_city

logger = logging.getLogger(__name__)

_HANDLER_NAME = "ingest.bnetza_etl"


# Column → canonical projection
_RENAME = {
    "Ladeeinrichtungs-ID": "ladeeinrichtungs_id",
    "Betreiber": "betreiber",
    "Status": "status",
    "Art der Ladeeinrichtung": "art_raw",
    "Anzahl Ladepunkte": "count_ports",
    "Nennleistung Ladeeinrichtung [kW]": "nennleistung_kw",
    "Inbetriebnahmedatum": "inbetriebnahme",
    "Postleitzahl": "plz",
    "Ort": "ort",
    "Kreis/kreisfreie Stadt": "kreis",
    "Bundesland": "bundesland",
    "Breitengrad": "lat",
    "Längengrad": "lon",
}


def _classify_art(art_raw: str | None, kw: float | None) -> str:
    """Map the registry's German taxonomy to AC / DC."""
    if art_raw:
        normalised = art_raw.casefold()
        if "schnell" in normalised:  # Schnellladeeinrichtung
            return "DC"
        if "normal" in normalised:  # Normalladeeinrichtung
            return "AC"
    # Fallback by power if the taxonomy field is missing.
    if kw is not None and kw > 22.0:
        return "DC"
    return "AC"


def run_bnetza_etl(context: IngestionContext) -> IngestionState:
    """Read the upstream XLSX, project + normalise, write parquet."""
    inputs = context.ingestion.inputs
    output = context.ingestion.output
    if not output.path:
        raise ValueError(
            f"ingestion {context.ingestion.id!r}: bnetza_etl requires output.path"
        )
    if len(inputs.paths) != 1:
        raise ValueError(
            f"ingestion {context.ingestion.id!r}: bnetza_etl expects exactly one input path"
        )

    upstream_xlsx = context.upstream_root / inputs.paths[0]
    if not upstream_xlsx.exists():
        raise FileNotFoundError(f"Bnetza source not found: {upstream_xlsx}")

    output_path = (context.project_root / output.path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("bnetza_etl: reading %s", upstream_xlsx)

    raw = pl.read_excel(upstream_xlsx, sheet_id=1, read_options={"header_row": 10})
    logger.info("bnetza_etl: %d rows, %d raw columns", raw.height, raw.width)

    keep = [c for c in _RENAME if c in raw.columns]
    missing = sorted(set(_RENAME) - set(keep))
    if missing:
        logger.warning("bnetza_etl: missing expected columns: %s", missing)

    df = raw.select(keep).rename({c: _RENAME[c] for c in keep})

    # Type coercion — some columns come in as strings depending on the workbook.
    df = df.with_columns(
        pl.col("ladeeinrichtungs_id").cast(pl.Int64, strict=False),
        pl.col("count_ports").cast(pl.Int32, strict=False),
        pl.col("nennleistung_kw").cast(pl.Float64, strict=False),
        pl.col("lat").cast(pl.Float64, strict=False),
        pl.col("lon").cast(pl.Float64, strict=False),
        pl.col("plz").cast(pl.Utf8, strict=False),
    )

    # Inbetriebnahme: tolerate ISO date strings as well as Excel datetimes.
    if "inbetriebnahme" in df.columns:
        col = df["inbetriebnahme"]
        if col.dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("inbetriebnahme").str.strptime(
                    pl.Date, format="%Y-%m-%d", strict=False
                )
            )
        else:
            df = df.with_columns(pl.col("inbetriebnahme").cast(pl.Date, strict=False))

    # Derived columns.
    df = df.with_columns(
        pl.struct(["art_raw", "nennleistung_kw"])
        .map_elements(
            lambda row: _classify_art(row["art_raw"], row["nennleistung_kw"]),
            return_dtype=pl.Utf8,
        )
        .alias("art"),
        pl.col("ort")
        .map_elements(
            lambda v: normalize_de_city(v) if v else "",
            return_dtype=pl.Utf8,
        )
        .alias("ort_normalized"),
    )

    # Reorder for stable parquet schema.
    columns = [
        "ladeeinrichtungs_id",
        "betreiber",
        "status",
        "art",
        "art_raw",
        "count_ports",
        "nennleistung_kw",
        "inbetriebnahme",
        "plz",
        "ort",
        "ort_normalized",
        "kreis",
        "bundesland",
        "lat",
        "lon",
    ]
    df = df.select([c for c in columns if c in df.columns])

    df.write_parquet(output_path, compression="snappy")
    output_size = output_path.stat().st_size
    logger.info(
        "bnetza_etl: wrote %s (%d rows, %.1f MB parquet)",
        output_path,
        df.height,
        output_size / (1024 * 1024),
    )

    distinct_cities = df["ort_normalized"].n_unique() if "ort_normalized" in df.columns else 0

    return IngestionState(
        ingestion_id=context.ingestion.id,
        source_id=context.source.id,
        last_ingested_at=datetime.now(timezone.utc).isoformat(),
        source_commit=context.resolved_commit,
        row_count=df.height,
        distinct_cities=distinct_cities,
        parquet_bytes=output_size,
        output_path=output_path.relative_to(context.project_root).as_posix(),
        schema_version=1,
    )


@register(_HANDLER_NAME)
def _entrypoint(context: IngestionContext) -> IngestionState:
    return run_bnetza_etl(context)


__all__ = ["run_bnetza_etl"]
