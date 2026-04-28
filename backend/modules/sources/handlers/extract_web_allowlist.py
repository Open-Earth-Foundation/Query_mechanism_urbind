"""Handler: extract tier-1 web allowlist from upstream documentation.

Strategy
--------
The list of trusted tier-1 sources + their coverage is *curated* in code
(``_CURATED``) rather than parsed from prose, because the upstream
documentation's structure varies and the cost of getting coverage hints
wrong (the search worker would either skip or pollute results) is higher
than the cost of editing this file when the upstream changes.

What the handler does on each run:
1. Read the upstream README + ``tier-5-eu-programmes/web-platforms-index.md``.
2. For each curated source, verify the upstream still mentions the domain
   somewhere — log a warning when a curated source is no longer cited
   (drift signal that we may want to remove or re-classify it).
3. Also log warnings for any *new* domains found in the upstream that
   don't yet have a curated entry — drift signal to add coverage.
4. Emit ``backend/data/tier1_web_sources.yaml`` from the curated list.

The yaml is the single source of truth at runtime; the upstream cross-
reference is a soft validation that surfaces drift to the operator.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import yaml

from backend.modules.sources.handlers import IngestionContext, register
from backend.modules.sources.state import IngestionState

logger = logging.getLogger(__name__)

_HANDLER_NAME = "ingest.extract_web_allowlist"


# ---------------------------------------------------------------------------
# Curated tier-1 web sources.
# Add entries here when the upstream README documents a new web-only platform.
# ---------------------------------------------------------------------------
_CURATED: list[dict] = [
    {
        "id": "bnetza_ladekarte",
        "name": "Bundesnetzagentur Ladesäulenkarte",
        "domain": "bundesnetzagentur.de",
        "urls": [
            "https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/Ladesaeulenkarte/"
        ],
        "access": "site_search",
        "coverage": {
            "countries": ["DE"],
            "fields": ["public_ac_charger_count", "public_dc_charger_count", "public_charger_count"],
            "scope": ["ev_charging"],
        },
        "notes": (
            "Live UI for the public-charger registry. The static XLSX export "
            "is already loaded into the bnetza_chargers structured lookup; "
            "this entry is for prose/news context."
        ),
    },
    {
        "id": "standorttool",
        "name": "StandortTOOL Ladeinfrastruktur",
        "domain": "standorttool.de",
        "urls": ["https://www.standorttool.de/strom/"],
        "access": "site_search",
        "coverage": {
            "countries": ["DE"],
            "fields": ["charger_demand_projection", "public_charger_count"],
            "scope": ["ev_charging"],
        },
        "notes": "Federal demand projections for public charging infrastructure.",
    },
    {
        "id": "mobilithek",
        "name": "Mobilithek",
        "domain": "mobilithek.info",
        "urls": ["https://mobilithek.info/"],
        "access": "site_search",
        "coverage": {
            "countries": ["DE"],
            "scope": ["transport_data", "mobility"],
        },
        "notes": "German national mobility data platform.",
    },
    {
        "id": "eafo",
        "name": "European Alternative Fuels Observatory",
        "domain": "alternative-fuels-observatory.ec.europa.eu",
        "urls": ["https://alternative-fuels-observatory.ec.europa.eu/"],
        "access": "api",
        "coverage": {
            "countries": ["EU_27"],
            "fields": [
                "public_charger_count",
                "public_ac_charger_count",
                "public_dc_charger_count",
                "ev_registrations",
                "ev_share",
            ],
            "scope": ["ev_charging", "ev_registrations"],
        },
        "notes": "EU-wide alternative-fuel infrastructure stats. Has an API.",
    },
    {
        "id": "iea_ev_explorer",
        "name": "IEA Global EV Data Explorer",
        "domain": "iea.org",
        "urls": [
            "https://www.iea.org/data-and-statistics/data-tools/global-ev-data-explorer"
        ],
        "access": "site_search",
        "coverage": {
            "countries": ["EU_27", "global"],
            "fields": ["ev_registrations", "ev_share", "public_charger_count"],
            "scope": ["ev_registrations", "ev_charging"],
        },
        "notes": "Global EV adoption + infrastructure dashboard.",
    },
    {
        "id": "eurostat_transport",
        "name": "Eurostat Transport Database",
        "domain": "ec.europa.eu",
        "urls": ["https://ec.europa.eu/eurostat/web/transport/database"],
        "access": "site_search",
        "coverage": {
            "countries": ["EU_27"],
            "scope": ["transport_data", "modal_split"],
        },
        "notes": "EU statistical office's transport database.",
    },
    {
        "id": "systems_change_lab_transport",
        "name": "Systems Change Lab Transport Dashboard",
        "domain": "systemschangelab.org",
        "urls": ["https://systemschangelab.org/transport"],
        "access": "site_search",
        "coverage": {
            "countries": ["global"],
            "scope": ["transport_data", "decarbonization"],
        },
        "notes": "Global transport-decarbonisation indicators.",
    },
    {
        "id": "klimadashboard_munster",
        "name": "Klimadashboard Münster",
        "domain": "klimadashboard.ms",
        "urls": ["https://www.klimadashboard.ms/"],
        "access": "site_search",
        "coverage": {
            "cities": ["munster", "muenster"],
            "scope": ["climate_indicators"],
        },
        "notes": "Münster city climate dashboard.",
    },
    {
        "id": "netzerocities_public",
        "name": "NetZeroCities (public pages)",
        "domain": "netzerocities.eu",
        "urls": [
            "https://netzerocities.eu/",
            "https://netzerocities.eu/climate-city-contract/",
            "https://netzerocities.eu/mission-cities/",
        ],
        "access": "site_search",
        "coverage": {
            "scope": ["ccc_summary", "mission_status", "eu_programme"],
        },
        "excluded_paths": ["/app/", "/knowledge-ccc/"],
        "notes": "Authenticated app paths are excluded — they're behind login.",
    },
    {
        "id": "netzerocities_app",
        "name": "NetZeroCities App",
        "domain": "netzerocities.app",
        "urls": ["https://netzerocities.app/"],
        "access": "auth_required",
        "coverage": {
            "scope": ["ccc_documents"],
        },
        "notes": "Auth-walled. Listed for visibility but not searchable by us.",
    },
    {
        "id": "eu_sump_database",
        "name": "EU SUMP Database",
        "domain": "transport.ec.europa.eu",
        "urls": [
            "https://urban-mobility-observatory.transport.ec.europa.eu/sustainable-urban-mobility-plans/eu-city-database-sumps_en"
        ],
        "access": "site_search",
        "coverage": {
            "countries": ["EU_27"],
            "scope": ["urban_mobility_plan"],
        },
        "notes": "European Commission database of Sustainable Urban Mobility Plans.",
    },
    {
        "id": "mobidata_bw",
        "name": "MobiData BW",
        "domain": "mobidata-bw.de",
        "urls": ["https://mobidata-bw.de/"],
        "access": "api",
        "coverage": {
            "cities": ["heidelberg", "mannheim"],
            "countries": ["DE"],
            "scope": ["transport_data", "mobility"],
        },
        "notes": "Open mobility data for Baden-Württemberg (covers Heidelberg, Mannheim).",
    },
    {
        "id": "open_data_sachsen",
        "name": "Open Data Sachsen",
        "domain": "opendata.sachsen.de",
        "urls": ["https://www.opendata.sachsen.de/"],
        "access": "site_search",
        "coverage": {
            "cities": ["dresden", "leipzig"],
            "countries": ["DE"],
            "scope": ["open_data"],
        },
        "notes": "Saxony state open data — relevant for Dresden + Leipzig.",
    },
    {
        "id": "govdata",
        "name": "GovData",
        "domain": "govdata.de",
        "urls": ["https://www.govdata.de/"],
        "access": "site_search",
        "coverage": {
            "countries": ["DE"],
            "scope": ["open_data"],
        },
        "notes": "Germany's central open data portal (federal + state + municipal).",
    },
]


def _read_upstream_text(upstream_root: Path, paths: list[str]) -> str:
    """Concatenate the contents of all referenced upstream markdown files."""
    chunks: list[str] = []
    for relative in paths:
        path = upstream_root / relative
        if not path.exists():
            logger.warning("extract_web_allowlist: upstream file missing: %s", relative)
            continue
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(chunks)


_URL_RE = re.compile(r"https?://[^\s)>\]]+")


def _domains_in_text(text: str) -> set[str]:
    """Extract distinct registered-ish domains mentioned in upstream text."""
    out: set[str] = set()
    for match in _URL_RE.finditer(text):
        try:
            host = urlparse(match.group(0)).netloc
        except ValueError:
            continue
        if not host:
            continue
        host = host.casefold().lstrip("www.")
        out.add(host)
    return out


def _domain_suffix_match(curated: str, upstream: str) -> bool:
    """True when ``upstream`` is the same domain or a subdomain of ``curated``.

    Google's ``site:`` operator already suffix-matches, so when a curated
    entry uses a parent domain we still consider any subdomain in the
    upstream text a satisfied reference.
    """
    return upstream == curated or upstream.endswith("." + curated)


def _check_drift(curated_domains: set[str], upstream_domains: set[str]) -> None:
    """Log warnings for curated/upstream domain divergence (suffix-aware)."""
    missing: list[str] = []
    for curated in sorted(curated_domains):
        if not any(_domain_suffix_match(curated, u) for u in upstream_domains):
            missing.append(curated)
    if missing:
        logger.warning(
            "extract_web_allowlist: curated domains not mentioned by upstream: %s",
            ", ".join(missing),
        )

    extra: list[str] = []
    for upstream in sorted(upstream_domains):
        if "github.com" in upstream:
            continue
        if any(_domain_suffix_match(c, upstream) for c in curated_domains):
            continue
        extra.append(upstream)
    if extra:
        logger.info(
            "extract_web_allowlist: upstream mentions domains not in curated set: %s",
            ", ".join(extra[:20]),
        )


def _build_yaml_payload() -> dict:
    return {"version": 1, "sources": _CURATED}


def run_extract_web_allowlist(context: IngestionContext) -> IngestionState:
    """Read upstream docs, verify curated set against them, write the yaml."""
    inputs = context.ingestion.inputs
    output = context.ingestion.output
    if not output.path:
        raise ValueError(
            f"ingestion {context.ingestion.id!r}: extract_web_allowlist requires output.path"
        )

    output_path = (context.project_root / output.path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    upstream_text = _read_upstream_text(context.upstream_root, inputs.paths)
    upstream_domains = _domains_in_text(upstream_text)
    curated_domains = {entry["domain"].casefold() for entry in _CURATED}
    _check_drift(curated_domains, upstream_domains)

    payload = _build_yaml_payload()
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info(
        "extract_web_allowlist: wrote %s (%d sources)",
        output_path,
        len(_CURATED),
    )

    return IngestionState(
        ingestion_id=context.ingestion.id,
        source_id=context.source.id,
        last_ingested_at=datetime.now(timezone.utc).isoformat(),
        source_commit=context.resolved_commit,
        extracted_entries=len(_CURATED),
        output_path=str(output_path.relative_to(context.project_root))
        if output_path.is_relative_to(context.project_root)
        else str(output_path),
        curated_domains=sorted(curated_domains),
        upstream_domains=sorted(upstream_domains),
    )


@register(_HANDLER_NAME)
def _entrypoint(context: IngestionContext) -> IngestionState:
    return run_extract_web_allowlist(context)


__all__ = ["run_extract_web_allowlist"]
