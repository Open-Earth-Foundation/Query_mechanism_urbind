"""Post-extraction validator: reject or correct implausible web findings."""

from __future__ import annotations

import logging
import math
from typing import Any

from backend.modules.web_researcher.models import WebFinding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INTEGER_FIELD_KEYWORDS: set[str] = {
    "bus", "taxi", "vehicle", "charger", "station", "fleet",
}

_FIELD_ABSOLUTE_CAPS: dict[str, int] = {
    "ev_bus_count": 5_000,
    "taxi_count": 50_000,
    "charging_station_count": 10_000,
    "ev_fleet_count": 20_000,
    "bike_sharing_station_count": 5_000,
}

_PER_CAPITA_CAPS: dict[str, float] = {
    "taxi_count": 500,        # per 100k inhabitants
    "ev_bus_count": 200,      # per 100k inhabitants
    "charging_station_count": 300,
    "ev_fleet_count": 500,
    "bike_sharing_station_count": 400,
}

# Approximate populations for cities in city_groups.json.
# Order-of-magnitude accuracy is sufficient for plausibility checks.
_CITY_POPULATIONS: dict[str, int] = {
    # DACH & Germany Core
    "aachen": 250_000,
    "dresden": 560_000,
    "heidelberg": 160_000,
    "leipzig": 600_000,
    "mannheim": 310_000,
    "munich": 1_500_000,
    "munster": 315_000,
    "klagenfurt": 100_000,
    # Benelux & Luxembourg
    "amsterdam": 870_000,
    "antwerp": 530_000,
    "eindhoven": 235_000,
    "groningen": 230_000,
    "leuven": 100_000,
    "rotterdam": 650_000,
    "the_hague": 545_000,
    "differdange": 28_000,
    # UK & Ireland
    "bristol": 470_000,
    "cork": 210_000,
    "dublin": 550_000,
    "glasgow": 635_000,
    # France Core
    "angers": 155_000,
    "bordeaux": 260_000,
    "dijon": 160_000,
    "dunkirk": 90_000,
    "grenoble": 160_000,
    "lyon": 520_000,
    "marseille": 870_000,
    "nantes": 320_000,
    "paris": 2_160_000,
    # Iberia
    "barcelona": 1_620_000,
    "madrid": 3_300_000,
    "valencia": 800_000,
    "seville": 690_000,
    "vitoria_gasteiz": 250_000,
    "zaragoza": 680_000,
    "valladolid": 300_000,
    "lisbon": 545_000,
    "porto": 250_000,
    "guimaraes": 160_000,
    # Italy Core
    "bergamo": 120_000,
    "bologna": 390_000,
    "florence": 380_000,
    "milan": 1_370_000,
    "padova": 210_000,
    "parma": 200_000,
    "prato": 195_000,
    "rome": 2_870_000,
    "turin": 870_000,
    # Scandinavia
    "aarhus": 350_000,
    "copenhagen": 630_000,
    "gothenburg": 580_000,
    "malmo": 350_000,
    "lund": 95_000,
    "helsingborg": 115_000,
    "oslo": 700_000,
    "stavanger": 145_000,
    "trondheim": 210_000,
    "stockholm": 980_000,
    "gavle": 75_000,
    "umea": 90_000,
    "sonderborg": 30_000,
    # Finland & Iceland
    "espoo": 300_000,
    "lahti": 120_000,
    "lappeenranta": 72_000,
    "tampere": 245_000,
    "turku": 195_000,
    "helsinki": 660_000,
    "reykjavik": 135_000,
    # Baltics
    "riga": 615_000,
    "liepaja": 70_000,
    "vilnius": 580_000,
    "taurage": 20_000,
    # Poland
    "krakow": 780_000,
    "lodz": 670_000,
    "rzeszow": 195_000,
    "warszawa": 1_790_000,
    "wroclaw": 640_000,
    # Central Europe
    "bratislava": 440_000,
    "budapest": 1_750_000,
    "kosice": 240_000,
    "miskolc": 155_000,
    "pecs": 145_000,
    "liberec": 105_000,
    "bucharest": 1_800_000,
    "cluj_napoca": 320_000,
    "suceava": 90_000,
    # Balkans & Eastern Mediterranean
    "athens": 660_000,
    "ioannina": 115_000,
    "kalamata": 70_000,
    "kozani": 50_000,
    "thessaloniki": 325_000,
    "trikala": 80_000,
    "istanbul": 15_500_000,
    "izmir": 4_400_000,
    "limassol": 185_000,
    "eilat": 55_000,
    "ljubljana": 290_000,
    "kranj": 55_000,
    "velenje": 25_000,
    "zagreb": 800_000,
    "sofia": 1_300_000,
    "gabrovo": 55_000,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_integer_field(field_name: str) -> bool:
    """Return True if the field name implies an integer count."""
    lower = field_name.lower()
    return any(kw in lower for kw in _INTEGER_FIELD_KEYWORDS)


def _numeric_value(value: Any) -> float | None:
    """Try to interpret value as a number. Return None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_finding(
    finding: WebFinding,
    population_lookup: dict[str, int],
) -> WebFinding | None:
    """Validate a single finding. Returns the finding or None to discard it.

    Checks run in order — first rejection wins.
    """
    # 1. Null / empty value
    if finding.value is None or (isinstance(finding.value, str) and not finding.value.strip()):
        logger.debug("Discarding null/empty finding: %s / %s", finding.city, finding.field)
        return None

    # 2. Cancellation passthrough
    if finding.unit == "cancelled_programme":
        return finding

    num = _numeric_value(finding.value)

    # 3. Data-type check: integer fields must not have fractional values
    if _is_integer_field(finding.field) and num is not None:
        if num != math.floor(num):
            logger.warning(
                "Rejected fractional count: %s / %s = %s",
                finding.city, finding.field, finding.value,
            )
            return None

    # 4. Absolute field caps
    field_lower = finding.field.lower()
    cap = _FIELD_ABSOLUTE_CAPS.get(field_lower)
    if cap is not None and num is not None and num > cap:
        logger.warning(
            "Rejected absolute cap violation: %s / %s = %s (cap=%d)",
            finding.city, finding.field, finding.value, cap,
        )
        return None

    # 5. Per-capita plausibility
    per_cap = _PER_CAPITA_CAPS.get(field_lower)
    if per_cap is not None and num is not None:
        city_pop = population_lookup.get(finding.city.lower())
        if city_pop is not None and city_pop > 0:
            ratio = num / city_pop * 100_000
            if ratio > per_cap:
                logger.warning(
                    "Rejected per-capita implausibility: %s / %s = %s "
                    "(ratio=%.1f per 100k, cap=%.1f)",
                    finding.city, finding.field, finding.value, ratio, per_cap,
                )
                return None

    return finding


def validate_findings(
    findings: list[WebFinding],
    cities: list[str],
    population_lookup: dict[str, int] | None = None,
) -> list[WebFinding]:
    """Validate a list of findings, returning only those that pass all checks."""
    if population_lookup is None:
        population_lookup = _CITY_POPULATIONS

    result: list[WebFinding] = []
    for f in findings:
        validated = validate_finding(f, population_lookup)
        if validated is not None:
            result.append(validated)

    rejected = len(findings) - len(result)
    if rejected:
        logger.info("Post-extraction validator: %d/%d findings rejected.", rejected, len(findings))

    return result


__all__ = ["validate_finding", "validate_findings"]
