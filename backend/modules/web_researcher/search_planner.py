"""Search Planner (Agent 2): formulate search queries from gap manifest."""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI

from backend.modules.web_researcher.models import GapManifest, SearchBatch
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)

_CITY_GROUPS_PATH = Path(__file__).resolve().parents[2] / "api" / "assets" / "city_groups.json"

# ---------------------------------------------------------------------------
# Source-type hints: pattern → likely document types
# ---------------------------------------------------------------------------
_SOURCE_HINTS: dict[str, list[str]] = {
    "vehicle_count|fleet|bus_count|tram_count": [
        "operator annual report", "fleet register",
        "tender award notice", "subsidy application",
    ],
    "capex|investment|cost|budget|expenditure": [
        "investment programme", "budget allocation",
        "press release procurement", "funding announcement",
    ],
    "capacity|energy|power|charging|charger": [
        "technical specification", "grid connection permit",
        "energy agency report", "infrastructure register",
    ],
    "emissions|co2|ghg|carbon": [
        "national GHG inventory", "city climate report",
        "environmental agency database",
    ],
}

# ---------------------------------------------------------------------------
# Field category mapping for batch splitting
# ---------------------------------------------------------------------------
_FIELD_CATEGORIES: dict[str, list[str]] = {
    "fleet_vehicle": ["vehicle", "fleet", "bus", "tram", "train", "ferry"],
    "infrastructure": ["charger", "charging", "depot", "station", "capacity"],
    "financial": ["capex", "opex", "cost", "budget", "investment", "expenditure", "funding"],
    "emissions": ["emission", "co2", "ghg", "carbon"],
}


def _get_source_hints(target_fields: list[str]) -> list[str]:
    """Return deduplicated source-type hints relevant to the given fields."""
    hints: list[str] = []
    joined = " ".join(target_fields).lower()
    for pattern, sources in _SOURCE_HINTS.items():
        if re.search(pattern, joined):
            hints.extend(sources)
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            deduped.append(h)
    return deduped


def _categorize_fields(fields: list[str]) -> dict[str, list[str]]:
    """Group fields into categories by keyword matching.

    Fields that don't match any category go into ``"general"``.
    """
    categorized: dict[str, list[str]] = {}
    for field in fields:
        field_lower = field.lower()
        matched = False
        for category, keywords in _FIELD_CATEGORIES.items():
            if any(kw in field_lower for kw in keywords):
                categorized.setdefault(category, []).append(field)
                matched = True
                break
        if not matched:
            categorized.setdefault("general", []).append(field)

    # Merge single-field categories back into general
    to_merge: list[str] = []
    for cat, cat_fields in list(categorized.items()):
        if cat != "general" and len(cat_fields) == 1:
            to_merge.append(cat)
    for cat in to_merge:
        categorized.setdefault("general", []).extend(categorized.pop(cat))

    return categorized


def plan_searches(
    gap_manifest: GapManifest,
    config: AppConfig,
    api_key: str,
    question: str = "",
) -> list[SearchBatch]:
    """Formulate batched search queries from the gap manifest.

    Groups queries by country/region, field category, and priority.
    Returns a list of ``SearchBatch`` objects ready for execution.
    On failure returns an empty list.
    """
    if not gap_manifest.city_gaps:
        return []

    city_groups = _load_city_groups()
    city_to_group = _build_city_group_index(city_groups)

    # Group city gaps by region
    region_gaps: dict[str, list[dict[str, Any]]] = {}
    for cg in gap_manifest.city_gaps:
        region = city_to_group.get(cg.city, "other")
        region_gaps.setdefault(region, []).append(cg.model_dump(mode="json"))

    # Determine which fields are searchable
    searchable_fields = {
        fc.field for fc in gap_manifest.query_fields if fc.searchable
    }

    batches: list[SearchBatch] = []
    max_total = config.enrichment.max_total_queries_per_run
    total_queries = 0

    for region, gaps in region_gaps.items():
        if total_queries >= max_total:
            break

        # Determine batch profile and budget
        batch_cities = [g["city"] for g in gaps]
        all_blank_fields: set[str] = set()
        all_stale_fields: set[str] = set()
        max_priority = "low"

        for g in gaps:
            all_blank_fields.update(f for f in g["blank_fields"] if f in searchable_fields)
            all_stale_fields.update(f for f in g["stale_flags"] if f in searchable_fields)
            if g["search_priority"] == "high":
                max_priority = "high"
            elif g["search_priority"] == "medium" and max_priority != "high":
                max_priority = "medium"

        target_fields = list(all_blank_fields | all_stale_fields)
        if not target_fields:
            continue

        # Split target fields by category for focused batches
        field_categories = _categorize_fields(target_fields)

        for category, cat_fields in field_categories.items():
            if total_queries >= max_total:
                break

            # Effort scaling per architecture doc
            cat_blank = [f for f in cat_fields if f in all_blank_fields]
            cat_stale = [f for f in cat_fields if f in all_stale_fields]

            budget = _compute_budget(
                blank_count=len(cat_blank),
                stale_count=len(cat_stale),
                city_count=len(batch_cities),
                priority=max_priority,
                config=config,
            )

            # Use LLM to formulate search queries
            queries = _formulate_queries(
                cities=batch_cities,
                target_fields=cat_fields,
                region=region,
                priority=max_priority,
                budget=budget,
                config=config,
                api_key=api_key,
                question_context=question,
            )

            if not queries:
                continue

            remaining = max_total - total_queries
            queries = queries[:remaining]
            total_queries += len(queries)

            # Determine search type
            search_type = "missing_entirely" if cat_blank else "freshness_check"
            if cat_blank and cat_stale:
                search_type = "mixed"

            batch_id = f"batch_{region}_{category}_{uuid.uuid4().hex[:8]}"
            batches.append(SearchBatch(
                batch_id=batch_id,
                cities=batch_cities[:8],  # max 8 cities per batch
                target_fields=cat_fields,
                search_type=search_type,
                queries=queries,
                budget=budget,
                priority=max_priority,
            ))

    # National/regional benchmark batches for Method A grounding
    national_batches, national_query_count = _plan_national_benchmark_batches(
        gap_manifest=gap_manifest,
        region_gaps=region_gaps,
        city_groups=city_groups,
        config=config,
        api_key=api_key,
        remaining_budget=max_total - total_queries,
    )
    batches.extend(national_batches)
    total_queries += national_query_count

    # Cross-country comparative batches for Method C grounding
    comparative_batches, comparative_query_count = _plan_comparative_batches(
        gap_manifest=gap_manifest,
        region_gaps=region_gaps,
        config=config,
        api_key=api_key,
        remaining_budget=max_total - total_queries,
    )
    batches.extend(comparative_batches)
    total_queries += comparative_query_count

    logger.info(
        "Search planner: %d batches (%d national, %d comparative), %d total queries across %d regions.",
        len(batches),
        len(national_batches),
        len(comparative_batches),
        total_queries,
        len(region_gaps),
    )
    return batches


def _compute_budget(
    blank_count: int,
    stale_count: int,
    city_count: int,
    priority: str,
    config: AppConfig,
) -> dict[str, object]:
    """Compute search budget based on batch profile."""
    max_per_batch = config.enrichment.max_queries_per_batch

    if blank_count == 0 and stale_count <= 2:
        # 1-2 partial gaps
        query_budget = min(5, max_per_batch)
        deep_dive = False
    elif stale_count > 0 and blank_count == 0:
        # Stale/freshness flags only
        query_budget = min(10, max_per_batch)
        deep_dive = False
    elif blank_count > 0 and priority == "high":
        # MISSING_ENTIRELY — scale to gap surface area
        gap_surface = blank_count * city_count
        query_budget = min(
            max(15, gap_surface * 2 // 3),  # ~2 queries per 3 gaps
            max_per_batch,
        )
        deep_dive = True
    elif city_count > 3 and blank_count > (city_count * 0.6):
        # >60% cohort same gap -> aggregate benchmarks
        query_budget = min(8, max_per_batch)
        deep_dive = False
    else:
        query_budget = min(8, max_per_batch)
        deep_dive = False

    return {
        "max_queries": query_budget,
        "deep_dive_allowed": deep_dive,
        "max_pages_per_deep_dive": config.enrichment.max_pages_per_deep_dive,
    }


def _formulate_queries(
    cities: list[str],
    target_fields: list[str],
    region: str,
    priority: str,
    budget: dict[str, object],
    config: AppConfig,
    api_key: str,
    question_context: str = "",
) -> list[str]:
    """Use LLM to formulate targeted search queries."""
    max_queries = budget.get("max_queries", 5)

    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)

    # Build source-type hints section
    source_hints = _get_source_hints(target_fields)
    source_hints_block = ""
    if source_hints:
        hint_lines = ", ".join(source_hints)
        source_hints_block = (
            f"\nRecommended source types for these fields:\n"
            f"- {hint_lines}\n"
            f"Target these document types in your queries.\n"
        )

    # Build question context section
    question_block = ""
    if question_context:
        question_block = (
            f"\nResearch context: {question_context}\n"
            "Use this to understand what kind of data sources and "
            "document types would contain these values.\n"
        )

    # Build variant instructions for high-priority batches
    variant_block = ""
    if priority == "high":
        variant_block = (
            "\nFor HIGH priority gaps, generate 2-3 variant queries per city+field:\n"
            "- Variant A: target the operator/owner directly "
            "(e.g. 'DVB Dresden fleet inventory 2025')\n"
            "- Variant B: target procurement/tender data "
            "(e.g. 'Dresden electric bus tender award')\n"
            "- Variant C: target subsidy/funding records "
            "(e.g. 'Saxony electric bus funding Dresden')\n"
        )

    system_prompt = (
        "You are a search query formulator for urban climate data research.\n"
        "Generate concise, targeted Google search queries (3-8 words each) to find\n"
        "specific data points for cities.\n\n"
        "RULES:\n"
        "- Each query should target a specific city + data field combination.\n"
        "- Use official terminology (e.g. 'electric bus fleet' not 'e-bus').\n"
        "- Include the city name in each query.\n"
        "- Prefer government, operator, and official sources.\n"
        "- For cost data, include currency context (EUR, GBP, etc.).\n"
        f"- Generate at most {max_queries} queries.\n"
        f"{question_block}"
        f"{source_hints_block}"
        f"{variant_block}\n"
        "OUTPUT: Return a JSON array of query strings. Nothing else.\n"
        'Example: ["Dresden electric bus fleet size 2024", "Dresden public transport capex budget"]\n'
    )

    user_prompt = (
        f"Region: {region}\n"
        f"Cities: {', '.join(cities)}\n"
        f"Target fields: {', '.join(target_fields)}\n"
        f"Priority: {priority}\n"
        f"Max queries: {max_queries}\n"
    )
    if question_context:
        user_prompt += f"Research question: {question_context}\n"
    user_prompt += "\nGenerate search queries to find these data points.\n"

    try:
        request_kwargs: dict[str, object] = {
            "model": config.enrichment.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        if config.enrichment.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = config.enrichment.reasoning_effort

        response = client.chat.completions.create(**request_kwargs)
        if not response.choices:
            return []

        content = extract_message_text(response.choices[0].message.content)
        candidate = extract_json_candidate(content)
        parsed = json.loads(candidate)

        if isinstance(parsed, list):
            queries = [str(q).strip() for q in parsed if isinstance(q, str) and q.strip()]
            return queries[:int(max_queries)]

        return []

    except Exception:
        logger.warning("Search query formulation failed.", exc_info=True)
        return []


def _load_city_groups() -> list[dict[str, Any]]:
    """Load city groups from the static asset file."""
    if not _CITY_GROUPS_PATH.exists():
        return []
    try:
        data = json.loads(_CITY_GROUPS_PATH.read_text(encoding="utf-8"))
        return data.get("groups", [])
    except (json.JSONDecodeError, OSError):
        return []


def _build_city_group_index(groups: list[dict[str, Any]]) -> dict[str, str]:
    """Build a city -> group_id index."""
    index: dict[str, str] = {}
    for group in groups:
        group_id = group.get("id", "unknown")
        for city in group.get("cities", []):
            index[city] = group_id
    return index


def _plan_national_benchmark_batches(
    gap_manifest: GapManifest,
    region_gaps: dict[str, list[dict[str, Any]]],
    city_groups: list[dict[str, Any]],
    config: AppConfig,
    api_key: str,
    remaining_budget: int,
) -> tuple[list[SearchBatch], int]:
    """Generate national/regional benchmark search batches for Method A grounding.

    For each region with estimable fields, uses the LLM to formulate 2-3
    targeted queries for national/regional average unit costs and benchmarks.

    Returns (batches, total_query_count).
    """
    if remaining_budget <= 0:
        return [], 0

    estimable_fields = [
        fc.field
        for fc in gap_manifest.query_fields
        if fc.classification != "non_estimable"
    ]
    if not estimable_fields:
        return [], 0

    # Build group_id → city list lookup
    group_cities: dict[str, list[str]] = {}
    for group in city_groups:
        group_id = group.get("id", "unknown")
        group_cities[group_id] = group.get("cities", [])

    batches: list[SearchBatch] = []
    total_queries = 0

    for region in region_gaps:
        if total_queries >= remaining_budget:
            break

        cities_in_region = group_cities.get(region, [])
        # Also include cities from the gaps themselves for country inference
        gap_cities = [g["city"] for g in region_gaps[region]]
        all_region_cities = list(set(cities_in_region) | set(gap_cities))

        queries = _formulate_national_benchmark_queries(
            region=region,
            region_cities=all_region_cities,
            target_fields=estimable_fields,
            config=config,
            api_key=api_key,
        )

        if not queries:
            continue

        remaining = remaining_budget - total_queries
        queries = queries[:min(3, remaining)]  # cap at 3 per region
        total_queries += len(queries)

        batch_id = f"batch_{region}_national_{uuid.uuid4().hex[:8]}"
        batches.append(SearchBatch(
            batch_id=batch_id,
            cities=[region],  # region name as the "city" for national benchmarks
            target_fields=estimable_fields,
            search_type="national_benchmark",
            queries=queries,
            budget={"max_queries": 3, "deep_dive_allowed": False},
            priority="medium",
        ))

    return batches, total_queries


def _formulate_national_benchmark_queries(
    region: str,
    region_cities: list[str],
    target_fields: list[str],
    config: AppConfig,
    api_key: str,
) -> list[str]:
    """Use LLM to generate concise Google queries for national/regional benchmarks.

    Targets average unit costs, fleet benchmarks, and infrastructure costs
    at the national/regional level for grounding Method A estimates.
    """
    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)

    system_prompt = (
        "You are a search query formulator for national/regional benchmark data.\n"
        "Generate concise Google search queries (4-8 words each) to find\n"
        "national or regional AVERAGE unit costs, fleet benchmarks, and\n"
        "infrastructure cost benchmarks.\n\n"
        "RULES:\n"
        "- Target national averages, not city-specific data.\n"
        "- Infer the country/countries from the city list provided.\n"
        "- Use terms like 'national average', 'benchmark', 'unit cost'.\n"
        "- Prefer government reports, industry associations, and statistical offices.\n"
        "- For cost data, include currency context (EUR, GBP, etc.).\n"
        "- Generate at most 3 queries.\n\n"
        "OUTPUT: Return a JSON array of query strings. Nothing else.\n"
        'Example: ["Germany electric bus procurement cost per unit 2024", '
        '"Germany charging infrastructure cost benchmark report"]\n'
    )

    user_prompt = (
        f"Region: {region}\n"
        f"Cities in this region: {', '.join(region_cities[:10])}\n"
        f"Target fields needing national benchmarks: {', '.join(target_fields)}\n"
        "\nGenerate 2-3 search queries for national/regional average values "
        "for these fields. Focus on unit costs and per-unit benchmarks.\n"
    )

    try:
        request_kwargs: dict[str, object] = {
            "model": config.enrichment.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        if config.enrichment.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = config.enrichment.reasoning_effort

        response = client.chat.completions.create(**request_kwargs)
        if not response.choices:
            return []

        content = extract_message_text(response.choices[0].message.content)
        candidate = extract_json_candidate(content)
        parsed = json.loads(candidate)

        if isinstance(parsed, list):
            queries = [str(q).strip() for q in parsed if isinstance(q, str) and q.strip()]
            return queries[:3]

        return []

    except Exception:
        logger.warning("National benchmark query formulation failed.", exc_info=True)
        return []


def _plan_comparative_batches(
    gap_manifest: GapManifest,
    region_gaps: dict[str, list[dict[str, Any]]],
    config: AppConfig,
    api_key: str,
    remaining_budget: int,
) -> tuple[list[SearchBatch], int]:
    """Generate cross-country comparative search batches for Method C grounding.

    Groups estimable fields by category, then generates 1-2 queries per category
    targeting cross-country comparison reports and per-capita indices.

    Returns (batches, total_query_count).
    """
    if remaining_budget <= 0:
        return [], 0

    estimable_fields = [
        fc.field
        for fc in gap_manifest.query_fields
        if fc.classification != "non_estimable"
    ]
    if not estimable_fields:
        return [], 0

    field_categories = _categorize_fields(estimable_fields)
    all_regions = list(region_gaps.keys())

    batches: list[SearchBatch] = []
    total_queries = 0

    for category, cat_fields in field_categories.items():
        if total_queries >= remaining_budget:
            break

        queries = _formulate_comparative_queries(
            category=category,
            target_fields=cat_fields,
            regions=all_regions,
            config=config,
            api_key=api_key,
        )

        if not queries:
            continue

        remaining = remaining_budget - total_queries
        queries = queries[:min(2, remaining)]  # cap at 2 per category
        total_queries += len(queries)

        batch_id = f"batch_comparative_{category}_{uuid.uuid4().hex[:8]}"
        batches.append(SearchBatch(
            batch_id=batch_id,
            cities=["cross_country"],
            target_fields=cat_fields,
            search_type="comparative_benchmark",
            queries=queries,
            budget={"max_queries": 2, "deep_dive_allowed": False},
            priority="medium",
        ))

    return batches, total_queries


def _formulate_comparative_queries(
    category: str,
    target_fields: list[str],
    regions: list[str],
    config: AppConfig,
    api_key: str,
) -> list[str]:
    """Use LLM to generate queries for cross-country comparison reports.

    Targets international benchmarks, per-capita/per-fleet indices, and
    publications from organizations like UITP, C40, ICLEI, Bloomberg NEF.
    """
    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)

    system_prompt = (
        "You are a search query formulator for cross-country comparative data.\n"
        "Generate concise Google search queries (4-10 words each) to find\n"
        "cross-country comparison reports, per-capita indices, and per-fleet\n"
        "benchmarks from international organizations.\n\n"
        "RULES:\n"
        "- Target international comparison reports, NOT single-country data.\n"
        "- Look for publications from UITP, C40, ICLEI, Bloomberg NEF, IEA, OECD.\n"
        "- Use terms like 'comparison', 'benchmark', 'per capita', 'per 100k inhabitants'.\n"
        "- Include the field category context (e.g. fleet, infrastructure, costs, emissions).\n"
        "- Generate at most 2 queries.\n\n"
        "OUTPUT: Return a JSON array of query strings. Nothing else.\n"
        'Example: ["European electric bus fleet comparison per capita UITP 2024", '
        '"C40 cities charging infrastructure benchmark report"]\n'
    )

    user_prompt = (
        f"Field category: {category}\n"
        f"Fields in this category: {', '.join(target_fields)}\n"
        f"Regions in this run: {', '.join(regions)}\n"
        "\nGenerate 1-2 search queries for cross-country comparison reports "
        "and per-capita/per-fleet benchmarks covering these fields.\n"
    )

    try:
        request_kwargs: dict[str, object] = {
            "model": config.enrichment.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        if config.enrichment.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = config.enrichment.reasoning_effort

        response = client.chat.completions.create(**request_kwargs)
        if not response.choices:
            return []

        content = extract_message_text(response.choices[0].message.content)
        candidate = extract_json_candidate(content)
        parsed = json.loads(candidate)

        if isinstance(parsed, list):
            queries = [str(q).strip() for q in parsed if isinstance(q, str) and q.strip()]
            return queries[:2]

        return []

    except Exception:
        logger.warning("Comparative query formulation failed.", exc_info=True)
        return []


__all__ = ["plan_searches"]
