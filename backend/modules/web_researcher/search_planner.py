"""Search Planner (Agent 2): formulate search queries from gap manifest."""

from __future__ import annotations

import json
import logging
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


def plan_searches(
    gap_manifest: GapManifest,
    config: AppConfig,
    api_key: str,
) -> list[SearchBatch]:
    """Formulate batched search queries from the gap manifest.

    Groups queries by country/region, field type, and priority.
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

        # Effort scaling per architecture doc
        budget = _compute_budget(
            blank_count=len(all_blank_fields),
            stale_count=len(all_stale_fields),
            city_count=len(batch_cities),
            priority=max_priority,
            config=config,
        )

        # Use LLM to formulate search queries
        queries = _formulate_queries(
            cities=batch_cities,
            target_fields=target_fields,
            region=region,
            priority=max_priority,
            budget=budget,
            config=config,
            api_key=api_key,
        )

        if not queries:
            continue

        remaining = max_total - total_queries
        queries = queries[:remaining]
        total_queries += len(queries)

        # Determine search type
        search_type = "missing_entirely" if all_blank_fields else "freshness_check"
        if all_blank_fields and all_stale_fields:
            search_type = "mixed"

        batches.append(SearchBatch(
            batch_id=f"batch_{uuid.uuid4().hex[:8]}",
            cities=batch_cities[:8],  # max 8 cities per batch
            target_fields=target_fields,
            search_type=search_type,
            queries=queries,
            budget=budget,
            priority=max_priority,
        ))

    logger.info(
        "Search planner: %d batches, %d total queries across %d regions.",
        len(batches),
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
        # MISSING_ENTIRELY
        query_budget = min(15, max_per_batch)
        deep_dive = True
    elif city_count > 3 and blank_count > (city_count * 0.6):
        # >60% cohort same gap → aggregate benchmarks
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
) -> list[str]:
    """Use LLM to formulate targeted search queries."""
    max_queries = budget.get("max_queries", 5)

    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)

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
        f"- Generate at most {max_queries} queries.\n\n"
        "OUTPUT: Return a JSON array of query strings. Nothing else.\n"
        'Example: ["Dresden electric bus fleet size 2024", "Dresden public transport capex budget"]\n'
    )

    user_prompt = (
        f"Region: {region}\n"
        f"Cities: {', '.join(cities)}\n"
        f"Target fields: {', '.join(target_fields)}\n"
        f"Priority: {priority}\n"
        f"Max queries: {max_queries}\n\n"
        "Generate search queries to find these data points.\n"
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
    """Build a city → group_id index."""
    index: dict[str, str] = {}
    for group in groups:
        group_id = group.get("id", "unknown")
        for city in group.get("cities", []):
            index[city] = group_id
    return index


__all__ = ["plan_searches"]
