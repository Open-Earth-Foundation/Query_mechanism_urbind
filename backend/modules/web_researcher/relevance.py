"""Relevance checker: LLM pre-filter for search results before scraping."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from backend.modules.web_researcher.search import SearchResult
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)
from backend.services.llm_observability import (
    LlmCallContext,
    LlmCallRecorder,
    record_openai_chat_completion,
)
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)


class RelevanceCheckOutput(BaseModel):
    """Structured output from relevance checker."""

    is_relevant: bool
    entity_check: str  # "city_confirmed" | "likely_false_match" | "uncertain"
    reason: str


def _build_entity_disambiguation_block(cities: list[str]) -> str:
    """Return a user-prompt fragment reminding the LLM to disambiguate city names.

    This fires for every batch — not limited to a hardcoded blocklist — so that
    all cities get the entity-collision check.
    """
    city_list = ", ".join(cities)
    return (
        f"\nENTITY CHECK: For each result, confirm the mention of "
        f"{city_list} refers to the city/municipality — not a person, "
        "company, brand, product, or financial entity that shares the name. "
        "City names frequently collide with corporations (e.g. Heidelberg → "
        "HeidelbergCement), people (e.g. Munster → Gene Munster), auction/"
        "trade platforms (e.g. Mannheim → Manheim Auctions), and cultural "
        "references. Reject any result where the primary subject is one of "
        "these non-municipal entities.\n"
    )


def check_relevance_batch(
    results: list[SearchResult],
    target_fields: list[str],
    cities: list[str],
    config: AppConfig,
    api_key: str,
    llm_recorder: LlmCallRecorder | None = None,
) -> list[tuple[SearchResult, bool]]:
    """Check relevance of search results against target fields and cities.

    Returns list of (result, is_relevant) tuples.
    On failure, all results are marked as potentially relevant (fail-open).
    """
    if not results:
        return []

    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)

    # Build result summaries for the LLM
    result_summaries = []
    for i, r in enumerate(results):
        result_summaries.append({
            "index": i,
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
        })

    system_prompt = (
        "You are a relevance checker for urban climate data research.\n"
        "For each search result, determine if it likely contains quantitative data\n"
        "about the specified fields for the specified cities.\n\n"
        "RULES:\n"
        "- Mark as relevant if the snippet or title suggests concrete numbers,\n"
        "  statistics, budgets, fleet sizes, or official reports.\n"
        "- Mark as irrelevant if it's clearly about a different topic, different city,\n"
        "  or contains only opinion/commentary without data.\n"
        "- When uncertain, mark as relevant (fail-open).\n\n"
        "ENTITY DISAMBIGUATION (critical):\n"
        "- Verify the city name in the result refers to the MUNICIPALITY, not a\n"
        "  person, company, brand, or product with a similar name.\n"
        "  Examples of FALSE MATCHES to reject:\n"
        "  * 'Munster' as a financial analyst (Gene Munster / Loup Ventures)\n"
        "  * 'Manheim' as Cox Automotive's vehicle auction brand\n"
        "  * 'Heidelberg' as HeidelbergCement / Heidelberg Materials (the company)\n"
        "  * 'Dresden' as a product line or model name\n"
        "- Check whether the snippet describes municipal/city-level policy,\n"
        "  procurement, or infrastructure — not corporate operations of a\n"
        "  company that happens to share the city's name.\n"
        "- If the URL domain suggests a non-municipal source (investing.com,\n"
        "  autodealertoday, corporate newsrooms), apply extra scrutiny to\n"
        "  confirm the content is about the city's transport/climate policy.\n\n"
        "OUTPUT: Return a JSON array of objects with 'index' (int), "
        "'is_relevant' (bool), 'entity_check' (str: 'city_confirmed' | "
        "'likely_false_match' | 'uncertain'), and 'reason' (str).\n"
    )

    entity_block = _build_entity_disambiguation_block(cities)

    user_prompt = (
        f"Cities: {', '.join(cities)}\n"
        f"Target fields: {', '.join(target_fields)}\n"
        f"{entity_block}\n"
        "Search results:\n"
        f"```json\n{json.dumps(result_summaries, indent=2)}\n```\n"
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

        response = record_openai_chat_completion(
            client,
            request_kwargs,
            context=LlmCallContext(
                stage_name="enrichment",
                stage_family="enrichment",
                agent="relevance_checker",
                call_kind="relevance_check",
                model=config.enrichment.model,
                metadata={
                    "result_count": len(results),
                    "city_count": len(cities),
                    "target_field_count": len(target_fields),
                },
            ),
            recorder=llm_recorder,
        )
        if not response.choices:
            return [(r, True) for r in results]

        content = extract_message_text(response.choices[0].message.content)
        candidate = extract_json_candidate(content)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # LLM sometimes emits trailing text after valid JSON — parse
            # only the first complete value and discard the rest.
            parsed, _ = json.JSONDecoder().raw_decode(candidate)

        # Build relevance map — override is_relevant when entity check
        # explicitly flags a false match.
        relevance_map: dict[int, bool] = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    idx = item.get("index")
                    relevant = item.get("is_relevant", True)
                    entity_check = str(item.get("entity_check", "")).lower()
                    if entity_check == "likely_false_match":
                        relevant = False
                    if isinstance(idx, int):
                        relevance_map[idx] = bool(relevant)

        output: list[tuple[SearchResult, bool]] = []
        for i, r in enumerate(results):
            output.append((r, relevance_map.get(i, True)))

        relevant_count = sum(1 for _, rel in output if rel)
        logger.info(
            "Relevance check: %d/%d results marked relevant.",
            relevant_count,
            len(results),
        )
        return output

    except Exception:
        logger.warning("Relevance check failed; marking all as relevant.", exc_info=True)
        return [(r, True) for r in results]


__all__ = ["RelevanceCheckOutput", "check_relevance_batch"]
