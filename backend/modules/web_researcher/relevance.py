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
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)


class RelevanceCheckOutput(BaseModel):
    """Structured output from relevance checker."""

    is_relevant: bool
    reason: str


def check_relevance_batch(
    results: list[SearchResult],
    target_fields: list[str],
    cities: list[str],
    config: AppConfig,
    api_key: str,
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
        "OUTPUT: Return a JSON array of objects with 'index' (int), "
        "'is_relevant' (bool), and 'reason' (str).\n"
    )

    user_prompt = (
        f"Cities: {', '.join(cities)}\n"
        f"Target fields: {', '.join(target_fields)}\n\n"
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

        response = client.chat.completions.create(**request_kwargs)
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

        # Build relevance map
        relevance_map: dict[int, bool] = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    idx = item.get("index")
                    relevant = item.get("is_relevant", True)
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
