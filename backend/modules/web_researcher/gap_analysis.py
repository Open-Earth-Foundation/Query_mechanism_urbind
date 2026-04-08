"""Gap Analyst (Agent 1): classify fields and identify city-level data gaps."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from backend.modules.web_researcher.models import (
    GapManifest,
    _GapManifestEnvelope,
)
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)

_MAX_RETRIES = 1


def run_gap_analysis(
    question: str,
    context_bundle: dict[str, Any],
    config: AppConfig,
    api_key: str,
) -> GapManifest:
    """Analyse the context bundle for data gaps and classify each query field.

    Returns a ``GapManifest`` with field classifications, per-city gaps, and
    non-estimable field list.  On failure, returns an empty manifest so the
    pipeline can continue without enrichment.
    """
    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)
    system_prompt = _build_system_prompt(config)
    user_prompt = _build_user_prompt(question, context_bundle, config)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            request_kwargs: dict[str, object] = {
                "model": config.enrichment.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": float(config.enrichment.temperature),
            }
            if config.enrichment.reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = config.enrichment.reasoning_effort

            logger.info(
                "Gap analysis attempt=%d model=%s",
                attempt + 1,
                config.enrichment.model,
            )
            response = client.chat.completions.create(**request_kwargs)
            if not response.choices:
                raise ValueError("Gap analyst returned no choices.")

            content = extract_message_text(response.choices[0].message.content)
            candidate = extract_json_candidate(content)
            parsed = json.loads(candidate)
            envelope = _GapManifestEnvelope.model_validate(parsed)
            manifest = GapManifest(
                query_fields=envelope.query_fields,
                city_gaps=envelope.city_gaps,
                non_estimable_fields=envelope.non_estimable_fields,
            )
            logger.info(
                "Gap analysis completed: fields=%d city_gaps=%d non_estimable=%d",
                len(manifest.query_fields),
                len(manifest.city_gaps),
                len(manifest.non_estimable_fields),
            )
            return manifest

        except Exception:
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "Gap analysis attempt %d failed; retrying with explicit schema hint.",
                    attempt + 1,
                    exc_info=True,
                )
                user_prompt = _add_schema_hint(user_prompt)
                continue
            logger.warning(
                "Gap analysis failed after %d attempts; returning empty manifest.",
                _MAX_RETRIES + 1,
                exc_info=True,
            )
            return _empty_manifest()

    return _empty_manifest()


def _build_system_prompt(config: AppConfig) -> str:
    max_fields = config.enrichment.max_fields_per_query
    freshness_days = config.enrichment.freshness_threshold_days
    return (
        "You are a data-gap analyst for urban climate action plans.\n"
        "Your task is to classify every field the user's question asks about and\n"
        "identify which cities are missing data for those fields.\n\n"
        "FIELD CLASSIFICATION (apply once per field, NOT per city):\n"
        '1. "estimable_numerical" — a concrete quantity (cost, count, capacity, area)\n'
        "   that can be estimated from peer city data or national averages.\n"
        '2. "derivable_from_ratio" — a value computable from another field the city\n'
        "   has, combined with a ratio observable in peers (e.g. per-unit cost from\n"
        "   total cost and fleet size).\n"
        '3. "non_estimable" — qualitative, unique-to-city, or legally specific data\n'
        "   (operator names, contract terms, specific policy text).\n\n"
        "PER-CITY GAP DETECTION:\n"
        "- For each city in the context, check every estimable/derivable field.\n"
        "- A field is blank if the context contains no concrete numeric value for it.\n"
        f"- A field is stale if the data is older than {freshness_days} days, or uses\n"
        "  aspirational language (\"plans to\", \"targets\") without concrete numbers.\n"
        "- search_priority: high = MISSING_ENTIRELY, medium = stale/partial, low = minor gap.\n\n"
        "RULES:\n"
        f"- Maximum {max_fields} fields per query.\n"
        "- Field classification is per-query, not per-city (consistency).\n"
        "- Return valid JSON matching the schema below. No extra keys.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "```json\n"
        "{\n"
        '  "query_fields": [\n'
        '    {"field": "...", "classification": "estimable_numerical|derivable_from_ratio|non_estimable",\n'
        '     "searchable": true|false, "rationale": "..."}\n'
        "  ],\n"
        '  "city_gaps": [\n'
        '    {"city": "...", "blank_fields": ["..."], "stale_flags": ["..."],\n'
        '     "search_priority": "high|medium|low"}\n'
        "  ],\n"
        '  "non_estimable_fields": ["..."]\n'
        "}\n"
        "```\n"
    )


def _build_user_prompt(
    question: str,
    context_bundle: dict[str, Any],
    config: AppConfig,
) -> str:
    context_json = json.dumps(context_bundle, ensure_ascii=True, indent=2, default=str)
    research_question = context_bundle.get("research_question", question)
    return (
        f"User question:\n{question.strip()}\n\n"
        f"Research question:\n{research_question}\n\n"
        "Context bundle (contains SQL results, markdown excerpts, and metadata):\n"
        "```json\n"
        f"{context_json}\n"
        "```\n\n"
        "Classify each field the question asks about, then list per-city gaps.\n"
        "Return only the JSON object described in your instructions.\n"
    )


def _add_schema_hint(user_prompt: str) -> str:
    return (
        user_prompt
        + "\n\nIMPORTANT: Your previous response was not valid JSON. "
        "Return ONLY the JSON object with keys: query_fields, city_gaps, non_estimable_fields. "
        "No markdown, no explanation, just the JSON object.\n"
    )


def _empty_manifest() -> GapManifest:
    return GapManifest(query_fields=[], city_gaps=[], non_estimable_fields=[])


__all__ = ["run_gap_analysis"]
