"""Gap Analyst (Agent 1): classify fields and identify city-level data gaps.

Two entry points:
- ``run_gap_analysis`` — original single-pass implementation: one LLM call
  produces both the field decomposition and the per-city gaps.  Kept for
  backward compatibility with existing callers and integration tests.
- ``decompose_fields`` (Phase 0) + ``detect_city_gaps`` (Phase 2) — the
  split flow used by the new orchestrator.  Decomposition runs against
  the question alone; per-city gap detection runs against the current
  context bundle. A future external-source stage sits between those steps
  but is a no-op in this cleanup.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from backend.modules.web_researcher.models import (
    CityGap,
    FieldClassification,
    FieldDecomposition,
    GapManifest,
    _CityGapsEnvelope,
    _FieldDecompositionEnvelope,
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
    user_prompt = _build_user_prompt(question, context_bundle)

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
        "FIELD DECOMPOSITION (critical — do this FIRST):\n"
        "- Break compound questions into their granular sub-fields BEFORE classifying.\n"
        "- If the question mentions sub-categories, types, modes, or variants, each\n"
        "  one becomes its own field. Also add the aggregate total if asked.\n"
        "- Always decompose: types, categories, sectors, modes, phases, fuel types,\n"
        "  vehicle classes, infrastructure variants, etc.\n"
        "- Use short snake_case names for fields.\n\n"
        "FIELD CLASSIFICATION (apply once per field, NOT per city):\n"
        '1. "estimable_numerical" — a concrete quantity (cost, count, capacity, area)\n'
        "   that can be estimated from peer city data or national averages.\n"
        '2. "derivable_from_ratio" — a value computable from another field the city\n'
        "   has, combined with a ratio observable in peers (e.g. per-unit cost from\n"
        "   total cost and fleet size, or charger count from fleet size using\n"
        "   charger-to-vehicle ratios).\n"
        '3. "non_estimable" — qualitative, unique-to-city, or legally specific data\n'
        "   (operator names, contract terms, specific policy text), OR quantities\n"
        "   that depend heavily on local context (housing stock mix, street layout,\n"
        "   local parking policy) where a per-capita or peer-city proxy would be\n"
        "   misleading.\n\n"
        "WORKED EXAMPLE:\n"
        "Question: 'What charging infrastructure volume targets by 2030 are in the\n"
        "CCCs — public charging points (AC vs DC), depot charging, bus charging\n"
        "depots, fast corridors, residential on-street?'\n\n"
        "Correct field decomposition & classification:\n"
        "  depot_charger_count — derivable_from_ratio\n"
        "    Rationale: Can be derived from fleet size using charger-to-vehicle\n"
        "    ratios (e.g. 1 depot charger per 3–5 buses).\n"
        "  bus_charging_depot_count — derivable_from_ratio\n"
        "    Rationale: Typically 1 depot per 50–80 buses. Derivable from fleet data.\n"
        "  fast_charging_corridor_points — estimable_numerical\n"
        "    Rationale: Some cities report this. Estimable from peers but wide ranges.\n"
        "  public_ac_charger_count — estimable_numerical\n"
        "    Rationale: Reported by some cities and national registries.\n"
        "  public_dc_charger_count — estimable_numerical\n"
        "    Rationale: Same as AC — estimable with caveats.\n"
        "  residential_onstreet_charging — non_estimable\n"
        "    Rationale: Depends heavily on housing stock (apartment vs detached),\n"
        "    street layout, and local parking policy. Per-capita proxy is misleading.\n\n"
        "WRONG: collapsing all of the above into a single field like\n"
        "'charging_infrastructure_targets'. Always decompose.\n\n"
        "FIELD SCOPE (apply once per field — drives aggregation safety):\n"
        "Pick exactly one scope per field.  The writer refuses to sum fields\n"
        "with different scopes; it presents per-scope subtotals instead.\n"
        '- "municipal" — costs/assets borne by the city government itself.\n'
        '- "public_transport" — public transit operators (bus, tram, metro, rail).\n'
        '- "private" — households, private companies, private vehicle owners.\n'
        '- "mixed" — explicit cross-scope aggregate; reserve for deliberate\n'
        "  cross-cuts, do NOT use as a default.\n"
        '- "unscoped" — only when truly ambiguous.\n\n'
        "PER-CITY GAP DETECTION:\n"
        "- For each city in the context, check every estimable/derivable field.\n"
        "- A field is **blank** if the context contains no concrete numeric value for it.\n"
        f"- A field is **stale** if the data is older than {freshness_days} days, or uses\n"
        '  aspirational language ("plans to", "targets") without concrete numbers.\n'
        "- A field is **bundled** if the city reports an aggregate / parent value that\n"
        "  contains the requested field, but does NOT report the disaggregated line.\n"
        '  Example: question asks for "per-vehicle CAPEX" but the CCC only states\n'
        '  "total fleet CAPEX = €100M". The total is bundled; the per-unit line is missing.\n'
        "  Put such fields in ``bundled_fields`` (NOT ``blank_fields``) so the\n"
        "  estimator can derive the line via peer per-unit ratios.\n"
        "- search_priority: high = MISSING_ENTIRELY, medium = stale/bundled/partial, low = minor gap.\n\n"
        "RULES:\n"
        f"- Maximum {max_fields} fields per query.\n"
        "- Field classification is per-query, not per-city (consistency).\n"
        "- Return valid JSON matching the schema below. No extra keys.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "```json\n"
        "{\n"
        '  "query_fields": [\n'
        '    {"field": "...", "classification": "estimable_numerical|derivable_from_ratio|non_estimable",\n'
        '     "searchable": true|false, "rationale": "...",\n'
        '     "scope": "municipal|public_transport|private|mixed|unscoped"}\n'
        "  ],\n"
        '  "city_gaps": [\n'
        '    {"city": "...", "blank_fields": ["..."], "stale_flags": ["..."],\n'
        '     "bundled_fields": ["..."], "search_priority": "high|medium|low"}\n'
        "  ],\n"
        '  "non_estimable_fields": ["..."]\n'
        "}\n"
        "```\n"
    )


def _slim_context_for_gap_analysis(context_bundle: dict[str, Any]) -> dict[str, Any]:
    """Strip retrieval metadata from the context bundle while keeping data values.

    The gap analyst needs excerpt content to identify gaps, but it does not
    need chunk IDs, retrieval distances, batch plans, or decision audit data.
    """
    slim: dict[str, Any] = {}
    for key in (
        "research_question",
        "original_question",
        "analysis_mode",
        "query_mode",
        "city_scope_mode",
        "selected_cities",
        "selected_city_names",
        "inspected_cities",
        "inspected_city_names",
    ):
        if key in context_bundle:
            slim[key] = context_bundle[key]

    # Markdown: keep excerpts only, drop run-level city scope metadata.
    markdown = context_bundle.get("markdown")
    if isinstance(markdown, dict):
        slim_md: dict[str, Any] = {}
        for keep_key in ("excerpts", "excerpt_count", "analysis_mode"):
            if keep_key in markdown:
                slim_md[keep_key] = markdown[keep_key]
        slim["markdown"] = slim_md

    return slim


def _build_user_prompt(
    question: str,
    context_bundle: dict[str, Any],
) -> str:
    slim = _slim_context_for_gap_analysis(context_bundle)
    context_json = json.dumps(slim, ensure_ascii=False, indent=2, default=str)
    research_question = context_bundle.get("research_question", question)
    return (
        f"User question:\n{question.strip()}\n\n"
        f"Research question:\n{research_question}\n\n"
        "Context bundle (contains markdown excerpts and run metadata):\n"
        "```json\n"
        f"{context_json}\n"
        "```\n\n"
        "Decompose the question into granular fields, classify each, then list per-city gaps.\n"
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


def _empty_decomposition() -> FieldDecomposition:
    return FieldDecomposition(query_fields=[], non_estimable_fields=[])


# ---------------------------------------------------------------------------
# Phase 0: field decomposition (no context bundle, no per-city gaps)
# ---------------------------------------------------------------------------


def _build_decompose_system_prompt(config: AppConfig) -> str:
    max_fields = config.enrichment.max_fields_per_query
    return (
        "You are a data-field decomposer for urban climate action plans.\n"
        "Your task is to break the user's question into granular, individually\n"
        "estimable fields and classify each one. You do NOT consider per-city\n"
        "data availability — that is a later step.\n\n"
        "FIELD DECOMPOSITION (do this FIRST):\n"
        "- Break compound questions into their granular sub-fields.\n"
        "- If the question mentions sub-categories, types, modes, or variants,\n"
        "  each one becomes its own field. Also add the aggregate total if asked.\n"
        "- Always decompose: types, categories, sectors, modes, phases, fuel\n"
        "  types, vehicle classes, infrastructure variants, etc.\n"
        "- Use short snake_case names for fields.\n\n"
        "FIELD CLASSIFICATION (apply once per field):\n"
        '1. "estimable_numerical" — a concrete quantity (cost, count, capacity, area)\n'
        "   that can be estimated from peer city data or national averages.\n"
        '2. "derivable_from_ratio" — a value computable from another field the\n'
        "   city has, combined with a ratio observable in peers (e.g. per-unit\n"
        "   cost from total cost and fleet size, or charger count from fleet\n"
        "   size using charger-to-vehicle ratios).\n"
        '3. "non_estimable" — qualitative, unique-to-city, or legally specific\n'
        "   data (operator names, contract terms, specific policy text), OR\n"
        "   quantities that depend heavily on local context (housing stock\n"
        "   mix, street layout, local parking policy) where a per-capita or\n"
        "   peer-city proxy would be misleading.\n\n"
        "FIELD SCOPE (apply once per field — drives aggregation safety):\n"
        "Pick exactly one scope per field.  The writer refuses to sum fields\n"
        "with different scopes; it presents per-scope subtotals instead.\n"
        '- "municipal" — costs/assets borne by the city government itself:\n'
        "  municipal fleet (cars, trucks, waste vehicles), city-owned buildings,\n"
        "  city-run programmes, city-funded infrastructure.\n"
        '- "public_transport" — public transit operators (buses, trams, metro,\n'
        "  light rail) regardless of legal ownership; commuter rail.\n"
        '- "private" — households, private companies, private vehicle owners,\n'
        "  privately-financed infrastructure.\n"
        '- "mixed" — the field is *deliberately* an aggregate across scopes\n'
        "  (e.g. total charging points city-wide, regardless of operator).\n"
        "  Reserve for explicit cross-cuts; do NOT use as a default.\n"
        '- "unscoped" — not applicable, or you cannot decide.\n\n'
        "WORKED EXAMPLE:\n"
        "Question: 'What charging infrastructure volume targets by 2030 are in\n"
        "the CCCs — public charging points (AC vs DC), depot charging, bus\n"
        "charging depots, fast corridors, residential on-street?'\n\n"
        "Correct field decomposition, classification, and scope:\n"
        "  depot_charger_count — derivable_from_ratio — municipal\n"
        "  bus_charging_depot_count — derivable_from_ratio — public_transport\n"
        "  fast_charging_corridor_points — estimable_numerical — mixed\n"
        "  public_ac_charger_count — estimable_numerical — mixed\n"
        "  public_dc_charger_count — estimable_numerical — mixed\n"
        "  residential_onstreet_charging — non_estimable — private\n\n"
        "RULES:\n"
        f"- Maximum {max_fields} fields per query.\n"
        "- Set ``searchable: true`` for any field that could plausibly be\n"
        "  found in public web sources.  Set ``false`` only for highly\n"
        "  internal/proprietary fields.\n"
        "- Always include ``scope`` (default to ``unscoped`` only if truly\n"
        "  ambiguous — prefer making a call).\n"
        "- Return valid JSON matching the schema below. No extra keys.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "```json\n"
        "{\n"
        '  "query_fields": [\n'
        '    {"field": "...", "classification": "estimable_numerical|derivable_from_ratio|non_estimable",\n'
        '     "searchable": true|false, "rationale": "...",\n'
        '     "scope": "municipal|public_transport|private|mixed|unscoped"}\n'
        "  ],\n"
        '  "non_estimable_fields": ["..."]\n'
        "}\n"
        "```\n"
    )


def _build_decompose_user_prompt(question: str) -> str:
    return (
        f"User question:\n{question.strip()}\n\n"
        "Decompose into granular fields and classify each.\n"
        "Return only the JSON object described in your instructions.\n"
    )


def decompose_fields(
    question: str,
    config: AppConfig,
    api_key: str,
) -> FieldDecomposition:
    """Phase 0: decompose the question into granular classified fields.

    Does NOT inspect any context bundle; per-city gap detection happens
    later (``detect_city_gaps``) once Phase 1 has enriched the bundle.
    On any failure, returns an empty decomposition so the pipeline can
    continue gracefully.
    """
    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)
    system_prompt = _build_decompose_system_prompt(config)
    user_prompt = _build_decompose_user_prompt(question)

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
                "Field decomposition attempt=%d model=%s",
                attempt + 1,
                config.enrichment.model,
            )
            response = client.chat.completions.create(**request_kwargs)
            if not response.choices:
                raise ValueError("Decomposer returned no choices.")

            content = extract_message_text(response.choices[0].message.content)
            candidate = extract_json_candidate(content)
            parsed = json.loads(candidate)
            envelope = _FieldDecompositionEnvelope.model_validate(parsed)
            decomposition = FieldDecomposition(
                query_fields=envelope.query_fields,
                non_estimable_fields=envelope.non_estimable_fields,
            )
            logger.info(
                "Field decomposition completed: fields=%d non_estimable=%d",
                len(decomposition.query_fields),
                len(decomposition.non_estimable_fields),
            )
            return decomposition

        except Exception:
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "Decomposition attempt %d failed; retrying with explicit schema hint.",
                    attempt + 1,
                    exc_info=True,
                )
                user_prompt = _add_schema_hint(user_prompt)
                continue
            logger.warning(
                "Field decomposition failed after %d attempts; returning empty decomposition.",
                _MAX_RETRIES + 1,
                exc_info=True,
            )
            return _empty_decomposition()

    return _empty_decomposition()


# ---------------------------------------------------------------------------
# Phase 2: per-city gap detection (given pre-decomposed fields + bundle)
# ---------------------------------------------------------------------------


def _build_detect_system_prompt(config: AppConfig) -> str:
    freshness_days = config.enrichment.freshness_threshold_days
    return (
        "You are a per-city gap detector for urban climate action plans.\n"
        "You have already received the full list of query fields and their\n"
        "classifications.  Your job is to decide, for each city in the\n"
        "context, which of those fields are blank or stale.\n\n"
        "PER-CITY GAP DETECTION:\n"
        "- For each city in the context, check every estimable/derivable\n"
        "  field given to you.\n"
        "- A field is **blank** if the context contains no concrete numeric\n"
        "  value for it in the available markdown excerpts or metadata.\n"
        f"- A field is **stale** if the data is older than {freshness_days} days,\n"
        '  or uses aspirational language ("plans to", "targets") without\n'
        "  concrete numbers.\n"
        "- A field is **bundled** if the city reports an aggregate / parent\n"
        "  value containing the requested field but no disaggregated line.\n"
        '  Example: question asks for "per-vehicle CAPEX" but the CCC only\n'
        '  states "total fleet CAPEX = €100M". Put it in ``bundled_fields``\n'
        "  (NOT ``blank_fields``) so the estimator derives the line via\n"
        "  peer per-unit ratios.\n"
        "- search_priority: high = MISSING_ENTIRELY, medium = stale/bundled/partial,\n"
        "  low = minor gap.\n"
        "- Skip fields classified as non_estimable; they don't appear in city_gaps.\n\n"
        "RULES:\n"
        "- Field classification is fixed by the upstream decomposer; do not\n"
        "  re-classify.\n"
        "- Return valid JSON matching the schema below. No extra keys.\n\n"
        "OUTPUT JSON SCHEMA:\n"
        "```json\n"
        "{\n"
        '  "city_gaps": [\n'
        '    {"city": "...", "blank_fields": ["..."], "stale_flags": ["..."],\n'
        '     "bundled_fields": ["..."], "search_priority": "high|medium|low"}\n'
        "  ]\n"
        "}\n"
        "```\n"
    )


def _build_detect_user_prompt(
    question: str,
    decomposition: FieldDecomposition,
    context_bundle: dict[str, Any],
) -> str:
    slim = _slim_context_for_gap_analysis(context_bundle)
    context_json = json.dumps(slim, ensure_ascii=False, indent=2, default=str)
    fields_json = json.dumps(
        [field.model_dump() for field in decomposition.query_fields],
        ensure_ascii=False,
        indent=2,
    )
    research_question = context_bundle.get("research_question", question)
    return (
        f"User question:\n{question.strip()}\n\n"
        f"Research question:\n{research_question}\n\n"
        "Pre-decomposed query fields (DO NOT re-classify):\n"
        "```json\n"
        f"{fields_json}\n"
        "```\n\n"
        "Context bundle (markdown excerpts and metadata):\n"
        "```json\n"
        f"{context_json}\n"
        "```\n\n"
        "Detect per-city gaps for the estimable / derivable fields.\n"
        "Return only the JSON object described in your instructions.\n"
    )


def detect_city_gaps(
    question: str,
    decomposition: FieldDecomposition,
    context_bundle: dict[str, Any],
    config: AppConfig,
    api_key: str,
) -> GapManifest:
    """Phase 2: detect per-city blank/stale gaps against the context bundle.

    Takes the field decomposition produced by ``decompose_fields`` and the
    current context bundle. Returns a full GapManifest by combining the
    upstream decomposition with per-city gaps from this pass. On any failure,
    returns a manifest with empty city_gaps so the pipeline can continue.
    """
    if not decomposition.query_fields:
        return GapManifest(
            query_fields=decomposition.query_fields,
            city_gaps=[],
            non_estimable_fields=decomposition.non_estimable_fields,
        )

    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)
    system_prompt = _build_detect_system_prompt(config)
    user_prompt = _build_detect_user_prompt(question, decomposition, context_bundle)

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
                "City-gap detection attempt=%d model=%s",
                attempt + 1,
                config.enrichment.model,
            )
            response = client.chat.completions.create(**request_kwargs)
            if not response.choices:
                raise ValueError("Detector returned no choices.")

            content = extract_message_text(response.choices[0].message.content)
            candidate = extract_json_candidate(content)
            parsed = json.loads(candidate)
            envelope = _CityGapsEnvelope.model_validate(parsed)
            manifest = GapManifest(
                query_fields=decomposition.query_fields,
                city_gaps=envelope.city_gaps,
                non_estimable_fields=decomposition.non_estimable_fields,
            )
            logger.info(
                "City-gap detection completed: fields=%d city_gaps=%d non_estimable=%d",
                len(manifest.query_fields),
                len(manifest.city_gaps),
                len(manifest.non_estimable_fields),
            )
            return manifest

        except Exception:
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "City-gap detection attempt %d failed; retrying with explicit schema hint.",
                    attempt + 1,
                    exc_info=True,
                )
                user_prompt = _add_schema_hint(user_prompt)
                continue
            logger.warning(
                "City-gap detection failed after %d attempts; returning empty city_gaps.",
                _MAX_RETRIES + 1,
                exc_info=True,
            )
            return GapManifest(
                query_fields=decomposition.query_fields,
                city_gaps=[],
                non_estimable_fields=decomposition.non_estimable_fields,
            )

    return GapManifest(
        query_fields=decomposition.query_fields,
        city_gaps=[],
        non_estimable_fields=decomposition.non_estimable_fields,
    )


__all__ = [
    "decompose_fields",
    "detect_city_gaps",
    "run_gap_analysis",
]
