"""Assumptions Estimator (Agent 6): produce model-estimated values for data gaps."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from backend.modules.web_researcher.models import (
    AssumptionRecord,
    EnrichedField,
    GapManifest,
    NonEstimableRecord,
    _AssumptionsEnvelope,
)
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)
from backend.services.progress_tracker import ProgressTracker
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)

_METHOD_C_SATURATION_THRESHOLD = 0.60


def run_assumptions_estimator(
    question: str,
    context_bundle: dict[str, Any],
    gap_manifest: GapManifest,
    enriched_fields: list[EnrichedField],
    config: AppConfig,
    api_key: str,
    progress: ProgressTracker | None = None,
) -> tuple[list[AssumptionRecord], list[NonEstimableRecord], str | None]:
    """Estimate values for remaining data gaps using a priority ladder.

    Returns (assumptions, non_estimable_records, saturation_warning).
    On failure returns empty lists.
    """
    # Determine which fields still need estimation
    still_missing = [f for f in enriched_fields if f.status == "still_missing"]
    partially_resolved = [f for f in enriched_fields if f.status == "partially_resolved"]
    fields_to_estimate = still_missing + partially_resolved

    if not fields_to_estimate:
        logger.info("Assumptions estimator: no gaps to estimate.")
        return [], [], None

    if progress:
        progress.add_item(
            "assumptions",
            f"{len(fields_to_estimate)} fields to estimate",
        )

    # Separate non-estimable fields immediately (no LLM needed)
    non_estimable_field_names = set(gap_manifest.non_estimable_fields)
    non_estimable_records: list[NonEstimableRecord] = []
    estimable_fields: list[EnrichedField] = []

    for field in fields_to_estimate:
        if field.field in non_estimable_field_names:
            non_estimable_records.append(
                NonEstimableRecord(
                    city=field.city,
                    field_name=field.field,
                    gap_description=f"Missing {field.field} for {field.city}",
                    explanation=(
                        "This field is classified as non-estimable: qualitative, "
                        "unique-to-city, or legally specific data that cannot be "
                        "reliably estimated from peer data."
                    ),
                    recommendation=(
                        f"Contact {field.city} directly or consult local policy "
                        "documents for authoritative data on this field."
                    ),
                )
            )
        else:
            estimable_fields.append(field)

    if progress and non_estimable_records:
        for nr in non_estimable_records:
            progress.add_item(
                "assumptions",
                f"{nr.city}: {nr.field_name} — non-estimable",
            )

    if not estimable_fields:
        logger.info(
            "Assumptions estimator: all %d gaps are non-estimable.",
            len(non_estimable_records),
        )
        return [], non_estimable_records, None

    # Pass 1: generate estimates
    if progress:
        progress.add_item("assumptions", "Pass 1: generating estimates...")
    assumptions = _call_estimator(
        question=question,
        context_bundle=context_bundle,
        gap_manifest=gap_manifest,
        estimable_fields=estimable_fields,
        config=config,
        api_key=api_key,
        pass_name="generate",
    )

    if progress and assumptions:
        for a in assumptions:
            mid = a.estimate.get("mid", "?") if isinstance(a.estimate, dict) else "?"
            progress.add_item(
                "assumptions",
                f"{a.city}: {a.field_name} ≈ {mid} ({a.confidence})",
            )
        progress.add_item(
            "assumptions",
            f"Pass 1 done: {len(assumptions)} estimates",
        )

    # Pass 2: critique and revise (capped at 1 cycle)
    if assumptions:
        if progress:
            progress.add_item("assumptions", "Pass 2: critique & revise...")
        revised = _call_estimator(
            question=question,
            context_bundle=context_bundle,
            gap_manifest=gap_manifest,
            estimable_fields=estimable_fields,
            config=config,
            api_key=api_key,
            pass_name="critique",
            prior_estimates=assumptions,
        )
        if revised:
            assumptions = revised
            if progress:
                progress.add_item(
                    "assumptions",
                    f"Pass 2 done: {len(assumptions)} revised estimates",
                )

    # Check for saturation warning
    saturation_warning = _check_saturation(assumptions)

    logger.info(
        "Assumptions estimator completed: assumptions=%d non_estimable=%d saturation=%s",
        len(assumptions),
        len(non_estimable_records),
        saturation_warning is not None,
    )
    return assumptions, non_estimable_records, saturation_warning


def _call_estimator(
    question: str,
    context_bundle: dict[str, Any],
    gap_manifest: GapManifest,
    estimable_fields: list[EnrichedField],
    config: AppConfig,
    api_key: str,
    pass_name: str,
    prior_estimates: list[AssumptionRecord] | None = None,
) -> list[AssumptionRecord]:
    """Make a single LLM call for estimation or critique."""
    model = config.enrichment.assumptions_estimator_model or config.enrichment.model
    temperature = config.enrichment.assumptions_estimator_temperature

    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)
    system_prompt = _build_system_prompt(pass_name)
    user_prompt = _build_user_prompt(
        question=question,
        context_bundle=context_bundle,
        gap_manifest=gap_manifest,
        estimable_fields=estimable_fields,
        pass_name=pass_name,
        prior_estimates=prior_estimates,
    )

    request_kwargs: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
    }
    if config.enrichment.reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = config.enrichment.reasoning_effort

    try:
        logger.info(
            "Assumptions estimator pass=%s model=%s fields=%d",
            pass_name,
            model,
            len(estimable_fields),
        )
        response = client.chat.completions.create(**request_kwargs)
        if not response.choices:
            raise ValueError("Assumptions estimator returned no choices.")

        content = extract_message_text(response.choices[0].message.content)
        candidate = extract_json_candidate(content)
        parsed = json.loads(candidate)
        envelope = _AssumptionsEnvelope.model_validate(parsed)
        return envelope.assumptions

    except Exception:
        logger.warning(
            "Assumptions estimator pass=%s failed; returning empty.",
            pass_name,
            exc_info=True,
        )
        return []


def _build_system_prompt(pass_name: str) -> str:
    base = (
        "You are an assumptions estimator for urban climate action data.\n"
        "You produce model-estimated values for missing data fields using a strict priority ladder.\n\n"
        "PRIORITY LADDER (apply in order — use the first method that applies):\n"
        "Method A (national_regional_average): City has the quantity but missing unit cost.\n"
        "  Use national/regional average unit costs. Confidence: HIGH, range: +/-15-20%.\n"
        "Method B (peer_city_proxy): City has a related value + peers show a derivable ratio.\n"
        "  Or city has KPI target + fleet size is knowable. Confidence: MEDIUM, range: +/-25-35%.\n"
        "Method C (expert_heuristic_scaling): Use comparable same-country/region peer cities.\n"
        "  Minimum 3 same-country peers; if fewer, widen to same-GDP-tier (+/-30%). "
        "  Confidence: LOW, range: +/-40-50%.\n"
        "If none apply → mark as non-estimable (skip, do not estimate).\n\n"
        "RULES:\n"
        "- Never average across different asset types (separate reference pools).\n"
        "- Zero-data cities get a single summary record tagged VERY_LOW.\n"
        "- Every estimate MUST be a range (low/mid/high), never a point estimate.\n"
        "- is_replaceable is always true.\n"
        "- Include reference_data (what peer data was used), rationale, and basis.\n\n"
    )
    if pass_name == "critique":
        base += (
            "CRITIQUE PASS:\n"
            "Review the prior estimates provided. For each:\n"
            "1. Check the method is correctly applied per the priority ladder.\n"
            "2. Verify the range is reasonable (not too narrow or wide).\n"
            "3. Ensure reference_data is specific (city names, values) not generic.\n"
            "4. If an estimate should be revised, include the revised version.\n"
            "5. If an estimate is correct, include it unchanged.\n"
            "Return the complete revised list.\n\n"
        )

    base += (
        "OUTPUT JSON SCHEMA:\n"
        "```json\n"
        "{\n"
        '  "assumptions": [\n'
        "    {\n"
        '      "city": "...",\n'
        '      "field_name": "...",\n'
        '      "gap_description": "...",\n'
        '      "method_used": "national_regional_average|peer_city_proxy|expert_heuristic_scaling",\n'
        '      "estimate": {"low": ..., "mid": ..., "high": ...},\n'
        '      "confidence": "HIGH|MEDIUM|LOW|VERY_LOW",\n'
        '      "reference_data": "...",\n'
        '      "rationale": "...",\n'
        '      "basis": "...",\n'
        '      "is_replaceable": true\n'
        "    }\n"
        "  ],\n"
        '  "non_estimable": [\n'
        "    {\n"
        '      "city": "...",\n'
        '      "field_name": "...",\n'
        '      "gap_description": "...",\n'
        '      "status": "NON_ESTIMABLE",\n'
        '      "explanation": "...",\n'
        '      "recommendation": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
    )
    return base


def _build_user_prompt(
    question: str,
    context_bundle: dict[str, Any],
    gap_manifest: GapManifest,
    estimable_fields: list[EnrichedField],
    pass_name: str,
    prior_estimates: list[AssumptionRecord] | None = None,
) -> str:
    context_json = json.dumps(context_bundle, ensure_ascii=True, indent=2, default=str)
    gap_json = json.dumps(gap_manifest.model_dump(mode="json"), indent=2, default=str)
    fields_json = json.dumps(
        [f.model_dump(mode="json") for f in estimable_fields],
        indent=2,
        default=str,
    )

    parts = [
        f"User question:\n{question.strip()}\n",
        f"Gap manifest:\n```json\n{gap_json}\n```\n",
        f"Fields needing estimation ({len(estimable_fields)}):\n```json\n{fields_json}\n```\n",
        f"Context bundle:\n```json\n{context_json}\n```\n",
    ]

    if pass_name == "critique" and prior_estimates:
        prior_json = json.dumps(
            [e.model_dump(mode="json") for e in prior_estimates],
            indent=2,
            default=str,
        )
        parts.append(
            f"Prior estimates to critique and revise:\n```json\n{prior_json}\n```\n"
        )
        parts.append(
            "Review each estimate, correct any issues, and return the complete revised list.\n"
        )
    else:
        parts.append(
            "Generate estimates for each field using the priority ladder. "
            "Return only the JSON object.\n"
        )

    return "\n".join(parts)


def _check_saturation(assumptions: list[AssumptionRecord]) -> str | None:
    """Return a warning if >60% of estimates use Method C (expert heuristic)."""
    if not assumptions:
        return None
    method_c_count = sum(
        1 for a in assumptions if a.method_used == "expert_heuristic_scaling"
    )
    ratio = method_c_count / len(assumptions)
    if ratio > _METHOD_C_SATURATION_THRESHOLD:
        return (
            f"Methodological caveat: {method_c_count}/{len(assumptions)} "
            f"({ratio:.0%}) of estimates rely on expert heuristic scaling "
            "(Method C), which carries wider uncertainty ranges (+/-40-50%). "
            "Results should be treated as indicative order-of-magnitude figures."
        )
    return None


__all__ = ["run_assumptions_estimator"]
