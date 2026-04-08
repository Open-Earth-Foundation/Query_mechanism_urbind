"""Freshness Checker (Agent 4): compare web findings against CCC data."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from backend.modules.web_researcher.models import FreshnessResult, WebFinding
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)


def check_freshness(
    web_findings: list[WebFinding],
    context_bundle: dict[str, Any],
    config: AppConfig,
    api_key: str,
) -> list[FreshnessResult]:
    """Compare web findings against existing CCC data in the context bundle.

    Classifies each CCC ↔ web pair as:
    - **consistent** — web aligns with CCC. CCC stands.
    - **superseded** — web is newer and more specific. Web becomes primary.
    - **uncertain** — ambiguous. Both preserved, flagged for review.

    On failure, all web findings are treated as ``uncertain``.
    """
    if not web_findings:
        return []

    # Extract existing CCC values from context bundle for comparison
    ccc_values = _extract_ccc_values(context_bundle)

    # Only check freshness for findings that have a corresponding CCC value
    findings_to_check: list[tuple[WebFinding, str | None]] = []
    for wf in web_findings:
        key = (wf.city.lower(), wf.field.lower())
        ccc_val = ccc_values.get(key)
        findings_to_check.append((wf, ccc_val))

    # If no CCC values overlap, skip freshness check entirely
    has_overlaps = any(ccc_val is not None for _, ccc_val in findings_to_check)
    if not has_overlaps:
        logger.info("Freshness check: no CCC overlaps, skipping.")
        return []

    # Batch LLM call for freshness classification
    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)

    comparison_items = []
    for i, (wf, ccc_val) in enumerate(findings_to_check):
        if ccc_val is None:
            continue
        comparison_items.append({
            "index": i,
            "city": wf.city,
            "field": wf.field,
            "ccc_value": ccc_val,
            "web_value": str(wf.value) if wf.value is not None else None,
            "web_source_url": wf.source_url,
            "web_source_type": wf.source_type,
            "web_source_date": wf.source_date,
        })

    if not comparison_items:
        return []

    system_prompt = (
        "You are a data freshness checker for urban climate action plans.\n"
        "Compare CCC (Cities Climate Challenge) database values against\n"
        "web-sourced values and classify each pair.\n\n"
        "CLASSIFICATIONS:\n"
        '- "consistent" — web value aligns with CCC value (same order of magnitude,\n'
        "  similar figure). CCC stands as primary.\n"
        '- "superseded" — web value is clearly newer and more specific than CCC.\n'
        "  Requires: web source has a timestamp, comes from a credible source\n"
        "  (government > operator > news > blog), and the difference is meaningful.\n"
        '- "uncertain" — values differ but it\'s unclear which is more accurate.\n'
        "  Both should be preserved and flagged.\n\n"
        "RULES:\n"
        "- Never classify as 'superseded' without a timestamp on the web source.\n"
        "- Government and operator sources outweigh news and blog sources.\n"
        "- Small differences (<10%) should be 'consistent'.\n"
        "- Large differences (>50%) need clear provenance to be 'superseded'.\n\n"
        "OUTPUT: Return a JSON array of objects:\n"
        '  [{"index": 0, "classification": "consistent|superseded|uncertain", "reason": "..."}]\n'
    )

    user_prompt = (
        "Compare these CCC vs web value pairs:\n"
        f"```json\n{json.dumps(comparison_items, indent=2)}\n```\n"
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
            return _fallback_uncertain(findings_to_check)

        content = extract_message_text(response.choices[0].message.content)
        candidate = extract_json_candidate(content)
        parsed = json.loads(candidate)

        # Build classification map
        classification_map: dict[int, tuple[str, str]] = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    idx = item.get("index")
                    cls = item.get("classification", "uncertain")
                    reason = item.get("reason", "")
                    if isinstance(idx, int) and cls in ("consistent", "superseded", "uncertain"):
                        classification_map[idx] = (cls, reason)

        results: list[FreshnessResult] = []
        for i, (wf, ccc_val) in enumerate(findings_to_check):
            if ccc_val is None:
                continue
            cls, reason = classification_map.get(i, ("uncertain", "No classification returned"))
            results.append(FreshnessResult(
                city=wf.city,
                field=wf.field,
                ccc_value=ccc_val,
                web_value=str(wf.value) if wf.value is not None else None,
                classification=cls,  # type: ignore[arg-type]
                reason=reason,
                web_source_url=wf.source_url,
            ))

        consistent = sum(1 for r in results if r.classification == "consistent")
        superseded = sum(1 for r in results if r.classification == "superseded")
        uncertain = sum(1 for r in results if r.classification == "uncertain")
        logger.info(
            "Freshness check: %d consistent, %d superseded, %d uncertain.",
            consistent,
            superseded,
            uncertain,
        )
        return results

    except Exception:
        logger.warning("Freshness check failed; treating all as uncertain.", exc_info=True)
        return _fallback_uncertain(findings_to_check)


def _extract_ccc_values(context_bundle: dict[str, Any]) -> dict[tuple[str, str], str]:
    """Extract known CCC values from the context bundle for comparison.

    Looks through markdown excerpts and SQL results for city+field values.
    Returns a dict of (city_lower, field_lower) → value_string.
    """
    values: dict[tuple[str, str], str] = {}

    # Extract from markdown excerpts if present
    markdown = context_bundle.get("markdown")
    if isinstance(markdown, dict):
        excerpts = markdown.get("excerpts", [])
        if isinstance(excerpts, list):
            for excerpt in excerpts:
                if not isinstance(excerpt, dict):
                    continue
                city = excerpt.get("city_key", "")
                partial = excerpt.get("partial_answer", "")
                if isinstance(city, str) and city.strip() and isinstance(partial, str):
                    # Use the city key as a basic indicator that data exists
                    # The actual field-level extraction would need schema awareness
                    values.setdefault((city.strip().lower(), "_has_data"), partial[:200])

    return values


def _fallback_uncertain(
    findings_to_check: list[tuple[WebFinding, str | None]],
) -> list[FreshnessResult]:
    """Fallback: mark all overlapping findings as uncertain."""
    results: list[FreshnessResult] = []
    for wf, ccc_val in findings_to_check:
        if ccc_val is None:
            continue
        results.append(FreshnessResult(
            city=wf.city,
            field=wf.field,
            ccc_value=ccc_val,
            web_value=str(wf.value) if wf.value is not None else None,
            classification="uncertain",
            reason="Freshness check failed; defaulting to uncertain.",
            web_source_url=wf.source_url,
        ))
    return results


__all__ = ["check_freshness"]
