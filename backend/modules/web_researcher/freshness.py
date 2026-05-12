"""Freshness Checker (Agent 4): compare web findings against CCC data."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from backend.modules.web_researcher.models import FreshnessResult, WebFinding
from backend.modules.web_researcher.utils.json_helpers import (
    extract_json_candidate,
    extract_message_text,
)
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import AppConfig
from backend.utils.llm_serialization import render_toon_block

logger = logging.getLogger(__name__)

# Per-city excerpts are truncated to this length before being sent to the LLM
# so a chatty markdown bundle doesn't blow up the prompt budget.
_MAX_EXCERPT_CHARS = 800
_MAX_EXCERPTS_PER_CITY = 6


def check_freshness(
    web_findings: list[WebFinding],
    context_bundle: dict[str, Any],
    config: AppConfig,
    api_key: str,
) -> list[FreshnessResult]:
    """Compare web findings against existing CCC data in the context bundle.

    CCC values live in the markdown bundle as prose excerpts. For each web
    finding whose city has at least one excerpt, we pass the excerpts and the
    web value to an LLM that extracts the CCC value (if present) and returns
    one of four classifications:
    - **consistent** — web aligns with the CCC excerpt. CCC stands.
    - **superseded** — web is newer and more specific. Web becomes primary.
    - **uncertain** — ambiguous. Both preserved, flagged for review.
    - **cancelled** — web source indicates the programme was cancelled.

    Findings for cities with no markdown evidence are skipped (no overlap to
    check against). On LLM failure, overlapping findings fall back to
    ``uncertain``.
    """
    if not web_findings:
        return []

    ccc_evidence = _extract_ccc_evidence(context_bundle)

    findings_to_check: list[tuple[WebFinding, list[str]]] = []
    for wf in web_findings:
        excerpts = ccc_evidence.get(normalize_city_key(wf.city), [])
        if excerpts:
            findings_to_check.append((wf, excerpts))

    if not findings_to_check:
        logger.info("Freshness check: no markdown CCC evidence for any web finding city, skipping.")
        return []

    client = OpenAI(api_key=api_key, base_url=config.openrouter_base_url)

    comparison_items = []
    for i, (wf, excerpts) in enumerate(findings_to_check):
        comparison_items.append({
            "index": i,
            "city": wf.city,
            "field": wf.field,
            "markdown_excerpts": excerpts,
            "web_value": str(wf.value) if wf.value is not None else None,
            "web_source_url": wf.source_url,
            "web_source_type": wf.source_type,
            "web_source_date": wf.source_date,
        })

    system_prompt = (
        "You are a data freshness checker for urban climate action plans.\n"
        "For each item you receive markdown excerpts sourced from the CCC\n"
        "(Cities Climate Challenge) knowledge base alongside a web-sourced\n"
        "value. Extract the CCC value relevant to the given field from the\n"
        "excerpts (if any), then classify the web value against that CCC\n"
        "value.\n\n"
        "CLASSIFICATIONS:\n"
        '- "consistent" — web value aligns with the CCC excerpt (same order of\n'
        "  magnitude, similar figure). CCC stands as primary.\n"
        '- "superseded" — web value is clearly newer and more specific than the\n'
        "  CCC excerpt. Requires: web source has a timestamp, comes from a\n"
        "  credible source (government > operator > news > blog), and the\n"
        "  difference is meaningful.\n"
        '- "uncertain" — values differ but it\'s unclear which is more accurate,\n'
        "  OR the excerpts do not contain a comparable value for this field.\n"
        "  Both should be preserved and flagged.\n"
        '- "cancelled" — web source indicates the programme/project was\n'
        "  cancelled, discontinued, or reversed. The CCC value is no longer\n"
        "  valid.\n\n"
        "RULES:\n"
        "- Never classify as 'superseded' without a timestamp on the web source.\n"
        "- Government and operator sources outweigh news and blog sources.\n"
        "- Small differences (<10%) should be 'consistent'.\n"
        "- Large differences (>50%) need clear provenance to be 'superseded'.\n"
        "- If the excerpts mention nothing about the field, set\n"
        "  ccc_value_extracted to null and classification to 'uncertain'.\n\n"
        "OUTPUT: Return a JSON array of objects, one per input item:\n"
        '  [{"index": 0, "classification": "consistent|superseded|uncertain|cancelled",\n'
        '    "ccc_value_extracted": "string or null",\n'
        '    "reason": "..."}]\n'
    )

    user_prompt = (
        "Classify each web finding against the paired CCC markdown excerpts:\n"
        f"{render_toon_block('Comparison items TOON', comparison_items)}\n"
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

        classification_map: dict[int, tuple[str, str, str | None]] = {}
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                cls = item.get("classification", "uncertain")
                reason = item.get("reason", "")
                ccc_extracted = item.get("ccc_value_extracted")
                if isinstance(idx, int) and cls in ("consistent", "superseded", "uncertain", "cancelled"):
                    ccc_val = str(ccc_extracted) if ccc_extracted not in (None, "") else None
                    classification_map[idx] = (cls, reason, ccc_val)

        results: list[FreshnessResult] = []
        for i, (wf, _excerpts) in enumerate(findings_to_check):
            cls, reason, ccc_val = classification_map.get(
                i, ("uncertain", "No classification returned", None)
            )
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
        cancelled = sum(1 for r in results if r.classification == "cancelled")
        logger.info(
            "Freshness check: %d consistent, %d superseded, %d uncertain, %d cancelled.",
            consistent,
            superseded,
            uncertain,
            cancelled,
        )
        return results

    except Exception:
        logger.warning("Freshness check failed; treating all as uncertain.", exc_info=True)
        return _fallback_uncertain(findings_to_check)


def _extract_ccc_evidence(context_bundle: dict[str, Any]) -> dict[str, list[str]]:
    """Collect per-city markdown excerpt snippets usable as CCC evidence.

    Returns ``{city_key_lower: [excerpt_text, ...]}``. Each excerpt is trimmed
    to ``_MAX_EXCERPT_CHARS`` and each city keeps at most
    ``_MAX_EXCERPTS_PER_CITY`` snippets so the downstream LLM prompt stays
    bounded. CCC values in this pipeline live only in markdown prose; there
    is no structured tabular source in the context bundle.
    """
    evidence: dict[str, list[str]] = {}

    markdown = context_bundle.get("markdown")
    if not isinstance(markdown, dict):
        return evidence

    excerpts = markdown.get("excerpts")
    if not isinstance(excerpts, list):
        return evidence

    for excerpt in excerpts:
        if not isinstance(excerpt, dict):
            continue
        raw_city = excerpt.get("city_key") or excerpt.get("city_name") or ""
        if not isinstance(raw_city, str):
            continue
        city_key = normalize_city_key(raw_city)
        if not city_key:
            continue

        text = excerpt.get("partial_answer") or excerpt.get("quote") or ""
        if not isinstance(text, str) or not text.strip():
            continue

        bucket = evidence.setdefault(city_key, [])
        if len(bucket) >= _MAX_EXCERPTS_PER_CITY:
            continue
        bucket.append(text.strip()[:_MAX_EXCERPT_CHARS])

    return evidence


def _fallback_uncertain(
    findings_to_check: list[tuple[WebFinding, list[str]]],
) -> list[FreshnessResult]:
    """Fallback: mark all overlapping findings as uncertain with no CCC value."""
    results: list[FreshnessResult] = []
    for wf, _excerpts in findings_to_check:
        results.append(FreshnessResult(
            city=wf.city,
            field=wf.field,
            ccc_value=None,
            web_value=str(wf.value) if wf.value is not None else None,
            classification="uncertain",
            reason="Freshness check failed; defaulting to uncertain.",
            web_source_url=wf.source_url,
        ))
    return results


__all__ = ["check_freshness"]
