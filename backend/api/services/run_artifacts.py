"""Assemble normalized, per-field pipeline artifacts for the audit UI.

The audit view needs one record per (city, field) merging three sources that
the live ``steps[]`` timeline does not expose:

* enrichment ``field_manifest`` -> field type (classification), scope, rationale
* assumptions bundle -> estimate range / non-estimable status + explanation
* external-source audit -> the evidence/quotes that back (or fail to back) a field

It also returns a lightweight per-stage summary (number, name, metrics) read
from Mirco's unified ``stages/*.json`` records so the UI can show how data
flowed through each stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Status -> the two colors the team agreed on, plus an explicit in-between bucket
# so unresolved gaps never masquerade as estimated or non-estimable.
STATUS_ESTIMATED = "estimated"
STATUS_NON_ESTIMABLE = "non_estimable"
STATUS_UNRESOLVED = "unresolved"

# External-source quotes are raw markdown chunks (can be many KB). The audit UI
# only needs a readable preview, so cap length and number per field.
_MAX_QUOTE_CHARS = 240
_MAX_SOURCES_PER_FIELD = 4


def _clip_quote(value: Any) -> str | None:
    """Collapse whitespace and clip an evidence quote to a readable preview."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    if not cleaned:
        return None
    if len(cleaned) <= _MAX_QUOTE_CHARS:
        return cleaned
    return cleaned[:_MAX_QUOTE_CHARS].rstrip() + "…"


def _load_json(path: Path) -> Any | None:
    """Best-effort JSON read; returns ``None`` on any failure."""
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - artifacts are advisory, never fatal
        logger.debug("run_artifacts: could not read %s", path, exc_info=True)
        return None


def _find_stage_file(run_dir: Path, stage_glob: str, filename: str) -> Path | None:
    """Locate ``filename`` inside the first matching ``stage_files/<stage>`` dir."""
    base = run_dir / "stage_files"
    if not base.exists():
        return None
    for stage_dir in sorted(base.glob(stage_glob)):
        candidate = stage_dir / filename
        if candidate.exists():
            return candidate
    return None


def _build_classification_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Map field name -> {type, scope, rationale} from the enrichment manifest."""
    path = _find_stage_file(run_dir, "*enrichment*", "enrichment_bundle.json")
    data = _load_json(path) if path else None
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict):
        return index
    manifest = data.get("field_manifest")
    query_fields = manifest.get("query_fields") if isinstance(manifest, dict) else None
    if not isinstance(query_fields, list):
        return index
    for entry in query_fields:
        if not isinstance(entry, dict):
            continue
        field = entry.get("field")
        if not isinstance(field, str) or not field:
            continue
        index[field] = {
            "type": entry.get("classification"),
            "scope": entry.get("scope"),
            "rationale": entry.get("rationale"),
        }
    return index


def _build_sources_index(run_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Map (city, field) -> list of source/evidence dicts from the external audit."""
    path = _find_stage_file(run_dir, "*enrichment*", "external_source_search_audit.json")
    data = _load_json(path) if path else None
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not isinstance(data, dict):
        return index

    seen: set[tuple[str, str, str]] = set()

    def _add(record: dict[str, Any], *, has_evidence: bool) -> None:
        city = str(record.get("city", "")).strip()
        field = str(record.get("field", "")).strip()
        if not city or not field:
            return
        bucket = index.setdefault((city, field), [])
        if len(bucket) >= _MAX_SOURCES_PER_FIELD:
            return
        source_id = record.get("source_id") or ""
        quote = _clip_quote(record.get("quote") or record.get("matched_text"))
        dedupe_key = (city, field, f"{source_id}:{(quote or '')[:48]}")
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        bucket.append(
            {
                "source_id": record.get("source_id"),
                "title": record.get("title"),
                "quote": quote,
                "has_evidence": has_evidence,
            }
        )

    # Validated claims first (real evidence), then candidates as weaker signal.
    for record in data.get("validated_claims", []) or []:
        if isinstance(record, dict):
            _add(record, has_evidence=True)
    for record in data.get("candidates", []) or []:
        if isinstance(record, dict):
            _add(record, has_evidence=False)
    return index


# Why a field ended up non-estimable, derived from the upstream signals.
REASON_LABELS = {
    "too_few_peers": "Too few comparable cities",
    "found_not_validated": "Found, but not validated",
    "shape_mismatch": "Source has related data, in a different shape",
    "no_source_data": "No source data found",
}

# Structural / generic tokens that don't identify a field's subject matter.
_TOPIC_STOPWORDS = {
    "public", "count", "total", "target", "year", "number", "private",
    "onstreet", "offstreet", "street", "city", "share", "value", "data",
}


def _build_evidence_counts(run_dir: Path) -> dict[tuple[str, str], dict[str, int]]:
    """Per (city, field): how many external candidates were found vs validated."""
    data = _load_json(
        _find_stage_file(run_dir, "*enrichment*", "external_source_search_audit.json")
    )
    counts: dict[tuple[str, str], dict[str, int]] = {}
    if not isinstance(data, dict):
        return counts

    def _bump(record: Any, *, validated: bool) -> None:
        if not isinstance(record, dict):
            return
        city = str(record.get("city", "")).strip()
        field = str(record.get("field", "")).strip()
        if not city or not field:
            return
        bucket = counts.setdefault((city, field), {"found": 0, "validated": 0})
        if validated:
            bucket["validated"] += 1
        else:
            bucket["found"] += 1

    for record in data.get("candidates", []) or []:
        _bump(record, validated=False)
    for record in data.get("validated_claims", []) or []:
        _bump(record, validated=True)
    return counts


def _build_numeric_topics(run_dir: Path) -> set[str]:
    """5-char stems of words that appear near a number in the CCC excerpts.

    Used to flag "the corpus has this topic, just in a different shape" — e.g.
    excerpts say "400 new charging stations" so `charg` is a numeric topic even
    though no `public_ac_charger_count` value exists.
    """
    import re

    data = _load_json(
        _find_stage_file(run_dir, "*markdown_extraction*", "accepted_excerpts.json")
    )
    blobs: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            blobs.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(data)
    text = " ".join(blobs).lower()
    topics: set[str] = set()
    for match in re.finditer(r"[a-z]{4,}", text):
        window = text[max(0, match.start() - 40) : match.end() + 40]
        if any(ch.isdigit() for ch in window):
            topics.add(match.group()[:5])
    return topics


def _field_topic_stems(field_name: str) -> set[str]:
    """Subject-matter stems for a field, dropping structural words."""
    return {
        token[:5]
        for token in field_name.lower().split("_")
        if len(token) >= 4 and token not in _TOPIC_STOPWORDS
    }


def _classify_reason(
    field_name: str,
    counts: dict[str, int],
    numeric_topics: set[str],
) -> tuple[str, str]:
    """Pick the most informative reason a field could not be estimated."""
    if counts.get("validated", 0) > 0:
        code = "too_few_peers"
    elif _field_topic_stems(field_name) & numeric_topics:
        code = "shape_mismatch"
    elif counts.get("found", 0) > 0:
        code = "found_not_validated"
    else:
        code = "no_source_data"
    return code, REASON_LABELS[code]


def _normalized_field(
    *,
    city: str,
    field_name: str,
    status: str,
    explanation: str | None,
    classification: dict[str, Any],
    sources: list[dict[str, Any]],
    gap_description: str | None = None,
    recommendation: str | None = None,
    estimate: dict[str, Any] | None = None,
    reason: str | None = None,
    reason_label: str | None = None,
) -> dict[str, Any]:
    """Shape one card record."""
    return {
        "city": city,
        "field": field_name,
        "type": classification.get("type"),
        "scope": classification.get("scope"),
        "status": status,
        "explanation": explanation or classification.get("rationale"),
        "gap_description": gap_description,
        "recommendation": recommendation,
        "estimate": estimate,
        "sources": sources,
        "reason": reason,
        "reason_label": reason_label,
    }


def _build_fields(run_dir: Path) -> list[dict[str, Any]]:
    """Merge assumptions + non-estimable records into normalized card records."""
    path = _find_stage_file(run_dir, "*assumptions*", "assumptions_bundle.json")
    bundle = _load_json(path) if path else None
    if not isinstance(bundle, dict):
        return []

    classifications = _build_classification_index(run_dir)
    sources_index = _build_sources_index(run_dir)
    evidence_counts = _build_evidence_counts(run_dir)
    numeric_topics = _build_numeric_topics(run_dir)
    fields: list[dict[str, Any]] = []

    for record in bundle.get("assumptions", []) or []:
        if not isinstance(record, dict):
            continue
        city = str(record.get("city", "")).strip()
        field_name = str(record.get("field_name", "")).strip()
        if not city or not field_name:
            continue
        estimate_range = record.get("estimate")
        estimate = None
        if isinstance(estimate_range, dict):
            estimate = {
                "low": estimate_range.get("low"),
                "mid": estimate_range.get("mid"),
                "high": estimate_range.get("high"),
                "method": record.get("method_used"),
                "confidence": record.get("confidence"),
                "basis": record.get("basis"),
            }
        fields.append(
            _normalized_field(
                city=city,
                field_name=field_name,
                status=STATUS_ESTIMATED,
                explanation=record.get("rationale"),
                classification=classifications.get(field_name, {}),
                sources=sources_index.get((city, field_name), []),
                gap_description=record.get("gap_description"),
                estimate=estimate,
            )
        )

    for record in bundle.get("non_estimable", []) or []:
        if not isinstance(record, dict):
            continue
        city = str(record.get("city", "")).strip()
        field_name = str(record.get("field_name", "")).strip()
        if not city or not field_name:
            continue
        reason_code, reason_label = _classify_reason(
            field_name,
            evidence_counts.get((city, field_name), {}),
            numeric_topics,
        )
        fields.append(
            _normalized_field(
                city=city,
                field_name=field_name,
                status=STATUS_NON_ESTIMABLE,
                explanation=record.get("explanation"),
                classification=classifications.get(field_name, {}),
                sources=sources_index.get((city, field_name), []),
                gap_description=record.get("gap_description"),
                recommendation=record.get("recommendation"),
                reason=reason_code,
                reason_label=reason_label,
            )
        )

    fields.sort(key=lambda f: (f["city"].lower(), f["field"]))
    return fields


def _build_stages(run_dir: Path) -> list[dict[str, Any]]:
    """Lightweight per-stage summary from the unified ``stages/*.json`` records."""
    stages_dir = run_dir / "stages"
    if not stages_dir.exists():
        return []
    stages: list[dict[str, Any]] = []
    for stage_path in sorted(stages_dir.glob("*.json")):
        data = _load_json(stage_path)
        if not isinstance(data, dict):
            continue
        decisions = data.get("decisions")
        status_value = "completed"
        if isinstance(decisions, list) and decisions and isinstance(decisions[0], dict):
            status_value = decisions[0].get("status", status_value)
        stages.append(
            {
                "stage_number": data.get("stage_number"),
                "name": data.get("step_name") or stage_path.stem,
                "status": status_value,
                "metrics": data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
                "inputs": data.get("inputs") if isinstance(data.get("inputs"), dict) else {},
            }
        )
    stages.sort(key=lambda s: (s.get("stage_number") is None, s.get("stage_number") or 0))
    return stages


def _as_int(value: Any) -> int:
    """Coerce a metric to a non-negative int, defaulting to 0."""
    return value if isinstance(value, int) and value >= 0 else 0


def _build_enrichment_steps(
    run_dir: Path,
    reason_breakdown: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Summarize the three enrichment sub-steps so the audit shows where the
    chain breaks (e.g. web findings that never validate into anchors)."""
    enrichment = _load_json(
        _find_stage_file(run_dir, "*enrichment*", "enrichment_bundle.json")
    )
    assumptions = _load_json(
        _find_stage_file(run_dir, "*assumptions*", "assumptions_bundle.json")
    )
    meta = enrichment.get("meta", {}) if isinstance(enrichment, dict) else {}
    no_evidence = (
        enrichment.get("external_no_evidence", []) if isinstance(enrichment, dict) else []
    )

    estimable = _as_int(meta.get("estimable_count"))
    non_estimable_classified = _as_int(meta.get("non_estimable_count"))
    web_findings = _as_int(meta.get("web_findings_count"))
    validated = _as_int(meta.get("external_evidence_count"))
    no_evidence_count = len(no_evidence) if isinstance(no_evidence, list) else 0

    estimated = (
        len(assumptions.get("assumptions", []))
        if isinstance(assumptions, dict)
        else 0
    )
    non_estimable = (
        len(assumptions.get("non_estimable", []))
        if isinstance(assumptions, dict)
        else 0
    )

    return [
        {
            "key": "gap_analysis",
            "label": "Gap analysis",
            "summary": f"{estimable + non_estimable_classified} fields classified",
            "metrics": {
                "classified_estimable": estimable,
                "classified_non_estimable": non_estimable_classified,
            },
            "warn": None,
        },
        {
            "key": "external_web",
            "label": "External + web search",
            "summary": f"{web_findings} found · {validated} validated",
            "metrics": {
                "web_findings": web_findings,
                "validated_evidence": validated,
                "no_evidence": no_evidence_count,
            },
            # The classic break: lots of hits, none usable as anchors.
            "warn": (
                "Found evidence but none validated into usable anchors."
                if web_findings > 0 and validated == 0
                else None
            ),
        },
        {
            "key": "assumptions",
            "label": "Assumptions",
            "summary": f"{estimated} estimated · {non_estimable} non-estimable",
            "metrics": {
                "estimated": estimated,
                "non_estimable": non_estimable,
                "reason_breakdown": reason_breakdown or {},
            },
            "warn": (
                "No fields could be estimated."
                if estimated == 0 and non_estimable > 0
                else None
            ),
        },
    ]


def build_run_artifacts(run_dir: Path) -> dict[str, Any]:
    """Assemble the normalized artifact payload for one run directory."""
    fields = _build_fields(run_dir)
    reason_breakdown: dict[str, int] = {}
    for field in fields:
        code = field.get("reason")
        if code:
            reason_breakdown[code] = reason_breakdown.get(code, 0) + 1
    return {
        "fields": fields,
        "stages": _build_stages(run_dir),
        "enrichment_steps": _build_enrichment_steps(run_dir, reason_breakdown),
    }


__all__ = ["build_run_artifacts", "STATUS_ESTIMATED", "STATUS_NON_ESTIMABLE", "STATUS_UNRESOLVED"]
