"""Assemble normalized, per-field pipeline artifacts for the audit UI.

The audit view needs one record per (city, field) merging three sources that
the live ``steps[]`` timeline does not expose:

* enrichment ``field_manifest`` -> field type (classification), scope, rationale
* assumptions bundle -> estimate range / non-estimable status + explanation
* external-source audit -> the evidence/quotes that back (or fail to back) a field

It also returns a lightweight per-stage summary (number, name, metrics) read
from Mirco's unified ``stages/*.json`` records so the UI can show how data
flowed through each stage.

Artifacts are resolved through the run ``manifest.json`` aliases first (the same
contract the other API readers rely on). The historical
``stage_files/<stage>/<file>`` glob is only used for legacy runs with no manifest
at all; once a manifest exists, a missing alias should stay visible instead of
being masked by a directory guess. Reads are best-effort but tracked: a missing
artifact (normal when a stage is disabled) is distinguished from one that exists
but could not be parsed, so consumers can tell "no data" apart from "artifact
bundle could not be read."
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from backend.utils.artifact_manifest import resolve_manifest_alias

logger = logging.getLogger(__name__)

# Status -> the two colors the team agreed on, plus an explicit in-between bucket
# so unresolved gaps never masquerade as estimated or non-estimable.
STATUS_ESTIMATED = "estimated"
STATUS_NON_ESTIMABLE = "non_estimable"
STATUS_UNRESOLVED = "unresolved"

# Per-artifact read outcome surfaced via ``artifact_health``.
HEALTH_OK = "ok"
HEALTH_MISSING = "missing"
HEALTH_UNREADABLE = "unreadable"

# External-source quotes are raw markdown chunks (can be many KB). The audit UI
# only needs a readable preview, so cap length and number per field.
_MAX_QUOTE_CHARS = 240
_MAX_SOURCES_PER_FIELD = 4

# Logical artifact key -> (manifest alias, stage glob, filename). The alias is
# the authoritative resolver; the glob is a legacy fallback only when no manifest
# exists for the run.
_ARTIFACT_SPECS: dict[str, tuple[str, str, str]] = {
    "enrichment_bundle": (
        "enrichment_bundle",
        "*enrichment*",
        "enrichment_bundle.json",
    ),
    "external_audit": (
        "enrichment_external_source_search_audit",
        "*enrichment*",
        "external_source_search_audit.json",
    ),
    "assumptions_bundle": (
        "assumptions_assumptions_bundle",
        "*assumptions*",
        "assumptions_bundle.json",
    ),
    "accepted_excerpts": (
        "markdown_excerpts",
        "*markdown_extraction*",
        "accepted_excerpts.json",
    ),
}


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


class _ArtifactStore:
    """Resolve + cache the audit artifacts and record how each read fared.

    Resolution prefers the manifest alias (robust to stage-folder renames or
    additional similarly named stages). If a manifest exists, missing aliases are
    treated as missing artifacts; ``stage_files`` globbing is limited to legacy
    runs that never wrote a manifest. Each logical artifact is read at most once
    and its outcome recorded in :attr:`health` (``ok`` / ``missing`` /
    ``unreadable``).
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._has_manifest = (run_dir / "manifest.json").exists()
        self._cache: dict[str, Any] = {}
        self.health: dict[str, str] = {}

    def _resolve(self, key: str) -> Path | None:
        alias, stage_glob, filename = _ARTIFACT_SPECS[key]
        manifest_path = resolve_manifest_alias(self.run_dir, alias)
        if manifest_path is not None or self._has_manifest:
            return manifest_path
        return _find_stage_file(self.run_dir, stage_glob, filename)

    def get(self, key: str) -> Any | None:
        """Return the parsed artifact (or ``None``), recording read health."""
        if key in self._cache:
            return self._cache[key]
        path = self._resolve(key)
        data: Any | None = None
        if path is None or not path.exists():
            # Absent is expected when a stage was disabled — not a failure.
            self.health[key] = HEALTH_MISSING
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.health[key] = HEALTH_OK
            except Exception:  # noqa: BLE001 - artifacts are advisory, never fatal
                # Present but unparseable: a real degradation worth surfacing.
                logger.warning("run_artifacts: could not parse %s", path, exc_info=True)
                self.health[key] = HEALTH_UNREADABLE
        self._cache[key] = data
        return data


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


def _build_classification_index(store: _ArtifactStore) -> dict[str, dict[str, Any]]:
    """Map field name -> {type, scope, rationale} from the enrichment manifest."""
    data = store.get("enrichment_bundle")
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


def _build_sources_index(store: _ArtifactStore) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Map (city, field) -> list of source/evidence dicts from the external audit."""
    data = store.get("external_audit")
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


# Why a field ended up non-estimable, derived from hard upstream signals only.
# (A weaker "the corpus may hold this in a different shape" signal is surfaced
# separately as a non-authoritative hint, never as one of these reason codes.)
REASON_LABELS = {
    "too_few_peers": "Too few comparable cities",
    "found_not_validated": "Found, but not validated",
    "no_source_data": "No source data found",
}

# Non-authoritative hint shown only on no-source-data fields: the corpus still
# mentions this field's subject near a number, so the data may exist in a form
# the extractor did not capture (a possible extraction gap, not a true absence).
# This rests on a broad stem-overlap heuristic, so it is presented as a soft
# "maybe" — always alongside the raw matched snippet so an auditor can judge it.
SHAPE_HINT_LABEL = "Possible extraction gap"

# How many corpus snippets to keep per topic stem / per field hint, and how long
# each snippet may be before clipping.
_MAX_HINT_EVIDENCE = 3
_MAX_SNIPPET_CHARS = 160

# Structural / generic tokens that don't identify a field's subject matter.
_TOPIC_STOPWORDS = {
    "public", "count", "total", "target", "year", "number", "private",
    "onstreet", "offstreet", "street", "city", "share", "value", "data",
}


def _build_evidence_counts(store: _ArtifactStore) -> dict[tuple[str, str], dict[str, int]]:
    """Per (city, field): how many external candidates were found vs validated."""
    data = store.get("external_audit")
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


def _clip_snippet(text: str, start: int, end: int) -> str | None:
    """Return a readable corpus fragment around a matched word."""
    lo = max(0, start - 80)
    hi = min(len(text), end + 80)
    fragment = " ".join(text[lo:hi].split())
    if not fragment:
        return None
    fragment = f"{'…' if lo > 0 else ''}{fragment}{'…' if hi < len(text) else ''}"
    if len(fragment) > _MAX_SNIPPET_CHARS:
        fragment = fragment[:_MAX_SNIPPET_CHARS].rstrip() + "…"
    return fragment


def _build_numeric_context(store: _ArtifactStore) -> dict[str, list[str]]:
    """Map topic stem -> example corpus snippets where it sits near a number.

    Powers the soft "possible extraction gap" hint *and* its hover evidence —
    e.g. excerpts say "400 new charging stations" so ``charg`` maps to that
    sentence. Deliberately broad, which is why it only ever feeds a
    non-authoritative hint shown with the raw snippet attached.
    """
    data = store.get("accepted_excerpts")
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
    context: dict[str, list[str]] = {}
    for blob in blobs:
        # Lowercasing preserves length, so match indices map back onto ``blob``.
        lowered = blob.lower()
        for match in re.finditer(r"[a-z]{4,}", lowered):
            window = lowered[max(0, match.start() - 40) : match.end() + 40]
            if not any(ch.isdigit() for ch in window):
                continue
            bucket = context.setdefault(match.group()[:5], [])
            if len(bucket) >= _MAX_HINT_EVIDENCE:
                continue
            snippet = _clip_snippet(blob, match.start(), match.end())
            if snippet and snippet not in bucket:
                bucket.append(snippet)
    return context


def _field_topic_stems(field_name: str) -> set[str]:
    """Subject-matter stems for a field, dropping structural words."""
    return {
        token[:5]
        for token in field_name.lower().split("_")
        if len(token) >= 4 and token not in _TOPIC_STOPWORDS
    }


def _classify_reason(counts: dict[str, int]) -> tuple[str, str]:
    """Pick the authoritative reason a field could not be estimated.

    Based on hard evidence-count signals only. The "data exists in a different
    shape" intuition is intentionally excluded here and surfaced as a soft hint.
    """
    if counts.get("validated", 0) > 0:
        code = "too_few_peers"
    elif counts.get("found", 0) > 0:
        code = "found_not_validated"
    else:
        code = "no_source_data"
    return code, REASON_LABELS[code]


def _shape_evidence(
    field_name: str, numeric_context: dict[str, list[str]]
) -> list[str]:
    """Corpus snippets suggesting the field's topic appears near a figure.

    Non-empty only when the field's subject stems overlap the numeric-context
    map — these are the raw fragments shown on hover so an auditor can decide
    whether the match is a real extraction gap or heuristic noise.
    """
    snippets: list[str] = []
    seen: set[str] = set()
    for stem in sorted(_field_topic_stems(field_name)):
        for snippet in numeric_context.get(stem, []):
            if snippet in seen:
                continue
            seen.add(snippet)
            snippets.append(snippet)
            if len(snippets) >= _MAX_HINT_EVIDENCE:
                return snippets
    return snippets


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
    reason_hint: str | None = None,
    reason_hint_evidence: list[str] | None = None,
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
        "reason_hint": reason_hint,
        "reason_hint_evidence": reason_hint_evidence or [],
    }


def _build_fields(store: _ArtifactStore) -> list[dict[str, Any]]:
    """Merge assumptions + non-estimable records into normalized card records."""
    bundle = store.get("assumptions_bundle")
    if not isinstance(bundle, dict):
        return []

    classifications = _build_classification_index(store)
    sources_index = _build_sources_index(store)
    evidence_counts = _build_evidence_counts(store)
    numeric_context = _build_numeric_context(store)
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
            evidence_counts.get((city, field_name), {})
        )
        # The extraction-gap hint is only meaningful (and non-redundant) when we
        # found NO source data at all yet the corpus still mentions the topic
        # near a figure. For found/validated fields it would just restate what we
        # already know, so scope it to no_source_data.
        hint_evidence = (
            _shape_evidence(field_name, numeric_context)
            if reason_code == "no_source_data"
            else []
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
                reason_hint=SHAPE_HINT_LABEL if hint_evidence else None,
                reason_hint_evidence=hint_evidence,
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
        try:
            data = json.loads(stage_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - one bad stage record never blocks the rest
            logger.debug("run_artifacts: could not read %s", stage_path, exc_info=True)
            continue
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
    store: _ArtifactStore,
    reason_breakdown: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Summarize the three enrichment sub-steps so the audit shows where the
    chain breaks (e.g. web findings that never validate into anchors)."""
    enrichment = store.get("enrichment_bundle")
    assumptions = store.get("assumptions_bundle")
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
            # The classic break: lots of hits, none usable as anchors. This is a
            # scoped statement about the external/web step only — it does NOT
            # claim the downstream assumptions step failed, which can still
            # estimate from peer/national data.
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


# Cap the unused-candidate list the API returns; the UI shows a subset + count.
_MAX_UNUSED_CANDIDATES = 40


def _build_gap_analysis(store: _ArtifactStore) -> dict[str, Any]:
    """Per-field classification + rationale and per-city gap priorities.

    Sourced from the enrichment ``field_manifest`` (the full classified set),
    independent of which fields later survived into assumptions — so a field
    dropped before the assumptions stage still shows here.
    """
    enrichment = store.get("enrichment_bundle")
    fields: list[dict[str, Any]] = []
    city_gaps: list[dict[str, Any]] = []
    if not isinstance(enrichment, dict):
        return {"fields": fields, "city_gaps": city_gaps}

    manifest = enrichment.get("field_manifest")
    query_fields = manifest.get("query_fields") if isinstance(manifest, dict) else None
    for entry in query_fields or []:
        if not isinstance(entry, dict) or not entry.get("field"):
            continue
        fields.append(
            {
                "field": entry.get("field"),
                "classification": entry.get("classification"),
                "scope": entry.get("scope"),
                "rationale": entry.get("rationale"),
            }
        )

    gap_manifest = enrichment.get("gap_manifest")
    raw_city_gaps = gap_manifest.get("city_gaps") if isinstance(gap_manifest, dict) else None
    for entry in raw_city_gaps or []:
        if not isinstance(entry, dict) or not entry.get("city"):
            continue
        city_gaps.append(
            {"city": entry.get("city"), "priority": entry.get("search_priority")}
        )
    return {"fields": fields, "city_gaps": city_gaps}


def _build_external_search(store: _ArtifactStore) -> dict[str, Any]:
    """Validated anchors, found-but-unused candidates, and no-evidence records."""
    data = store.get("external_audit")
    if not isinstance(data, dict):
        return {"validated": [], "unused": [], "unused_total": 0, "no_evidence": []}

    validated = [
        {
            "city": r.get("city"),
            "field": r.get("field"),
            "value": r.get("value"),
            "unit": r.get("unit"),
            "source_id": r.get("source_id"),
            "source_type": r.get("source_type"),
            "publication_year": r.get("publication_year"),
            "quote": _clip_quote(r.get("quote")),
        }
        for r in (data.get("validated_claims") or [])
        if isinstance(r, dict) and r.get("city") and r.get("field")
    ]

    unused_source = data.get("unused_candidates") or data.get("candidates") or []
    unused = [
        {
            "city": r.get("city"),
            "field": r.get("field"),
            "source_id": r.get("source_id"),
            "title": r.get("title"),
            "matched_text": r.get("matched_text"),
            "quote": _clip_quote(r.get("quote")),
        }
        for r in unused_source[:_MAX_UNUSED_CANDIDATES]
        if isinstance(r, dict) and r.get("city") and r.get("field")
    ]

    no_evidence = [
        {"city": r.get("city"), "field": r.get("field")}
        for r in (data.get("no_evidence") or [])
        if isinstance(r, dict) and r.get("city") and r.get("field")
    ]

    return {
        "validated": validated,
        "unused": unused,
        "unused_total": len(unused_source),
        "no_evidence": no_evidence,
    }


def build_run_artifacts(run_dir: Path) -> dict[str, Any]:
    """Assemble the normalized artifact payload for one run directory."""
    store = _ArtifactStore(run_dir)
    fields = _build_fields(store)
    reason_breakdown: dict[str, int] = {}
    for field in fields:
        code = field.get("reason")
        if code:
            reason_breakdown[code] = reason_breakdown.get(code, 0) + 1
    # Touch the remaining artifacts so their health is recorded too.
    gap_analysis = _build_gap_analysis(store)
    external_search = _build_external_search(store)
    enrichment_steps = _build_enrichment_steps(store, reason_breakdown)
    artifact_health = {key: store.health.get(key, HEALTH_MISSING) for key in _ARTIFACT_SPECS}
    return {
        "fields": fields,
        "stages": _build_stages(run_dir),
        "enrichment_steps": enrichment_steps,
        "gap_analysis": gap_analysis,
        "external_search": external_search,
        "artifact_health": artifact_health,
        # ``degraded`` means an artifact existed but could not be parsed — a real
        # read regression, distinct from a stage simply being absent/disabled.
        "degraded": any(v == HEALTH_UNREADABLE for v in artifact_health.values()),
    }


__all__ = ["build_run_artifacts", "STATUS_ESTIMATED", "STATUS_NON_ESTIMABLE", "STATUS_UNRESOLVED"]
