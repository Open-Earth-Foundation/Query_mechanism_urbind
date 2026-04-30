"""Context Merger (Agent 5) + Output Assembler (Agent 7).

Merges enrichment results into the context bundle and serializes artifacts.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.modules.web_researcher.derived_metrics import compute_derived_metrics
from backend.modules.web_researcher.models import (
    AssumptionRecord,
    DerivedMetric,
    EnrichedField,
    EnrichmentBundle,
    EnrichmentMeta,
    FreshnessResult,
    GapManifest,
    NonEstimableRecord,
    WebFinding,
)
from backend.services.run_logger import RunLogger
from backend.utils.json_io import write_json

logger = logging.getLogger(__name__)


def _ensure_resolved_has_value(ef: EnrichedField) -> EnrichedField:
    """Invariant: 'resolved' requires a non-null value."""
    if ef.status == "resolved" and ef.value is None:
        return ef.model_copy(update={"status": "partially_resolved"})
    return ef


def _build_source_name_index() -> dict[str, str]:
    """Map ``source_id`` → human-readable name from the manifest + allowlist.

    Best-effort: returns an empty dict when neither file is present yet so
    runs without any ingestion still work.  The lookup is small and is
    rebuilt per ``compute_field_statuses`` call (not hot enough to cache).
    """
    name_index: dict[str, str] = {}

    try:  # noqa: SIM105 — keep failures isolated to each loader
        from backend.modules.web_researcher.tier1_web import (
            load_tier1_web_allowlist,
        )

        allowlist = load_tier1_web_allowlist()
    except Exception:  # noqa: BLE001
        allowlist = None

    if allowlist is not None:
        for source in allowlist.sources:
            if source.id and source.name:
                name_index[source.id] = source.name

    try:
        from backend.modules.sources.manifest import load_manifest

        manifest = load_manifest()
    except Exception:  # noqa: BLE001
        manifest = None

    if manifest is not None:
        for source in manifest.sources:
            if source.id and source.name and source.id not in name_index:
                name_index[source.id] = source.name

    return name_index


def _attach_source_name(
    provenance: dict[str, object], source_id: str | None, name_index: dict[str, str]
) -> dict[str, object]:
    """Return a new provenance dict with ``source_name`` populated when known."""
    if not source_id:
        return provenance
    name = name_index.get(source_id)
    if not name:
        return provenance
    enriched = dict(provenance)
    enriched["source_name"] = name
    return enriched


def compute_field_statuses(
    gap_manifest: GapManifest,
    web_findings: list[WebFinding],
    freshness_results: list[FreshnessResult],
    context_bundle: dict[str, Any],
) -> list[EnrichedField]:
    """Determine the status of each city x field combination after web research.

    Merger logic:
    - Web superseded → web value primary, status=resolved
    - Web consistent → CCC confirmed, status=resolved
    - Web uncertain → CCC primary, flagged, status=partially_resolved
    - No web + CCC exists → CCC stands, status=resolved
    - Web found + no CCC → web fills blank, status=resolved
    - Has peer reference data → status=partially_resolved
    - Nothing → status=still_missing
    """
    # Index web findings by (city, field)
    web_index: dict[tuple[str, str], WebFinding] = {}
    for wf in web_findings:
        key = (wf.city.lower(), wf.field.lower())
        # Keep highest confidence finding per city+field
        existing = web_index.get(key)
        if existing is None or wf.extraction_confidence > existing.extraction_confidence:
            web_index[key] = wf

    # Index freshness results by (city, field)
    freshness_index: dict[tuple[str, str], FreshnessResult] = {}
    for fr in freshness_results:
        freshness_index[(fr.city.lower(), fr.field.lower())] = fr

    # Source-id → human-readable name lookup, populated from the manifest +
    # tier-1 allowlist when present.  Used to enrich provenance so the
    # writer can attribute by name without re-resolving ids itself.
    source_name_index = _build_source_name_index()

    # Field → scope index from the gap manifest.  Scope is the single source
    # of truth on FieldClassification; we propagate it onto each EnrichedField
    # at the end so the writer can enforce per-scope subtotals without joining.
    scope_by_field: dict[str, str] = {
        fc.field.lower(): fc.scope for fc in gap_manifest.query_fields
    }

    enriched_fields: list[EnrichedField] = []

    for city_gap in gap_manifest.city_gaps:
        city = city_gap.city
        bundled_set = set(city_gap.bundled_fields)
        all_gap_fields = (
            set(city_gap.blank_fields) | set(city_gap.stale_flags) | bundled_set
        )

        for field in all_gap_fields:
            key = (city.lower(), field.lower())
            wf = web_index.get(key)
            fr = freshness_index.get(key)

            if wf and fr:
                # Both web finding and freshness check exist
                if fr.classification == "cancelled":
                    ef = EnrichedField(
                        city=city,
                        field=field,
                        status="still_missing",
                        source="none",
                        freshness_flag="cancelled",
                    )
                elif fr.classification == "superseded":
                    ef = EnrichedField(
                        city=city,
                        field=field,
                        status="resolved",
                        value=wf.value,
                        source="web",
                        source_id=wf.source_id,
                        source_tier=wf.source_tier,
                        provenance=_attach_source_name(
                            {
                                "source_url": wf.source_url,
                                "source_type": wf.source_type,
                                "source_date": wf.source_date,
                                "extraction_confidence": wf.extraction_confidence,
                            },
                            wf.source_id,
                            source_name_index,
                        ),
                        freshness_flag="superseded",
                    )
                elif fr.classification == "consistent":
                    ef = EnrichedField(
                        city=city,
                        field=field,
                        status="resolved",
                        value=fr.ccc_value or wf.value,
                        source="ccc",
                        source_id=wf.source_id,
                        source_tier=wf.source_tier,
                        provenance=_attach_source_name(
                            {"confirmed_by_web": wf.source_url},
                            wf.source_id,
                            source_name_index,
                        ),
                        freshness_flag="consistent",
                    )
                else:  # uncertain
                    ef = EnrichedField(
                        city=city,
                        field=field,
                        status="partially_resolved",
                        value=fr.ccc_value,
                        source="ccc",
                        source_id=wf.source_id,
                        source_tier=wf.source_tier,
                        provenance=_attach_source_name(
                            {
                                "web_alternative": wf.source_url,
                                "web_value": (
                                    str(wf.value) if wf.value is not None else None
                                ),
                            },
                            wf.source_id,
                            source_name_index,
                        ),
                        freshness_flag="uncertain",
                    )
                enriched_fields.append(_ensure_resolved_has_value(ef))
            elif wf and not fr:
                # Web finding exists but no freshness check (no CCC value to compare)
                ef = EnrichedField(
                    city=city,
                    field=field,
                    status="resolved",
                    value=wf.value,
                    source="web",
                    source_id=wf.source_id,
                    source_tier=wf.source_tier,
                    provenance=_attach_source_name(
                        {
                            "source_url": wf.source_url,
                            "source_type": wf.source_type,
                            "extraction_confidence": wf.extraction_confidence,
                        },
                        wf.source_id,
                        source_name_index,
                    ),
                )
                enriched_fields.append(_ensure_resolved_has_value(ef))
            elif field in city_gap.stale_flags and field not in city_gap.blank_fields:
                # Stale but present in CCC, no web update found
                enriched_fields.append(
                    EnrichedField(
                        city=city,
                        field=field,
                        status="partially_resolved",
                        source="ccc",
                        freshness_flag="stale_no_update",
                    )
                )
            elif field in bundled_set:
                # Bundled-only: the CCC has a parent / aggregate value but
                # the requested disaggregated line is missing.  Route to the
                # estimator with bundled provenance so the LLM can apply
                # peer per-unit ratios rather than treating it as resolved.
                enriched_fields.append(
                    EnrichedField(
                        city=city,
                        field=field,
                        status="bundled_only",
                        source="ccc",
                        freshness_flag="bundled_only",
                        provenance={
                            "note": (
                                "Aggregate / bundled value present in CCC; "
                                "disaggregated line not reported. Estimator "
                                "should derive via peer per-unit ratios."
                            )
                        },
                    )
                )
            else:
                # Nothing found
                enriched_fields.append(
                    EnrichedField(
                        city=city,
                        field=field,
                        status="still_missing",
                        source="none",
                    )
                )

    # Propagate scope from gap_manifest to enriched_fields in one place so
    # the per-branch constructors above stay focused on status/value logic.
    return [
        ef.model_copy(
            update={"scope": scope_by_field.get(ef.field.lower(), "unscoped")}
        )
        for ef in enriched_fields
    ]


def merge_enrichment_into_context(
    context_bundle: dict[str, Any],
    gap_manifest: GapManifest,
    web_findings: list[WebFinding],
    freshness_results: list[FreshnessResult],
    assumptions: list[AssumptionRecord],
    non_estimable: list[NonEstimableRecord],
    saturation_warning: str | None,
    config_model: str,
    assumptions_model: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Create a new context bundle with enrichment data merged in.

    Never mutates the original context_bundle.
    """
    enriched = deepcopy(context_bundle)

    # Count gaps
    total_gaps = sum(
        len(cg.blank_fields) + len(cg.stale_flags) for cg in gap_manifest.city_gaps
    )
    estimable_count = total_gaps - len(gap_manifest.non_estimable_fields)
    non_estimable_count = len(gap_manifest.non_estimable_fields)

    meta = EnrichmentMeta(
        created_at=datetime.now(timezone.utc),
        gap_analyst_model=config_model,
        assumptions_estimator_model=assumptions_model,
        total_gaps=total_gaps,
        estimable_count=max(0, estimable_count),
        non_estimable_count=non_estimable_count,
        web_findings_count=len(web_findings),
        elapsed_seconds=elapsed_seconds,
    )

    # Compute enriched field statuses
    enriched_fields = compute_field_statuses(
        gap_manifest, web_findings, freshness_results, context_bundle
    )

    # Derived metrics (per-capita, per-unit) — computed once from the
    # resolved dataset.  Skipped silently on any failure so a bad metric
    # doesn't take down the whole enrichment bundle.
    derived_metrics: list[DerivedMetric] = []
    try:
        derived_metrics = compute_derived_metrics(enriched_fields)
    except Exception:  # noqa: BLE001
        logger.warning("derived_metrics: failed; continuing without", exc_info=True)

    bundle = EnrichmentBundle(
        gap_manifest=gap_manifest,
        enriched_fields=enriched_fields,
        web_findings=web_findings,
        freshness_results=freshness_results,
        assumptions=assumptions,
        non_estimable=non_estimable,
        derived_metrics=derived_metrics,
        saturation_warning=saturation_warning,
        meta=meta,
    )

    enriched["enrichment"] = bundle.model_dump(mode="json")
    return enriched


def serialize_enrichment_artifacts(
    enriched_context: dict[str, Any],
    base_dir: Path,
    run_logger: RunLogger,
) -> None:
    """Write enrichment artifacts to disk and register them in the run log."""
    enrichment_data = enriched_context.get("enrichment")
    if not isinstance(enrichment_data, dict):
        return

    enrichment_dir = base_dir / "enrichment"
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    artifact_map = {
        "gap_manifest": enrichment_data.get("gap_manifest"),
        "assumptions": enrichment_data.get("assumptions"),
        "non_estimable": enrichment_data.get("non_estimable"),
        "enrichment_bundle": enrichment_data,
    }

    # Only write web artifacts if they contain data
    web_findings = enrichment_data.get("web_findings", [])
    if web_findings:
        artifact_map["web_findings"] = web_findings

    freshness_results = enrichment_data.get("freshness_results", [])
    if freshness_results:
        artifact_map["freshness_results"] = freshness_results

    for name, payload in artifact_map.items():
        if payload is None:
            continue
        artifact_path = enrichment_dir / f"{name}.json"
        write_json(artifact_path, payload, default=str)
        run_logger.record_artifact(f"enrichment_{name}", artifact_path)

    logger.info("Enrichment artifacts written to %s", enrichment_dir)


__all__ = [
    "compute_field_statuses",
    "merge_enrichment_into_context",
    "serialize_enrichment_artifacts",
]
