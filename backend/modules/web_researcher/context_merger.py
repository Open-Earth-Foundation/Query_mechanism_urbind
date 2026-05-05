"""Context Merger (Agent 5) + Output Assembler (Agent 7).

Merges enrichment results into the context bundle and serializes artifacts.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.modules.web_researcher.models import (
    AssumptionRecord,
    EnrichedField,
    EnrichmentBundle,
    EnrichmentMeta,
    ExternalEvidenceClaim,
    ExternalEvidenceResolution,
    FreshnessResult,
    GapManifest,
    NoEvidenceRecord,
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


def compute_field_statuses(
    gap_manifest: GapManifest,
    web_findings: list[WebFinding],
    freshness_results: list[FreshnessResult],
    context_bundle: dict[str, Any],
    external_resolutions: list[ExternalEvidenceResolution] | None = None,
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

    enriched_fields: list[EnrichedField] = []

    for city_gap in gap_manifest.city_gaps:
        city = city_gap.city
        all_gap_fields = set(city_gap.blank_fields) | set(city_gap.stale_flags)

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
                        provenance={
                            "source_url": wf.source_url,
                            "source_type": wf.source_type,
                            "source_date": wf.source_date,
                            "extraction_confidence": wf.extraction_confidence,
                        },
                        freshness_flag="superseded",
                    )
                elif fr.classification == "consistent":
                    ef = EnrichedField(
                        city=city,
                        field=field,
                        status="resolved",
                        value=fr.ccc_value or wf.value,
                        source="ccc",
                        provenance={"confirmed_by_web": wf.source_url},
                        freshness_flag="consistent",
                    )
                else:  # uncertain
                    ef = EnrichedField(
                        city=city,
                        field=field,
                        status="partially_resolved",
                        value=fr.ccc_value,
                        source="ccc",
                        provenance={
                            "web_alternative": wf.source_url,
                            "web_value": str(wf.value) if wf.value is not None else None,
                        },
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
                    provenance={
                        "source_url": wf.source_url,
                        "source_type": wf.source_type,
                        "extraction_confidence": wf.extraction_confidence,
                    },
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

    return _apply_external_resolutions(enriched_fields, external_resolutions or [])


def _apply_external_resolutions(
    enriched_fields: list[EnrichedField],
    external_resolutions: list[ExternalEvidenceResolution],
) -> list[EnrichedField]:
    """Overlay external Markdown resolver decisions on enriched field statuses."""
    if not external_resolutions:
        return enriched_fields

    by_key = {
        (field.city.lower(), field.field.lower()): index
        for index, field in enumerate(enriched_fields)
    }
    merged = list(enriched_fields)
    for resolution in external_resolutions:
        key = (resolution.city.lower(), resolution.field.lower())
        current = merged[by_key[key]] if key in by_key else None
        if resolution.action == "unresolved" and current is not None and current.status == "resolved":
            continue
        field = _field_from_external_resolution(resolution, current)
        if key in by_key:
            merged[by_key[key]] = field
        else:
            by_key[key] = len(merged)
            merged.append(field)
    return merged


def _field_from_external_resolution(
    resolution: ExternalEvidenceResolution,
    current: EnrichedField | None,
) -> EnrichedField:
    """Build an enriched field from one external resolver decision."""
    if resolution.action == "unresolved":
        return EnrichedField(
            city=resolution.city,
            field=resolution.field,
            status="still_missing",
            source="none",
            provenance={"external_resolution": resolution.rationale},
        )

    provenance = {
        "external_resolution_action": resolution.action,
        "source_id": resolution.source_id,
        "line_start": resolution.line_start,
        "line_end": resolution.line_end,
        "quote": resolution.quote,
        "confidence": resolution.confidence,
        "rationale": resolution.rationale,
    }
    if resolution.action == "conflict_review_required":
        return EnrichedField(
            city=resolution.city,
            field=resolution.field,
            status="partially_resolved",
            value=resolution.external_value,
            source="external_markdown",
            provenance={**provenance, "ccc_value": resolution.ccc_value},
            freshness_flag="conflict_review_required",
        )

    source = "ccc" if resolution.action == "confirm" else "external_markdown"
    value = resolution.ccc_value if resolution.action == "confirm" else resolution.external_value
    if value is None and current is not None:
        value = current.value
    return EnrichedField(
        city=resolution.city,
        field=resolution.field,
        status="resolved",
        value=value,
        source=source,
        provenance=provenance,
        freshness_flag=resolution.action,
    )


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
    external_evidence: list[ExternalEvidenceClaim] | None = None,
    external_resolutions: list[ExternalEvidenceResolution] | None = None,
    external_no_evidence: list[NoEvidenceRecord] | None = None,
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
        external_evidence_count=len(external_evidence or []),
        elapsed_seconds=elapsed_seconds,
    )

    # Compute enriched field statuses
    enriched_fields = compute_field_statuses(
        gap_manifest,
        web_findings,
        freshness_results,
        context_bundle,
        external_resolutions=external_resolutions,
    )

    bundle = EnrichmentBundle(
        gap_manifest=gap_manifest,
        enriched_fields=enriched_fields,
        web_findings=web_findings,
        external_evidence=external_evidence or [],
        external_resolutions=external_resolutions or [],
        external_no_evidence=external_no_evidence or [],
        freshness_results=freshness_results,
        assumptions=assumptions,
        non_estimable=non_estimable,
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

    external_evidence = enrichment_data.get("external_evidence", [])
    if external_evidence:
        artifact_map["external_evidence"] = external_evidence

    external_resolutions = enrichment_data.get("external_resolutions", [])
    if external_resolutions:
        artifact_map["external_resolutions"] = external_resolutions

    external_no_evidence = enrichment_data.get("external_no_evidence", [])
    if external_no_evidence:
        artifact_map["external_no_evidence"] = external_no_evidence

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
