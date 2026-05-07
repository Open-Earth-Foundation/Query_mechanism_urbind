"""Context Merger (Agent 5) and Output Assembler (Agent 7)."""

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


def _ensure_resolved_has_value(enriched: EnrichedField) -> EnrichedField:
    """Downgrade resolved fields that have no value."""
    if enriched.status == "resolved" and enriched.value is None:
        return enriched.model_copy(update={"status": "partially_resolved"})
    return enriched


def _build_source_name_index() -> dict[str, str]:
    """Map tier-1 ``source_id`` values to human-readable names."""
    try:
        from backend.modules.web_researcher.tier1_web import (
            load_tier1_web_allowlist,
        )

        allowlist = load_tier1_web_allowlist()
    except Exception:  # noqa: BLE001
        return {}

    return {
        source.id: source.name
        for source in allowlist.sources
        if source.id and source.name
    }


def _attach_source_name(
    provenance: dict[str, object],
    source_id: str | None,
    name_index: dict[str, str],
) -> dict[str, object]:
    """Return provenance with ``source_name`` when the tier-1 id is known."""
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
    external_resolutions: list[ExternalEvidenceResolution] | None = None,
) -> list[EnrichedField]:
    """Determine city-field statuses after web research."""
    web_index: dict[tuple[str, str], WebFinding] = {}
    for finding in web_findings:
        key = (finding.city.lower(), finding.field.lower())
        existing = web_index.get(key)
        if existing is None or finding.extraction_confidence > existing.extraction_confidence:
            web_index[key] = finding

    freshness_index: dict[tuple[str, str], FreshnessResult] = {}
    for result in freshness_results:
        freshness_index[(result.city.lower(), result.field.lower())] = result

    source_name_index = _build_source_name_index()
    scope_by_field = {
        field.field.lower(): field.scope
        for field in getattr(gap_manifest, "query_fields", [])
    }
    enriched_fields: list[EnrichedField] = []

    for city_gap in gap_manifest.city_gaps:
        city = city_gap.city
        bundled_set = set(city_gap.bundled_fields)
        all_gap_fields = set(city_gap.blank_fields) | set(city_gap.stale_flags) | bundled_set

        for field in all_gap_fields:
            key = (city.lower(), field.lower())
            finding = web_index.get(key)
            freshness = freshness_index.get(key)

            if finding and freshness:
                enriched_fields.append(
                    _ensure_resolved_has_value(
                        _field_from_freshness(
                            city,
                            field,
                            finding,
                            freshness,
                            source_name_index,
                        )
                    )
                )
            elif finding:
                enriched_fields.append(
                    _ensure_resolved_has_value(
                        EnrichedField(
                            city=city,
                            field=field,
                            status="resolved",
                            value=finding.value,
                            source="web",
                            source_id=finding.source_id,
                            source_tier=finding.source_tier,
                            provenance=_attach_source_name(
                                {
                                    "source_url": finding.source_url,
                                    "source_type": finding.source_type,
                                    "extraction_confidence": finding.extraction_confidence,
                                },
                                finding.source_id,
                                source_name_index,
                            ),
                        )
                    )
                )
            elif field in city_gap.stale_flags and field not in city_gap.blank_fields:
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
                enriched_fields.append(
                    EnrichedField(
                        city=city,
                        field=field,
                        status="bundled_only",
                        source="ccc",
                        freshness_flag="bundled_only",
                        provenance={
                            "note": (
                                "Aggregate value present in CCC; requested "
                                "disaggregated line not reported."
                            )
                        },
                    )
                )
            else:
                enriched_fields.append(
                    EnrichedField(
                        city=city,
                        field=field,
                        status="still_missing",
                        source="none",
                    )
                )

    scoped_fields = [
        enriched.model_copy(
            update={"scope": scope_by_field.get(enriched.field.lower(), "unscoped")}
        )
        for enriched in enriched_fields
    ]
    return _apply_external_resolutions(scoped_fields, external_resolutions or [])


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
        scope = current.scope if current is not None else "unscoped"
        return EnrichedField(
            city=resolution.city,
            field=resolution.field,
            status="still_missing",
            source="none",
            provenance={"external_resolution": resolution.rationale},
            scope=scope,
        )

    scope = current.scope if current is not None else "unscoped"
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
            scope=scope,
        )

    source = "ccc" if resolution.action == "confirm" else "external_markdown"
    if resolution.action == "confirm":
        value = resolution.ccc_value
        if value is None and current is not None:
            value = current.value
        if value is None:
            value = resolution.external_value
    else:
        value = resolution.external_value
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
        scope=scope,
    )


def _field_from_freshness(
    city: str,
    field: str,
    finding: WebFinding,
    freshness: FreshnessResult,
    source_name_index: dict[str, str],
) -> EnrichedField:
    """Build an enriched field when both web finding and freshness result exist."""
    if freshness.classification == "cancelled":
        return EnrichedField(
            city=city,
            field=field,
            status="still_missing",
            source="none",
            freshness_flag="cancelled",
        )

    if freshness.classification == "superseded":
        return EnrichedField(
            city=city,
            field=field,
            status="resolved",
            value=finding.value,
            source="web",
            source_id=finding.source_id,
            source_tier=finding.source_tier,
            provenance=_attach_source_name(
                {
                    "source_url": finding.source_url,
                    "source_type": finding.source_type,
                    "source_date": finding.source_date,
                    "extraction_confidence": finding.extraction_confidence,
                },
                finding.source_id,
                source_name_index,
            ),
            freshness_flag="superseded",
        )

    if freshness.classification == "consistent":
        return EnrichedField(
            city=city,
            field=field,
            status="resolved",
            value=freshness.ccc_value or finding.value,
            source="ccc",
            source_id=finding.source_id,
            source_tier=finding.source_tier,
            provenance=_attach_source_name(
                {"confirmed_by_web": finding.source_url},
                finding.source_id,
                source_name_index,
            ),
            freshness_flag="consistent",
        )

    return EnrichedField(
        city=city,
        field=field,
        status="partially_resolved",
        value=freshness.ccc_value,
        source="ccc",
        source_id=finding.source_id,
        source_tier=finding.source_tier,
        provenance=_attach_source_name(
            {
                "web_alternative": finding.source_url,
                "web_value": str(finding.value) if finding.value is not None else None,
            },
            finding.source_id,
            source_name_index,
        ),
        freshness_flag="uncertain",
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
    """Create a new context bundle with enrichment data merged in."""
    enriched = deepcopy(context_bundle)

    total_gaps = sum(
        len(gap.blank_fields) + len(gap.stale_flags) + len(gap.bundled_fields)
        for gap in gap_manifest.city_gaps
    )
    meta = EnrichmentMeta(
        created_at=datetime.now(timezone.utc),
        gap_analyst_model=config_model,
        assumptions_estimator_model=assumptions_model,
        total_gaps=total_gaps,
        estimable_count=max(0, total_gaps - len(gap_manifest.non_estimable_fields)),
        non_estimable_count=len(gap_manifest.non_estimable_fields),
        web_findings_count=len(web_findings),
        external_evidence_count=len(external_evidence or []),
        elapsed_seconds=elapsed_seconds,
    )

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
