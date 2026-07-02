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
from backend.modules.web_researcher.assumptions_context import build_assumptions_payload
from backend.services.run_logger import RunLogger
from backend.utils.artifact_writer import stage_file_dir_name

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
        if _should_keep_current_field(current, resolution):
            continue
        field = _field_from_external_resolution(resolution, current)
        if key in by_key:
            merged[by_key[key]] = field
        else:
            by_key[key] = len(merged)
            merged.append(field)
    return merged


def _should_keep_current_field(
    current: EnrichedField | None,
    resolution: ExternalEvidenceResolution,
) -> bool:
    """Return True when an external resolution should not replace current evidence."""
    if current is None:
        return False

    if resolution.action == "unresolved":
        return current.status != "still_missing"

    if resolution.action != "confirm":
        return False

    if current.source == "web" or current.freshness_flag == "superseded":
        return True

    return resolution.ccc_value is None and not (
        current.source == "ccc" and current.value is not None
    )


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
    enriched = merge_enrichment_evidence_into_context(
        context_bundle=context_bundle,
        gap_manifest=gap_manifest,
        web_findings=web_findings,
        freshness_results=freshness_results,
        config_model=config_model,
        elapsed_seconds=elapsed_seconds,
        external_evidence=external_evidence,
        external_resolutions=external_resolutions,
        external_no_evidence=external_no_evidence,
    )
    enriched["assumptions"] = build_assumptions_payload(
        assumptions_model=assumptions_model,
        assumptions=assumptions,
        non_estimable=non_estimable,
        saturation_warning=saturation_warning,
        elapsed_seconds=elapsed_seconds,
    )
    return enriched


def merge_enrichment_evidence_into_context(
    context_bundle: dict[str, Any],
    gap_manifest: GapManifest,
    web_findings: list[WebFinding],
    freshness_results: list[FreshnessResult],
    config_model: str,
    elapsed_seconds: float,
    external_evidence: list[ExternalEvidenceClaim] | None = None,
    external_resolutions: list[ExternalEvidenceResolution] | None = None,
    external_no_evidence: list[NoEvidenceRecord] | None = None,
) -> dict[str, Any]:
    """Create a new context bundle with evidence-only enrichment data merged in."""
    enriched = deepcopy(context_bundle)

    total_gaps = sum(
        len(gap.blank_fields) + len(gap.stale_flags) + len(gap.bundled_fields)
        for gap in gap_manifest.city_gaps
    )
    meta = EnrichmentMeta(
        created_at=datetime.now(timezone.utc),
        gap_analyst_model=config_model,
        total_gaps=total_gaps,
        estimable_count=max(0, total_gaps - len(gap_manifest.non_estimable_fields)),
        non_estimable_count=0,
        classified_non_estimable_field_count=len(gap_manifest.non_estimable_fields),
        non_estimable_output_count=0,
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
        assumptions=[],
        non_estimable=[],
        saturation_warning=None,
        meta=meta,
    )

    enriched["enrichment"] = _serialize_enrichment_bundle(bundle)
    return enriched


def _serialize_enrichment_bundle(bundle: EnrichmentBundle) -> dict[str, Any]:
    """Return persisted enrichment payload with field metadata outside gaps."""
    payload = bundle.model_dump(mode="json")
    gap_payload = payload.pop("gap_manifest")
    payload.pop("assumptions", None)
    payload.pop("non_estimable", None)
    payload.pop("saturation_warning", None)
    field_manifest = {
        "query_fields": gap_payload.pop("query_fields", []),
        "non_estimable_fields": gap_payload.pop("non_estimable_fields", []),
    }
    return {
        "field_manifest": field_manifest,
        "gap_manifest": gap_payload,
        **payload,
    }


def _path_for_stage_file(stage_name: str, filename: str) -> str:
    """Return the canonical run-local path label for one stage file."""
    return f"stage_files/{stage_file_dir_name(stage_name)}/{filename}"


def serialize_enrichment_artifacts(
    enriched_context: dict[str, Any],
    base_dir: Path,
    run_logger: RunLogger,
    *,
    stage_flags: dict[str, Any] | None = None,
    substage_artifacts: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write enrichment artifacts to disk and register them in the run log."""
    enrichment_data = enriched_context.get("enrichment")
    if not isinstance(enrichment_data, dict):
        return

    run_logger.write_stage_file(
        "enrichment",
        "enrichment_bundle.json",
        enrichment_data,
        alias="enrichment_bundle",
    )

    substage_payloads = dict(substage_artifacts or {})
    outputs: dict[str, Any] = {
        "enrichment_bundle": _path_for_stage_file("enrichment", "enrichment_bundle.json"),
    }

    web_research_stage = substage_payloads.get("web_research")
    if isinstance(web_research_stage, dict):
        web_research_audit = _build_web_research_audit(web_research_stage)
        if web_research_audit:
            audit_path = run_logger.write_stage_file(
                "enrichment",
                "web_research_audit.json",
                web_research_audit,
                alias="enrichment_web_research_audit",
            )
            outputs["web_research_audit"] = run_logger.artifact_label(audit_path)

    field_manifest_payload = (
        enrichment_data.get("field_manifest")
        if isinstance(enrichment_data.get("field_manifest"), dict)
        else {}
    )
    gap_manifest_payload = (
        enrichment_data.get("gap_manifest")
        if isinstance(enrichment_data.get("gap_manifest"), dict)
        else {}
    )
    gap_metrics = _build_gap_metrics(field_manifest_payload, gap_manifest_payload)
    external_stage = substage_payloads.get("external_sources", {})
    external_stage_metrics = (
        external_stage.get("metrics") if isinstance(external_stage, dict) else {}
    )
    if not isinstance(external_stage_metrics, dict):
        external_stage_metrics = {}
    external_stage_outputs = (
        external_stage.get("outputs") if isinstance(external_stage, dict) else {}
    )
    if not isinstance(external_stage_outputs, dict):
        external_stage_outputs = {}
    web_research_stage_metrics = (
        web_research_stage.get("metrics")
        if isinstance(web_research_stage, dict)
        else {}
    )
    if not isinstance(web_research_stage_metrics, dict):
        web_research_stage_metrics = {}
    search_audit_artifact = external_stage_outputs.get("search_audit_artifact")
    if search_audit_artifact:
        outputs["external_source_search_audit"] = search_audit_artifact

    flags = dict(stage_flags or {})
    substage_summaries = _build_substage_summaries(substage_payloads)
    if substage_summaries:
        outputs["substages"] = substage_summaries

    run_logger.write_stage_detail(
        "enrichment",
        {
            "inputs": {
                "has_markdown_context": isinstance(
                    enriched_context.get("markdown"), dict
                ),
                **flags,
            },
            "outputs": outputs,
            "metrics": {
                **gap_metrics,
                "web_finding_count": len(enrichment_data.get("web_findings") or []),
                "external_evidence_count": len(
                    enrichment_data.get("external_evidence") or []
                ),
                "external_resolution_count": len(
                    enrichment_data.get("external_resolutions") or []
                ),
                "external_no_evidence_count": len(
                    enrichment_data.get("external_no_evidence") or []
                ),
                "unresolved_external_source_field_count": (
                    external_stage_metrics.get("unresolved_searched_city_field_count")
                ),
                "max_turn_exceeded_count": external_stage_metrics.get(
                    "max_turn_exceeded_count"
                ),
                "fallback_finalization_count": external_stage_metrics.get(
                    "fallback_finalization_count"
                ),
                "external_source_token_count": run_logger.llm_token_count_for_agents(
                    {"External Source Researcher", "External Source Finalizer"}
                ),
                "freshness_result_count": len(
                    enrichment_data.get("freshness_results") or []
                ),
                "scrape_attempt_count": web_research_stage_metrics.get(
                    "scrape_attempt_count"
                ),
                "scrape_success_count": web_research_stage_metrics.get(
                    "scrape_success_count"
                ),
                "scrape_failure_count": web_research_stage_metrics.get(
                    "scrape_failure_count"
                ),
                "scrape_warning_count": web_research_stage_metrics.get(
                    "scrape_warning_count"
                ),
                "actual_serper_query_count": web_research_stage_metrics.get(
                    "actual_serper_query_count"
                ),
                "actual_serper_call_count": web_research_stage_metrics.get(
                    "actual_serper_call_count"
                ),
                "serper_call_count": web_research_stage_metrics.get(
                    "serper_call_count"
                ),
                "successful_serper_query_count": web_research_stage_metrics.get(
                    "successful_serper_query_count"
                ),
                "successful_serper_call_count": web_research_stage_metrics.get(
                    "successful_serper_call_count"
                ),
                "tier1_site_query_count": web_research_stage_metrics.get(
                    "tier1_site_query_count"
                ),
                "tier1_site_call_count": web_research_stage_metrics.get(
                    "tier1_site_call_count"
                ),
                "open_query_count": web_research_stage_metrics.get(
                    "open_query_count"
                ),
                "open_call_count": web_research_stage_metrics.get(
                    "open_call_count"
                ),
                "open_query_skipped_count": web_research_stage_metrics.get(
                    "open_query_skipped_count"
                ),
                "skipped_open_call_count": web_research_stage_metrics.get(
                    "skipped_open_call_count"
                ),
                "estimated_max_serper_query_count": web_research_stage_metrics.get(
                    "estimated_max_serper_query_count"
                ),
                "estimated_max_serper_call_count": web_research_stage_metrics.get(
                    "estimated_max_serper_call_count"
                ),
                "web_research_executed": bool(flags.get("web_research_executed")),
                "external_source_search_executed": bool(
                    flags.get("external_source_search_executed")
                ),
            },
        },
    )

    logger.info(
        "Enrichment artifacts written to stage_files/%s",
        stage_file_dir_name("enrichment"),
    )


def _build_gap_metrics(
    field_manifest: dict[str, Any],
    gap_manifest: dict[str, Any],
) -> dict[str, int]:
    """Build compact gap-analysis metrics from the canonical enrichment bundle."""
    query_fields = field_manifest.get("query_fields")
    non_estimable_fields = field_manifest.get("non_estimable_fields")
    city_gaps = gap_manifest.get("city_gaps")
    city_gap_entries = (
        [item for item in city_gaps if isinstance(item, dict)]
        if isinstance(city_gaps, list)
        else []
    )
    blank_field_count = sum(
        len(item.get("blank_fields", []))
        for item in city_gap_entries
        if isinstance(item.get("blank_fields"), list)
    )
    stale_field_count = sum(
        len(item.get("stale_flags", []))
        for item in city_gap_entries
        if isinstance(item.get("stale_flags"), list)
    )
    bundled_field_count = sum(
        len(item.get("bundled_fields", []))
        for item in city_gap_entries
        if isinstance(item.get("bundled_fields"), list)
    )
    return {
        "query_field_count": len(query_fields) if isinstance(query_fields, list) else 0,
        "city_gap_count": len(city_gap_entries),
        "gap_field_count": blank_field_count + stale_field_count + bundled_field_count,
        "blank_field_count": blank_field_count,
        "stale_field_count": stale_field_count,
        "bundled_field_count": bundled_field_count,
        "classified_non_estimable_field_count": (
            len(non_estimable_fields) if isinstance(non_estimable_fields, list) else 0
        ),
    }


def _build_substage_summaries(
    substage_payloads: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return compact substage status/flag/metric summaries for stage detail."""
    summaries: dict[str, dict[str, Any]] = {}
    for name, payload in substage_payloads.items():
        if not isinstance(payload, dict):
            continue
        summary: dict[str, Any] = {}
        for key in ("status", "skip_reason", "flags", "inputs", "metrics"):
            value = payload.get(key)
            if value is not None:
                summary[key] = value
        if summary:
            summaries[name] = summary
    return summaries


def _build_web_research_audit(stage_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Build the web-research trace artifact from non-bundle stage outputs."""
    outputs = stage_payload.get("outputs")
    if not isinstance(outputs, dict):
        return None

    trace_keys = (
        "search_batches",
        "national_findings",
        "comparative_findings",
        "added_city_fields",
        "freshness_touched_city_fields",
        "scrape_failures",
        "scrape_failure_summary",
        "search_execution_summary",
        "failed_batch_groups",
        "serper_billing_summary",
    )
    trace_outputs = {
        key: outputs.get(key)
        for key in trace_keys
        if outputs.get(key) not in (None, [], {})
    }
    if not trace_outputs:
        return None

    return {
        "status": stage_payload.get("status"),
        "skip_reason": stage_payload.get("skip_reason"),
        "flags": stage_payload.get("flags", {}),
        "outputs": trace_outputs,
        "metrics": stage_payload.get("metrics", {}),
    }


__all__ = [
    "compute_field_statuses",
    "merge_enrichment_evidence_into_context",
    "merge_enrichment_into_context",
    "serialize_enrichment_artifacts",
]
