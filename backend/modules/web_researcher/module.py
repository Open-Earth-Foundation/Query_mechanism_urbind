"""Enrichment pipeline orchestrator.

Top-level entry point called from the main pipeline orchestrator to run
gap analysis, web research, and assumptions estimation.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from backend.modules.web_researcher.assumptions_estimator import run_assumptions_estimator
from backend.modules.web_researcher.assumptions_context import (
    build_assumptions_payload,
    city_field_pairs,
    serialize_assumptions_artifacts,
)
from backend.modules.web_researcher.context_merger import (
    compute_field_statuses,
    merge_enrichment_evidence_into_context,
    serialize_enrichment_artifacts,
)
from backend.modules.web_researcher.external_agent import run_external_source_enrichment
from backend.modules.web_researcher.external_sources import EXTERNAL_SOURCE_SEARCH_AUDIT_FILENAME
from backend.modules.web_researcher.freshness import check_freshness
from backend.modules.web_researcher.gap_analysis import (
    decompose_fields,
    detect_city_gaps,
    run_gap_analysis,
)
from backend.modules.web_researcher.search_planner import plan_searches
from backend.modules.web_researcher.search_worker import execute_search_batches
from backend.services.progress_tracker import ProgressTracker
from backend.services.run_logger import RunLogger
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)


def _build_enrichment_stage_flags(
    *,
    config: AppConfig,
    gap_manifest: Any,
) -> dict[str, Any]:
    """Build the stable flag payload for enrichment logging."""
    return {
        "enrichment_enabled": bool(config.enrichment.enabled),
        "use_split_gap_flow": bool(config.enrichment.use_split_gap_flow),
        "web_research_enabled": bool(config.enrichment.web_research_enabled),
        "external_source_search_enabled": bool(
            config.enrichment.external_source_search_enabled
        ),
        "gap_analysis_executed": True,
        "web_research_executed": bool(
            config.enrichment.web_research_enabled and gap_manifest.city_gaps
        ),
        "external_source_search_executed": bool(
            config.enrichment.external_source_search_enabled and gap_manifest.city_gaps
        ),
        "enrichment_model": config.enrichment.model,
    }


def _build_gap_analysis_artifact(
    *,
    question: str,
    context_bundle: dict[str, Any],
    gap_manifest: Any,
    config: AppConfig,
) -> dict[str, Any]:
    """Build the gap-analysis enrichment sub-artifact."""
    city_gap_payloads = [gap.model_dump(mode="json") for gap in gap_manifest.city_gaps]
    query_field_payloads = [field.model_dump(mode="json") for field in gap_manifest.query_fields]
    blank_field_count = sum(len(gap.blank_fields) for gap in gap_manifest.city_gaps)
    stale_field_count = sum(len(gap.stale_flags) for gap in gap_manifest.city_gaps)
    bundled_field_count = sum(len(gap.bundled_fields) for gap in gap_manifest.city_gaps)
    gap_field_count = blank_field_count + stale_field_count + bundled_field_count
    return {
        "status": "completed",
        "flags": {
            "use_split_gap_flow": bool(config.enrichment.use_split_gap_flow),
        },
        "inputs": {
            "question": question,
            "has_markdown_context": isinstance(context_bundle.get("markdown"), dict),
        },
        "outputs": {
            "query_fields": query_field_payloads,
            "city_gaps": city_gap_payloads,
            "non_estimable_fields": list(gap_manifest.non_estimable_fields),
        },
        "metrics": {
            "query_field_count": len(query_field_payloads),
            "city_gap_count": len(city_gap_payloads),
            "gap_field_count": gap_field_count,
            "blank_field_count": blank_field_count,
            "stale_field_count": stale_field_count,
            "bundled_field_count": bundled_field_count,
            "non_estimable_field_count": len(gap_manifest.non_estimable_fields),
        },
    }


def _build_external_sources_artifact(
    *,
    enabled: bool,
    executed: bool,
    external_evidence: list[dict[str, Any]],
    external_resolutions: list[dict[str, Any]],
    external_no_evidence: list[dict[str, Any]],
    tool_calls: list[dict[str, object]],
    search_audit: dict[str, Any] | None = None,
    search_audit_path: str | None = None,
) -> dict[str, Any]:
    """Build the external-sources enrichment sub-artifact."""
    filled_city_fields = city_field_pairs(
        [
            record
            for record in external_resolutions
            if str(record.get("action", "")).strip().casefold() == "fill"
        ]
    )
    unresolved_city_fields = city_field_pairs(
        [
            record
            for record in external_resolutions
            if str(record.get("action", "")).strip().casefold() == "unresolved"
        ]
    )
    no_evidence_city_fields = city_field_pairs(external_no_evidence)
    audit_payload = search_audit if isinstance(search_audit, dict) else {}
    metrics_payload = audit_payload.get("metrics")
    audit_metrics = metrics_payload if isinstance(metrics_payload, dict) else {}
    return {
        "status": "completed" if executed else "skipped",
        "skip_reason": None if executed else "disabled_or_no_gaps",
        "flags": {
            "external_source_search_enabled": enabled,
            "external_source_search_executed": executed,
        },
        "outputs": {
            "external_source_validated_claims": external_evidence,
            "external_source_resolutions": external_resolutions,
            "external_source_no_evidence": external_no_evidence,
            "tool_calls": tool_calls,
            "filled_city_fields": filled_city_fields,
            "unresolved_city_fields": unresolved_city_fields,
            "no_evidence_city_fields": no_evidence_city_fields,
            "searched_city_fields": audit_payload.get("searched_city_fields", []),
            "rejected_claims": audit_payload.get("rejected_claims", []),
            "unused_candidates": audit_payload.get("unused_candidates", []),
            "unresolved_searched_city_fields": audit_payload.get(
                "unresolved_searched_city_fields", []
            ),
            "search_audit_artifact": search_audit_path,
        },
        "metrics": {
            "external_evidence_count": len(external_evidence),
            "external_resolution_count": len(external_resolutions),
            "external_no_evidence_count": len(external_no_evidence),
            "tool_call_count": len(tool_calls),
            "filled_city_field_count": len(filled_city_fields),
            "unresolved_city_field_count": len(unresolved_city_fields),
            "no_evidence_city_field_count": len(no_evidence_city_fields),
            "searched_city_field_count": audit_metrics.get("searched_city_field_count"),
            "candidate_count": audit_metrics.get("candidate_count"),
            "validated_claim_count": audit_metrics.get("validated_claim_count"),
            "rejected_claim_count": audit_metrics.get("rejected_claim_count"),
            "unused_candidate_count": audit_metrics.get("unused_candidate_count"),
            "unresolved_searched_city_field_count": audit_metrics.get(
                "unresolved_searched_city_field_count"
            ),
            "max_turn_exceeded_count": audit_metrics.get("max_turn_exceeded_count"),
            "fallback_finalization_count": audit_metrics.get(
                "fallback_finalization_count"
            ),
        },
    }


def _build_web_research_artifact(
    *,
    enabled: bool,
    executed: bool,
    search_batches: list[dict[str, Any]],
    web_findings: list[dict[str, Any]],
    freshness_results: list[dict[str, Any]],
    national_findings: list[dict[str, Any]],
    comparative_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the web-research enrichment sub-artifact."""
    added_city_fields = city_field_pairs(web_findings)
    freshness_city_fields = city_field_pairs(freshness_results)
    return {
        "status": "completed" if executed else "skipped",
        "skip_reason": None if executed else "disabled_or_no_gaps",
        "flags": {
            "web_research_enabled": enabled,
            "web_research_executed": executed,
            "freshness_check_executed": bool(freshness_results),
        },
        "outputs": {
            "search_batches": search_batches,
            "web_findings": web_findings,
            "freshness_results": freshness_results,
            "national_findings": national_findings,
            "comparative_findings": comparative_findings,
            "added_city_fields": added_city_fields,
            "freshness_touched_city_fields": freshness_city_fields,
        },
        "metrics": {
            "search_batch_count": len(search_batches),
            "search_query_count": sum(
                len(batch.get("queries", []))
                for batch in search_batches
                if isinstance(batch.get("queries"), list)
            ),
            "web_finding_count": len(web_findings),
            "freshness_result_count": len(freshness_results),
            "national_finding_count": len(national_findings),
            "comparative_finding_count": len(comparative_findings),
            "added_city_field_count": len(added_city_fields),
            "freshness_touched_city_field_count": len(freshness_city_fields),
        },
    }


def _write_context_handoff_stage(
    *,
    run_logger: RunLogger,
    progress: ProgressTracker | None,
    stage_name: str,
    snapshot_filename: str,
    payload_filename: str,
    payload_key: str,
    payload: dict[str, Any] | None,
    progress_label: str,
    metrics: dict[str, Any],
) -> None:
    """Write a context handoff stage from inside the enrichment pipeline."""
    if progress:
        progress.start_step(stage_name, f"Freezing {stage_name.replace('_', ' ')}")
    context_snapshot_path = run_logger.write_stage_file(
        stage_name,
        snapshot_filename,
        run_logger.context_bundle,
        alias=f"{stage_name}_context_snapshot",
    )
    outputs = {
        "context_bundle_snapshot": run_logger.artifact_label(context_snapshot_path),
    }
    if isinstance(payload, dict):
        payload_path = run_logger.write_stage_file(
            stage_name,
            payload_filename,
            payload,
            alias=f"{stage_name}_{payload_key}",
        )
        outputs[payload_key] = run_logger.artifact_label(payload_path)
    if progress:
        progress.add_item(stage_name, progress_label)
        progress.complete_step(stage_name)
    run_logger.write_stage_detail(
        stage_name,
        {
            "inputs": {f"has_{payload_key}": isinstance(payload, dict)},
            "outputs": outputs,
            "metrics": metrics,
        },
    )


def run_enrichment_pipeline(
    question: str,
    context_bundle: dict[str, Any],
    base_dir: Path,
    run_logger: RunLogger,
    config: AppConfig,
    api_key: str,
    progress: ProgressTracker | None = None,
) -> dict[str, Any]:
    """Run the enrichment pipeline: gap analysis -> web research -> assumptions.

    On any failure, returns the original ``context_bundle`` unmodified so the
    pipeline can continue gracefully.
    """
    start_time = time.monotonic()

    try:
        if progress:
            progress.start_step("gap_analysis", "Analyzing data gaps")
            progress.add_item("gap_analysis", "Classifying fields and detecting gaps...")
        logger.info("Enrichment pipeline: starting gap analysis.")

        if config.enrichment.use_split_gap_flow:
            decomposition = decompose_fields(question, config, api_key)
            if progress:
                progress.add_item(
                    "gap_analysis",
                    f"{len(decomposition.query_fields)} fields decomposed",
                )

            gap_manifest = detect_city_gaps(
                question, decomposition, context_bundle, config, api_key
            )
        else:
            gap_manifest = run_gap_analysis(question, context_bundle, config, api_key)

        if progress:
            n_fields = len(gap_manifest.query_fields)
            for qf in gap_manifest.query_fields:
                progress.add_item(
                    "gap_analysis",
                    f"Field: {qf.field} ({qf.classification})",
                    item_type="field",
                    title=qf.field,
                    metadata={
                        "classification": qf.classification,
                        "scope": qf.scope,
                    },
                )
            field_word = "field" if n_fields == 1 else "fields"
            progress.add_item("gap_analysis", f"{n_fields} {field_word} classified")

            for cg in gap_manifest.city_gaps:
                n_blank = len(cg.blank_fields)
                n_stale = len(cg.stale_flags)
                n_bundled = len(cg.bundled_fields)
                parts = []
                if n_blank:
                    parts.append(f"{n_blank} blank")
                if n_stale:
                    parts.append(f"{n_stale} stale")
                if n_bundled:
                    parts.append(f"{n_bundled} bundled")
                detail = ", ".join(parts) if parts else "gap detected"
                progress.add_item(
                    "gap_analysis",
                    f"{cg.city}: {detail} [{cg.search_priority}]",
                    item_type="gap",
                    title=cg.city,
                    count=n_blank + n_stale + n_bundled,
                    metadata={
                        "priority": cg.search_priority,
                        "blank": n_blank,
                        "stale": n_stale,
                        "bundled": n_bundled,
                    },
                )
            n_gaps = len(gap_manifest.city_gaps)
            gap_word = "city" if n_gaps == 1 else "cities"
            progress.add_item("gap_analysis", f"{n_gaps} {gap_word} with gaps")
            progress.complete_step("gap_analysis")

        if not gap_manifest.city_gaps and not gap_manifest.query_fields:
            logger.info("Enrichment pipeline: no gaps found, skipping enrichment.")
            if progress:
                progress.start_step("web_research", "Running web research")
                progress.add_item("web_research", "Skipped (no gaps)")
                progress.complete_step("web_research", status="skipped")
                progress.start_step("assumptions", "Estimating assumptions")
                progress.add_item("assumptions", "Skipped (no gaps)")
                progress.complete_step("assumptions", status="skipped")
            run_logger.record_decision(
                {
                    "step": "enrichment",
                    "status": "skipped",
                    "reason": "no_gaps_found",
                }
            )
            return context_bundle

        web_findings = []
        freshness_results = []
        national_findings = []
        comparative_findings = []
        external_evidence = []
        external_resolutions = []
        external_no_evidence = []
        external_tool_calls: list[dict[str, object]] = []
        external_search_audit: dict[str, Any] = {}
        search_batches = []

        # Governed external Markdown search runs by default when tagged sources exist.
        if config.enrichment.external_source_search_enabled and gap_manifest.city_gaps:
            if progress:
                progress.start_step("external_sources", "Searching tagged external sources")
                progress.add_item("external_sources", "Running governed Markdown search...")
            (
                external_evidence,
                external_resolutions,
                external_no_evidence,
                external_tool_calls,
                external_search_audit,
            ) = (
                run_external_source_enrichment(
                    question=question,
                    context_bundle=context_bundle,
                    gap_manifest=gap_manifest,
                    base_dir=base_dir,
                    config=config,
                    api_key=api_key,
                    run_id=base_dir.name,
                )
            )
            if progress:
                progress.add_item(
                    "external_sources",
                    f"{len(external_evidence)} external evidence claims",
                )
                if external_no_evidence:
                    progress.add_item(
                        "external_sources",
                        f"{len(external_no_evidence)} no-evidence records",
                    )
                progress.complete_step("external_sources")
            logger.info(
                "External source search complete: claims=%d resolutions=%d no_evidence=%d",
                len(external_evidence),
                len(external_resolutions),
                len(external_no_evidence),
            )

        external_search_audit_path: str | None = None
        if external_search_audit:
            audit_path = run_logger.write_stage_file(
                "enrichment",
                EXTERNAL_SOURCE_SEARCH_AUDIT_FILENAME,
                external_search_audit,
                alias="enrichment_external_source_search_audit",
            )
            external_search_audit_path = run_logger.artifact_label(audit_path)

        if config.enrichment.web_research_enabled and gap_manifest.city_gaps:
            if progress:
                progress.start_step("web_research", "Running web research")
                progress.add_item("web_research", "Planning search queries...")
            logger.info("Enrichment pipeline: starting web research.")

            search_batches = plan_searches(gap_manifest, config, api_key, question=question)
            if progress:
                total_queries = sum(len(b.queries) for b in search_batches)
                progress.add_item(
                    "web_research",
                    f"{len(search_batches)} batches, {total_queries} queries planned",
                )
                progress.add_item("web_research", "Executing searches...")

            city_batches = [
                b
                for b in search_batches
                if b.search_type not in ("national_benchmark", "comparative_benchmark")
            ]
            national_batches = [
                b for b in search_batches if b.search_type == "national_benchmark"
            ]
            comparative_batches = [
                b for b in search_batches if b.search_type == "comparative_benchmark"
            ]

            batch_groups = {}
            if city_batches:
                batch_groups["city"] = city_batches
            if national_batches:
                batch_groups["national"] = national_batches
            if comparative_batches:
                batch_groups["comparative"] = comparative_batches

            if batch_groups:
                with ThreadPoolExecutor(max_workers=len(batch_groups)) as pool:
                    futures = {
                        pool.submit(
                            execute_search_batches,
                            batches,
                            config,
                            api_key,
                            progress,
                        ): label
                        for label, batches in batch_groups.items()
                    }
                    for future in as_completed(futures):
                        label = futures[future]
                        try:
                            findings = future.result()
                        except Exception:
                            logger.warning(
                                "Batch group %s failed.",
                                label,
                                exc_info=True,
                            )
                            continue
                        if label == "city":
                            web_findings = findings
                        elif label == "national":
                            national_findings = findings
                            if progress and national_findings:
                                progress.add_item(
                                    "web_research",
                                    f"{len(national_findings)} national benchmark findings",
                                )
                        elif label == "comparative":
                            comparative_findings = findings
                            if progress and comparative_findings:
                                progress.add_item(
                                    "web_research",
                                    f"{len(comparative_findings)} comparative benchmark findings",
                                )

            if search_batches and web_findings:
                if progress:
                    progress.add_item("web_research", "Checking freshness vs CCC data...")
                freshness_results = check_freshness(
                    web_findings, context_bundle, config, api_key
                )
                if progress and freshness_results:
                    n_superseded = sum(
                        1 for r in freshness_results if r.classification == "superseded"
                    )
                    n_consistent = sum(
                        1 for r in freshness_results if r.classification == "consistent"
                    )
                    progress.add_item(
                        "web_research",
                        f"Freshness: {n_consistent} consistent, {n_superseded} superseded",
                    )
            if progress:
                progress.add_item("web_research", f"{len(web_findings)} total findings")
                progress.complete_step("web_research")
            logger.info(
                "Web research complete: batches=%d findings=%d freshness=%d",
                len(search_batches),
                len(web_findings),
                len(freshness_results),
            )
        else:
            if progress:
                progress.start_step("web_research", "Running web research")
                progress.add_item("web_research", "Skipped (disabled or no gaps)")
                progress.complete_step("web_research", status="skipped")

        enriched_fields = compute_field_statuses(
            gap_manifest,
            web_findings,
            freshness_results,
            context_bundle,
            external_resolutions=external_resolutions,
        )

        assumptions_model = (
            config.enrichment.assumptions_estimator_model or config.enrichment.model
        )
        enrichment_elapsed = time.monotonic() - start_time
        stage_flags = _build_enrichment_stage_flags(
            config=config,
            gap_manifest=gap_manifest,
        )
        substage_artifacts = {
            "gap_analysis": _build_gap_analysis_artifact(
                question=question,
                context_bundle=context_bundle,
                gap_manifest=gap_manifest,
                config=config,
            ),
            "external_sources": _build_external_sources_artifact(
                enabled=bool(config.enrichment.external_source_search_enabled),
                executed=bool(
                    config.enrichment.external_source_search_enabled
                    and gap_manifest.city_gaps
                ),
                external_evidence=[
                    record.model_dump(mode="json") for record in external_evidence
                ],
                external_resolutions=[
                    record.model_dump(mode="json") for record in external_resolutions
                ],
                external_no_evidence=[
                    record.model_dump(mode="json") for record in external_no_evidence
                ],
                tool_calls=external_tool_calls,
                search_audit=external_search_audit,
                search_audit_path=external_search_audit_path,
            ),
            "web_research": _build_web_research_artifact(
                enabled=bool(config.enrichment.web_research_enabled),
                executed=bool(
                    config.enrichment.web_research_enabled and gap_manifest.city_gaps
                ),
                search_batches=[
                    batch.model_dump(mode="json") for batch in search_batches
                ],
                web_findings=[
                    record.model_dump(mode="json") for record in web_findings
                ],
                freshness_results=[
                    record.model_dump(mode="json") for record in freshness_results
                ],
                national_findings=[
                    record.model_dump(mode="json") for record in national_findings
                ],
                comparative_findings=[
                    record.model_dump(mode="json") for record in comparative_findings
                ],
            ),
        }

        enriched = merge_enrichment_evidence_into_context(
            context_bundle=context_bundle,
            gap_manifest=gap_manifest,
            web_findings=web_findings,
            freshness_results=freshness_results,
            external_evidence=external_evidence,
            external_resolutions=external_resolutions,
            external_no_evidence=external_no_evidence,
            config_model=config.enrichment.model,
            elapsed_seconds=enrichment_elapsed,
        )

        serialize_enrichment_artifacts(
            enriched,
            base_dir,
            run_logger,
            stage_flags=stage_flags,
            substage_artifacts=substage_artifacts,
        )
        run_logger.context_bundle = enriched
        run_logger.write_context_bundle()
        enrichment_payload = enriched.get("enrichment")
        _write_context_handoff_stage(
            run_logger=run_logger,
            progress=progress,
            stage_name="enrichment_context_handoff",
            snapshot_filename="context_bundle_after_enrichment.json",
            payload_filename="enrichment_context_payload.json",
            payload_key="enrichment_context_payload",
            payload=enrichment_payload if isinstance(enrichment_payload, dict) else None,
            progress_label="Enrichment context snapshot written",
            metrics={
                "context_bundle_top_level_keys": len(enriched),
                "enriched_field_count": len(enriched_fields),
                "web_finding_count": len(web_findings),
                "external_evidence_count": len(external_evidence),
            },
        )

        gap_field_count = sum(
            len(gap.blank_fields) + len(gap.stale_flags) + len(gap.bundled_fields)
            for gap in gap_manifest.city_gaps
        )
        run_logger.record_decision(
            {
                "step": "enrichment",
                "status": "completed",
                "city_gap_count": len(gap_manifest.city_gaps),
                "gap_field_count": gap_field_count,
                "web_findings": len(web_findings),
                "external_evidence": len(external_evidence),
                "elapsed_seconds": round(enrichment_elapsed, 2),
            }
        )

        if progress:
            progress.start_step("assumptions", "Estimating assumptions")
        assumptions, non_estimable, saturation_warning = run_assumptions_estimator(
            question=question,
            context_bundle=enriched,
            gap_manifest=gap_manifest,
            enriched_fields=enriched_fields,
            config=config,
            api_key=api_key,
            progress=progress,
            national_benchmarks=national_findings or None,
            comparative_data=comparative_findings or None,
        )

        if progress:
            if assumptions:
                method_counts = Counter(a.method_used for a in assumptions)
                method_parts = [f"{m}: {c}" for m, c in method_counts.most_common()]
                progress.add_item(
                    "assumptions",
                    f"{len(assumptions)} estimates ({', '.join(method_parts)})",
                )
            else:
                progress.add_item("assumptions", "0 estimates")
            if non_estimable:
                progress.add_item("assumptions", f"{len(non_estimable)} non-estimable")
            progress.complete_step("assumptions")

        elapsed = time.monotonic() - start_time
        assumptions_payload = build_assumptions_payload(
            assumptions_model=assumptions_model,
            assumptions=assumptions,
            non_estimable=non_estimable,
            saturation_warning=saturation_warning,
            elapsed_seconds=elapsed,
        )
        enriched["assumptions"] = assumptions_payload
        serialize_assumptions_artifacts(assumptions_payload, base_dir, run_logger)
        run_logger.context_bundle = enriched
        run_logger.write_context_bundle()
        _write_context_handoff_stage(
            run_logger=run_logger,
            progress=progress,
            stage_name="assumptions_context_handoff",
            snapshot_filename="context_bundle_after_assumptions.json",
            payload_filename="assumptions_context_payload.json",
            payload_key="assumptions_context_payload",
            payload=assumptions_payload,
            progress_label="Assumptions context snapshot written",
            metrics={
                "context_bundle_top_level_keys": len(enriched),
                "assumption_count": len(assumptions),
                "non_estimable_output_count": len(non_estimable),
            },
        )
        run_logger.record_decision(
            {
                "step": "assumptions",
                "status": "completed",
                "assumptions_produced": len(assumptions),
                "non_estimable_flagged": len(non_estimable),
                "elapsed_seconds": round(elapsed, 2),
            }
        )

        logger.info(
            "Enrichment pipeline completed in %.1fs: gaps=%d assumptions=%d non_estimable=%d",
            elapsed,
            len(gap_manifest.city_gaps),
            len(assumptions),
            len(non_estimable),
        )
        return enriched

    except Exception as exc:
        elapsed = time.monotonic() - start_time
        logger.warning(
            "Enrichment pipeline failed after %.1fs; falling back. error=%s",
            elapsed,
            exc,
            exc_info=True,
        )
        run_logger.record_decision(
            {
                "step": "enrichment",
                "status": "fallback",
                "reason": str(exc),
                "elapsed_seconds": round(elapsed, 2),
            }
        )
        return context_bundle


__all__ = ["run_enrichment_pipeline"]
