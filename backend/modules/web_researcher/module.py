"""Enrichment pipeline orchestrator.

Top-level entry point called from the main pipeline orchestrator to run
gap analysis, web research (Phase 2), and assumptions estimation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from backend.modules.web_researcher.assumptions_estimator import run_assumptions_estimator
from backend.modules.web_researcher.context_merger import (
    compute_field_statuses,
    merge_enrichment_into_context,
    serialize_enrichment_artifacts,
)
from backend.modules.web_researcher.external_agent import run_external_source_enrichment
from backend.modules.web_researcher.freshness import check_freshness
from backend.modules.web_researcher.gap_analysis import run_gap_analysis
from backend.modules.web_researcher.search_planner import plan_searches
from backend.modules.web_researcher.search_worker import execute_search_batches
from backend.services.progress_tracker import ProgressTracker
from backend.services.run_logger import RunLogger
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)


def run_enrichment_pipeline(
    question: str,
    context_bundle: dict[str, Any],
    base_dir: Path,
    run_logger: RunLogger,
    config: AppConfig,
    api_key: str,
    progress: ProgressTracker | None = None,
) -> dict[str, Any]:
    """Run the enrichment pipeline: gap analysis → (web research) → assumptions.

    On any failure, returns the original ``context_bundle`` unmodified so the
    pipeline can continue gracefully.
    """
    start_time = time.monotonic()

    try:
        # Step 5: Gap Analysis
        if progress:
            progress.start_step("gap_analysis", "Analyzing data gaps")
            progress.add_item("gap_analysis", "Classifying fields and detecting gaps...")
        logger.info("Enrichment pipeline: starting gap analysis.")
        gap_manifest = run_gap_analysis(question, context_bundle, config, api_key)

        if progress:
            # Show each field with its classification
            n_fields = len(gap_manifest.query_fields)
            for qf in gap_manifest.query_fields:
                progress.add_item(
                    "gap_analysis",
                    f"Field: {qf.field} ({qf.classification})",
                    item_type="field",
                    title=qf.field,
                    metadata={"classification": qf.classification},
                )
            field_word = "field" if n_fields == 1 else "fields"
            progress.add_item("gap_analysis", f"{n_fields} {field_word} classified")

            # Show per-city gap summary with blank field counts
            for cg in gap_manifest.city_gaps:
                n_blank = len(cg.blank_fields)
                n_stale = len(cg.stale_flags)
                parts = []
                if n_blank:
                    parts.append(f"{n_blank} blank")
                if n_stale:
                    parts.append(f"{n_stale} stale")
                detail = ", ".join(parts) if parts else "gap detected"
                progress.add_item(
                    "gap_analysis",
                    f"{cg.city}: {detail} [{cg.search_priority}]",
                    item_type="gap",
                    title=cg.city,
                    count=n_blank + n_stale,
                    metadata={"priority": cg.search_priority, "blank": n_blank, "stale": n_stale},
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
            run_logger.record_decision({
                "step": "enrichment",
                "status": "skipped",
                "reason": "no_gaps_found",
            })
            return context_bundle

        web_findings = []
        freshness_results = []
        national_findings = []
        comparative_findings = []
        external_evidence = []
        external_resolutions = []
        external_no_evidence = []

        # Governed external Markdown search runs by default when tagged sources exist.
        if config.enrichment.external_source_search_enabled and gap_manifest.city_gaps:
            if progress:
                progress.start_step("external_sources", "Searching tagged external sources")
                progress.add_item("external_sources", "Running governed Markdown search...")
            external_evidence, external_resolutions, external_no_evidence, _tool_calls = (
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

        # Steps 6-8: Web Research (only if enabled AND gaps found)
        if config.enrichment.web_research_enabled and gap_manifest.city_gaps:
            if progress:
                progress.start_step("web_research", "Running web research")
                progress.add_item("web_research", "Planning search queries...")
            logger.info("Enrichment pipeline: starting web research.")
            # Step 6: Search Planner → formulate queries
            search_batches = plan_searches(gap_manifest, config, api_key, question=question)
            if progress:
                total_queries = sum(len(b.queries) for b in search_batches)
                progress.add_item(
                    "web_research",
                    f"{len(search_batches)} batches, {total_queries} queries planned",
                )
                progress.add_item("web_research", "Executing searches...")
            # Split city-specific vs national vs comparative benchmark batches
            city_batches = [b for b in search_batches if b.search_type not in ("national_benchmark", "comparative_benchmark")]
            national_batches = [b for b in search_batches if b.search_type == "national_benchmark"]
            comparative_batches = [b for b in search_batches if b.search_type == "comparative_benchmark"]

            # Step 6 cont: Search Workers → execute all batch types in parallel
            national_findings = []
            comparative_findings = []

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
                            execute_search_batches, batches, config, api_key, progress,
                        ): label
                        for label, batches in batch_groups.items()
                    }
                    for future in as_completed(futures):
                        label = futures[future]
                        try:
                            findings = future.result()
                        except Exception:
                            logger.warning(
                                "Batch group %s failed.", label, exc_info=True,
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

            if search_batches:
                # Step 7: Freshness Checker → compare web vs CCC
                # (national findings skip freshness — they're reference data)
                if web_findings:
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

        # Determine enriched field statuses (which are still_missing after web research)
        enriched_fields = compute_field_statuses(
            gap_manifest,
            web_findings,
            freshness_results,
            context_bundle,
            external_resolutions=external_resolutions,
        )

        # Step 9: Assumptions Model on remaining blanks
        if progress:
            progress.start_step("assumptions", "Estimating assumptions")
        assumptions_model = (
            config.enrichment.assumptions_estimator_model or config.enrichment.model
        )
        assumptions, non_estimable, saturation_warning = run_assumptions_estimator(
            question=question,
            context_bundle=context_bundle,
            gap_manifest=gap_manifest,
            enriched_fields=enriched_fields,
            config=config,
            api_key=api_key,
            progress=progress,
            national_benchmarks=national_findings or None,
            comparative_data=comparative_findings or None,
        )

        if progress:
            # Method breakdown summary
            if assumptions:
                from collections import Counter
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

        # Merge everything into enriched context bundle
        enriched = merge_enrichment_into_context(
            context_bundle=context_bundle,
            gap_manifest=gap_manifest,
            web_findings=web_findings,
            freshness_results=freshness_results,
            external_evidence=external_evidence,
            external_resolutions=external_resolutions,
            external_no_evidence=external_no_evidence,
            assumptions=assumptions,
            non_estimable=non_estimable,
            saturation_warning=saturation_warning,
            config_model=config.enrichment.model,
            assumptions_model=assumptions_model,
            elapsed_seconds=elapsed,
        )

        # Serialize artifacts to disk
        serialize_enrichment_artifacts(enriched, base_dir, run_logger)

        run_logger.record_decision({
            "step": "enrichment",
            "status": "completed",
            "total_gaps": len(gap_manifest.city_gaps),
            "web_findings": len(web_findings),
            "external_evidence": len(external_evidence),
            "assumptions_produced": len(assumptions),
            "non_estimable_flagged": len(non_estimable),
            "elapsed_seconds": round(elapsed, 2),
        })

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
        run_logger.record_decision({
            "step": "enrichment",
            "status": "fallback",
            "reason": str(exc),
            "elapsed_seconds": round(elapsed, 2),
        })
        return context_bundle  # Original, unmodified


__all__ = ["run_enrichment_pipeline"]
