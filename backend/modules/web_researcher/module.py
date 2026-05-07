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
from backend.modules.web_researcher.context_merger import (
    compute_field_statuses,
    merge_enrichment_into_context,
    serialize_enrichment_artifacts,
)
from backend.modules.web_researcher.external_sources import run_external_source_stage
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

            run_external_source_stage(decomposition, context_bundle)
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
            gap_manifest, web_findings, freshness_results, context_bundle
        )

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

        enriched = merge_enrichment_into_context(
            context_bundle=context_bundle,
            gap_manifest=gap_manifest,
            web_findings=web_findings,
            freshness_results=freshness_results,
            assumptions=assumptions,
            non_estimable=non_estimable,
            saturation_warning=saturation_warning,
            config_model=config.enrichment.model,
            assumptions_model=assumptions_model,
            elapsed_seconds=elapsed,
        )

        serialize_enrichment_artifacts(enriched, base_dir, run_logger)

        run_logger.record_decision(
            {
                "step": "enrichment",
                "status": "completed",
                "total_gaps": len(gap_manifest.city_gaps),
                "web_findings": len(web_findings),
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
