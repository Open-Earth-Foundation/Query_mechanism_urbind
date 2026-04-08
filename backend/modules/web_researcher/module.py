"""Enrichment pipeline orchestrator.

Top-level entry point called from the main pipeline orchestrator to run
gap analysis, web research (Phase 2), and assumptions estimation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from backend.modules.web_researcher.assumptions_estimator import run_assumptions_estimator
from backend.modules.web_researcher.context_merger import (
    compute_field_statuses,
    merge_enrichment_into_context,
    serialize_enrichment_artifacts,
)
from backend.modules.web_researcher.freshness import check_freshness
from backend.modules.web_researcher.gap_analysis import run_gap_analysis
from backend.modules.web_researcher.search_planner import plan_searches
from backend.modules.web_researcher.search_worker import execute_search_batches
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
) -> dict[str, Any]:
    """Run the enrichment pipeline: gap analysis → (web research) → assumptions.

    On any failure, returns the original ``context_bundle`` unmodified so the
    pipeline can continue gracefully.
    """
    start_time = time.monotonic()

    try:
        # Step 5: Gap Analysis
        logger.info("Enrichment pipeline: starting gap analysis.")
        gap_manifest = run_gap_analysis(question, context_bundle, config, api_key)

        if not gap_manifest.city_gaps and not gap_manifest.query_fields:
            logger.info("Enrichment pipeline: no gaps found, skipping enrichment.")
            run_logger.record_decision({
                "step": "enrichment",
                "status": "skipped",
                "reason": "no_gaps_found",
            })
            return context_bundle

        web_findings = []
        freshness_results = []

        # Steps 6-8: Web Research (only if enabled AND gaps found)
        if config.enrichment.web_research_enabled and gap_manifest.city_gaps:
            logger.info("Enrichment pipeline: starting web research.")
            # Step 6: Search Planner → formulate queries
            search_batches = plan_searches(gap_manifest, config, api_key)
            # Step 6 cont: Search Workers → execute queries, scrape, extract
            if search_batches:
                web_findings = execute_search_batches(search_batches, config, api_key)
                # Step 7: Freshness Checker → compare web vs CCC
                if web_findings:
                    freshness_results = check_freshness(
                        web_findings, context_bundle, config, api_key
                    )
            logger.info(
                "Web research complete: batches=%d findings=%d freshness=%d",
                len(search_batches),
                len(web_findings),
                len(freshness_results),
            )

        # Determine enriched field statuses (which are still_missing after web research)
        enriched_fields = compute_field_statuses(
            gap_manifest, web_findings, freshness_results, context_bundle
        )

        # Step 9: Assumptions Model on remaining blanks
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
        )

        elapsed = time.monotonic() - start_time

        # Merge everything into enriched context bundle
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

        # Serialize artifacts to disk
        serialize_enrichment_artifacts(enriched, base_dir, run_logger)

        run_logger.record_decision({
            "step": "enrichment",
            "status": "completed",
            "total_gaps": len(gap_manifest.city_gaps),
            "web_findings": len(web_findings),
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
