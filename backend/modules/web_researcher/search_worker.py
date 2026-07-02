"""Search Worker: per-batch search orchestration with retry and extraction."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

from backend.modules.web_researcher.deep_diver import DeepDiver
from backend.modules.web_researcher.extractor import extract_fields_from_content
from backend.modules.web_researcher.models import SearchBatch, WebFinding
from backend.modules.web_researcher.post_extraction_validator import validate_findings
from backend.modules.web_researcher.relevance import check_relevance_batch
from backend.modules.web_researcher.scraper import FirecrawlScraper
from backend.modules.web_researcher.search import SerperSearchClient
from backend.modules.web_researcher.tier1_web import (
    Tier1WebAllowlist,
    Tier1WebSource,
    load_tier1_web_allowlist,
)
from backend.services.llm_observability import LlmCallRecorder
from backend.services.progress_tracker import ProgressTracker
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)

_MAX_URLS_PER_DOMAIN_PER_BATCH = 3


def _load_tier1_allowlist_safe() -> Tier1WebAllowlist | None:
    """Load the tier-1 allowlist, returning None when not yet generated."""
    try:
        return load_tier1_web_allowlist()
    except FileNotFoundError:
        logger.info(
            "search_worker: tier-1 allowlist not present; skipping tier-1 pre-pass."
        )
        return None
    except Exception:  # noqa: BLE001
        logger.warning("search_worker: failed to load tier-1 allowlist", exc_info=True)
        return None


def _matching_tier1_sources(
    allowlist: Tier1WebAllowlist | None,
    batch: SearchBatch,
) -> list[Tier1WebSource]:
    """Return tier-1 sources whose coverage overlaps the batch's target fields/cities."""
    if allowlist is None:
        return []
    matches = allowlist.matching(
        cities=batch.cities,
        fields=batch.target_fields,
    )
    # Drop auth-walled sources — we can't site-search them.
    return [s for s in matches if s.access != "auth_required"]


def _process_results_for_query(
    *,
    query: str,
    results: list,
    batch: SearchBatch,
    scraper: FirecrawlScraper,
    config: AppConfig,
    api_key: str,
    scraped_urls: set[str],
    domain_url_counts: dict[str, int],
    bypass_domain_cap: bool,
    tag_source_id: str | None,
    tag_source_tier: str | None,
    excluded_paths: list[str] | None = None,
    llm_recorder: LlmCallRecorder | None = None,
) -> list[WebFinding]:
    """Run relevance → scrape → extract for one query's results.

    ``bypass_domain_cap`` is True for tier-1 site: queries because the cap
    was designed for diversifying open results, not for limiting how many
    pages we read off a single trusted domain.
    """
    if not results:
        return []

    # Relevance check (skip for national/comparative benchmarks — they
    # don't target specific cities so entity disambiguation would
    # incorrectly reject them).
    if batch.search_type in ("national_benchmark", "comparative_benchmark"):
        checked = [(r, True) for r in results]
    else:
        checked = check_relevance_batch(
            results,
            batch.target_fields,
            batch.cities,
            config,
            api_key,
            llm_recorder=llm_recorder,
        )

    out: list[WebFinding] = []
    for result, is_relevant in checked:
        if not is_relevant:
            continue
        if result.url in scraped_urls:
            continue
        if excluded_paths and any(p and p in result.url for p in excluded_paths):
            continue

        domain = urlparse(result.url).netloc.lower()
        if not bypass_domain_cap:
            if domain_url_counts.get(domain, 0) >= _MAX_URLS_PER_DOMAIN_PER_BATCH:
                continue

        scrape_result = scraper.scrape(
            result.url,
            query=query,
            batch_id=batch.batch_id,
        )
        if not scrape_result.success:
            continue

        scraped_urls.add(result.url)
        domain_url_counts[domain] = domain_url_counts.get(domain, 0) + 1

        findings = extract_fields_from_content(
            content=scrape_result.content,
            source_url=result.url,
            target_fields=batch.target_fields,
            cities=batch.cities,
            config=config,
            api_key=api_key,
            llm_recorder=llm_recorder,
        )
        findings = validate_findings(findings, batch.cities)
        for finding in findings:
            if tag_source_id is not None:
                finding.source_id = tag_source_id
            if tag_source_tier is not None:
                finding.source_tier = tag_source_tier
        out.extend(findings)

    return out


def _coverage_set_from_findings(findings: list[WebFinding]) -> set[tuple[str, str]]:
    return {(f.city.lower(), f.field.lower()) for f in findings}


def _empty_batch_search_stats(batch: SearchBatch, max_retries: int) -> dict[str, Any]:
    """Build the per-batch search execution stats skeleton."""
    return {
        "batch_id": batch.batch_id,
        "search_type": batch.search_type,
        "cities": batch.cities,
        "target_fields": batch.target_fields,
        "planned_query_count": len(batch.queries),
        "max_attempts": max_retries + 1,
        "matching_tier1_source_count": 0,
        "matching_tier1_source_ids": [],
        "estimated_max_serper_query_count": 0,
        "actual_serper_query_count": 0,
        "serper_call_count": 0,
        "tier1_site_query_count": 0,
        "open_query_count": 0,
        "open_query_skipped_count": 0,
    }


def _record_serper_call(batch_stats: dict[str, Any]) -> None:
    """Count one search call that should send a request to Serper."""
    batch_stats["actual_serper_query_count"] += 1
    batch_stats["serper_call_count"] += 1


def _successful_serper_count(search_client: SerperSearchClient) -> int:
    """Return successful Serper calls, falling back for older test doubles."""
    value = getattr(search_client, "successful_query_count", None)
    if isinstance(value, int):
        return value
    return int(search_client.query_count)


def _merge_search_execution_stats(
    *,
    batches: list[SearchBatch],
    config: AppConfig,
    batch_stats: list[dict[str, Any]],
    actual_serper_query_count: int,
    successful_serper_query_count: int,
) -> dict[str, Any]:
    """Aggregate per-batch search execution stats for the audit artifact."""
    return {
        "config": {
            "tier1_first_search": bool(config.enrichment.tier1_first_search),
            "tier1_confidence_threshold": float(
                config.enrichment.tier1_confidence_threshold
            ),
            "max_total_queries_per_run": int(
                config.enrichment.max_total_queries_per_run
            ),
            "max_queries_per_batch": int(config.enrichment.max_queries_per_batch),
            "max_retries_per_worker": int(config.enrichment.max_retries_per_worker),
            "max_workers": int(config.enrichment.max_workers),
        },
        "metrics": {
            "search_batch_count": len(batch_stats),
            "planned_query_count": sum(len(batch.queries) for batch in batches),
            "actual_serper_query_count": actual_serper_query_count,
            "serper_call_count": actual_serper_query_count,
            "successful_serper_query_count": successful_serper_query_count,
            "tier1_site_query_count": sum(
                int(item.get("tier1_site_query_count") or 0)
                for item in batch_stats
            ),
            "open_query_count": sum(
                int(item.get("open_query_count") or 0) for item in batch_stats
            ),
            "open_query_skipped_count": sum(
                int(item.get("open_query_skipped_count") or 0)
                for item in batch_stats
            ),
            "estimated_max_serper_query_count": sum(
                int(item.get("estimated_max_serper_query_count") or 0)
                for item in batch_stats
            ),
        },
        "batches": batch_stats,
    }


def execute_search_batch(
    batch: SearchBatch,
    search_client: SerperSearchClient,
    scraper: FirecrawlScraper,
    deep_diver: DeepDiver,
    config: AppConfig,
    api_key: str,
    search_stats: dict[str, Any] | None = None,
    llm_recorder: LlmCallRecorder | None = None,
) -> list[WebFinding]:
    """Execute a single search batch: search → filter → scrape → extract.

    When ``config.enrichment.tier1_first_search`` is true, each query runs
    a ``site:<domain>`` pre-pass against curated tier-1 web sources whose
    coverage matches the batch.  If tier-1 fully resolves a (city, field)
    pair (with extraction confidence ≥ ``tier1_confidence_threshold``),
    the open Serper pass is skipped for that query.

    Returns all ``WebFinding`` objects extracted from this batch.  Each
    finding from the tier-1 pre-pass is tagged with
    ``source_id = <allowlist_entry_id>`` and ``source_tier = "tier1"``;
    findings from the open pass carry ``source_tier = "open"``.
    """
    all_findings: list[WebFinding] = []
    domain_url_counts: dict[str, int] = {}
    scraped_urls: set[str] = set()
    max_retries = config.enrichment.max_retries_per_worker
    batch_stats = _empty_batch_search_stats(batch, max_retries)

    use_tier1 = bool(config.enrichment.tier1_first_search)
    confidence_threshold = float(config.enrichment.tier1_confidence_threshold)
    tier1_allowlist = _load_tier1_allowlist_safe() if use_tier1 else None
    tier1_sources = (
        _matching_tier1_sources(tier1_allowlist, batch) if use_tier1 else []
    )
    batch_stats["matching_tier1_source_count"] = len(tier1_sources)
    batch_stats["matching_tier1_source_ids"] = [source.id for source in tier1_sources]
    batch_stats["estimated_max_serper_query_count"] = (
        len(batch.queries) * (len(tier1_sources) + 1) * (max_retries + 1)
    )
    if use_tier1:
        logger.info(
            "search_worker: batch %s tier1 pre-pass — %d matching sources: %s",
            batch.batch_id,
            len(tier1_sources),
            ", ".join(s.id for s in tier1_sources) or "(none)",
        )

    for attempt in range(max_retries + 1):
        for query in batch.queries:
            tier1_findings: list[WebFinding] = []
            tier1_resolved: set[tuple[str, str]] = set()

            # Tier-1 pre-pass.
            for source in tier1_sources:
                scoped_query = f"site:{source.domain} {query}"
                if search_client.can_search:
                    batch_stats["tier1_site_query_count"] += 1
                    _record_serper_call(batch_stats)
                results = search_client.search(scoped_query)
                if not results:
                    continue
                source_findings = _process_results_for_query(
                    query=scoped_query,
                    results=results,
                    batch=batch,
                    scraper=scraper,
                    config=config,
                    api_key=api_key,
                    scraped_urls=scraped_urls,
                    domain_url_counts=domain_url_counts,
                    bypass_domain_cap=True,
                    tag_source_id=source.id,
                    tag_source_tier="tier1",
                    excluded_paths=source.excluded_paths,
                    llm_recorder=llm_recorder,
                )
                tier1_findings.extend(source_findings)
                # Only count high-confidence findings as "resolving" the gap.
                for f in source_findings:
                    if f.extraction_confidence >= confidence_threshold:
                        tier1_resolved.add((f.city.lower(), f.field.lower()))

            all_findings.extend(tier1_findings)

            # Decide whether to also run the open Serper pass for this query.
            needed_for_query = {
                (city.lower(), field.lower())
                for city in batch.cities
                for field in batch.target_fields
            }
            if use_tier1 and tier1_resolved >= needed_for_query:
                batch_stats["open_query_skipped_count"] += 1
                logger.info(
                    "search_worker: tier-1 fully covered query %r; skipping open pass.",
                    query,
                )
                continue

            # Open pass — same path as before, but findings are tagged.
            if search_client.can_search:
                batch_stats["open_query_count"] += 1
                _record_serper_call(batch_stats)
            results = search_client.search(query)
            open_findings = _process_results_for_query(
                query=query,
                results=results,
                batch=batch,
                scraper=scraper,
                config=config,
                api_key=api_key,
                scraped_urls=scraped_urls,
                domain_url_counts=domain_url_counts,
                bypass_domain_cap=False,
                tag_source_id=None,
                tag_source_tier="open" if use_tier1 else None,
                llm_recorder=llm_recorder,
            )
            all_findings.extend(open_findings)

        # Evaluate coverage
        covered_fields = {(f.city.lower(), f.field.lower()) for f in all_findings}
        needed = {
            (city.lower(), field.lower())
            for city in batch.cities
            for field in batch.target_fields
        }
        remaining_gaps = needed - covered_fields

        if not remaining_gaps:
            break

        if attempt < max_retries:
            logger.info(
                "Search batch %s: %d gaps remain after attempt %d, retrying.",
                batch.batch_id,
                len(remaining_gaps),
                attempt + 1,
            )
        # On retry, the same queries will run again (in practice the LLM
        # planner would reformulate; for now we rely on the initial queries)

    # Step 5: Deep dive on promising domains
    budget = batch.budget
    deep_dive_allowed = budget.get("deep_dive_allowed", False)

    if deep_dive_allowed and deep_diver.can_dive() and scraped_urls:
        # Find domains with the most findings
        domain_finding_counts: dict[str, int] = {}
        domain_urls: dict[str, str] = {}
        for finding in all_findings:
            domain = urlparse(finding.source_url).netloc.lower()
            domain_finding_counts[domain] = domain_finding_counts.get(domain, 0) + 1
            domain_urls[domain] = finding.source_url

        # Deep dive the most promising domain
        if domain_finding_counts:
            best_domain = max(domain_finding_counts, key=domain_finding_counts.get)  # type: ignore[arg-type]
            seed_url = domain_urls[best_domain]
            logger.info("Deep diving domain=%r from url=%r", best_domain, seed_url)
            dive_result = deep_diver.dive(seed_url)

            for page in dive_result.pages:
                if page.url in scraped_urls:
                    continue
                findings = extract_fields_from_content(
                    content=page.content,
                    source_url=page.url,
                    target_fields=batch.target_fields,
                    cities=batch.cities,
                    config=config,
                    api_key=api_key,
                    llm_recorder=llm_recorder,
                )
                findings = validate_findings(findings, batch.cities)
                all_findings.extend(findings)

    logger.info(
        "Search batch %s completed: planned_queries=%d serper_calls=%d "
        "tier1_site_calls=%d open_calls=%d skipped_open_calls=%d "
        "estimated_max_serper_calls=%d findings=%d scraped_urls=%d.",
        batch.batch_id,
        len(batch.queries),
        int(batch_stats["actual_serper_query_count"]),
        int(batch_stats["tier1_site_query_count"]),
        int(batch_stats["open_query_count"]),
        int(batch_stats["open_query_skipped_count"]),
        int(batch_stats["estimated_max_serper_query_count"]),
        len(all_findings),
        len(scraped_urls),
    )
    if search_stats is not None:
        search_stats.update(batch_stats)
    return all_findings


def execute_search_batches(
    batches: list[SearchBatch],
    config: AppConfig,
    api_key: str,
    progress: ProgressTracker | None = None,
    scrape_failures: list[dict[str, Any]] | None = None,
    scrape_stats: dict[str, int] | None = None,
    search_execution_summary: dict[str, Any] | None = None,
    llm_recorder: LlmCallRecorder | None = None,
) -> list[WebFinding]:
    """Execute all search batches through a bounded thread pool.

    Shares a single GoogleSearchClient, FirecrawlScraper, and DeepDiver
    across workers for global quota tracking. Optional collectors receive
    structured scrape failures and compact scrape counters for artifacts.
    """
    if not batches:
        return []

    search_client = SerperSearchClient()
    scraper = FirecrawlScraper()
    deep_diver = DeepDiver(
        scraper=scraper,
        max_dives_per_run=config.enrichment.max_deep_dives_per_run,
        max_pages_per_dive=config.enrichment.max_pages_per_deep_dive,
    )

    all_findings: list[WebFinding] = []
    max_workers = min(config.enrichment.max_workers, len(batches))

    # Map batch_id → batch for progress reporting
    batch_by_id = {b.batch_id: b for b in batches}

    # Resolve tier-1 source_id → display name for progress items so the UI
    # can attribute findings to a curated source rather than a bare URL.
    name_by_source_id: dict[str, str] = {}
    if config.enrichment.tier1_first_search:
        allowlist = _load_tier1_allowlist_safe()
        if allowlist is not None:
            for s in allowlist.sources:
                if s.id and s.name:
                    name_by_source_id[s.id] = s.name

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        search_stats_by_future: dict[Any, dict[str, Any]] = {}
        for batch in batches:
            batch_search_stats: dict[str, Any] = {}
            future = pool.submit(
                execute_search_batch,
                batch,
                search_client,
                scraper,
                deep_diver,
                config,
                api_key,
                search_stats=batch_search_stats,
                llm_recorder=llm_recorder,
            )
            futures[future] = batch.batch_id
            search_stats_by_future[future] = batch_search_stats

        completed_count = 0
        batch_search_stats: list[dict[str, Any]] = []
        for future in as_completed(futures):
            batch_id = futures[future]
            completed_count += 1
            try:
                findings = future.result()
                batch_search_stats.append(search_stats_by_future.get(future, {}))
                all_findings.extend(findings)
                if progress:
                    batch = batch_by_id.get(batch_id)
                    cities_label = ", ".join(batch.cities[:3]) if batch else batch_id
                    if batch and len(batch.cities) > 3:
                        cities_label += f" +{len(batch.cities) - 3}"
                    progress.add_item(
                        "web_research",
                        f"Batch {completed_count}/{len(batches)} ({cities_label}): {len(findings)} findings",
                        item_type="batch_summary",
                        count=len(findings),
                        metadata={"cities": cities_label, "batch": f"{completed_count}/{len(batches)}"},
                    )
                    for f in findings:
                        val = f"{f.value} {f.unit}" if f.unit else str(f.value)
                        parsed_domain = urlparse(f.source_url).netloc.lower()
                        meta: dict[str, object] = {}
                        if f.source_tier:
                            meta["source_tier"] = f.source_tier
                        if f.source_id:
                            meta["source_id"] = f.source_id
                            name = name_by_source_id.get(f.source_id)
                            if name:
                                meta["source_name"] = name
                        progress.add_item(
                            "web_research",
                            f"  Found: {f.city} / {f.field} = {val} — {f.source_url}",
                            item_type="search_result",
                            title=f"{f.city} / {f.field} = {val}",
                            domain=parsed_domain,
                            url=f.source_url,
                            metadata=meta or None,
                        )
            except Exception:
                logger.warning(
                    "Search batch %s failed.", batch_id, exc_info=True
                )
                if progress:
                    progress.add_item(
                        "web_research",
                        f"Batch {completed_count}/{len(batches)}: failed",
                        item_type="batch_summary",
                        metadata={"batch": f"{completed_count}/{len(batches)}", "error": True},
                    )

    # Deduplicate findings by (city, field, source_url)
    seen: set[tuple[str, str, str]] = set()
    deduped: list[WebFinding] = []
    for f in all_findings:
        key = (f.city.lower(), f.field.lower(), f.source_url)
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    successful_serper_query_count = _successful_serper_count(search_client)
    logger.info(
        "All search batches complete: findings=%d deduped_findings=%d "
        "actual_serper_calls=%d successful_serper_calls=%d total_scrapes=%d.",
        len(all_findings),
        len(deduped),
        search_client.query_count,
        successful_serper_query_count,
        scraper.scrape_count,
    )
    if scrape_failures is not None:
        scrape_failures.extend(scraper.scrape_failures)
    if scrape_stats is not None:
        scrape_stats["scrape_success_count"] = scraper.scrape_count
        scrape_stats["scrape_failure_count"] = len(scraper.scrape_failures)
    if search_execution_summary is not None:
        search_execution_summary.update(
            _merge_search_execution_stats(
                batches=batches,
                config=config,
                batch_stats=batch_search_stats,
                actual_serper_query_count=search_client.query_count,
                successful_serper_query_count=successful_serper_query_count,
            )
        )
    return deduped


__all__ = ["execute_search_batch", "execute_search_batches"]
