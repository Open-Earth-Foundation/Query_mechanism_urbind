"""Search Worker: per-batch search orchestration with retry and extraction."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

from backend.modules.web_researcher.deep_diver import DeepDiver
from backend.modules.web_researcher.extractor import extract_fields_from_content
from backend.modules.web_researcher.models import SearchBatch, WebFinding
from backend.modules.web_researcher.relevance import check_relevance_batch
from backend.modules.web_researcher.scraper import FirecrawlScraper
from backend.modules.web_researcher.search import SerperSearchClient
from backend.services.progress_tracker import ProgressTracker
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)

_MAX_URLS_PER_DOMAIN_PER_BATCH = 3


def execute_search_batch(
    batch: SearchBatch,
    search_client: SerperSearchClient,
    scraper: FirecrawlScraper,
    deep_diver: DeepDiver,
    config: AppConfig,
    api_key: str,
) -> list[WebFinding]:
    """Execute a single search batch: search → filter → scrape → extract.

    Per-worker flow:
    1. Run each query through Google CSE
    2. Relevance-check results
    3. Scrape relevant URLs via Firecrawl
    4. Extract target fields from scraped content
    5. If gaps remain and retries available, reformulate
    6. If promising domains found and deep dive allowed, crawl deeper

    Returns all ``WebFinding`` objects extracted from this batch.
    """
    all_findings: list[WebFinding] = []
    domain_url_counts: dict[str, int] = {}
    scraped_urls: set[str] = set()
    max_retries = config.enrichment.max_retries_per_worker

    for attempt in range(max_retries + 1):
        for query in batch.queries:
            # Step 1: Search
            results = search_client.search(query)
            if not results:
                continue

            # Step 2: Relevance check
            checked = check_relevance_batch(
                results, batch.target_fields, batch.cities, config, api_key
            )

            # Step 3: Scrape relevant results
            for result, is_relevant in checked:
                if not is_relevant:
                    continue
                if result.url in scraped_urls:
                    continue

                # Domain diversity: max 3 URLs per domain per batch
                domain = urlparse(result.url).netloc.lower()
                if domain_url_counts.get(domain, 0) >= _MAX_URLS_PER_DOMAIN_PER_BATCH:
                    continue

                scrape_result = scraper.scrape(result.url)
                if not scrape_result.success:
                    continue

                scraped_urls.add(result.url)
                domain_url_counts[domain] = domain_url_counts.get(domain, 0) + 1

                # Step 4: Extract fields
                findings = extract_fields_from_content(
                    content=scrape_result.content,
                    source_url=result.url,
                    target_fields=batch.target_fields,
                    cities=batch.cities,
                    config=config,
                    api_key=api_key,
                )
                all_findings.extend(findings)

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
                )
                all_findings.extend(findings)

    logger.info(
        "Search batch %s completed: %d findings from %d scraped URLs.",
        batch.batch_id,
        len(all_findings),
        len(scraped_urls),
    )
    return all_findings


def execute_search_batches(
    batches: list[SearchBatch],
    config: AppConfig,
    api_key: str,
    progress: ProgressTracker | None = None,
) -> list[WebFinding]:
    """Execute all search batches through a bounded thread pool.

    Shares a single GoogleSearchClient, FirecrawlScraper, and DeepDiver
    across workers for global quota tracking.
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

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                execute_search_batch,
                batch,
                search_client,
                scraper,
                deep_diver,
                config,
                api_key,
            ): batch.batch_id
            for batch in batches
        }

        completed_count = 0
        for future in as_completed(futures):
            batch_id = futures[future]
            completed_count += 1
            try:
                findings = future.result()
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
                        progress.add_item(
                            "web_research",
                            f"  Found: {f.city} / {f.field} = {val} — {f.source_url}",
                            item_type="search_result",
                            title=f"{f.city} / {f.field} = {val}",
                            domain=parsed_domain,
                            url=f.source_url,
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

    logger.info(
        "All search batches complete: %d findings (%d after dedup), "
        "%d total queries, %d total scrapes.",
        len(all_findings),
        len(deduped),
        search_client.query_count,
        scraper.scrape_count,
    )
    return deduped


__all__ = ["execute_search_batch", "execute_search_batches"]
