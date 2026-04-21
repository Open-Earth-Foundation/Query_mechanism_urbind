"""Deep diver: domain-focused crawling for promising sources."""

from __future__ import annotations

import logging
import threading
from urllib.parse import urljoin, urlparse

from backend.modules.web_researcher.scraper import FirecrawlScraper, ScrapeResult

logger = logging.getLogger(__name__)

_DEFAULT_MAX_PAGES = 10


class DeepDiveResult:
    """Aggregated content from a deep dive into a single domain."""

    __slots__ = ("domain", "pages", "total_chars")

    def __init__(self, domain: str) -> None:
        self.domain = domain
        self.pages: list[ScrapeResult] = []
        self.total_chars = 0


class DeepDiver:
    """Domain-focused crawler that follows links within a single domain."""

    def __init__(
        self,
        scraper: FirecrawlScraper,
        max_dives_per_run: int = 3,
        max_pages_per_dive: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        self._scraper = scraper
        self._max_dives = max_dives_per_run
        self._max_pages = max_pages_per_dive
        self._dive_count = 0
        self._lock = threading.Lock()

    @property
    def dives_remaining(self) -> int:
        return max(0, self._max_dives - self._dive_count)

    def can_dive(self) -> bool:
        return self._dive_count < self._max_dives

    def dive(self, seed_url: str, max_pages: int | None = None) -> DeepDiveResult:
        """Crawl a domain starting from seed_url.

        Scrapes the seed page, then follows same-domain links up to
        ``max_pages`` total. Thread-safe via atomic dive count.
        """
        with self._lock:
            if self._dive_count >= self._max_dives:
                logger.info("Deep dive limit reached; skipping url=%r", seed_url)
                return DeepDiveResult(domain=_extract_domain(seed_url))
            self._dive_count += 1

        page_limit = min(max_pages or self._max_pages, self._max_pages)
        domain = _extract_domain(seed_url)
        result = DeepDiveResult(domain=domain)

        visited: set[str] = set()
        queue: list[str] = [seed_url]

        while queue and len(result.pages) < page_limit:
            url = queue.pop(0)
            normalized = _normalize_url(url)

            if normalized in visited:
                continue
            visited.add(normalized)

            if _extract_domain(url) != domain:
                continue

            scrape_result = self._scraper.scrape(url)
            if not scrape_result.success:
                continue

            result.pages.append(scrape_result)
            result.total_chars += len(scrape_result.content)

            # Extract same-domain links from content (simple heuristic)
            new_links = _extract_links_from_markdown(scrape_result.content, domain, url)
            for link in new_links:
                if _normalize_url(link) not in visited:
                    queue.append(link)

        logger.info(
            "Deep dive domain=%r pages=%d chars=%d",
            domain,
            len(result.pages),
            result.total_chars,
        )
        return result


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    parsed = urlparse(url)
    return parsed.netloc.lower()


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication (strip fragment, trailing slash)."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}".lower()


def _extract_links_from_markdown(
    content: str,
    target_domain: str,
    base_url: str,
) -> list[str]:
    """Extract same-domain links from markdown content.

    Simple regex-free heuristic: look for markdown links [text](url)
    and bare URLs.
    """
    links: list[str] = []

    # Find markdown-style links
    i = 0
    while i < len(content):
        start = content.find("](", i)
        if start == -1:
            break
        end = content.find(")", start + 2)
        if end == -1:
            break
        url = content[start + 2 : end].strip()
        i = end + 1

        if not url or url.startswith("#") or url.startswith("mailto:"):
            continue

        # Handle relative URLs
        if url.startswith("/"):
            url = urljoin(base_url, url)
        elif not url.startswith("http"):
            continue

        if _extract_domain(url) == target_domain:
            links.append(url)

    return links[:20]  # Cap to prevent runaway link extraction


__all__ = ["DeepDiveResult", "DeepDiver"]
