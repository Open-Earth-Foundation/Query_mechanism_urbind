"""Web scraper using Firecrawl for JavaScript-rendered content extraction."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

_FIRECRAWL_ENDPOINT = "https://api.firecrawl.dev/v1/scrape"
_CONCURRENT_SCRAPE_LIMIT = 4
_DEFAULT_TIMEOUT = 30.0
_MAX_CONTENT_CHARS = 7000

# File extensions that Firecrawl cannot meaningfully render.
_SKIP_EXTENSIONS = frozenset({
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
})


def _has_skip_extension(url: str) -> bool:
    """Return True if the URL path ends with a non-scrapable file extension."""
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _SKIP_EXTENSIONS)


class ScrapeResult:
    """Content extracted from a single URL."""

    __slots__ = ("url", "content", "title", "success", "error")

    def __init__(
        self,
        url: str,
        content: str = "",
        title: str = "",
        success: bool = True,
        error: str | None = None,
    ) -> None:
        self.url = url
        self.content = content
        self.title = title
        self.success = success
        self.error = error


class FirecrawlScraper:
    """Firecrawl-based web scraper with concurrency limiting."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the scraper with a trimmed explicit or environment key."""
        raw_api_key = (
            api_key if api_key is not None else os.getenv("FIRECRAWL_API_KEY", "")
        )
        self.api_key = raw_api_key.strip()
        self._semaphore = threading.Semaphore(_CONCURRENT_SCRAPE_LIMIT)
        self._failure_lock = threading.Lock()
        self._failures: list[dict[str, Any]] = []
        self._scrape_count = 0

    @property
    def scrape_count(self) -> int:
        return self._scrape_count

    @property
    def scrape_failures(self) -> list[dict[str, Any]]:
        """Return a snapshot of structured scrape failures captured this run."""
        with self._failure_lock:
            return list(self._failures)

    def scrape(
        self,
        url: str,
        max_chars: int = _MAX_CONTENT_CHARS,
        timeout: float = _DEFAULT_TIMEOUT,
        query: str | None = None,
        batch_id: str | None = None,
    ) -> ScrapeResult:
        """Scrape a single URL via Firecrawl.

        Respects the concurrency semaphore. Returns a ``ScrapeResult``
        with ``success=False`` on any error.
        """
        if not self.api_key:
            logger.warning("Firecrawl API key not configured; skipping scrape.")
            self._record_failure(
                url=url,
                error_type="no_api_key",
                error="Firecrawl API key not configured.",
                query=query,
                batch_id=batch_id,
            )
            return ScrapeResult(url=url, success=False, error="no_api_key")

        if _has_skip_extension(url):
            logger.info("Skipping non-scrapable URL extension: %r", url)
            return ScrapeResult(url=url, success=False, error="unsupported_file_type")

        with self._semaphore:
            return self._do_scrape(url, max_chars, timeout, query, batch_id)

    def _do_scrape(
        self,
        url: str,
        max_chars: int,
        timeout: float,
        query: str | None,
        batch_id: str | None,
    ) -> ScrapeResult:
        """Internal scrape implementation."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "url": url,
            "formats": ["markdown"],
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    _FIRECRAWL_ENDPOINT,
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()

            self._scrape_count += 1
            data = resp.json()

            success = data.get("success", False)
            if not success:
                error = data.get("error", "firecrawl_failure")
                self._record_failure(
                    url=url,
                    error_type="firecrawl_failure",
                    error=str(error),
                    query=query,
                    batch_id=batch_id,
                )
                return ScrapeResult(
                    url=url,
                    success=False,
                    error=error,
                )

            result_data = data.get("data", {})
            content = result_data.get("markdown", "") or ""
            title = result_data.get("metadata", {}).get("title", "") or ""

            # Truncate content to limit
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[...content truncated...]"

            logger.info(
                "Firecrawl scrape url=%r chars=%d total_scrapes=%d",
                url,
                len(content),
                self._scrape_count,
            )
            return ScrapeResult(url=url, content=content, title=title, success=True)

        except httpx.HTTPStatusError as exc:
            error_type = f"http_{exc.response.status_code}"
            logger.warning(
                "Firecrawl HTTP error: status=%d url=%r",
                exc.response.status_code,
                url,
            )
            self._record_failure(
                url=url,
                error_type=error_type,
                error=str(exc),
                query=query,
                batch_id=batch_id,
            )
            return ScrapeResult(url=url, success=False, error=error_type)
        except Exception as exc:
            logger.warning("Firecrawl scrape failed for url=%r", url, exc_info=True)
            self._record_failure(
                url=url,
                error_type=type(exc).__name__,
                error=str(exc),
                query=query,
                batch_id=batch_id,
            )
            return ScrapeResult(url=url, success=False, error=str(exc))

    def _record_failure(
        self,
        *,
        url: str,
        error_type: str,
        error: str,
        query: str | None,
        batch_id: str | None,
    ) -> None:
        """Store a compact failure record for the web-research audit artifact."""
        record = {
            "url": url,
            "domain": urlparse(url).netloc.lower(),
            "provider": "firecrawl",
            "batch_id": batch_id,
            "query": query,
            "error_type": error_type,
            "error": error,
            "severity": "warning",
        }
        with self._failure_lock:
            self._failures.append(record)


def scrape_url_simple(url: str, max_chars: int = _MAX_CONTENT_CHARS) -> ScrapeResult:
    """Convenience function for one-off scrapes using httpx fallback.

    Uses plain HTTP GET when Firecrawl is not available. Does not render JS.
    """
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "URBIND-Research/1.0"})
            resp.raise_for_status()

        content = resp.text
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[...content truncated...]"

        return ScrapeResult(url=url, content=content, success=True)

    except Exception as exc:
        return ScrapeResult(url=url, success=False, error=str(exc))


__all__ = ["FirecrawlScraper", "ScrapeResult", "scrape_url_simple"]
