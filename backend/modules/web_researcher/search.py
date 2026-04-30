"""Serper.dev web search wrapper for web research."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SERPER_ENDPOINT = "https://google.serper.dev/search"
_DEFAULT_NUM_RESULTS = 5


class SearchResult:
    """A single web search result."""

    __slots__ = ("title", "url", "snippet")

    def __init__(self, title: str, url: str, snippet: str) -> None:
        self.title = title
        self.url = url
        self.snippet = snippet

    def __repr__(self) -> str:
        return f"SearchResult(title={self.title!r}, url={self.url!r})"


class SerperSearchClient:
    """Serper.dev search client with quota tracking.

    Uses the Serper Google Search API for full web search results.
    Requires a single ``SERPER_API_KEY`` environment variable.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("SERPER_API_KEY", "")
        self._query_count = 0
        # Circuit breaker: once Serper reports the account is out of credits
        # we stop calling it for the rest of this client's lifetime.  Otherwise
        # one credit-exhausted run hammers the API with hundreds of failures.
        self._credits_exhausted = False

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def credits_exhausted(self) -> bool:
        return self._credits_exhausted

    def search(
        self,
        query: str,
        num_results: int = _DEFAULT_NUM_RESULTS,
    ) -> list[SearchResult]:
        """Execute a synchronous search query against Serper.

        Returns an empty list on any error.
        """
        if not self.api_key:
            logger.warning("Serper API key not configured; skipping search.")
            return []
        if self._credits_exhausted:
            return []

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "q": query,
            "num": min(num_results, 10),
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(_SERPER_ENDPOINT, headers=headers, json=payload)
                resp.raise_for_status()

            self._query_count += 1
            data = resp.json()
            organic = data.get("organic", [])

            results: list[SearchResult] = []
            for item in organic:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                ))

            logger.info(
                "Serper query=%r results=%d total_queries=%d",
                query,
                len(results),
                self._query_count,
            )
            return results

        except httpx.HTTPStatusError as exc:
            # Capture the response body so 4xx errors are diagnosable —
            # Serper returns the actual reason (auth, quota, malformed) here.
            try:
                body = exc.response.text[:500]
            except Exception:  # noqa: BLE001
                body = "<unreadable>"
            # Trip the circuit breaker on credit exhaustion so the rest of
            # the run doesn't hammer the API with hundreds of failures.
            if "not enough credits" in body.lower() or "credits" in body.lower() and exc.response.status_code == 400:
                if not self._credits_exhausted:
                    logger.error(
                        "Serper credits exhausted — disabling further Serper calls "
                        "for this run.  Top up at https://serper.dev/billing or set "
                        "enrichment.tier1_first_search=false to reduce call volume."
                    )
                self._credits_exhausted = True
            logger.warning(
                "Serper HTTP error: status=%d query=%r body=%s",
                exc.response.status_code,
                query,
                body,
            )
            return []
        except Exception:
            logger.warning("Serper search failed for query=%r", query, exc_info=True)
            return []


# Backward-compatible aliases
GoogleSearchResult = SearchResult
GoogleSearchClient = SerperSearchClient

__all__ = ["SerperSearchClient", "SearchResult", "GoogleSearchClient", "GoogleSearchResult"]
