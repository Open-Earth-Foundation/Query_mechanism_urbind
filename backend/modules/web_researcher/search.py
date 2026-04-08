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

    @property
    def query_count(self) -> int:
        return self._query_count

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
            logger.warning(
                "Serper HTTP error: status=%d query=%r",
                exc.response.status_code,
                query,
            )
            return []
        except Exception:
            logger.warning("Serper search failed for query=%r", query, exc_info=True)
            return []


# Backward-compatible aliases
GoogleSearchResult = SearchResult
GoogleSearchClient = SerperSearchClient

__all__ = ["SerperSearchClient", "SearchResult", "GoogleSearchClient", "GoogleSearchResult"]
