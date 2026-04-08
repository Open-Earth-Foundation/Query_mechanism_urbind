"""Google Custom Search Engine wrapper for web research."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
_DEFAULT_NUM_RESULTS = 5


class GoogleSearchResult:
    """A single search result from Google CSE."""

    __slots__ = ("title", "url", "snippet")

    def __init__(self, title: str, url: str, snippet: str) -> None:
        self.title = title
        self.url = url
        self.snippet = snippet

    def __repr__(self) -> str:
        return f"GoogleSearchResult(title={self.title!r}, url={self.url!r})"


class GoogleSearchClient:
    """Async-capable Google Custom Search client with quota tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        cse_id: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID", "")
        self._query_count = 0

    @property
    def query_count(self) -> int:
        return self._query_count

    def search(
        self,
        query: str,
        num_results: int = _DEFAULT_NUM_RESULTS,
    ) -> list[GoogleSearchResult]:
        """Execute a synchronous search query against Google CSE.

        Returns an empty list on any error.
        """
        if not self.api_key or not self.cse_id:
            logger.warning("Google CSE credentials not configured; skipping search.")
            return []

        params: dict[str, Any] = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(num_results, 10),
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(_CSE_ENDPOINT, params=params)
                resp.raise_for_status()

            self._query_count += 1
            data = resp.json()
            items = data.get("items", [])

            results: list[GoogleSearchResult] = []
            for item in items:
                results.append(GoogleSearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                ))

            logger.info(
                "Google CSE query=%r results=%d total_queries=%d",
                query,
                len(results),
                self._query_count,
            )
            return results

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Google CSE HTTP error: status=%d query=%r",
                exc.response.status_code,
                query,
            )
            return []
        except Exception:
            logger.warning("Google CSE search failed for query=%r", query, exc_info=True)
            return []


__all__ = ["GoogleSearchClient", "GoogleSearchResult"]
