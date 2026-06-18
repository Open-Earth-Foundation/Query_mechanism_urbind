"""Fail-fast readiness checks for provider-backed run features."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)

_SERPER_PROBE_ENDPOINT = "https://google.serper.dev/search"
_FIRECRAWL_PROBE_ENDPOINT = "https://api.firecrawl.dev/v1/scrape"
_FIRECRAWL_PROBE_URL = "https://example.com/"
_DEFAULT_CACHE_TTL_SECONDS = 300


@dataclass(frozen=True)
class ProviderCheckResult:
    """One cached provider readiness result."""

    ready: bool
    message: str
    checked_at: datetime


class FeatureReadinessService:
    """Validate provider-backed features before a run is accepted."""

    def __init__(self, cache_ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS) -> None:
        self._cache_ttl = timedelta(seconds=max(1, cache_ttl_seconds))
        self._lock = threading.Lock()
        self._serper_cache: dict[str, ProviderCheckResult] = {}
        self._firecrawl_cache: dict[str, ProviderCheckResult] = {}

    def validate_run_request(
        self,
        *,
        config: AppConfig,
        api_key_override: str | None,
    ) -> list[str]:
        """Return blocking readiness errors for one requested run."""
        errors: list[str] = []

        if not self._has_openrouter_api_key(api_key_override):
            errors.append(
                "This run cannot start because no OpenRouter API key is configured. "
                "Set OPENROUTER_API_KEY or send X-OpenRouter-Api-Key with the request."
            )

        if not self._is_web_research_requested(config):
            return errors

        serper_api_key = os.getenv("SERPER_API_KEY", "").strip()
        if not serper_api_key:
            errors.append(
                "Web research was requested, but SERPER_API_KEY is not configured. "
                "Add a valid Serper key or disable the web research sub-step."
            )
        else:
            serper_check = self._check_serper_api_key(serper_api_key)
            if not serper_check.ready:
                errors.append(serper_check.message)

        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
        if not firecrawl_api_key:
            errors.append(
                "Web research was requested, but FIRECRAWL_API_KEY is not configured. "
                "Add a valid Firecrawl key or disable the web research sub-step."
            )
        else:
            firecrawl_check = self._check_firecrawl_api_key(firecrawl_api_key)
            if not firecrawl_check.ready:
                errors.append(firecrawl_check.message)

        return errors

    def _has_openrouter_api_key(self, api_key_override: str | None) -> bool:
        """Return True when the run has an OpenRouter key available."""
        if api_key_override and api_key_override.strip():
            return True
        return bool(os.getenv("OPENROUTER_API_KEY", "").strip())

    def _is_web_research_requested(self, config: AppConfig) -> bool:
        """Return True when this run will execute the web research sub-step."""
        return bool(config.enrichment.enabled and config.enrichment.web_research_enabled)

    def _check_serper_api_key(self, api_key: str) -> ProviderCheckResult:
        """Return a cached Serper readiness result for one API key."""
        return self._check_cached_provider_result(
            cache=self._serper_cache,
            api_key=api_key,
            probe=self._probe_serper_api_key,
        )

    def _check_firecrawl_api_key(self, api_key: str) -> ProviderCheckResult:
        """Return a cached Firecrawl readiness result for one API key."""
        return self._check_cached_provider_result(
            cache=self._firecrawl_cache,
            api_key=api_key,
            probe=self._probe_firecrawl_api_key,
        )

    def _check_cached_provider_result(
        self,
        *,
        cache: dict[str, ProviderCheckResult],
        api_key: str,
        probe: callable,
    ) -> ProviderCheckResult:
        """Return a cached provider readiness result keyed by API-key hash."""
        cache_key = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        with self._lock:
            cached = cache.get(cache_key)
            if cached is not None and now - cached.checked_at < self._cache_ttl:
                return cached

        result = probe(api_key)
        with self._lock:
            cache[cache_key] = result
        return result

    def _probe_serper_api_key(self, api_key: str) -> ProviderCheckResult:
        """Probe Serper with a minimal request to catch invalid credentials early."""
        now = datetime.now(timezone.utc)
        try:
            response = httpx.post(
                _SERPER_PROBE_ENDPOINT,
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={"q": "test", "num": 1},
                timeout=10.0,
            )
        except httpx.TimeoutException:
            logger.warning("Serper readiness probe timed out.")
            return ProviderCheckResult(
                ready=False,
                message=(
                    "Web research is unavailable because the Serper readiness check timed out. "
                    "Retry after Serper is reachable or disable the web research sub-step."
                ),
                checked_at=now,
            )
        except Exception:
            logger.warning("Serper readiness probe failed.", exc_info=True)
            return ProviderCheckResult(
                ready=False,
                message=(
                    "Web research is unavailable because the Serper readiness check failed. "
                    "Verify Serper connectivity or disable the web research sub-step."
                ),
                checked_at=now,
            )

        if response.status_code in {401, 403}:
            logger.warning(
                "Serper readiness probe rejected the configured API key status=%d",
                response.status_code,
            )
            return ProviderCheckResult(
                ready=False,
                message=(
                    "Web research is unavailable because the configured SERPER_API_KEY was "
                    f"rejected by Serper (HTTP {response.status_code}). Update the key or "
                    "disable the web research sub-step."
                ),
                checked_at=now,
            )

        if response.status_code >= 400:
            logger.warning(
                "Serper readiness probe returned unexpected status=%d body=%s",
                response.status_code,
                response.text[:300],
            )
            return ProviderCheckResult(
                ready=False,
                message=(
                    "Web research is unavailable because the Serper readiness check returned "
                    f"HTTP {response.status_code}. Retry after Serper is healthy or disable "
                    "the web research sub-step."
                ),
                checked_at=now,
            )

        return ProviderCheckResult(
            ready=True,
            message="Serper is reachable.",
            checked_at=now,
        )

    def _probe_firecrawl_api_key(self, api_key: str) -> ProviderCheckResult:
        """Probe Firecrawl with a minimal scrape request to catch invalid credentials."""
        now = datetime.now(timezone.utc)
        try:
            response = httpx.post(
                _FIRECRAWL_PROBE_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "url": _FIRECRAWL_PROBE_URL,
                    "formats": ["markdown"],
                },
                timeout=15.0,
            )
        except httpx.TimeoutException:
            logger.warning("Firecrawl readiness probe timed out.")
            return ProviderCheckResult(
                ready=False,
                message=(
                    "Web research is unavailable because the Firecrawl readiness check "
                    "timed out. Retry after Firecrawl is reachable or disable the web "
                    "research sub-step."
                ),
                checked_at=now,
            )
        except Exception:
            logger.warning("Firecrawl readiness probe failed.", exc_info=True)
            return ProviderCheckResult(
                ready=False,
                message=(
                    "Web research is unavailable because the Firecrawl readiness check "
                    "failed. Verify Firecrawl connectivity or disable the web research "
                    "sub-step."
                ),
                checked_at=now,
            )

        if response.status_code in {401, 403}:
            logger.warning(
                "Firecrawl readiness probe rejected the configured API key status=%d",
                response.status_code,
            )
            return ProviderCheckResult(
                ready=False,
                message=(
                    "Web research is unavailable because the configured FIRECRAWL_API_KEY "
                    f"was rejected by Firecrawl (HTTP {response.status_code}). Update the "
                    "key or disable the web research sub-step."
                ),
                checked_at=now,
            )

        if response.status_code >= 400:
            logger.warning(
                "Firecrawl readiness probe returned unexpected status=%d body=%s",
                response.status_code,
                response.text[:300],
            )
            return ProviderCheckResult(
                ready=False,
                message=(
                    "Web research is unavailable because the Firecrawl readiness check "
                    f"returned HTTP {response.status_code}. Retry after Firecrawl is healthy "
                    "or disable the web research sub-step."
                ),
                checked_at=now,
            )

        return ProviderCheckResult(
            ready=True,
            message="Firecrawl is reachable.",
            checked_at=now,
        )


__all__ = ["FeatureReadinessService", "ProviderCheckResult"]
