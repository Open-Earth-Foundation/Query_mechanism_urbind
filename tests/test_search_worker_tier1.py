"""Tests for the search worker's tier-1-first behaviour."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from backend.modules.web_researcher.models import SearchBatch, WebFinding
from backend.modules.web_researcher.search import SearchResult
from backend.modules.web_researcher.search_worker import execute_search_batch
from tests.support import build_test_app_config


# ---------------------------------------------------------------------------
# Allowlist fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def allowlist_yaml(tmp_path: Path, monkeypatch) -> Path:
    """Write a minimal tier-1 allowlist and patch the loader's default path."""
    payload = {
        "version": 1,
        "sources": [
            {
                "id": "bnetza_ladekarte",
                "name": "Bnetza",
                "domain": "bundesnetzagentur.de",
                "urls": ["https://bundesnetzagentur.de/foo"],
                "access": "site_search",
                "coverage": {
                    "countries": ["DE"],
                    "fields": ["public_dc_charger_count"],
                },
            },
            {
                "id": "auth_walled",
                "name": "Walled",
                "domain": "netzerocities.app",
                "access": "auth_required",
                "coverage": {"scope": ["ccc_documents"]},
            },
        ],
    }
    target = tmp_path / "tier1.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")

    monkeypatch.setattr(
        "backend.modules.web_researcher.search_worker.load_tier1_web_allowlist",
        lambda path=None: __import__(
            "backend.modules.web_researcher.tier1_web", fromlist=["load_tier1_web_allowlist"]
        ).load_tier1_web_allowlist(target),
    )
    return target


def _config(*, tier1: bool, threshold: float = 0.6):
    return build_test_app_config(
        enrichment_overrides={
            "enabled": True,
            "tier1_first_search": tier1,
            "tier1_confidence_threshold": threshold,
            "max_retries_per_worker": 0,
            "max_deep_dives_per_run": 0,
            "max_pages_per_deep_dive": 0,
        }
    )


def _batch() -> SearchBatch:
    return SearchBatch(
        batch_id="b1",
        cities=["Dresden"],
        target_fields=["public_dc_charger_count"],
        search_type="missing_entirely",
        queries=["Dresden DC charger count 2024"],
        budget={"max_queries": 5, "deep_dive_allowed": False},
        priority="high",
    )


def _scrape_success(content: str = "fake content"):
    obj = MagicMock()
    obj.success = True
    obj.content = content
    return obj


def _make_finding(value: float, confidence: float = 0.9) -> WebFinding:
    return WebFinding(
        city="Dresden",
        field="public_dc_charger_count",
        value=value,
        unit="stations",
        source_url="https://example.com/x",
        source_type="government_report",
        extraction_confidence=confidence,
    )


def _patch_relevance_pass_through() -> None:
    """The relevance check returns (result, True) for everything we hand it."""
    return patch(
        "backend.modules.web_researcher.search_worker.check_relevance_batch",
        side_effect=lambda results, *a, **k: [(r, True) for r in results],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_legacy_path_unchanged_when_flag_off(allowlist_yaml: Path) -> None:
    """With the flag off, the worker behaves exactly as before — one open search per query."""
    config = _config(tier1=False)
    batch = _batch()
    search_client = MagicMock()
    search_client.search.return_value = [
        SearchResult(title="t", url="https://example.com/x", snippet="s")
    ]
    scraper = MagicMock()
    scraper.scrape.return_value = _scrape_success()

    with _patch_relevance_pass_through(), patch(
        "backend.modules.web_researcher.search_worker.extract_fields_from_content",
        return_value=[_make_finding(100.0)],
    ), patch(
        "backend.modules.web_researcher.search_worker.validate_findings",
        side_effect=lambda findings, cities: findings,
    ):
        findings = execute_search_batch(
            batch, search_client, scraper, MagicMock(), config, api_key="k"
        )

    # Exactly one search call for the single query, no site: prefix.
    queries_called = [c.args[0] for c in search_client.search.call_args_list]
    assert queries_called == ["Dresden DC charger count 2024"]
    assert len(findings) == 1
    # No tier tagging on the legacy path.
    assert findings[0].source_tier is None
    assert findings[0].source_id is None


def test_tier1_skips_open_pass_when_high_confidence_match(
    allowlist_yaml: Path,
) -> None:
    """Tier-1 high-confidence finding → no open Serper call for that query."""
    config = _config(tier1=True, threshold=0.6)
    batch = _batch()
    search_client = MagicMock()
    search_client.search.return_value = [
        SearchResult(
            title="Bnetza Dresden", url="https://bundesnetzagentur.de/dresden", snippet="s"
        )
    ]
    scraper = MagicMock()
    scraper.scrape.return_value = _scrape_success()

    with _patch_relevance_pass_through(), patch(
        "backend.modules.web_researcher.search_worker.extract_fields_from_content",
        return_value=[_make_finding(159.0, confidence=0.9)],
    ), patch(
        "backend.modules.web_researcher.search_worker.validate_findings",
        side_effect=lambda findings, cities: findings,
    ):
        findings = execute_search_batch(
            batch, search_client, scraper, MagicMock(), config, api_key="k"
        )

    queries_called = [c.args[0] for c in search_client.search.call_args_list]
    # Tier-1 site query was issued; open follow-up was skipped.
    assert any(q.startswith("site:bundesnetzagentur.de") for q in queries_called)
    assert "Dresden DC charger count 2024" not in queries_called

    assert len(findings) == 1
    assert findings[0].source_tier == "tier1"
    assert findings[0].source_id == "bnetza_ladekarte"


def test_tier1_falls_through_when_no_findings(allowlist_yaml: Path) -> None:
    """Tier-1 returns nothing → open Serper still runs for the query."""
    config = _config(tier1=True, threshold=0.6)
    batch = _batch()
    search_client = MagicMock()

    def search_side_effect(query: str):
        if query.startswith("site:"):
            return []  # tier-1 returns nothing
        return [SearchResult(title="t", url="https://example.com/x", snippet="s")]

    search_client.search.side_effect = search_side_effect
    scraper = MagicMock()
    scraper.scrape.return_value = _scrape_success()

    with _patch_relevance_pass_through(), patch(
        "backend.modules.web_researcher.search_worker.extract_fields_from_content",
        return_value=[_make_finding(180.0, confidence=0.8)],
    ), patch(
        "backend.modules.web_researcher.search_worker.validate_findings",
        side_effect=lambda findings, cities: findings,
    ):
        findings = execute_search_batch(
            batch, search_client, scraper, MagicMock(), config, api_key="k"
        )

    queries_called = [c.args[0] for c in search_client.search.call_args_list]
    # Both tier-1 site query and the open query were issued.
    assert any(q.startswith("site:bundesnetzagentur.de") for q in queries_called)
    assert "Dresden DC charger count 2024" in queries_called

    assert len(findings) == 1
    assert findings[0].source_tier == "open"
    assert findings[0].source_id is None


def test_tier1_runs_open_when_confidence_below_threshold(
    allowlist_yaml: Path,
) -> None:
    """Low-confidence tier-1 finding doesn't satisfy the gap → open pass runs."""
    config = _config(tier1=True, threshold=0.9)
    batch = _batch()
    search_client = MagicMock()

    def search_side_effect(query: str):
        if query.startswith("site:"):
            return [
                SearchResult(
                    title="Bnetza",
                    url="https://bundesnetzagentur.de/dresden",
                    snippet="s",
                )
            ]
        return [
            SearchResult(
                title="Other", url="https://other.example.com/page", snippet="s"
            )
        ]

    search_client.search.side_effect = search_side_effect

    extract_results = [
        [_make_finding(159.0, confidence=0.5)],  # tier-1 (low confidence)
        [_make_finding(160.0, confidence=0.8)],  # open
    ]
    scraper = MagicMock()
    scraper.scrape.return_value = _scrape_success()

    with _patch_relevance_pass_through(), patch(
        "backend.modules.web_researcher.search_worker.extract_fields_from_content",
        side_effect=extract_results,
    ), patch(
        "backend.modules.web_researcher.search_worker.validate_findings",
        side_effect=lambda findings, cities: findings,
    ):
        findings = execute_search_batch(
            batch, search_client, scraper, MagicMock(), config, api_key="k"
        )

    queries_called = [c.args[0] for c in search_client.search.call_args_list]
    assert any(q.startswith("site:bundesnetzagentur.de") for q in queries_called)
    # Open pass DID run because tier-1 didn't clear the threshold.
    assert "Dresden DC charger count 2024" in queries_called
    # Both findings present (tier-1 + open, neither was duplicated; URLs differ).
    tiers = [f.source_tier for f in findings]
    assert "tier1" in tiers
    assert "open" in tiers


def test_tier1_skips_auth_walled_sources(allowlist_yaml: Path) -> None:
    """Sources with access=auth_required are not site-searched."""
    config = _config(tier1=True)
    batch = _batch()
    search_client = MagicMock()
    search_client.search.return_value = []
    scraper = MagicMock()

    with _patch_relevance_pass_through():
        execute_search_batch(
            batch, search_client, scraper, MagicMock(), config, api_key="k"
        )

    queries_called = [c.args[0] for c in search_client.search.call_args_list]
    # Bnetza pre-pass query expected; netzerocities.app should not appear.
    assert any("bundesnetzagentur.de" in q for q in queries_called)
    assert not any("netzerocities.app" in q for q in queries_called)


def test_tier1_only_runs_for_matching_coverage(allowlist_yaml: Path) -> None:
    """A field outside the curated coverage doesn't trigger any tier-1 query."""
    config = _config(tier1=True)
    batch = SearchBatch(
        batch_id="b1",
        cities=["Dresden"],
        target_fields=["unrelated_field_xyz"],
        search_type="missing_entirely",
        queries=["unrelated query"],
        budget={"max_queries": 5, "deep_dive_allowed": False},
        priority="high",
    )
    search_client = MagicMock()
    search_client.search.return_value = []
    scraper = MagicMock()

    with _patch_relevance_pass_through():
        execute_search_batch(
            batch, search_client, scraper, MagicMock(), config, api_key="k"
        )

    queries_called = [c.args[0] for c in search_client.search.call_args_list]
    # No site: prefix at all — tier-1 didn't match.
    assert not any(q.startswith("site:") for q in queries_called)
    # Open query still ran.
    assert "unrelated query" in queries_called
