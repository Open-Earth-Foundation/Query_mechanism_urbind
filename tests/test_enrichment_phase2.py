"""Tests for Phase 2 web research components."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.modules.web_researcher.deep_diver import (
    DeepDiver,
    _extract_domain,
    _extract_links_from_markdown,
    _normalize_url,
)
from backend.modules.web_researcher.extractor import extract_fields_from_content
from backend.modules.web_researcher.freshness import check_freshness, _extract_ccc_evidence
from backend.modules.web_researcher.models import (
    FreshnessResult,
    SearchBatch,
    WebFinding,
)
from backend.modules.web_researcher.relevance import check_relevance_batch
from backend.modules.web_researcher.scraper import (
    FirecrawlScraper,
    ScrapeResult,
    scrape_url_simple,
)
from backend.modules.web_researcher.search import SerperSearchClient, SearchResult
from backend.modules.web_researcher.search_planner import (
    plan_searches,
    _compute_budget,
)
from backend.modules.web_researcher.search_worker import (
    execute_search_batch,
    execute_search_batches,
)
from backend.modules.web_researcher.models import GapManifest, CityGap, FieldClassification
from tests.support import build_test_app_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_openai_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_config(**overrides: Any) -> Any:
    return build_test_app_config(
        enrichment_overrides={"enabled": True, "model": "test-model", **overrides}
    )


# ---------------------------------------------------------------------------
# SerperSearchClient
# ---------------------------------------------------------------------------


class TestSerperSearchClient:
    def test_no_credentials_returns_empty(self) -> None:
        client = SerperSearchClient(api_key="")
        results = client.search("test query")
        assert results == []
        assert client.query_count == 0

    def test_successful_search(self) -> None:
        client = SerperSearchClient(api_key="key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {"title": "Test", "link": "https://example.com", "snippet": "A result"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("backend.modules.web_researcher.search.httpx.Client") as mock_httpx:
            mock_httpx_instance = MagicMock()
            mock_httpx_instance.__enter__ = MagicMock(return_value=mock_httpx_instance)
            mock_httpx_instance.__exit__ = MagicMock(return_value=False)
            mock_httpx_instance.post.return_value = mock_response
            mock_httpx.return_value = mock_httpx_instance

            results = client.search("Dresden electric bus fleet")
            assert len(results) == 1
            assert results[0].title == "Test"
            assert results[0].url == "https://example.com"
            assert client.query_count == 1
            assert client.successful_query_count == 1


# ---------------------------------------------------------------------------
# FirecrawlScraper
# ---------------------------------------------------------------------------


class TestFirecrawlScraper:
    def test_no_api_key_returns_failure(self) -> None:
        with patch.dict("os.environ", {"FIRECRAWL_API_KEY": ""}, clear=False):
            scraper = FirecrawlScraper(api_key="")
            result = scraper.scrape("https://example.com")
            assert result.success is False
            assert result.error == "no_api_key"

    def test_content_truncation(self) -> None:
        scraper = FirecrawlScraper(api_key="key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "x" * 10000,
                "metadata": {"title": "Test"},
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch("backend.modules.web_researcher.scraper.httpx.Client") as mock_httpx:
            mock_httpx_instance = MagicMock()
            mock_httpx_instance.__enter__ = MagicMock(return_value=mock_httpx_instance)
            mock_httpx_instance.__exit__ = MagicMock(return_value=False)
            mock_httpx_instance.post.return_value = mock_response
            mock_httpx.return_value = mock_httpx_instance

            result = scraper.scrape("https://example.com", max_chars=100)
            assert result.success is True
            assert len(result.content) < 200  # truncated + notice
            assert "[...content truncated...]" in result.content

    def test_timeout_records_structured_failure(self) -> None:
        scraper = FirecrawlScraper(api_key="key")

        with patch("backend.modules.web_researcher.scraper.httpx.Client") as mock_httpx:
            mock_httpx_instance = MagicMock()
            mock_httpx_instance.__enter__ = MagicMock(return_value=mock_httpx_instance)
            mock_httpx_instance.__exit__ = MagicMock(return_value=False)
            mock_httpx_instance.post.side_effect = httpx.ReadTimeout(
                "The read operation timed out"
            )
            mock_httpx.return_value = mock_httpx_instance

            result = scraper.scrape(
                "https://www.researchgate.net/publication/test",
                query="Aachen geothermal permit",
                batch_id="batch_001",
            )

        assert result.success is False
        failures = scraper.scrape_failures
        assert failures == [
            {
                "url": "https://www.researchgate.net/publication/test",
                "domain": "www.researchgate.net",
                "provider": "firecrawl",
                "batch_id": "batch_001",
                "query": "Aachen geothermal permit",
                "error_type": "ReadTimeout",
                "error": "The read operation timed out",
                "severity": "warning",
            }
        ]


# ---------------------------------------------------------------------------
# Deep Diver
# ---------------------------------------------------------------------------


class TestDeepDiver:
    def test_extract_domain(self) -> None:
        assert _extract_domain("https://www.example.com/page") == "www.example.com"
        assert _extract_domain("https://EXAMPLE.COM/page") == "example.com"

    def test_normalize_url(self) -> None:
        assert _normalize_url("https://example.com/page/") == "https://example.com/page"
        assert _normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_extract_links_from_markdown(self) -> None:
        content = "See [report](https://example.com/report) and [other](/other)"
        links = _extract_links_from_markdown(content, "example.com", "https://example.com")
        assert len(links) == 2
        assert "https://example.com/report" in links

    def test_dive_limit_respected(self) -> None:
        scraper = MagicMock(spec=FirecrawlScraper)
        diver = DeepDiver(scraper=scraper, max_dives_per_run=1)

        scraper.scrape.return_value = ScrapeResult(
            url="https://example.com", content="test", success=True
        )
        result1 = diver.dive("https://example.com")
        assert len(result1.pages) >= 0

        # Second dive should be blocked
        result2 = diver.dive("https://other.com")
        assert len(result2.pages) == 0
        assert diver.dives_remaining == 0


# ---------------------------------------------------------------------------
# Search Planner
# ---------------------------------------------------------------------------


class TestSearchPlanner:
    def test_empty_gaps_returns_empty(self) -> None:
        manifest = GapManifest(query_fields=[], city_gaps=[], non_estimable_fields=[])
        config = _make_config()
        batches = plan_searches(manifest, config, "key")
        assert batches == []

    def test_compute_budget_missing_entirely(self) -> None:
        config = _make_config()
        budget = _compute_budget(
            blank_count=5, stale_count=0, city_count=2,
            priority="high", config=config,
        )
        assert budget["deep_dive_allowed"] is True
        assert budget["max_queries"] <= 15

    def test_compute_budget_stale_only(self) -> None:
        config = _make_config()
        budget = _compute_budget(
            blank_count=0, stale_count=3, city_count=1,
            priority="medium", config=config,
        )
        assert budget["deep_dive_allowed"] is False

    def test_plan_searches_with_mocked_llm(self) -> None:
        config = _make_config()
        manifest = GapManifest(
            query_fields=[
                FieldClassification(
                    field="total_capex",
                    classification="estimable_numerical",
                    searchable=True,
                    rationale="R",
                ),
            ],
            city_gaps=[
                CityGap(
                    city="Dresden",
                    blank_fields=["total_capex"],
                    stale_flags=[],
                    search_priority="high",
                ),
            ],
            non_estimable_fields=[],
        )
        mock_resp = _mock_openai_response(
            '["Dresden electric bus fleet capex 2024"]'
        )
        with patch("backend.modules.web_researcher.search_planner.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_cls.return_value = mock_client

            batches = plan_searches(manifest, config, "key")
            assert len(batches) >= 1
            assert len(batches[0].queries) >= 1


# ---------------------------------------------------------------------------
# Relevance Checker
# ---------------------------------------------------------------------------


class TestRelevanceChecker:
    def test_empty_results_returns_empty(self) -> None:
        config = _make_config()
        result = check_relevance_batch([], ["capex"], ["Dresden"], config, "key")
        assert result == []

    def test_relevance_with_mocked_llm(self) -> None:
        config = _make_config()
        search_results = [
            SearchResult("Electric bus report", "https://example.com", "50M EUR budget"),
        ]
        mock_resp = _mock_openai_response(
            '[{"index": 0, "is_relevant": true, "reason": "Contains budget data"}]'
        )
        with patch("backend.modules.web_researcher.relevance.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_cls.return_value = mock_client

            result = check_relevance_batch(
                search_results, ["total_capex"], ["Dresden"], config, "key"
            )
            assert len(result) == 1
            assert result[0][1] is True


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class TestExtractor:
    def test_empty_content_returns_empty(self) -> None:
        config = _make_config()
        result = extract_fields_from_content("", "https://x.com", ["capex"], ["Dresden"], config, "k")
        assert result == []

    def test_extraction_with_mocked_llm(self) -> None:
        config = _make_config()
        mock_resp = _mock_openai_response(json.dumps({
            "findings": [{
                "city": "Dresden",
                "field": "total_capex",
                "value": 50000000,
                "unit": "EUR",
                "source_type": "government_report",
                "source_date": "2024",
                "extraction_confidence": 0.9,
            }]
        }))
        with patch("backend.modules.web_researcher.extractor.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_cls.return_value = mock_client

            result = extract_fields_from_content(
                "Dresden allocated 50M EUR for electric buses.",
                "https://example.com/report",
                ["total_capex"],
                ["Dresden"],
                config,
                "key",
            )
            assert len(result) == 1
            assert result[0].city == "Dresden"
            assert result[0].value == 50000000

    def test_extraction_rejects_wrong_city(self) -> None:
        config = _make_config()
        mock_resp = _mock_openai_response(json.dumps({
            "findings": [{
                "city": "Berlin",
                "field": "total_capex",
                "value": 100,
                "unit": "EUR",
                "source_type": "other",
                "extraction_confidence": 0.5,
            }]
        }))
        with patch("backend.modules.web_researcher.extractor.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_cls.return_value = mock_client

            result = extract_fields_from_content(
                "content", "https://x.com", ["total_capex"], ["Dresden"], config, "key"
            )
            assert result == []


# ---------------------------------------------------------------------------
# Freshness Checker
# ---------------------------------------------------------------------------


class TestFreshnessChecker:
    def test_no_findings_returns_empty(self) -> None:
        config = _make_config()
        result = check_freshness([], {}, config, "key")
        assert result == []

    def test_no_ccc_overlap_returns_empty(self) -> None:
        config = _make_config()
        findings = [WebFinding(
            city="Dresden", field="capex", value=50, source_url="https://x.com",
            source_type="report", extraction_confidence=0.9,
        )]
        # Context bundle with no matching CCC data
        result = check_freshness(findings, {"markdown": None}, config, "key")
        assert result == []

    def test_extract_ccc_evidence(self) -> None:
        context = {
            "markdown": {
                "excerpts": [
                    {"city_key": "Dresden", "partial_answer": "Dresden allocated 50M EUR to climate capex."},
                ]
            }
        }
        evidence = _extract_ccc_evidence(context)
        assert "dresden" in evidence
        assert evidence["dresden"][0].startswith("Dresden allocated 50M EUR")


# ---------------------------------------------------------------------------
# Search Worker (integration-level with mocks)
# ---------------------------------------------------------------------------


class TestSearchWorker:
    def test_execute_search_batches_empty(self) -> None:
        config = _make_config()
        result = execute_search_batches([], config, "key")
        assert result == []

    def test_execute_search_batch_with_mocked_infra(self) -> None:
        config = _make_config()
        batch = SearchBatch(
            batch_id="test_001",
            cities=["Dresden"],
            target_fields=["total_capex"],
            search_type="missing_entirely",
            queries=["Dresden electric bus capex"],
            budget={"max_queries": 5, "deep_dive_allowed": False, "max_pages_per_deep_dive": 10},
            priority="high",
        )

        # Mock all infrastructure
        search_client = MagicMock(spec=SerperSearchClient)
        search_client.search.return_value = [
            SearchResult("Report", "https://example.com/report", "50M EUR"),
        ]

        scraper = MagicMock(spec=FirecrawlScraper)
        scraper.scrape.return_value = ScrapeResult(
            url="https://example.com/report",
            content="Dresden allocated 50M EUR for electric buses.",
            success=True,
        )

        deep_diver = MagicMock(spec=DeepDiver)
        deep_diver.can_dive.return_value = False

        # Mock relevance checker (all relevant)
        with patch("backend.modules.web_researcher.search_worker.check_relevance_batch") as mock_rel:
            mock_rel.return_value = [
                (SearchResult("Report", "https://example.com/report", "50M EUR"), True),
            ]
            # Mock extractor
            with patch("backend.modules.web_researcher.search_worker.extract_fields_from_content") as mock_ext:
                mock_ext.return_value = [
                    WebFinding(
                        city="Dresden", field="total_capex", value=50000000,
                        source_url="https://example.com/report",
                        source_type="government_report",
                        extraction_confidence=0.9,
                    ),
                ]

                findings = execute_search_batch(
                    batch, search_client, scraper, deep_diver, config, "key"
                )

        assert len(findings) == 1
        assert findings[0].city == "Dresden"
        assert findings[0].value == 50000000
