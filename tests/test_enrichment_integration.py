"""Integration tests for the enrichment pipeline and config loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.modules.web_researcher.models import SearchBatch
from backend.modules.web_researcher.module import run_enrichment_pipeline
from backend.services.run_logger import RunLogger
from backend.utils.artifact_writer import stage_file_dir_name
from backend.utils.config import EnrichmentConfig, load_config
from backend.utils.paths import create_run_paths
from tests.support import build_test_app_config


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestEnrichmentConfig:
    def test_default_config_has_enrichment_disabled(self) -> None:
        config = build_test_app_config()
        assert config.enrichment.enabled is False
        assert config.enrichment.web_research_enabled is False
        assert config.enrichment.model == "openai/gpt-5.4-mini"

    def test_config_overrides_via_build_test(self) -> None:
        config = build_test_app_config(
            enrichment_overrides={"enabled": True, "max_workers": 4}
        )
        assert config.enrichment.enabled is True
        assert config.enrichment.max_workers == 4

    def test_enrichment_config_defaults(self) -> None:
        ec = EnrichmentConfig(model="test-model")
        assert ec.enabled is False
        assert ec.web_research_enabled is False
        assert ec.max_workers == 6
        assert ec.max_queries_per_batch == 10
        assert ec.max_total_queries_per_run == 50
        assert ec.max_retries_per_worker == 2
        assert ec.max_deep_dives_per_run == 3
        assert ec.max_pages_per_deep_dive == 10
        assert ec.freshness_threshold_days == 730
        assert ec.max_fields_per_query == 20
        assert ec.assumptions_estimator_model == ""
        assert ec.assumptions_estimator_temperature == 0.0

    def test_load_config_with_enrichment_section(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENRICHMENT_ENABLED", "")
        monkeypatch.setenv("WEB_RESEARCH_ENABLED", "")
        config_path = tmp_path / "llm_config.yaml"
        config_path.write_text(
            "\n".join([
                "orchestrator:",
                "  model: test-model",
                "markdown_researcher:",
                "  model: test-model",
                "  chunk_overlap_tokens: 2000",
                "  batch_max_chunks: 32",
                "  max_workers: 8",
                "  request_backoff_base_seconds: 0.5",
                "  request_backoff_max_seconds: 2.0",
                "writer:",
                "  model: test-model",
                "chat:",
                "  model: openai/gpt-5.4-mini",
                "  provider_timeout_seconds: 60.0",
                "  followup_router_max_excerpts_per_source: 50",
                "assumptions_reviewer:",
                "  model: openai/gpt-5.4-mini",
                "enrichment:",
                "  model: openai/gpt-5.4-mini",
                "  enabled: true",
                "  web_research_enabled: false",
                "  max_workers: 4",
                "retry:",
                "  backoff_base_seconds: 1.0",
                "  backoff_max_seconds: 30.0",
            ]),
            encoding="utf-8",
        )
        config = load_config(config_path)
        assert config.enrichment.enabled is True
        assert config.enrichment.web_research_enabled is False
        assert config.enrichment.max_workers == 4

    def test_env_var_overrides_enrichment_enabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "llm_config.yaml"
        config_path.write_text(
            "\n".join([
                "orchestrator:",
                "  model: test-model",
                "markdown_researcher:",
                "  model: test-model",
                "  chunk_overlap_tokens: 2000",
                "  batch_max_chunks: 32",
                "  max_workers: 8",
                "  request_backoff_base_seconds: 0.5",
                "  request_backoff_max_seconds: 2.0",
                "writer:",
                "  model: test-model",
                "chat:",
                "  model: openai/gpt-5.4-mini",
                "  provider_timeout_seconds: 60.0",
                "  followup_router_max_excerpts_per_source: 50",
                "assumptions_reviewer:",
                "  model: openai/gpt-5.4-mini",
                "retry:",
                "  backoff_base_seconds: 1.0",
                "  backoff_max_seconds: 30.0",
            ]),
            encoding="utf-8",
        )
        monkeypatch.setenv("ENRICHMENT_ENABLED", "true")
        monkeypatch.setenv("WEB_RESEARCH_ENABLED", "1")
        config = load_config(config_path)
        assert config.enrichment.enabled is True
        assert config.enrichment.web_research_enabled is True


# ---------------------------------------------------------------------------
# Pipeline integration (mocked LLM calls)
# ---------------------------------------------------------------------------


def _mock_gap_analysis_response() -> dict[str, Any]:
    """Return a valid gap manifest JSON response."""
    return {
        "query_fields": [
            {
                "field": "total_capex",
                "classification": "estimable_numerical",
                "searchable": True,
                "rationale": "Concrete cost figure",
            },
            {
                "field": "operator_name",
                "classification": "non_estimable",
                "searchable": False,
                "rationale": "Unique to city",
            },
        ],
        "city_gaps": [
            {
                "city": "Dresden",
                "blank_fields": ["total_capex"],
                "stale_flags": [],
                "search_priority": "high",
            },
        ],
        "non_estimable_fields": ["operator_name"],
    }


def _mock_assumptions_response() -> dict[str, Any]:
    """Return a valid assumptions envelope JSON response."""
    return {
        "assumptions": [
            {
                "city": "Dresden",
                "field_name": "total_capex",
                "gap_description": "Missing total CAPEX for Dresden",
                "method_used": "peer_city_proxy",
                "estimate": {"low": 40000000, "mid": 50000000, "high": 60000000},
                "confidence": "MEDIUM",
                "reference_data": "Munich: 55M EUR, Hamburg: 48M EUR",
                "rationale": "Based on similar German cities with comparable fleet sizes",
                "basis": "Peer city comparison using 3 German cities",
                "is_replaceable": True,
            }
        ],
        "non_estimable": [],
    }


def _make_mock_openai_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestEnrichmentPipeline:
    def test_pipeline_with_mocked_llm(self, tmp_path: Path) -> None:
        """Full pipeline run with mocked LLM calls."""
        config = build_test_app_config(
            enrichment_overrides={"enabled": True, "model": "test-model"}
        )
        run_paths = create_run_paths(tmp_path, "run_test", "context_bundle.json")
        run_logger = RunLogger(run_paths, "What is the total CAPEX for Dresden?")
        context_bundle = {
            "markdown": {"status": "success", "excerpts": []},
            "research_question": "What is the total CAPEX for Dresden?",
        }

        gap_response = _make_mock_openai_response(
            json.dumps(_mock_gap_analysis_response())
        )
        assumptions_response = _make_mock_openai_response(
            json.dumps(_mock_assumptions_response())
        )

        call_count = 0

        def mock_create(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            # First two calls are gap analysis (1 attempt + possible retry)
            # Then assumptions estimator calls
            if call_count <= 1:
                return gap_response
            return assumptions_response

        with patch("backend.modules.web_researcher.gap_analysis.OpenAI") as mock_gap_cls, \
             patch("backend.modules.web_researcher.assumptions_estimator.OpenAI") as mock_est_cls:
            mock_gap_client = MagicMock()
            mock_gap_client.chat.completions.create = mock_create
            mock_gap_cls.return_value = mock_gap_client

            mock_est_client = MagicMock()
            mock_est_client.chat.completions.create.return_value = assumptions_response
            mock_est_cls.return_value = mock_est_client

            result = run_enrichment_pipeline(
                question="What is the total CAPEX for Dresden?",
                context_bundle=context_bundle,
                base_dir=run_paths.base_dir,
                run_logger=run_logger,
                config=config,
                api_key="test-key",
            )

        assert "enrichment" in result
        enrichment = result["enrichment"]
        assert "gap_manifest" in enrichment
        assert "assumptions" not in enrichment
        assert "non_estimable" not in enrichment
        assert "assumptions" in result
        assumptions_payload = result["assumptions"]
        assert isinstance(assumptions_payload, dict)
        assert "meta" in enrichment
        assert enrichment["meta"]["gap_analyst_model"] == "test-model"

        # Single-city query with no peer data: the anchor sufficiency check
        # correctly routes the field to non-estimable (insufficient anchors).
        assert len(assumptions_payload["assumptions"]) == 0
        non_est = assumptions_payload["non_estimable"]
        assert len(non_est) >= 1
        anchor_records = [
            r for r in non_est
            if "insufficient" in r.get("explanation", "").lower()
            or "insufficient" in r.get("gap_description", "").lower()
        ]
        assert len(anchor_records) >= 1

        # Artifacts should be on disk
        enrichment_dir = run_paths.base_dir / "stage_files" / stage_file_dir_name(
            "enrichment"
        )
        assert enrichment_dir.exists()
        assert (enrichment_dir / "enrichment_bundle.json").exists()
        assert not (enrichment_dir / "gap_manifest.json").exists()
        assert not (enrichment_dir / "external_source_search_stage.json").exists()
        assert not (enrichment_dir / "web_research_stage.json").exists()
        assert not (enrichment_dir / "assumptions_stage.json").exists()
        assumptions_dir = run_paths.base_dir / "stage_files" / stage_file_dir_name(
            "assumptions"
        )
        enrichment_handoff_dir = (
            run_paths.base_dir
            / "stage_files"
            / stage_file_dir_name("enrichment_context_handoff")
        )
        assumptions_handoff_dir = (
            run_paths.base_dir
            / "stage_files"
            / stage_file_dir_name("assumptions_context_handoff")
        )
        assert (assumptions_dir / "assumptions_stage.json").exists()
        assert (
            enrichment_handoff_dir / "context_bundle_after_enrichment.json"
        ).exists()
        assert (
            assumptions_handoff_dir / "context_bundle_after_assumptions.json"
        ).exists()
        assert (run_paths.base_dir / "stages" / "008_enrichment.json").exists()
        assert (
            run_paths.base_dir / "stages" / "009_enrichment_context_handoff.json"
        ).exists()
        assert (run_paths.base_dir / "stages" / "010_assumptions.json").exists()
        assert (
            run_paths.base_dir / "stages" / "011_assumptions_context_handoff.json"
        ).exists()

        enrichment_snapshot = json.loads(
            (
                enrichment_handoff_dir / "context_bundle_after_enrichment.json"
            ).read_text(encoding="utf-8")
        )
        assumptions_snapshot = json.loads(
            (
                assumptions_handoff_dir / "context_bundle_after_assumptions.json"
            ).read_text(encoding="utf-8")
        )
        assert "enrichment" in enrichment_snapshot
        assert "assumptions" not in enrichment_snapshot
        assert "assumptions" in assumptions_snapshot

    def test_pipeline_fallback_on_error(self, tmp_path: Path) -> None:
        """Pipeline returns original context_bundle on failure."""
        config = build_test_app_config(
            enrichment_overrides={"enabled": True, "model": "test-model"}
        )
        run_paths = create_run_paths(tmp_path, "run_fallback", "context_bundle.json")
        run_logger = RunLogger(run_paths, "test question")
        context_bundle = {"markdown": None, "marker": "original"}

        with patch("backend.modules.web_researcher.gap_analysis.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("LLM down")
            mock_cls.return_value = mock_client

            result = run_enrichment_pipeline(
                question="test",
                context_bundle=context_bundle,
                base_dir=run_paths.base_dir,
                run_logger=run_logger,
                config=config,
                api_key="test-key",
            )

        # Should return original context bundle
        assert result.get("marker") == "original"
        assert "enrichment" not in result

    def test_pipeline_skipped_when_no_gaps(self, tmp_path: Path) -> None:
        """Pipeline returns original context when gap analysis finds no gaps."""
        config = build_test_app_config(
            enrichment_overrides={"enabled": True, "model": "test-model"}
        )
        run_paths = create_run_paths(tmp_path, "run_no_gaps", "context_bundle.json")
        run_logger = RunLogger(run_paths, "test question")
        context_bundle = {"markdown": None}

        empty_manifest = {
            "query_fields": [],
            "city_gaps": [],
            "non_estimable_fields": [],
        }
        gap_response = _make_mock_openai_response(json.dumps(empty_manifest))

        with patch("backend.modules.web_researcher.gap_analysis.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = gap_response
            mock_cls.return_value = mock_client

            result = run_enrichment_pipeline(
                question="test",
                context_bundle=context_bundle,
                base_dir=run_paths.base_dir,
                run_logger=run_logger,
                config=config,
                api_key="test-key",
            )

        # Should return original without enrichment
        assert "enrichment" not in result

    def test_web_research_failed_batch_group_still_persists_diagnostics(
        self, tmp_path: Path
    ) -> None:
        """Late batch-group failures should still contribute audit diagnostics."""
        config = build_test_app_config(
            enrichment_overrides={
                "enabled": True,
                "model": "test-model",
                "web_research_enabled": True,
                "external_source_search_enabled": False,
            }
        )
        run_paths = create_run_paths(tmp_path, "run_failed_batch_group", "context_bundle.json")
        run_logger = RunLogger(run_paths, "What is the total CAPEX for Dresden?")
        context_bundle = {
            "markdown": {"status": "success", "excerpts": []},
            "research_question": "What is the total CAPEX for Dresden?",
        }

        gap_response = _make_mock_openai_response(
            json.dumps(_mock_gap_analysis_response())
        )
        assumptions_response = _make_mock_openai_response(
            json.dumps(_mock_assumptions_response())
        )
        search_batches = [
            SearchBatch(
                batch_id="city_batch",
                cities=["Dresden"],
                target_fields=["total_capex"],
                search_type="city",
                queries=["capex Dresden"],
                budget={"max_pages": 3},
                priority="high",
            ),
            SearchBatch(
                batch_id="national_batch",
                cities=["Germany"],
                target_fields=["total_capex"],
                search_type="national_benchmark",
                queries=["national capex benchmark"],
                budget={"max_pages": 3},
                priority="medium",
            ),
        ]

        call_count = 0

        def mock_create(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return gap_response
            return assumptions_response

        def mock_execute_search_batches(
            batches: list[SearchBatch],
            cfg: Any,
            key: str,
            progress: Any = None,
            scrape_failures: list[dict[str, Any]] | None = None,
            scrape_stats: dict[str, int] | None = None,
            search_execution_summary: dict[str, Any] | None = None,
        ) -> list[Any]:
            batch = batches[0]
            if search_execution_summary is not None:
                search_execution_summary.update(
                    {
                        "config": {
                            "tier1_first_search": False,
                            "max_retries_per_worker": 1,
                        },
                        "metrics": {
                            "planned_query_count": len(batch.queries),
                            "actual_serper_query_count": 2 if batch.search_type == "city" else 1,
                            "successful_serper_query_count": 1,
                            "tier1_site_query_count": 0,
                            "open_query_count": 2 if batch.search_type == "city" else 1,
                            "open_query_skipped_count": 0,
                            "estimated_max_serper_query_count": 2 if batch.search_type == "city" else 1,
                            "batch_count": 1,
                        },
                        "batches": [
                            {
                                "batch_id": batch.batch_id,
                                "planned_query_count": len(batch.queries),
                                "actual_serper_query_count": 2 if batch.search_type == "city" else 1,
                            }
                        ],
                    }
                )
            if scrape_stats is not None:
                scrape_stats["scrape_success_count"] = 1 if batch.search_type == "city" else 2
                scrape_stats["scrape_failure_count"] = 1 if batch.search_type == "city" else 0
            if batch.search_type == "city":
                if scrape_failures is not None:
                    scrape_failures.append(
                        {
                            "url": "https://example.com/city-capex",
                            "domain": "example.com",
                            "provider": "firecrawl",
                            "batch_id": batch.batch_id,
                            "query": batch.queries[0],
                            "error_type": "ReadTimeout",
                            "error": "The read operation timed out",
                            "severity": "warning",
                        }
                    )
                raise RuntimeError("city search failed late")
            return []

        with patch("backend.modules.web_researcher.gap_analysis.OpenAI") as mock_gap_cls, \
             patch("backend.modules.web_researcher.assumptions_estimator.OpenAI") as mock_est_cls, \
             patch("backend.modules.web_researcher.module.plan_searches", return_value=search_batches), \
             patch("backend.modules.web_researcher.module.execute_search_batches", side_effect=mock_execute_search_batches):
            mock_gap_client = MagicMock()
            mock_gap_client.chat.completions.create = mock_create
            mock_gap_cls.return_value = mock_gap_client

            mock_est_client = MagicMock()
            mock_est_client.chat.completions.create.return_value = assumptions_response
            mock_est_cls.return_value = mock_est_client

            result = run_enrichment_pipeline(
                question="What is the total CAPEX for Dresden?",
                context_bundle=context_bundle,
                base_dir=run_paths.base_dir,
                run_logger=run_logger,
                config=config,
                api_key="test-key",
            )

        assert "enrichment" in result
        enrichment_dir = run_paths.base_dir / "stage_files" / stage_file_dir_name(
            "enrichment"
        )
        audit_payload = json.loads(
            (enrichment_dir / "web_research_audit.json").read_text(encoding="utf-8")
        )
        assert audit_payload["outputs"]["scrape_failures"][0]["error_type"] == "ReadTimeout"
        assert audit_payload["outputs"]["search_execution_summary"]["metrics"][
            "actual_serper_query_count"
        ] == 3
        assert audit_payload["outputs"]["search_execution_summary"]["failed_batch_groups"] == [
            {
                "group": "city",
                "error_type": "RuntimeError",
                "error": "city search failed late",
            }
        ]
        assert audit_payload["outputs"]["failed_batch_groups"] == [
            {
                "group": "city",
                "error_type": "RuntimeError",
                "error": "city search failed late",
            }
        ]
        assert audit_payload["metrics"]["scrape_success_count"] == 3
        assert audit_payload["metrics"]["scrape_failure_count"] == 1
        assert audit_payload["metrics"]["failed_batch_group_count"] == 1


# ---------------------------------------------------------------------------
# RunLogger enrichment method
# ---------------------------------------------------------------------------


class TestRunLoggerEnrichment:
    def test_update_enrichment_bundle(self, tmp_path: Path) -> None:
        run_paths = create_run_paths(tmp_path, "run_logger_test", "context_bundle.json")
        run_logger = RunLogger(run_paths, "test question")

        payload = {"gap_manifest": {}, "assumptions": [], "meta": {}}
        run_logger.update_enrichment_bundle(payload)

        assert run_logger.context_bundle["enrichment"] == payload

        # Verify persisted to disk
        persisted = json.loads(
            run_paths.context_bundle.read_text(encoding="utf-8")
        )
        assert "enrichment" in persisted
