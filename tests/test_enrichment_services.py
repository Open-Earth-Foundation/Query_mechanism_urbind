"""Tests for enrichment service functions (gap analysis, assumptions, context merger)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from backend.modules.web_researcher.context_merger import (
    compute_field_statuses,
    merge_enrichment_into_context,
    serialize_enrichment_artifacts,
)
from backend.modules.web_researcher.assumptions_context import (
    serialize_assumptions_artifacts,
)
from backend.modules.web_researcher.models import (
    CityGap,
    EnrichedField,
    FieldClassification,
    FreshnessResult,
    GapManifest,
    WebFinding,
)
from backend.services.run_logger import RunLogger
from backend.utils.artifact_writer import stage_file_dir_name
from backend.utils.paths import create_run_paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gap_manifest(
    *,
    fields: list[dict[str, Any]] | None = None,
    city_gaps: list[dict[str, Any]] | None = None,
    non_estimable: list[str] | None = None,
) -> GapManifest:
    return GapManifest(
        query_fields=[FieldClassification(**f) for f in (fields or [])],
        city_gaps=[CityGap(**cg) for cg in (city_gaps or [])],
        non_estimable_fields=non_estimable or [],
    )


def _make_web_finding(**kwargs: Any) -> WebFinding:
    defaults = {
        "city": "Dresden",
        "field": "total_capex",
        "value": 50_000_000,
        "source_url": "https://example.com",
        "source_type": "government_report",
        "extraction_confidence": 0.9,
    }
    defaults.update(kwargs)
    return WebFinding(**defaults)


def _make_freshness_result(**kwargs: Any) -> FreshnessResult:
    defaults = {
        "city": "Dresden",
        "field": "total_capex",
        "ccc_value": "45000000",
        "web_value": "50000000",
        "classification": "consistent",
        "reason": "Values align",
    }
    defaults.update(kwargs)
    return FreshnessResult(**defaults)


# ---------------------------------------------------------------------------
# compute_field_statuses
# ---------------------------------------------------------------------------


class TestComputeFieldStatuses:
    def test_no_gaps_returns_empty(self) -> None:
        manifest = _make_gap_manifest()
        result = compute_field_statuses(manifest, [], [], {})
        assert result == []

    def test_blank_field_no_web_is_still_missing(self) -> None:
        manifest = _make_gap_manifest(
            city_gaps=[{
                "city": "Dresden",
                "blank_fields": ["total_capex"],
                "stale_flags": [],
                "search_priority": "high",
            }]
        )
        result = compute_field_statuses(manifest, [], [], {})
        assert len(result) == 1
        assert result[0].status == "still_missing"
        assert result[0].source == "none"

    def test_web_finding_fills_blank(self) -> None:
        manifest = _make_gap_manifest(
            city_gaps=[{
                "city": "Dresden",
                "blank_fields": ["total_capex"],
                "stale_flags": [],
                "search_priority": "high",
            }]
        )
        wf = _make_web_finding()
        result = compute_field_statuses(manifest, [wf], [], {})
        assert len(result) == 1
        assert result[0].status == "resolved"
        assert result[0].source == "web"
        assert result[0].value == 50_000_000

    def test_web_superseded_uses_web_value(self) -> None:
        manifest = _make_gap_manifest(
            city_gaps=[{
                "city": "Dresden",
                "blank_fields": ["total_capex"],
                "stale_flags": [],
                "search_priority": "high",
            }]
        )
        wf = _make_web_finding(value=55_000_000)
        fr = _make_freshness_result(classification="superseded")
        result = compute_field_statuses(manifest, [wf], [fr], {})
        assert result[0].status == "resolved"
        assert result[0].source == "web"
        assert result[0].freshness_flag == "superseded"
        assert result[0].value == 55_000_000

    def test_web_consistent_uses_ccc(self) -> None:
        manifest = _make_gap_manifest(
            city_gaps=[{
                "city": "Dresden",
                "blank_fields": ["total_capex"],
                "stale_flags": [],
                "search_priority": "high",
            }]
        )
        wf = _make_web_finding()
        fr = _make_freshness_result(classification="consistent", ccc_value="45000000")
        result = compute_field_statuses(manifest, [wf], [fr], {})
        assert result[0].status == "resolved"
        assert result[0].source == "ccc"
        assert result[0].freshness_flag == "consistent"

    def test_web_uncertain_is_partially_resolved(self) -> None:
        manifest = _make_gap_manifest(
            city_gaps=[{
                "city": "Dresden",
                "blank_fields": ["total_capex"],
                "stale_flags": [],
                "search_priority": "high",
            }]
        )
        wf = _make_web_finding()
        fr = _make_freshness_result(classification="uncertain")
        result = compute_field_statuses(manifest, [wf], [fr], {})
        assert result[0].status == "partially_resolved"
        assert result[0].freshness_flag == "uncertain"

    def test_stale_field_no_web_is_partially_resolved(self) -> None:
        manifest = _make_gap_manifest(
            city_gaps=[{
                "city": "Dresden",
                "blank_fields": [],
                "stale_flags": ["timeline"],
                "search_priority": "medium",
            }]
        )
        result = compute_field_statuses(manifest, [], [], {})
        assert len(result) == 1
        assert result[0].status == "partially_resolved"
        assert result[0].freshness_flag == "stale_no_update"

    def test_multiple_cities_handled(self) -> None:
        manifest = _make_gap_manifest(
            city_gaps=[
                {
                    "city": "Dresden",
                    "blank_fields": ["capex"],
                    "stale_flags": [],
                    "search_priority": "high",
                },
                {
                    "city": "Berlin",
                    "blank_fields": ["capex"],
                    "stale_flags": [],
                    "search_priority": "high",
                },
            ]
        )
        result = compute_field_statuses(manifest, [], [], {})
        assert len(result) == 2
        cities = {r.city for r in result}
        assert cities == {"Dresden", "Berlin"}


# ---------------------------------------------------------------------------
# merge_enrichment_into_context
# ---------------------------------------------------------------------------


class TestMergeEnrichmentIntoContext:
    def test_does_not_mutate_original(self) -> None:
        original = {"markdown": None}
        manifest = _make_gap_manifest()
        enriched = merge_enrichment_into_context(
            context_bundle=original,
            gap_manifest=manifest,
            web_findings=[],
            freshness_results=[],
            assumptions=[],
            non_estimable=[],
            saturation_warning=None,
            config_model="test-model",
            assumptions_model="test-model",
            elapsed_seconds=1.0,
        )
        assert "enrichment" not in original
        assert "enrichment" in enriched

    def test_enrichment_structure(self) -> None:
        manifest = _make_gap_manifest(
            city_gaps=[{
                "city": "Dresden",
                "blank_fields": ["capex"],
                "stale_flags": [],
                "search_priority": "high",
            }]
        )
        enriched = merge_enrichment_into_context(
            context_bundle={},
            gap_manifest=manifest,
            web_findings=[],
            freshness_results=[],
            assumptions=[],
            non_estimable=[],
            saturation_warning=None,
            config_model="test-model",
            assumptions_model="test-model",
            elapsed_seconds=2.5,
        )
        e = enriched["enrichment"]
        assert "gap_manifest" in e
        assert "field_manifest" in e
        assert "enriched_fields" in e
        assert "assumptions" not in e
        assert "non_estimable" not in e
        assert "saturation_warning" not in e
        assert "assumptions" in enriched
        assumptions_payload = enriched["assumptions"]
        assert isinstance(assumptions_payload, dict)
        assert assumptions_payload["assumptions"] == []
        assert assumptions_payload["non_estimable"] == []
        assert "query_fields" in e["field_manifest"]
        assert "non_estimable_fields" in e["field_manifest"]
        assert "query_fields" not in e["gap_manifest"]
        assert "non_estimable_fields" not in e["gap_manifest"]
        assert list(e["gap_manifest"]) == ["city_gaps"]
        assert "meta" in e
        assert e["meta"]["gap_analyst_model"] == "test-model"
        assert e["meta"]["elapsed_seconds"] == 2.5
        assert e["meta"]["total_gaps"] == 1


# ---------------------------------------------------------------------------
# serialize_enrichment_artifacts
# ---------------------------------------------------------------------------


class TestSerializeEnrichmentArtifacts:
    def test_writes_artifacts_to_disk(self, tmp_path: Path) -> None:
        run_paths = create_run_paths(tmp_path, "run_001", "context_bundle.json")
        base_dir = run_paths.base_dir
        run_logger = RunLogger(run_paths, "test question")

        manifest = _make_gap_manifest(
            city_gaps=[{
                "city": "Dresden",
                "blank_fields": ["capex"],
                "stale_flags": [],
                "search_priority": "high",
            }]
        )
        enriched = merge_enrichment_into_context(
            context_bundle={},
            gap_manifest=manifest,
            web_findings=[],
            freshness_results=[],
            assumptions=[],
            non_estimable=[],
            saturation_warning=None,
            config_model="test-model",
            assumptions_model="test-model",
            elapsed_seconds=1.0,
        )

        serialize_enrichment_artifacts(
            enriched,
            base_dir,
            run_logger,
            stage_flags={
                "enrichment_enabled": True,
                "web_research_enabled": False,
                "web_research_executed": False,
                "external_source_search_enabled": False,
                "external_source_search_executed": False,
            },
            substage_artifacts={
                "gap_analysis": {
                    "status": "completed",
                    "flags": {"use_split_gap_flow": False},
                    "outputs": {"city_gaps": []},
                    "metrics": {"city_gap_count": 1},
                },
                "external_sources": {
                    "status": "skipped",
                    "flags": {"external_source_search_enabled": False},
                    "outputs": {
                        "filled_city_fields": [],
                        "unresolved_city_fields": [],
                        "no_evidence_city_fields": [],
                    },
                    "metrics": {"external_evidence_count": 0},
                },
                "web_research": {
                    "status": "completed",
                    "flags": {"web_research_enabled": True},
                    "outputs": {
                        "search_batches": [{"queries": ["capex Dresden"]}],
                        "added_city_fields": [],
                    },
                    "metrics": {"web_finding_count": 0, "search_batch_count": 1},
                },
            },
        )
        assumptions_payload = enriched.get("assumptions")
        assert isinstance(assumptions_payload, dict)
        serialize_assumptions_artifacts(assumptions_payload, base_dir, run_logger)

        enrichment_dir = base_dir / "stage_files" / stage_file_dir_name("enrichment")
        assumptions_dir = base_dir / "stage_files" / stage_file_dir_name("assumptions")
        assert enrichment_dir.exists()
        assert (enrichment_dir / "enrichment_bundle.json").exists()
        assert not (enrichment_dir / "field_manifest.json").exists()
        assert not (enrichment_dir / "gap_manifest.json").exists()
        assert not (enrichment_dir / "gap_analysis_stage.json").exists()
        assert not (enrichment_dir / "external_source_search_stage.json").exists()
        assert not (enrichment_dir / "web_research_stage.json").exists()
        assert (enrichment_dir / "web_research_audit.json").exists()
        assert not (enrichment_dir / "assumptions.json").exists()
        assert not (enrichment_dir / "non_estimable.json").exists()
        assert not (enrichment_dir / "assumptions_stage.json").exists()
        assert (assumptions_dir / "assumptions.json").exists()
        assert (assumptions_dir / "non_estimable.json").exists()
        assert (assumptions_dir / "assumptions_stage.json").exists()
        # Projection files should not exist; the bundle is the canonical payload.
        assert not (enrichment_dir / "web_findings.json").exists()
        assert not (enrichment_dir / "freshness_results.json").exists()
        enrichment_stage = json.loads(
            (base_dir / "stages" / "008_enrichment.json").read_text(encoding="utf-8")
        )
        assert enrichment_stage["outputs"]["enrichment_bundle"].endswith(
            "enrichment_bundle.json"
        )
        assert "field_manifest" not in enrichment_stage["outputs"]
        assert "gap_manifest" not in enrichment_stage["outputs"]
        assert "substages" in enrichment_stage["outputs"]
        assert enrichment_stage["outputs"]["web_research_audit"].endswith(
            "web_research_audit.json"
        )
        assert enrichment_stage["metrics"]["city_gap_count"] == 1
        assert enrichment_stage["metrics"]["gap_field_count"] == 1
        assert enrichment_stage["metrics"]["blank_field_count"] == 1
        assert enrichment_stage["metrics"]["stale_field_count"] == 0
        assert enrichment_stage["metrics"]["bundled_field_count"] == 0

    def test_no_enrichment_key_is_noop(self, tmp_path: Path) -> None:
        run_paths = create_run_paths(tmp_path, "run_002", "context_bundle.json")
        base_dir = run_paths.base_dir
        run_logger = RunLogger(run_paths, "test question")

        serialize_enrichment_artifacts({}, base_dir, run_logger)

        enrichment_dir = base_dir / "stage_files" / stage_file_dir_name("enrichment")
        assert not enrichment_dir.exists()
