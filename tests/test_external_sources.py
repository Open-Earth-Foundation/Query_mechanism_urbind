"""Tests for governed external Markdown source search and resolution."""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from pathlib import Path

import pytest

from backend.scripts.benchmark_external_source_pipeline import _score_case
import backend.modules.web_researcher.external_agent as external_agent_module
from backend.modules.web_researcher.context_merger import compute_field_statuses
from backend.modules.web_researcher.external_agent import (
    _claim_contains_field_requirements,
    _extract_external_ccc_values,
    build_external_source_research_agent,
    run_external_source_enrichment,
)
from backend.modules.web_researcher.external_resolver import resolve_external_evidence
from backend.modules.web_researcher.external_sources import (
    EXTERNAL_SOURCE_SEARCH_AUDIT_FILENAME,
    ExternalSearchSession,
    ExternalSourceToolError,
    SourceRegistry,
    build_external_search_limits,
    try_load_external_source_registry,
)
from backend.modules.web_researcher.models import (
    CityGap,
    EvidenceCandidateInput,
    ExternalEvidenceClaim,
    ExternalSourceAgentResult,
    ExternalEvidenceResolution,
    FieldClassification,
    FreshnessResult,
    GapManifest,
    NoEvidenceRecord,
    WebFinding,
)
from backend.modules.writer.utils.multi_pass import build_writer_context_bundle
from tests.support import build_test_app_config


def _test_workspace(name: str) -> Path:
    """Create an isolated temp workspace for external-source tests."""
    return Path(tempfile.mkdtemp(prefix=f"external_sources_{name}_{uuid.uuid4().hex}_"))


def _write_source_library(root: Path) -> None:
    """Create a minimal tagged source library for tests."""
    root.mkdir(parents=True)
    (root / "sources.yaml").write_text(
        "\n".join(
            [
                "sources:",
                "  - source_id: krakow-target",
                "    title: Krakow Target Plan",
                "    upstream_group: tier_1_city_plans",
                "    geographic_scope: city",
                "    city: [Krakow]",
                "    country: [Poland]",
                "    publication_year: 2025",
                "    description: Krakow target test source.",
                "    source_type: city_cap",
                "    publisher: City of Krakow",
                "    verticals: [mobility]",
                "    tef_sectors: [transport]",
                "  - source_id: krakow-context",
                "    title: Krakow Context Plan",
                "    upstream_group: tier_1_city_plans",
                "    geographic_scope: city",
                "    city: [Krakow]",
                "    country: [Poland]",
                "    publication_year: 2024",
                "    description: Supportive context without a target value.",
                "    source_type: mobility_plan",
                "    publisher: City of Krakow",
                "    verticals: [mobility]",
                "    tef_sectors: [transport]",
            ]
        ),
        encoding="utf-8",
    )
    (root / "krakow-target.md").write_text(
        "\n".join(
            [
                "# Climate targets",
                "Krakow sets a local CO2 reduction target of 30% by 2030.",
                "The baseline year is 2018.",
            ]
        ),
        encoding="utf-8",
    )
    (root / "krakow-context.md").write_text(
        "# Mobility context\nKrakow is expanding public transport planning.\n",
        encoding="utf-8",
    )


def _build_external_session(workspace: Path, source_root: Path) -> ExternalSearchSession:
    """Build a search session over the test source library."""
    config = build_test_app_config(
        enrichment_overrides={"external_source_search_enabled": True}
    )
    return ExternalSearchSession(
        run_id="test_run",
        registry=SourceRegistry.load(source_root),
        limits=build_external_search_limits(config),
        artifact_dir=workspace / "artifacts",
    )


def test_registry_filters_sources_by_tags() -> None:
    """Candidate listing validates and applies metadata filters."""
    source_root = _test_workspace("registry") / "source_library"
    _write_source_library(source_root)

    registry = SourceRegistry.load(source_root)
    options = registry.get_tag_options()
    assert options.cities == ["Krakow"]
    assert options.verticals == ["mobility"]

    candidates = registry.list_candidate_sources(cities=["krakow"], verticals=["Mobility"])
    assert [candidate.source_id for candidate in candidates] == [
        "krakow-target",
        "krakow-context",
    ]

    with pytest.raises(ExternalSourceToolError) as exc:
        registry.list_candidate_sources(cities=["Krakov"])
    assert exc.value.code == "INVALID_FILTER"


def test_invalid_source_registry_skips_only_external_sources(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid source mappings fail soft so the wider enrichment pipeline can continue."""
    source_root = _test_workspace("invalid_registry") / "source_library"
    source_root.mkdir(parents=True)
    (source_root / "sources.yaml").write_text(
        "\n".join(
            [
                "sources:",
                "  - source_id: missing-markdown",
                "    title: Missing Markdown",
                "    upstream_group: tier_1_city_plans",
                "    description: Metadata points to a file that is absent.",
                "    source_type: city_cap",
            ]
        ),
        encoding="utf-8",
    )
    caplog.set_level(logging.WARNING, logger="backend.modules.web_researcher.external_sources")

    registry = try_load_external_source_registry(source_root)

    assert registry is None
    assert any("invalid sources.yaml" in record.message for record in caplog.records)


def test_regex_search_expand_and_evidence_persistence() -> None:
    """Search tools return line-grounded hits and persist selected evidence."""
    workspace = _test_workspace("session")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    config = build_test_app_config(
        enrichment_overrides={"external_source_search_enabled": True}
    )
    session = ExternalSearchSession(
        run_id="test_run",
        registry=SourceRegistry.load(source_root),
        limits=build_external_search_limits(config),
        artifact_dir=workspace / "artifacts",
    )
    session.set_active_task("Krakow", "secap_local_co2_reduction_2030_target")

    with pytest.raises(ExternalSourceToolError) as exc:
        session.regex_search(pattern="2030")
    assert exc.value.code == "SOURCE_SCOPE_REQUIRED"

    hits = session.regex_search(
        pattern=r"30%.{0,80}2030",
        cities=["Krakow"],
        verticals=["mobility"],
        max_matches=5,
    )
    assert len(hits) == 1
    assert hits[0].hit_id == "h1"
    assert hits[0].line_start <= 2 <= hits[0].line_end
    assert "30%" in hits[0].snippet

    expanded = session.expand_hits(["h1"])
    assert expanded[0].heading_path == ["Climate targets"]

    saved = session.add_evidence_candidates(
        [
            EvidenceCandidateInput(
                hit_id="h1",
                city="Krakow",
                field="secap_local_co2_reduction_2030_target",
                reason="Contains the 2030 target value.",
                confidence=0.92,
            )
        ]
    )
    assert saved[0].candidate_id == "e1"
    payload = json.loads(
        (workspace / "artifacts" / EXTERNAL_SOURCE_SEARCH_AUDIT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["candidates"][0]["candidate_id"] == "e1"


def test_set_active_task_clears_prior_candidate_scope() -> None:
    """A new city-field task cannot reuse the previous task's candidate source list."""
    workspace = _test_workspace("task_scope_reset")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    session = _build_external_session(workspace, source_root)
    session.set_active_task("Krakow", "secap_local_co2_reduction_2030_target")
    session.list_candidate_sources(cities=["Krakow"])
    assert session.regex_search(pattern="30%")

    session.set_active_task("Krakow", "public_ev_chargers_2030_target")

    with pytest.raises(ExternalSourceToolError) as exc:
        session.regex_search(pattern="30%")
    assert exc.value.code == "SOURCE_SCOPE_REQUIRED"


def test_expand_and_save_reject_hits_from_previous_active_task() -> None:
    """Hit expansion and evidence saving are scoped to the current active task."""
    workspace = _test_workspace("task_hit_scope")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    session = _build_external_session(workspace, source_root)
    session.set_active_task("Krakow", "secap_local_co2_reduction_2030_target")
    session.regex_search(pattern="30%", cities=["Krakow"])

    session.set_active_task("Krakow", "public_ev_chargers_2030_target")

    with pytest.raises(ExternalSourceToolError) as expand_exc:
        session.expand_hits(["h1"])
    assert expand_exc.value.code == "HIT_NOT_FOUND"

    with pytest.raises(ExternalSourceToolError) as save_exc:
        session.add_evidence_candidates(
            [
                EvidenceCandidateInput(
                    hit_id="h1",
                    city="Krakow",
                    field="public_ev_chargers_2030_target",
                    reason="Old hit should not be reusable.",
                    confidence=0.7,
                )
            ]
        )
    assert save_exc.value.code == "HIT_NOT_FOUND"


def test_visibility_starts_with_base_tools_before_active_task_hits() -> None:
    """Only discovery/search tools are visible before the active task has hits."""
    workspace = _test_workspace("visibility_base")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    session = _build_external_session(workspace, source_root)

    session.set_active_task("Krakow", "secap_local_co2_reduction_2030_target")

    assert session.has_hits_for_active_task() is False
    assert session.allowed_tool_names_for_active_task() == [
        "get_tag_options",
        "list_candidate_sources",
        "regex_search",
        "list_evidence_candidates",
        "mark_no_evidence_found",
    ]


def test_visibility_adds_expansion_tools_after_active_task_hit() -> None:
    """A non-empty regex result unlocks anchored tools for that active task."""
    workspace = _test_workspace("visibility_hit")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    session = _build_external_session(workspace, source_root)
    session.set_active_task("Krakow", "secap_local_co2_reduction_2030_target")

    hits = session.regex_search(pattern=r"30%.{0,80}2030", cities=["Krakow"])

    assert [hit.hit_id for hit in hits] == ["h1"]
    assert session.has_hits_for_active_task() is True
    visible_tools = session.allowed_tool_names_for_active_task()
    assert "expand_hits" in visible_tools
    assert "add_evidence_candidates" in visible_tools


def test_visibility_does_not_unlock_after_zero_hit_search() -> None:
    """A scoped search with zero matches leaves anchored tools hidden."""
    workspace = _test_workspace("visibility_zero")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    session = _build_external_session(workspace, source_root)
    session.set_active_task("Krakow", "missing_target")

    hits = session.regex_search(pattern="not-present-in-library", cities=["Krakow"])

    assert hits == []
    assert session.has_hits_for_active_task() is False
    visible_tools = session.allowed_tool_names_for_active_task()
    assert "expand_hits" not in visible_tools
    assert "add_evidence_candidates" not in visible_tools


def test_visibility_is_task_scoped() -> None:
    """Hits for one city-field task do not unlock tools for another task."""
    workspace = _test_workspace("visibility_task_scoped")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    session = _build_external_session(workspace, source_root)
    session.set_active_task("Krakow", "secap_local_co2_reduction_2030_target")
    session.regex_search(pattern=r"30%.{0,80}2030", cities=["Krakow"])
    assert session.has_hits_for_active_task() is True

    session.set_active_task("Krakow", "public_ev_chargers_2030_target")

    assert session.has_hits_for_active_task() is False
    visible_tools = session.allowed_tool_names_for_active_task()
    assert "expand_hits" not in visible_tools
    assert "add_evidence_candidates" not in visible_tools


def test_research_agent_gates_anchored_tools_with_dynamic_callbacks() -> None:
    """Agent tools use SDK-level dynamic visibility for hit-anchored tools."""
    workspace = _test_workspace("agent_visibility")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    session = _build_external_session(workspace, source_root)
    session.set_active_task("Krakow", "secap_local_co2_reduction_2030_target")
    config = build_test_app_config(
        enrichment_overrides={"external_source_search_enabled": True}
    )

    agent = build_external_source_research_agent(config, "test-api-key", session)
    tools = {tool.name: tool for tool in agent.tools}

    assert callable(tools["expand_hits"].is_enabled)
    assert callable(tools["add_evidence_candidates"].is_enabled)
    assert tools["expand_hits"].is_enabled(None, agent) is False
    assert tools["add_evidence_candidates"].is_enabled(None, agent) is False

    session.regex_search(pattern=r"30%.{0,80}2030", cities=["Krakow"])

    assert tools["expand_hits"].is_enabled(None, agent) is True
    assert tools["add_evidence_candidates"].is_enabled(None, agent) is True


def test_expanded_hit_fallback_keeps_match_centered() -> None:
    """Expanded hits can be staged after overrun without truncating out the match."""
    workspace = _test_workspace("expanded_fallback")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    (source_root / "krakow-target.md").write_text(
        "# Climate targets\n"
        + ("noise " * 80)
        + "Krakow sets a local CO2 reduction target of 30% by 2030. "
        + ("noise " * 80),
        encoding="utf-8",
    )
    config = build_test_app_config(
        enrichment_overrides={
            "external_source_search_enabled": True,
            "external_max_snippet_chars": 120,
            "external_max_context_words": 100,
        }
    )
    session = ExternalSearchSession(
        run_id="test_run",
        registry=SourceRegistry.load(source_root),
        limits=build_external_search_limits(config),
        artifact_dir=workspace / "artifacts",
    )

    session.set_active_task("Krakow", "secap_local_co2_reduction_2030_target")
    hits = session.regex_search(
        pattern=r"30%.{0,20}2030",
        cities=["Krakow"],
        max_matches=1,
        context_words=100,
        context_lines=0,
    )
    assert hits[0].truncated is True
    assert "30%" in hits[0].snippet
    assert "2030" in hits[0].snippet

    session.expand_hits(["h1"], context_words=100, context_lines=0)
    saved = session.stage_expanded_hits_for_active_task()
    assert saved[0].field == "secap_local_co2_reduction_2030_target"
    assert "30%" in saved[0].quote
    assert "2030" in saved[0].quote


def test_recent_hit_fallback_stages_candidates_without_expansion() -> None:
    """Fallback finalization can inspect recent hits even if the model skipped expand."""
    workspace = _test_workspace("recent_hit_fallback")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    config = build_test_app_config(
        enrichment_overrides={"external_source_search_enabled": True}
    )
    session = ExternalSearchSession(
        run_id="test_run",
        registry=SourceRegistry.load(source_root),
        limits=build_external_search_limits(config),
        artifact_dir=workspace / "artifacts",
    )

    session.set_active_task("Krakow", "secap_local_co2_reduction_2030_target")
    session.regex_search(
        pattern=r"30%.{0,80}2030",
        cities=["Krakow"],
        max_matches=5,
    )

    saved = session.stage_expanded_hits_for_active_task()

    assert saved[0].field == "secap_local_co2_reduction_2030_target"
    assert saved[0].hit_id == "h1"
    assert "30%" in saved[0].quote


def test_run_external_source_enrichment_collects_validated_claims_without_nesting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validated claims from each task should aggregate as claims, not nested lists."""
    workspace = _test_workspace("benchmark_runner")
    source_root = workspace / "source_library"
    _write_source_library(source_root)
    config = build_test_app_config(
        enrichment_overrides={
            "external_source_search_enabled": True,
            "external_source_dir": source_root,
        }
    )
    base_dir = workspace / "run"
    base_dir.mkdir(parents=True, exist_ok=True)
    context_bundle = {
        "analysis_mode": "city_by_city",
        "selected_cities": ["Krakow"],
        "markdown": {"status": "success", "excerpts": [], "excerpt_count": 0},
        "enrichment": {},
    }
    gap_manifest = GapManifest(
        query_fields=[
            FieldClassification(
                field="secap_local_co2_reduction_2030_target",
                classification="estimable_numerical",
                searchable=True,
                rationale="Benchmark field.",
            )
        ],
        city_gaps=[
            CityGap(
                city="Krakow",
                blank_fields=["secap_local_co2_reduction_2030_target"],
                stale_flags=[],
                search_priority="high",
            )
        ],
        non_estimable_fields=[],
    )

    class _DummyAgent:
        def __init__(self, session: ExternalSearchSession | None = None) -> None:
            self.session = session

    def _fake_build_research_agent(
        config: object,
        api_key: str,
        session: ExternalSearchSession,
    ) -> _DummyAgent:
        return _DummyAgent(session)

    def _fake_run_agent_sync(
        agent: _DummyAgent,
        prompt: str,
        max_turns: int,
    ) -> object:
        task = json.loads(prompt)
        hits = agent.session.regex_search(
            pattern=r"30%.{0,80}2030",
            cities=[task["city"]],
            max_matches=5,
        )
        assert hits
        saved = agent.session.add_evidence_candidates(
            [
                EvidenceCandidateInput(
                    hit_id=hits[0].hit_id,
                    city=task["city"],
                    field=task["field"],
                    reason="Contains the requested target.",
                    confidence=0.95,
                )
            ]
        )
        final_output = ExternalSourceAgentResult(
            claims=[
                ExternalEvidenceClaim(
                    city=task["city"],
                    field=task["field"],
                    value=30,
                    unit="%",
                    source_id="placeholder-source",
                    source_type="city_cap",
                    line_start=1,
                    line_end=1,
                    quote="30% by 2030",
                    confidence=0.95,
                    claim_role="fills_missing",
                    candidate_id=saved[0].candidate_id,
                )
            ],
            no_evidence=[],
            notes=[],
        )
        return type("_Result", (), {"final_output": final_output})()

    monkeypatch.setattr(
        external_agent_module,
        "build_external_source_research_agent",
        _fake_build_research_agent,
    )
    monkeypatch.setattr(
        external_agent_module,
        "build_external_source_finalizer_agent",
        lambda config, api_key: _DummyAgent(),
    )
    monkeypatch.setattr(external_agent_module, "run_agent_sync", _fake_run_agent_sync)

    claims, resolutions, no_evidence, tool_calls, audit_payload = (
        run_external_source_enrichment(
            question="What is Krakow's local CO2 reduction target?",
            context_bundle=context_bundle,
            gap_manifest=gap_manifest,
            base_dir=base_dir,
            config=config,
            api_key="test-api-key",
            run_id="test_run",
        )
    )

    assert len(claims) == 1
    assert claims[0].candidate_id == "e1"
    assert claims[0].source_id == "krakow-target"
    assert len(resolutions) == 1
    assert resolutions[0].action == "fill"
    assert no_evidence == []
    assert tool_calls
    assert audit_payload["metrics"]["validated_claim_count"] == 1
    assert audit_payload["metrics"]["rejected_claim_count"] == 0
    assert audit_payload["metrics"]["max_turn_exceeded_count"] == 0
    assert audit_payload["metrics"]["fallback_finalization_count"] == 0


def test_regex_safety_rejects_unsafe_patterns() -> None:
    """Unsafe MVP regex forms are rejected before scanning files."""
    source_root = _test_workspace("regex_safety") / "source_library"
    _write_source_library(source_root)
    config = build_test_app_config(
        enrichment_overrides={"external_source_search_enabled": True}
    )
    session = ExternalSearchSession(
        run_id="test_run",
        registry=SourceRegistry.load(source_root),
        limits=build_external_search_limits(config),
    )

    with pytest.raises(ExternalSourceToolError) as exc:
        session.regex_search(pattern=r"(.*)+", cities=["Krakow"])
    assert exc.value.code == "REGEX_UNSAFE"


def test_external_resolver_covers_fill_conflict_and_unresolved() -> None:
    """Resolver emits the planned confirm/fill/conflict/unresolved action set."""
    city_gaps = [
        CityGap(
            city="Krakow",
            blank_fields=["target_2030", "missing_field"],
            stale_flags=["stale_target"],
            search_priority="high",
        )
    ]
    claims = [
        ExternalEvidenceClaim(
            city="Krakow",
            field="target_2030",
            value=30,
            unit="%",
            source_id="krakow-target",
            source_type="city_cap",
            line_start=2,
            line_end=3,
            quote="Krakow sets a local CO2 reduction target of 30% by 2030.",
            confidence=0.9,
            claim_role="fills_missing",
        ),
        ExternalEvidenceClaim(
            city="Krakow",
            field="stale_target",
            value=80,
            unit="%",
            source_id="krakow-target",
            source_type="city_cap",
            line_start=2,
            line_end=3,
            quote="Krakow sets a different target.",
            confidence=0.8,
            claim_role="challenges_ccc",
        ),
    ]
    no_evidence = [
        NoEvidenceRecord(
            record_id="n1",
            city="Krakow",
            field="missing_field",
            searched_source_ids=["krakow-target"],
            search_summary="No public EV charging target found.",
        )
    ]

    resolutions = resolve_external_evidence(city_gaps, claims, no_evidence)
    actions = {resolution.field: resolution.action for resolution in resolutions}
    assert actions == {
        "target_2030": "fill",
        "stale_target": "conflict_review_required",
        "missing_field": "unresolved",
    }


def test_external_resolver_threads_structured_ccc_values_into_confirmations() -> None:
    """Resolver confirmations should carry the structured CCC value when available."""
    city_gaps = [
        CityGap(
            city="Krakow",
            blank_fields=[],
            stale_flags=["target_2030"],
            search_priority="high",
        )
    ]
    claims = [
        ExternalEvidenceClaim(
            city="Krakow",
            field="target_2030",
            value=30,
            unit="%",
            source_id="krakow-target",
            source_type="city_cap",
            line_start=2,
            line_end=3,
            quote="Krakow sets a local CO2 reduction target of 30% by 2030.",
            confidence=0.9,
            claim_role="confirms_ccc",
        )
    ]

    resolutions = resolve_external_evidence(
        city_gaps,
        claims,
        [],
        ccc_values={("krakow", "target_2030"): "30"},
    )

    assert resolutions[0].action == "confirm"
    assert resolutions[0].ccc_value == "30"


def test_context_merger_overlays_external_resolutions() -> None:
    """External fill decisions become writer-visible enriched fields."""
    manifest_gap = CityGap(
        city="Krakow",
        blank_fields=["target_2030"],
        stale_flags=[],
        search_priority="high",
    )
    resolution = ExternalEvidenceResolution(
        city="Krakow",
        field="target_2030",
        action="fill",
        external_value=30,
        unit="%",
        source_id="krakow-target",
        line_start=2,
        line_end=3,
        quote="Krakow sets a local CO2 reduction target of 30% by 2030.",
        confidence=0.9,
        rationale="External evidence fills the CCC gap.",
    )

    result = compute_field_statuses(
        gap_manifest=type(
            "Manifest",
            (),
            {"city_gaps": [manifest_gap]},
        )(),
        web_findings=[],
        freshness_results=[],
        context_bundle={},
        external_resolutions=[resolution],
    )
    assert result[0].status == "resolved"
    assert result[0].source == "external_markdown"
    assert result[0].provenance["source_id"] == "krakow-target"


def test_context_merger_keeps_superseded_web_value_when_external_confirm_has_no_ccc_value() -> None:
    """External CCC confirmation must not relabel a newer web value as CCC."""
    manifest = GapManifest(
        query_fields=[
            FieldClassification(
                field="target_2030",
                classification="estimable_numerical",
                searchable=True,
                rationale="Benchmark field.",
            )
        ],
        city_gaps=[
            CityGap(
                city="Krakow",
                blank_fields=[],
                stale_flags=["target_2030"],
                search_priority="high",
            )
        ],
        non_estimable_fields=[],
    )
    web_findings = [
        WebFinding(
            city="Krakow",
            field="target_2030",
            value="55",
            unit="%",
            source_id="web-source",
            source_name="Web Source",
            source_tier="open",
            source_type="report",
            source_url="https://example.com/target",
            rationale="Newer public report.",
            extraction_confidence=0.9,
        )
    ]
    freshness_results = [
        FreshnessResult(
            city="Krakow",
            field="target_2030",
            ccc_value="30",
            web_value="55",
            classification="superseded",
            reason="Newer web source.",
            web_source_url="https://example.com/target",
        )
    ]
    external_resolutions = [
        ExternalEvidenceResolution(
            city="Krakow",
            field="target_2030",
            action="confirm",
            ccc_value=None,
            external_value="30",
            unit="%",
            source_id="krakow-target",
            line_start=2,
            line_end=3,
            quote="Krakow sets a local CO2 reduction target of 30% by 2030.",
            confidence=0.9,
            rationale="External evidence confirms the CCC target.",
        )
    ]

    result = compute_field_statuses(
        manifest,
        web_findings,
        freshness_results,
        {},
        external_resolutions=external_resolutions,
    )

    assert result[0].status == "resolved"
    assert result[0].value == "55"
    assert result[0].source == "web"
    assert result[0].freshness_flag == "superseded"


def test_context_merger_keeps_partial_web_evidence_when_external_search_is_unresolved() -> None:
    """No-evidence external results must not erase partially resolved freshness output."""
    manifest = GapManifest(
        query_fields=[
            FieldClassification(
                field="target_2030",
                classification="estimable_numerical",
                searchable=True,
                rationale="Benchmark field.",
            )
        ],
        city_gaps=[
            CityGap(
                city="Krakow",
                blank_fields=[],
                stale_flags=["target_2030"],
                search_priority="high",
            )
        ],
        non_estimable_fields=[],
    )
    web_findings = [
        WebFinding(
            city="Krakow",
            field="target_2030",
            value="55",
            unit="%",
            source_id="web-source",
            source_name="Web Source",
            source_tier="open",
            source_type="report",
            source_url="https://example.com/target",
            rationale="Newer public report.",
            extraction_confidence=0.9,
        )
    ]
    freshness_results = [
        FreshnessResult(
            city="Krakow",
            field="target_2030",
            ccc_value="30",
            web_value="55",
            classification="uncertain",
            reason="CCC phrasing is qualitative.",
            web_source_url="https://example.com/target",
        )
    ]
    external_resolutions = [
        ExternalEvidenceResolution(
            city="Krakow",
            field="target_2030",
            action="unresolved",
            ccc_value=None,
            external_value=None,
            unit=None,
            source_id=None,
            line_start=None,
            line_end=None,
            quote=None,
            confidence=None,
            rationale="Tagged external sources were searched but no usable evidence was found.",
        )
    ]

    result = compute_field_statuses(
        manifest,
        web_findings,
        freshness_results,
        {},
        external_resolutions=external_resolutions,
    )

    assert result[0].status == "partially_resolved"
    assert result[0].value == "30"
    assert result[0].source == "ccc"
    assert result[0].freshness_flag == "uncertain"


def test_extract_external_ccc_values_reads_structured_context_records() -> None:
    """Structured CCC values in enrichment context should reach the resolver."""
    context_bundle = {
        "enrichment": {
            "external_ccc_context": [
                {
                    "city": "Krakow",
                    "field": "target_2030",
                    "context": "CCC target context",
                    "ccc_value": "30",
                }
            ],
            "freshness_results": [
                {
                    "city": "Dresden",
                    "field": "capex",
                    "ccc_value": "45000000",
                }
            ],
        }
    }

    assert _extract_external_ccc_values(context_bundle) == {
        ("krakow", "target_2030"): "30",
        ("dresden", "capex"): "45000000",
    }


def test_writer_context_preserves_enrichment() -> None:
    """Writer batching keeps enrichment evidence available to prompts."""
    context_bundle = {
        "analysis_mode": "city_by_city",
        "markdown": {"status": "success", "excerpts": [], "excerpt_count": 0},
        "enrichment": {
            "field_manifest": {
                "query_fields": [{"field": "target_2030", "scope": "municipal"}],
                "non_estimable_fields": [],
            },
            "gap_manifest": {"city_gaps": []},
            "external_evidence": [{"city": "Krakow", "source_id": "krakow-target"}],
        },
    }

    writer_context = build_writer_context_bundle(
        context_bundle=context_bundle,
        excerpts=[],
        city_names=["Krakow"],
    )
    assert "field_manifest" in writer_context["enrichment"]
    assert writer_context["enrichment"]["gap_manifest"] == {"city_gaps": []}
    assert writer_context["enrichment"]["external_evidence"][0]["source_id"] == "krakow-target"


def test_external_claim_validation_allows_infrastructure_program_timing() -> None:
    """Infrastructure targets may state program timing without repeating the horizon year."""
    claim = ExternalEvidenceClaim(
        city="Krakow",
        field="public_ev_chargers_2030_target",
        value=150,
        unit="publicznych stacji ladowania",
        source_id="krakow-action-plan",
        source_type="city_cap",
        publication_year=2024,
        line_start=10,
        line_end=12,
        quote="Instalacja co najmniej 150 nowych publicznych stacji ladowania w pierwszych pieciu latach programu.",
        confidence=0.9,
        claim_role="fills_missing",
    )

    assert _claim_contains_field_requirements(claim) is True


def test_external_claim_validation_requires_year_for_reduction_targets() -> None:
    """Reduction target fields still need the requested horizon year in evidence."""
    claim = ExternalEvidenceClaim(
        city="Krakow",
        field="climate_city_contract_ghg_reduction_2030_target",
        value=80,
        unit="%",
        source_id="krakow-action-plan",
        source_type="city_cap",
        publication_year=2024,
        line_start=10,
        line_end=12,
        quote="Celem jest redukcja emisji gazow cieplarnianych o 80% w stosunku do roku 2018.",
        confidence=0.9,
        claim_role="fills_missing",
    )

    assert _claim_contains_field_requirements(claim) is False


def test_benchmark_score_case_uses_best_matching_claim_not_first_claim() -> None:
    """Benchmark scoring should pass when a later claim is the correct one."""
    case = {
        "field": "target_2030",
        "expected": {
            "source_id": "right-source",
            "value_terms": ["30"],
            "quote_terms": ["2030"],
        },
    }
    claims = [
        ExternalEvidenceClaim(
            city="Krakow",
            field="target_2030",
            value="25",
            unit="%",
            source_id="wrong-source",
            source_type="city_cap",
            line_start=1,
            line_end=1,
            quote="25% by 2030",
            confidence=0.6,
            claim_role="fills_missing",
        ),
        ExternalEvidenceClaim(
            city="Krakow",
            field="target_2030",
            value="30",
            unit="%",
            source_id="right-source",
            source_type="city_cap",
            line_start=2,
            line_end=2,
            quote="30% by 2030",
            confidence=0.9,
            claim_role="fills_missing",
        ),
    ]

    result = _score_case(case, claims, [])

    assert result["passed"] is True
    assert result["source_ok"] is True
    assert result["value_ok"] is True
    assert result["quote_ok"] is True
    assert result["best_claim"]["source_id"] == "right-source"
