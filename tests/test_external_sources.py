"""Tests for governed external Markdown source search and resolution."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from backend.modules.web_researcher.context_merger import compute_field_statuses
from backend.modules.web_researcher.external_agent import (
    _claim_contains_field_requirements,
    build_external_source_research_agent,
)
from backend.modules.web_researcher.external_resolver import resolve_external_evidence
from backend.modules.web_researcher.external_sources import (
    ExternalSearchSession,
    ExternalSourceToolError,
    SourceRegistry,
    build_external_search_limits,
)
from backend.modules.web_researcher.models import (
    CityGap,
    EvidenceCandidateInput,
    ExternalEvidenceClaim,
    ExternalEvidenceResolution,
    NoEvidenceRecord,
)
from backend.modules.writer.utils.multi_pass import build_writer_context_bundle
from tests.support import build_test_app_config


def _test_workspace(name: str) -> Path:
    """Create a repo-local test workspace without relying on system temp dirs."""
    path = Path(".pytest_tmp_external_sources") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        (workspace / "artifacts" / "external_evidence.json").read_text(encoding="utf-8")
    )
    assert payload["candidates"][0]["candidate_id"] == "e1"


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


def test_writer_context_preserves_enrichment() -> None:
    """Writer batching keeps enrichment evidence available to prompts."""
    context_bundle = {
        "analysis_mode": "city_by_city",
        "markdown": {"status": "success", "excerpts": [], "excerpt_count": 0},
        "enrichment": {"external_evidence": [{"source_id": "krakow-target"}]},
    }

    writer_context = build_writer_context_bundle(
        context_bundle=context_bundle,
        excerpts=[],
        city_names=["Krakow"],
    )
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
