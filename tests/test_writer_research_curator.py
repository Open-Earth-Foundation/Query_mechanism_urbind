from pathlib import Path

import pytest

from backend.modules.writer.utils import research_curator
from backend.modules.writer.utils.research_context import (
    apply_saved_evidence_to_context,
    build_writer_context_index,
    build_writer_references_payload,
)
from backend.modules.writer.utils.research_session import (
    WriterResearchSession,
    WriterResearchToolError,
    build_writer_research_limits,
)
from backend.modules.writer.utils.section_first import build_section_planner_payload
from backend.services.run_logger import RunLogger
from backend.utils.paths import create_run_paths
from tests.support import build_test_app_config


def _build_context_bundle() -> dict[str, object]:
    """Build a compact writer-safe context bundle with CCC and external evidence."""
    return {
        "analysis_mode": "aggregate",
        "selected_cities": ["Munich"],
        "markdown": {
            "excerpt_count": 1,
            "selected_city_names": ["Munich"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "city_key": "munich",
                    "quote": "Munich targets 250 public chargers by 2030.",
                    "partial_answer": "Munich has a 2030 public charger target.",
                    "source_chunk_ids": ["munich_chunk_1"],
                }
            ],
        },
        "enrichment": {
            "external_evidence": [
                {
                    "city": "Munich",
                    "field": "public charger target",
                    "value": "300",
                    "unit": "chargers",
                    "source_id": "external_plan",
                    "line_start": 12,
                    "line_end": 15,
                    "quote": "The external plan lists 300 chargers.",
                    "confidence": 0.8,
                }
            ]
        },
    }


def test_writer_research_session_search_save_and_build_refs(tmp_path: Path) -> None:
    """Saved CCC refs are preserved and non-CCC evidence gets new refs."""
    config = build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
        vector_store_overrides={"enabled": False},
    )
    context_bundle = _build_context_bundle()
    index = build_writer_context_index(
        context_bundle=context_bundle,
        run_dir=None,
        markdown_dir=config.markdown_dir,
        config=config,
        use_source_chunks=False,
    )
    session = WriterResearchSession(
        run_id="run-writer-research",
        items=index.items,
        limits=build_writer_research_limits(config),
    )

    ccc_hits = session.regex_search_context("2030", cities=["Munich"])
    ccc_saved = session.save_context_evidence(
        hit_ids=[ccc_hits[0].hit_id],
        reason="The CCC excerpt directly states the target year.",
    )
    external_hits = session.regex_search_context(
        "300 chargers",
        source_kinds=["external_markdown_claim"],
    )
    external_saved = session.save_context_evidence(
        hit_ids=[external_hits[0].hit_id],
        reason="External evidence gives a conflicting target value.",
    )
    duplicate_saved = session.save_context_evidence(
        hit_ids=[external_hits[0].hit_id],
        reason="Duplicate save should reuse the existing saved record.",
    )

    assert ccc_saved[0].ref_id == "ref_1"
    assert external_saved[0].ref_id == "ref_2"
    assert duplicate_saved[0].saved_id == external_saved[0].saved_id
    assert len(session.saved_evidence()) == 2

    curated = apply_saved_evidence_to_context(
        context_bundle=context_bundle,
        saved_evidence=session.saved_evidence(),
    )
    excerpts = curated["markdown"]["excerpts"]
    assert len(excerpts) == 2
    assert excerpts[0]["writer_saved_id"] == ccc_saved[0].saved_id
    assert excerpts[1]["ref_id"] == "ref_2"
    assert excerpts[1]["source_kind"] == "external_markdown_claim"

    references = build_writer_references_payload(
        run_id="run-writer-research",
        saved_evidence=session.saved_evidence(),
    )
    assert references["reference_count"] == 2
    assert references["references"][1]["source_kind"] == "external_markdown_claim"


def test_writer_research_session_rejects_unsafe_regex(tmp_path: Path) -> None:
    """Unsafe regex syntax is rejected before searching context."""
    config = build_test_app_config(runs_dir=tmp_path / "output")
    index = build_writer_context_index(
        context_bundle=_build_context_bundle(),
        run_dir=None,
        markdown_dir=config.markdown_dir,
        config=config,
        use_source_chunks=False,
    )
    session = WriterResearchSession(
        run_id="run-writer-research",
        items=index.items,
        limits=build_writer_research_limits(config),
    )

    with pytest.raises(WriterResearchToolError, match="Backreferences"):
        session.regex_search_context(r"(charger) \1")


def test_section_planner_payload_exposes_saved_and_fallback_catalogs() -> None:
    """The chapter planner can prefer saved evidence without losing fallback excerpts."""
    context_bundle = _build_context_bundle()
    context_bundle["markdown"]["excerpts"].append(
        {
            "ref_id": "ref_2",
            "city_name": "Munich",
            "quote": "Saved source chunk says the plan has a delivery milestone.",
            "partial_answer": "Saved source chunk has milestone detail.",
            "source_kind": "ccc_source_chunk",
            "writer_saved_id": "ws_1",
        }
    )
    context_bundle["markdown"]["excerpt_count"] = 2

    payload = build_section_planner_payload(
        question="Compare charger targets.",
        context_bundle=context_bundle,
        analysis_mode="aggregate",
        selected_city_names=["Munich"],
        max_input_tokens=10_000,
    ).payload

    assert [entry["ref_id"] for entry in payload["saved_evidence_catalog"]] == ["ref_2"]
    assert [entry["ref_id"] for entry in payload["fallback_evidence_catalog"]] == ["ref_1"]


def test_writer_research_curator_failure_falls_back_to_excerpt_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Curator runtime failures should not replace the baseline writer context."""
    config = build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
        writer_overrides={
            "evidence_curator_enabled": True,
            "evidence_curator_use_source_chunks": False,
        },
    )
    paths = create_run_paths(
        config.runs_dir,
        "run-curator-fallback",
        config.orchestrator.context_bundle_name,
    )
    run_logger = RunLogger(paths, "Compare charger targets.")

    monkeypatch.setattr(
        research_curator,
        "build_writer_research_curator_agent",
        lambda **_kwargs: object(),
    )

    def _raise_runtime_error(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("curator failed")

    monkeypatch.setattr(research_curator, "run_agent_sync", _raise_runtime_error)

    result = research_curator.run_writer_research_curator(
        question="Compare charger targets.",
        context_bundle=_build_context_bundle(),
        analysis_mode="aggregate",
        selected_city_names=["Munich"],
        config=config,
        api_key="test-key",
        run_id="run-curator-fallback",
        paths=paths,
        run_logger=run_logger,
        log_llm_payload=False,
    )

    assert result.status == "failed"
    assert result.saved_evidence == []
    assert result.context_bundle["markdown"]["excerpt_count"] == 1
    assert run_logger.run_log["writer_saved_evidence"]["curator_status"] == "failed"
