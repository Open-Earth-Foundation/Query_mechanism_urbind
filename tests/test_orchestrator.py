import json
import logging
from pathlib import Path

import pytest

from backend.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.orchestrator.module import _build_retrieval_queries, run_pipeline
from backend.modules.writer.models import WriterCitationCoverage, WriterOutput
from backend.utils.config import AppConfig
from backend.utils.logging_config import setup_logger
from backend.utils.paths import RunPaths
from tests.support import build_test_app_config


def _research_question_path(paths: RunPaths) -> Path:
    """Return the canonical research-question artifact path for one run."""
    return (
        paths.base_dir
        / "stage_files"
        / "002_query_preparation"
        / "research_question.json"
    )


def _stage_file_path(paths: RunPaths, folder: str, filename: str) -> Path:
    """Return one stage-file path under the numbered stage folder."""
    return paths.base_dir / "stage_files" / folder / filename


def _build_test_config(*, runs_dir: Path, markdown_dir: Path) -> AppConfig:
    """Build an orchestrator test config with vector retrieval disabled."""
    return build_test_app_config(
        runs_dir=runs_dir,
        markdown_dir=markdown_dir,
        vector_store_overrides={"enabled": False},
    )


def _stub_markdown(
    question: str,
    documents: list[dict[str, str]],
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> MarkdownResearchResult:
    """Return one grounded markdown excerpt for pipeline tests."""
    _ = question, documents, config, api_key
    excerpt = MarkdownExcerpt(
        quote="Munich has deployed 43 existing public chargers as of 2024.",
        city_name="Munich",
        partial_answer="Munich has deployed 43 existing public chargers as of 2024.",
    )
    return MarkdownResearchResult(excerpts=[excerpt])


def _stub_writer(
    question: str,
    context_bundle: dict,
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> WriterOutput:
    """Return one deterministic writer output for pipeline tests."""
    _ = question, context_bundle, config, api_key
    return WriterOutput(content="# Answer\n\nStub")


def _reset_root_handlers() -> None:
    """Remove and close all handlers from the root logger."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()


def test_build_retrieval_queries_trims_dedupes_case_insensitively_and_caps() -> None:
    assert _build_retrieval_queries(
        "  Main query  ",
        "main query",
        "  Second query  ",
        "",
        "SECOND QUERY",
        "Third query",
        "Fourth ignored query",
    ) == [
        "Main query",
        "Second query",
        "Third query",
    ]


def test_run_pipeline_creates_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    final_output = paths.final_output.read_text(encoding="utf-8")
    run_log = json.loads(paths.api_state.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed"
    assert "Finish reason:" not in final_output
    assert _stage_file_path(
        paths,
        "007_markdown_context_handoff",
        "context_bundle_after_markdown.json",
    ).exists()
    assert _stage_file_path(
        paths,
        "007_markdown_context_handoff",
        "markdown_context_payload.json",
    ).exists()
    stage_paths = {path.name for path in paths.stages_dir.iterdir() if path.is_file()}
    assert "007_markdown_context_handoff.json" in stage_paths
    summary_events = [
        json.loads(line)
        for line in paths.summary_events.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stage_events = [
        event
        for event in summary_events
        if event.get("event_type") == "stage_completed"
    ]
    assert stage_events[0]["payload"]["step"] == "input_snapshot"
    assert stage_events[0]["stage_number"] == 1
    assert stage_events[1]["payload"]["step"] == "query_preparation"
    assert stage_events[1]["stage_number"] == 2


def test_run_pipeline_keeps_city_scope_on_root_context_not_markdown_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
        selected_cities=["Munich"],
    )

    context_bundle = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    markdown_payload = json.loads(
        _stage_file_path(
            paths,
            "007_markdown_context_handoff",
            "markdown_context_payload.json",
        ).read_text(encoding="utf-8")
    )

    assert context_bundle["selected_cities"] == ["munich"]
    assert context_bundle["selected_city_names"] == ["Munich"]
    assert context_bundle["inspected_cities"] == ["munich"]
    assert context_bundle["inspected_city_names"] == ["Munich"]
    assert context_bundle["city_scope_mode"] == "selected_cities"
    assert "selected_cities" not in markdown_payload
    assert "selected_city_names" not in markdown_payload
    assert "inspected_cities" not in markdown_payload
    assert "inspected_city_names" not in markdown_payload
    assert "retrieval_mode" not in markdown_payload
    assert "analysis_mode" not in markdown_payload
    assert "retrieval_queries" not in markdown_payload


def test_run_pipeline_writes_enrichment_context_handoff_when_enrichment_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
        vector_store_overrides={"enabled": False},
        enrichment_overrides={"enabled": True},
    )

    def _stub_enrichment(
        question: str,
        context_bundle: dict[str, object],
        **_kwargs: dict[str, object],
    ) -> dict[str, object]:
        del question
        updated = dict(context_bundle)
        updated["enrichment"] = {
            "field_manifest": {"query_fields": [], "non_estimable_fields": []},
            "gap_manifest": {"city_gaps": []},
            "enriched_fields": [],
            "web_findings": [],
            "external_evidence": [],
            "external_resolutions": [],
            "external_no_evidence": [],
            "freshness_results": [],
            "meta": {
                "created_at": "2026-01-01T00:00:00Z",
                "gap_analyst_model": "test-model",
                "total_gaps": 0,
                "estimable_count": 0,
                "non_estimable_count": 0,
                "web_findings_count": 0,
                "external_evidence_count": 0,
                "elapsed_seconds": 0.0,
            },
        }
        updated["assumptions"] = {
            "assumptions": [],
            "non_estimable": [],
            "saturation_warning": None,
            "meta": {"assumption_count": 0},
        }
        return updated

    monkeypatch.setattr(
        "backend.modules.orchestrator.module.run_enrichment_pipeline",
        _stub_enrichment,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )

    context_bundle = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    assert "assumptions" in context_bundle
    assert "assumptions" not in context_bundle["enrichment"]


def test_run_pipeline_persists_partial_writer_output_as_completed_with_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    def _stub_partial_writer(
        question: str,
        context_bundle: dict,
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> WriterOutput:
        del question, context_bundle, config, api_key
        return WriterOutput(
            content="# Answer\n\nPartial",
            citation_coverage=WriterCitationCoverage(
                status="partial",
                attempt=2,
                max_attempts=2,
                coverage_confirmed=1,
                coverage_required=2,
                coverage_ratio="1/2",
                missing_cities=["Berlin"],
                analysis_mode="aggregate",
            ),
        )

    paths = run_pipeline(
        question="What initiatives exist for Munich and Berlin?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_stub_partial_writer,
    )

    assert paths.final_output.exists()
    run_log = json.loads(paths.api_state.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed_with_gaps"
    assert run_log["finish_reason"].startswith(
        "completed_with_gaps (writer partial citation coverage 1/2)"
    )
    assert run_log["writer_citation_coverage"]["missing_cities"] == ["Berlin"]


def test_run_pipeline_passes_run_logger_and_paths_to_writer_when_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )
    captured: dict[str, object] = {}

    def _writer_with_runtime_context(
        question: str,
        context_bundle: dict,
        config: AppConfig,
        api_key: str,
        run_logger: object,
        paths: object,
        **_kwargs: dict[str, object],
    ) -> WriterOutput:
        del question, context_bundle, config, api_key
        captured["run_logger"] = run_logger
        captured["paths"] = paths
        return WriterOutput(content="# Answer\n\nStub")

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_writer_with_runtime_context,
    )

    assert paths.final_output.exists()
    assert captured["run_logger"] is not None
    assert captured["paths"] == paths


def test_run_pipeline_detaches_run_log_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _reset_root_handlers()
    setup_logger()

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    try:
        first_paths = run_pipeline(
            question="First question?",
            config=config,
            run_id="run1",
            markdown_func=_stub_markdown,
            writer_func=_stub_writer,
        )
        first_run_log_path = str(first_paths.base_dir / "run.log")
        assert all(
            not isinstance(handler, logging.FileHandler)
            or handler.baseFilename != first_run_log_path
            for handler in logging.getLogger().handlers
        )
        logging.getLogger(__name__).warning("MARKER_AFTER_RUN1")

        second_paths = run_pipeline(
            question="Second question?",
            config=config,
            run_id="run2",
            markdown_func=_stub_markdown,
            writer_func=_stub_writer,
        )
        second_run_log_path = str(second_paths.base_dir / "run.log")
        assert all(
            not isinstance(handler, logging.FileHandler)
            or handler.baseFilename not in {first_run_log_path, second_run_log_path}
            for handler in logging.getLogger().handlers
        )
        logging.getLogger(__name__).warning("MARKER_AFTER_RUN2")

        first_log = (first_paths.base_dir / "run.log").read_text(encoding="utf-8")

        assert "MARKER_AFTER_RUN2" not in first_log
    finally:
        _reset_root_handlers()


def test_run_pipeline_standard_mode_uses_verbatim_question_before_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )
    captured: dict[str, str] = {}

    def _capture_markdown_question(
        question: str,
        documents: list[dict[str, str]],
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> MarkdownResearchResult:
        captured["question"] = question
        return _stub_markdown(question, documents, config, api_key, **_kwargs)

    paths = run_pipeline(
        question="What initiatives exist for Munich as typed?",
        config=config,
        markdown_func=_capture_markdown_question,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    assert captured["question"] == "What initiatives exist for Munich as typed?"
    context_bundle = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    assert context_bundle["research_question"] == "What initiatives exist for Munich as typed?"
    research_payload = json.loads(_research_question_path(paths).read_text(encoding="utf-8"))
    assert research_payload["retrieval_queries"] == [
        "What initiatives exist for Munich as typed?"
    ]
    assert (
        research_payload["canonical_research_query"]
        == "What initiatives exist for Munich as typed?"
    )
    assert research_payload["query_mode"] == "standard"


def test_run_pipeline_standard_mode_uses_optional_queries_when_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")
    (docs_dir / "Leipzig.md").write_text("# Leipzig\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )
    captured: dict[str, str] = {}

    def _capture_markdown_question(
        question: str,
        documents: list[dict[str, str]],
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> MarkdownResearchResult:
        captured["question"] = question
        return _stub_markdown(question, documents, config, api_key, **_kwargs)

    paths = run_pipeline(
        question="Compare Munich and Leipzig initiatives as written.",
        config=config,
        selected_cities=["Munich", "Leipzig"],
        query_2="implementation milestones",
        query_3="reported budget metrics",
        markdown_func=_capture_markdown_question,
        writer_func=_stub_writer,
    )

    assert captured["question"] == "Compare Munich and Leipzig initiatives as written."
    research_payload = json.loads(_research_question_path(paths).read_text(encoding="utf-8"))
    assert research_payload["query_mode"] == "standard"
    assert research_payload["retrieval_queries"] == [
        "Compare Munich and Leipzig initiatives as written.",
        "implementation milestones",
        "reported budget metrics",
    ]
    assert research_payload["retrieval_query_2"] == "implementation milestones"
    assert research_payload["retrieval_query_3"] == "reported budget metrics"


def test_run_pipeline_dev_mode_uses_direct_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )
    captured: dict[str, str] = {}

    def _capture_markdown_question(
        question: str,
        documents: list[dict[str, str]],
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> MarkdownResearchResult:
        captured["question"] = question
        return _stub_markdown(question, documents, config, api_key, **_kwargs)

    paths = run_pipeline(
        question="Main direct query",
        config=config,
        query_mode="dev",
        query_2="Second direct query",
        query_3="Third direct query",
        markdown_func=_capture_markdown_question,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    assert captured["question"] == "Main direct query"
    research_payload = json.loads(_research_question_path(paths).read_text(encoding="utf-8"))
    assert research_payload["query_mode"] == "dev"
    assert research_payload["retrieval_queries"] == [
        "Main direct query",
        "Second direct query",
        "Third direct query",
    ]


def test_run_pipeline_logs_resolved_run_id_when_requested_id_is_suffixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    first_paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        run_id="duplicate-run",
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )
    second_paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        run_id="duplicate-run",
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )

    assert first_paths.base_dir.name == "duplicate-run"
    assert second_paths.base_dir.name == "duplicate-run_01"

    second_run_log = (second_paths.base_dir / "run.log").read_text(encoding="utf-8")
    assert "run_id=duplicate-run_01 query_mode=standard query_count=1" in second_run_log
    assert "run_id=duplicate-run_01 markdown_source_mode=standard_chunking" in second_run_log
    assert "run_id=duplicate-run query_mode=standard query_count=1" not in second_run_log
