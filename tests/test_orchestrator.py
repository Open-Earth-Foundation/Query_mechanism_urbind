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
from tests.support import build_test_app_config


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
    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed"
    assert Path(run_log["artifacts"]["final_output"]).exists()
    assert "Finish reason:" not in final_output


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
    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
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
    research_payload = json.loads(paths.research_question.read_text(encoding="utf-8"))
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
    research_payload = json.loads(paths.research_question.read_text(encoding="utf-8"))
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
    research_payload = json.loads(paths.research_question.read_text(encoding="utf-8"))
    assert research_payload["query_mode"] == "dev"
    assert research_payload["retrieval_queries"] == [
        "Main direct query",
        "Second direct query",
        "Third direct query",
    ]
