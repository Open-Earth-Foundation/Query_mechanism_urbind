import json
import logging
import sqlite3
from pathlib import Path

import pytest

from app.utils.config import (
    AppConfig,
    AgentConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)
from app.modules.orchestrator.module import run_pipeline
from app.modules.orchestrator.models import (
    OrchestratorDecision,
    ResearchQuestionRefinement,
)
from app.modules.sql_researcher.models import SqlQuery, SqlQueryPlan
from app.modules.markdown_researcher.models import MarkdownExcerpt, MarkdownResearchResult
from app.modules.writer.models import WriterOutput
from app.utils.logging_config import setup_logger


def _stub_sql_plan(
    question: str,
    schema_summary: dict,
    city_names: list[str],
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> SqlQueryPlan:
    return SqlQueryPlan(
        queries=[SqlQuery(query_id="q1", query="SELECT cityName FROM City")],
    )


def _stub_markdown(
    question: str,
    documents: list[dict[str, str]],
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> MarkdownResearchResult:
    excerpt = MarkdownExcerpt(
        snippet="Sample",
        city_name="Munich",
        partial_answer="Stub answer",
        relevant="yes",
    )
    return MarkdownResearchResult(excerpts=[excerpt])


def _stub_refine_question(
    question: str,
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> ResearchQuestionRefinement:
    return ResearchQuestionRefinement(research_question=question)


def _stub_decision(
    question: str,
    context_bundle: dict,
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> OrchestratorDecision:
    return OrchestratorDecision(action="write", reason="Enough")


def _stub_decision_stop(
    question: str,
    context_bundle: dict,
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> OrchestratorDecision:
    return OrchestratorDecision(action="stop", reason="No answer possible")


def _stub_writer(
    question: str,
    context_bundle: dict,
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> WriterOutput:
    return WriterOutput(content="# Answer\n\nStub")


def _reset_root_handlers() -> None:
    """Remove and close all handlers from the root logger."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()


def test_run_pipeline_creates_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    db_path = tmp_path / "source.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE City (cityId TEXT, cityName TEXT)")
    conn.execute("INSERT INTO City (cityId, cityName) VALUES ('1', 'Munich')")
    conn.commit()
    conn.close()

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test", max_result_tokens=100000),
        markdown_researcher=MarkdownResearcherConfig(model="test"),
        writer=AgentConfig(model="test"),
        runs_dir=tmp_path / "output",
        source_db_path=db_path,
        markdown_dir=docs_dir,
        enable_sql=True,
    )

    paths = run_pipeline(
        question="What cities exist?",
        config=config,
        sql_plan_func=_stub_sql_plan,
        markdown_func=_stub_markdown,
        refine_question_func=_stub_refine_question,
        decide_func=_stub_decision,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed"
    assert Path(run_log["artifacts"]["final_output"]).exists()


def test_run_pipeline_sql_disabled_skips_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test", max_result_tokens=100000),
        markdown_researcher=MarkdownResearcherConfig(model="test"),
        writer=AgentConfig(model="test"),
        runs_dir=tmp_path / "output",
        source_db_path=tmp_path / "missing.db",
        markdown_dir=docs_dir,
        enable_sql=False,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        sql_plan_func=_stub_sql_plan,
        markdown_func=_stub_markdown,
        refine_question_func=_stub_refine_question,
        decide_func=_stub_decision,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed"


def test_run_pipeline_stop_with_sql_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test", max_result_tokens=100000),
        markdown_researcher=MarkdownResearcherConfig(model="test"),
        writer=AgentConfig(model="test"),
        runs_dir=tmp_path / "output",
        source_db_path=tmp_path / "missing.db",
        markdown_dir=docs_dir,
        enable_sql=False,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        sql_plan_func=_stub_sql_plan,
        markdown_func=_stub_markdown,
        refine_question_func=_stub_refine_question,
        decide_func=_stub_decision_stop,
        writer_func=_stub_writer,
    )

    assert not paths.final_output.exists()
    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert run_log["status"] == "stopped"
    assert run_log["finish_reason"] == "stopped_by_orchestrator"


def test_run_pipeline_detaches_run_log_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _reset_root_handlers()
    setup_logger()

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test", max_result_tokens=100000),
        markdown_researcher=MarkdownResearcherConfig(model="test"),
        writer=AgentConfig(model="test"),
        runs_dir=tmp_path / "output",
        source_db_path=tmp_path / "missing.db",
        markdown_dir=docs_dir,
        enable_sql=False,
    )

    try:
        first_paths = run_pipeline(
            question="First question?",
            config=config,
            run_id="run1",
            sql_plan_func=_stub_sql_plan,
            markdown_func=_stub_markdown,
            refine_question_func=_stub_refine_question,
            decide_func=_stub_decision,
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
            sql_plan_func=_stub_sql_plan,
            markdown_func=_stub_markdown,
            refine_question_func=_stub_refine_question,
            decide_func=_stub_decision,
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


def test_run_pipeline_refines_question_before_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test", max_result_tokens=100000),
        markdown_researcher=MarkdownResearcherConfig(model="test"),
        writer=AgentConfig(model="test"),
        runs_dir=tmp_path / "output",
        source_db_path=tmp_path / "missing.db",
        markdown_dir=docs_dir,
        enable_sql=False,
    )

    captured: dict[str, str] = {}

    def _refine_for_test(
        question: str,
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> ResearchQuestionRefinement:
        return ResearchQuestionRefinement(
            research_question="For Munich, list concrete documented initiatives with direct evidence."
        )

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
        question="What initiatives exist for Munich?",
        config=config,
        sql_plan_func=_stub_sql_plan,
        markdown_func=_capture_markdown_question,
        refine_question_func=_refine_for_test,
        decide_func=_stub_decision,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    assert (
        captured["question"]
        == "For Munich, list concrete documented initiatives with direct evidence."
    )
    context_bundle = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    assert (
        context_bundle["research_question"]
        == "For Munich, list concrete documented initiatives with direct evidence."
    )
