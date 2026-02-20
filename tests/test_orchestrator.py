import json
import logging
import sqlite3
from pathlib import Path

import pytest

from backend.utils.config import (
    AppConfig,
    AgentConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)
from backend.modules.orchestrator.module import run_pipeline
from backend.modules.orchestrator.models import (
    ResearchQuestionRefinement,
)
from backend.modules.sql_researcher.models import SqlQuery, SqlQueryPlan
from backend.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.vector_store.models import RetrievedChunk
from backend.modules.writer.models import WriterOutput
from backend.utils.logging_config import setup_logger


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
        quote="Munich has deployed 43 existing public chargers as of 2024.",
        city_name="Munich",
        partial_answer="Munich has deployed 43 existing public chargers as of 2024.",
    )
    return MarkdownResearchResult(excerpts=[excerpt])


def _stub_refine_question(
    question: str,
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> ResearchQuestionRefinement:
    return ResearchQuestionRefinement(
        research_question=question,
        retrieval_queries=[],
    )


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
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed"


def test_run_pipeline_writes_output_with_sql_disabled(
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
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed"
    assert run_log["finish_reason"] == "completed (write)"


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
            research_question="For Munich, list concrete documented initiatives with direct evidence.",
            retrieval_queries=[],
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


def test_run_pipeline_end_to_end_propagates_query_markdown_and_writer_output(
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

    input_question = "What initiatives exist for Munich?"
    refined_question = "For Munich, list concrete documented initiatives with direct evidence."
    expected_quote = "Munich has deployed 43 existing public chargers as of 2024."
    expected_partial_answer = (
        "Munich has deployed 43 existing public chargers as of 2024."
    )
    observed: dict[str, object] = {}

    def _refine_for_test(
        question: str,
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> ResearchQuestionRefinement:
        assert question == input_question
        return ResearchQuestionRefinement(
            research_question=refined_question,
            retrieval_queries=[],
        )

    def _markdown_for_test(
        question: str,
        documents: list[dict[str, str]],
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> MarkdownResearchResult:
        observed["markdown_question"] = question
        observed["markdown_documents"] = documents
        return MarkdownResearchResult(
            excerpts=[
                MarkdownExcerpt(
                    quote=expected_quote,
                    city_name="Munich",
                    partial_answer=expected_partial_answer,
                )
            ]
        )

    def _writer_for_test(
        question: str,
        context_bundle: dict,
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> WriterOutput:
        observed["writer_question"] = question
        observed["writer_context_bundle"] = context_bundle
        excerpt = context_bundle["markdown"]["excerpts"][0]
        content = f"# Answer\n\n{excerpt['partial_answer']}"
        return WriterOutput(content=content)

    paths = run_pipeline(
        question=input_question,
        config=config,
        sql_plan_func=_stub_sql_plan,
        markdown_func=_markdown_for_test,
        refine_question_func=_refine_for_test,
        writer_func=_writer_for_test,
    )

    assert paths.final_output.exists()
    assert observed["markdown_question"] == refined_question
    assert observed["writer_question"] == input_question

    markdown_documents = observed["markdown_documents"]
    assert isinstance(markdown_documents, list)
    assert len(markdown_documents) == 1

    writer_bundle = observed["writer_context_bundle"]
    assert isinstance(writer_bundle, dict)
    markdown_bundle = writer_bundle["markdown"]
    assert isinstance(markdown_bundle, dict)
    assert markdown_bundle["status"] == "success"
    assert markdown_bundle["inspected_cities"] == ["Munich"]
    assert markdown_bundle["excerpt_count"] == 1
    excerpts = markdown_bundle["excerpts"]
    assert isinstance(excerpts, list)
    assert len(excerpts) == 1
    first_excerpt = excerpts[0]
    assert "quote" in first_excerpt
    assert "partial_answer" in first_excerpt
    assert "snippet" not in first_excerpt
    assert first_excerpt["quote"] == expected_quote
    assert first_excerpt["partial_answer"] == expected_partial_answer

    assert writer_bundle["markdown"] == markdown_bundle

    final_output = paths.final_output.read_text(encoding="utf-8")
    assert f"# Question\n{input_question}\n\n" in final_output
    assert expected_partial_answer in final_output

    persisted_context_bundle = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    assert persisted_context_bundle["research_question"] == refined_question
    persisted_markdown = persisted_context_bundle["markdown"]
    assert isinstance(persisted_markdown, dict)
    assert persisted_markdown["inspected_cities"] == ["Munich"]
    assert persisted_markdown["excerpt_count"] == 1
    assert persisted_markdown["excerpts"][0]["quote"] == expected_quote
    assert persisted_markdown["excerpts"][0]["partial_answer"] == expected_partial_answer

    markdown_artifact = json.loads(paths.markdown_excerpts.read_text(encoding="utf-8"))
    assert markdown_artifact["inspected_cities"] == ["Munich"]
    assert markdown_artifact["excerpt_count"] == 1
    artifact_excerpt = markdown_artifact["excerpts"][0]
    assert "quote" in artifact_excerpt
    assert "partial_answer" in artifact_excerpt
    assert "snippet" not in artifact_excerpt

    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert run_log["inputs"]["markdown_chunk_count"] == 1
    assert run_log["inputs"]["markdown_excerpt_count"] == 1

    run_summary = paths.run_summary.read_text(encoding="utf-8")
    assert "Markdown chunk count: 1" in run_summary
    assert "Markdown excerpt count: 1" in run_summary


def test_run_pipeline_vector_store_enabled_uses_retriever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    config.vector_store.enabled = True

    captured: dict[str, object] = {}

    def _fake_retrieve(
        queries: list[str],
        config: AppConfig,
        docs_dir: Path,
        selected_cities: list[str] | None,
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        captured["queries"] = queries
        captured["selected_cities"] = selected_cities
        captured["docs_dir"] = docs_dir
        return (
            [
                RetrievedChunk(
                    city_name="Munich",
                    raw_text="Retriever chunk content",
                    source_path="documents/Munich.md",
                    heading_path="H1",
                    block_type="paragraph",
                    distance=0.111111,
                    chunk_id="chunk-1",
                )
            ],
            {"queries": queries, "per_city": []},
        )

    monkeypatch.setattr(
        "backend.modules.orchestrator.module.retrieve_chunks_for_queries",
        _fake_retrieve,
    )

    observed: dict[str, object] = {}

    def _capture_markdown_documents(
        question: str,
        documents: list[dict[str, str]],
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> MarkdownResearchResult:
        observed["documents"] = documents
        return _stub_markdown(question, documents, config, api_key, **_kwargs)

    def _refine_with_retrieval_queries(
        question: str,
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> ResearchQuestionRefinement:
        return ResearchQuestionRefinement(
            research_question=question,
            retrieval_queries=[
                "Munich initiatives charging retrofit policy",
                "Munich charging counts retrofit targets timeline metrics",
            ],
        )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        selected_cities=["Munich"],
        sql_plan_func=_stub_sql_plan,
        markdown_func=_capture_markdown_documents,
        refine_question_func=_refine_with_retrieval_queries,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    assert captured["selected_cities"] == ["Munich"]
    assert captured["queries"] == [
        "What initiatives exist for Munich?",
        "Munich initiatives charging retrofit policy",
        "Munich charging counts retrofit targets timeline metrics",
    ]

    markdown_documents = observed["documents"]
    assert isinstance(markdown_documents, list)
    assert len(markdown_documents) == 1
    assert markdown_documents[0]["content"] == "Retriever chunk content"
    assert markdown_documents[0]["chunk_id"] == "chunk-1"

    retrieval_path = paths.markdown_dir / "retrieval.json"
    assert retrieval_path.exists()
    retrieval_payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
    assert retrieval_payload["queries"] == [
        "What initiatives exist for Munich?",
        "Munich initiatives charging retrofit policy",
        "Munich charging counts retrofit targets timeline metrics",
    ]
    assert retrieval_payload["retrieved_count"] == 1

    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert run_log["inputs"]["markdown_source_mode"] == "vector_store_retrieval"

    markdown_artifact = json.loads(paths.markdown_excerpts.read_text(encoding="utf-8"))
    assert markdown_artifact["retrieval_mode"] == "vector_store_retrieval"
    assert markdown_artifact["retrieval_queries"] == [
        "What initiatives exist for Munich?",
        "Munich initiatives charging retrofit policy",
        "Munich charging counts retrofit targets timeline metrics",
    ]
