import json
import sqlite3
from pathlib import Path

from app.utils.config import AppConfig, AgentConfig, MarkdownResearcherConfig, OrchestratorConfig, SqlResearcherConfig
from app.modules.orchestrator.module import run_pipeline
from app.modules.orchestrator.models import OrchestratorDecision
from app.modules.sql_researcher.models import SqlQuery, SqlQueryPlan
from app.modules.markdown_researcher.models import MarkdownExcerpt, MarkdownResearchResult
from app.modules.writer.models import WriterOutput


def _stub_sql_plan(question, schema_summary, city_names, run_id, config, api_key, **_kwargs):
    return SqlQueryPlan(
        run_id=run_id,
        queries=[SqlQuery(query_id="q1", query="SELECT cityName FROM City")],
    )


def _stub_markdown(question, documents, run_id, config, api_key):
    excerpt = MarkdownExcerpt(
        snippet="Sample",
        city_name="Munich",
        answer="Stub answer",
        relevant="yes",
    )
    return MarkdownResearchResult(run_id=run_id, excerpts=[excerpt])


def _stub_decision(question, context_bundle, run_id, config, api_key):
    return OrchestratorDecision(run_id=run_id, action="write", reason="Enough")


def _stub_writer(question, context_bundle, run_id, config, api_key):
    return WriterOutput(run_id=run_id, content="# Answer\n\nStub")


def test_run_pipeline_creates_artifacts(tmp_path, monkeypatch):
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
    )

    paths = run_pipeline(
        question="What cities exist?",
        config=config,
        sql_plan_func=_stub_sql_plan,
        markdown_func=_stub_markdown,
        decide_func=_stub_decision,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    run_log = json.loads(paths.run_log.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed"
    assert Path(run_log["artifacts"]["final_output"]).exists()
