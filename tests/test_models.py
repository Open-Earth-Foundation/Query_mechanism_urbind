from app.models import ErrorInfo
from app.modules.markdown_researcher.models import (
    MarkdownCityScope,
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from app.modules.orchestrator.models import OrchestratorDecision
from app.modules.sql_researcher.models import SqlQuery, SqlQueryPlan, SqlQueryResult, SqlResearchResult
from app.modules.writer.models import WriterOutput


def test_model_validation() -> None:
    query = SqlQuery(query_id="q1", query="SELECT 1")
    plan = SqlQueryPlan(run_id="run1", queries=[query])
    result = SqlQueryResult(
        query_id="q1",
        columns=["value"],
        rows=[[1]],
        row_count=1,
        elapsed_ms=5,
        token_count=3,
    )
    research = SqlResearchResult(
        run_id="run1",
        queries=[query],
        results=[result],
        total_token_count=3,
        truncation_applied=False,
    )

    excerpt = MarkdownExcerpt(
        snippet="Hello",
        city_name="Munich",
        answer="Sample answer",
        relevant="yes",
    )
    scope = MarkdownCityScope(run_id="run1", scope="subset", city_names=["Munich"])
    md_result = MarkdownResearchResult(run_id="run1", excerpts=[excerpt], city_scope=scope)

    decision = OrchestratorDecision(run_id="run1", action="write", reason="Enough data")

    writer = WriterOutput(run_id="run1", content="# Answer")

    assert plan.run_id == "run1"
    assert research.total_token_count == 3
    assert md_result.excerpts[0].city_name == "Munich"
    assert md_result.city_scope and md_result.city_scope.scope == "subset"
    assert decision.action == "write"
    assert writer.content.startswith("#")

    error = ErrorInfo(code="E1", message="fail")
    assert error.code == "E1"
