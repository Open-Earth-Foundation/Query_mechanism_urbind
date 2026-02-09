from app.models import ErrorInfo
from app.modules.markdown_researcher.models import MarkdownExcerpt, MarkdownResearchResult
from app.modules.orchestrator.models import OrchestratorDecision
from app.modules.sql_researcher.models import SqlQuery, SqlQueryPlan, SqlQueryResult, SqlResearchResult
from app.modules.writer.models import WriterOutput


def test_model_validation() -> None:
    query = SqlQuery(query_id="q1", query="SELECT 1")
    plan = SqlQueryPlan(queries=[query])
    result = SqlQueryResult(
        query_id="q1",
        columns=["value"],
        rows=[[1]],
        row_count=1,
        elapsed_ms=5,
        token_count=3,
    )
    research = SqlResearchResult(
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
    md_result = MarkdownResearchResult(excerpts=[excerpt])

    decision = OrchestratorDecision(action="write", reason="Enough data")

    writer = WriterOutput(content="# Answer")

    assert plan.queries[0].query == "SELECT 1"
    assert research.total_token_count == 3
    assert md_result.excerpts[0].city_name == "Munich"
    assert decision.action == "write"
    assert writer.content.startswith("#")

    error = ErrorInfo(code="E1", message="fail")
    assert error.code == "E1"
