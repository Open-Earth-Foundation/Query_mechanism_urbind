import pytest
from pydantic import ValidationError

from app.models import ErrorInfo
from app.modules.markdown_researcher.models import MarkdownExcerpt, MarkdownResearchResult
from app.modules.orchestrator.models import (
    OrchestratorDecision,
    ResearchQuestionRefinement,
)
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
        partial_answer="Sample answer",
        relevant="yes",
    )
    md_result = MarkdownResearchResult(excerpts=[excerpt])

    decision = OrchestratorDecision(action="write", reason="Enough data")
    refinement = ResearchQuestionRefinement(
        research_question="For Munich, list documented initiatives with evidence."
    )

    writer = WriterOutput(content="# Answer")

    assert plan.queries[0].query == "SELECT 1"
    assert research.total_token_count == 3
    assert md_result.excerpts[0].city_name == "Munich"
    assert decision.action == "write"
    assert refinement.research_question.startswith("For Munich")
    assert writer.content.startswith("#")

    error = ErrorInfo(code="E1", message="fail")
    assert error.code == "E1"


def test_markdown_excerpt_accepts_legacy_answer_field() -> None:
    excerpt = MarkdownExcerpt.model_validate(
        {
            "snippet": "City report excerpt",
            "city_name": "Munich",
            "answer": "The answer is: Munich has 43 existing public chargers.",
            "relevant": "yes",
        }
    )

    assert excerpt.partial_answer == "Munich has 43 existing public chargers."


def test_orchestrator_decision_rejects_legacy_actions() -> None:
    with pytest.raises(ValidationError):
        OrchestratorDecision(action="run_sql", reason="Need more data")
