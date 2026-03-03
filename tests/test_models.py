import pytest
from pydantic import ValidationError

from backend.models import ErrorInfo
from backend.modules.calculation_researcher.models import (
    CalculationRequest,
    CalculationSubagentOutput,
)
from backend.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.orchestrator.models import (
    OrchestratorDecision,
    ResearchQuestionRefinement,
)
from backend.modules.sql_researcher.models import (
    SqlQuery,
    SqlQueryPlan,
    SqlQueryResult,
    SqlResearchResult,
)
from backend.modules.writer.models import WriterOutput


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
        quote="Munich has deployed 43 existing public chargers as of 2024.",
        city_name="Munich",
        partial_answer="Munich has deployed 43 existing public chargers as of 2024.",
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


def test_markdown_excerpt_accepts_quote_and_partial_answer_fields() -> None:
    excerpt = MarkdownExcerpt.model_validate(
        {
            "quote": "Munich has deployed 43 existing public chargers as of 2024.",
            "city_name": "Munich",
            "partial_answer": "Munich has deployed 43 existing public chargers as of 2024.",
        }
    )

    assert (
        excerpt.partial_answer
        == "Munich has deployed 43 existing public chargers as of 2024."
    )


def test_orchestrator_decision_rejects_legacy_actions() -> None:
    with pytest.raises(ValidationError):
        OrchestratorDecision(action="run_sql", reason="Need more data")


def test_calculation_request_requires_target_year_for_user_specified_rule() -> None:
    with pytest.raises(ValidationError):
        CalculationRequest(
            calculation_goal="Calculate total electric cars.",
            metric_name="electric cars",
            operation="sum",
            city_scope=["Munich"],
            inclusion_rule="registered passenger EV stock",
            exclusion_rule="municipal fleet",
            year_rule="user_specified_year",
            target_year=None,
            unit_rule="vehicles_count",
        )


def test_calculation_subagent_output_rejects_invalid_ref_ids() -> None:
    with pytest.raises(ValidationError):
        CalculationSubagentOutput(
            status="success",
            metric_name="electric cars",
            operation="sum",
            total_value=10,
            unit="vehicles_count",
            coverage_observed=1,
            coverage_total=1,
            included_cities=[
                {
                    "city_name": "Munich",
                    "year": 2024,
                    "value": 10,
                    "unit": "vehicles_count",
                    "ref_ids": ["ref_invalid"],
                }
            ],
            excluded_policy_cities=[],
            assumptions=[],
            final_ref_ids=["ref_1"],
            error=None,
        )
