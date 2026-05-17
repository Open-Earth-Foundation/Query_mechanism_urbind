import pytest
from pydantic import ValidationError

from backend.models import ErrorInfo
from backend.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.orchestrator.models import (
    ChatFollowupDecision,
    OrchestratorDecision,
    RetrievalQueryOverride,
)
from backend.modules.writer.models import WriterOutput


def test_model_validation() -> None:
    excerpt = MarkdownExcerpt(
        quote="Munich has deployed 43 existing public chargers as of 2024.",
        city_name="Munich",
        partial_answer="Munich has deployed 43 existing public chargers as of 2024.",
    )
    md_result = MarkdownResearchResult(excerpts=[excerpt])

    decision = OrchestratorDecision(action="write", reason="Enough data")
    query_override = RetrievalQueryOverride(
        primary_query="For Munich, list documented initiatives with evidence."
    )
    followup_decision = ChatFollowupDecision(
        action="search_single_city",
        reason="Fresh context is needed for Munich.",
        target_city="Munich",
        rewritten_question="What does Munich report?",
    )

    writer = WriterOutput(content="# Answer")

    assert md_result.excerpts[0].city_name == "Munich"
    assert decision.action == "write"
    assert query_override.primary_query.startswith("For Munich")
    assert followup_decision.target_city == "Munich"
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
        OrchestratorDecision(action="draft", reason="Need more data")
