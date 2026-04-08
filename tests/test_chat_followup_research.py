import json
from pathlib import Path

import pytest

from backend.api.services import chat_followup_research
from backend.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.orchestrator.models import ResearchQuestionRefinement
from backend.utils.config import (
    AppConfig,
    AssumptionsReviewerConfig,
    ChatConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    RetryConfig,
    WriterConfig,
)


def _build_test_config(tmp_path: Path) -> AppConfig:
    """Build a minimal markdown-only config for follow-up search tests."""
    return AppConfig(
        orchestrator=OrchestratorConfig(
            model="test-model",
            context_bundle_name="context_bundle.json",
        ),
        markdown_researcher=MarkdownResearcherConfig(
            model="test-model",
            chunk_overlap_tokens=2000,
            batch_max_chunks=32,
            max_workers=8,
            request_backoff_base_seconds=0.5,
            request_backoff_max_seconds=2.0,
        ),
        writer=WriterConfig(model="test-model"),
        chat=ChatConfig(
            model="test-model",
            provider_timeout_seconds=60.0,
            followup_router_max_excerpts_per_source=50,
        ),
        assumptions_reviewer=AssumptionsReviewerConfig(model="test-model"),
        retry=RetryConfig(backoff_base_seconds=1.0, backoff_max_seconds=30.0),
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
    )


def test_run_chat_followup_search_persists_standard_markdown_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    loaded = {"called": False}

    monkeypatch.setattr(
        chat_followup_research,
        "list_city_names",
        lambda _markdown_dir: ["Munich"],
    )
    monkeypatch.setattr(
        chat_followup_research,
        "refine_research_question",
        lambda **kwargs: ResearchQuestionRefinement(
            research_question="What does Munich report?",
            retrieval_queries=["Munich report"],
        ),
    )

    def _fake_load_markdown_documents(
        markdown_dir: Path,
        markdown_config: MarkdownResearcherConfig,
        selected_cities: list[str] | None = None,
    ) -> list[dict[str, object]]:
        _ = markdown_config
        loaded["called"] = True
        assert markdown_dir == config.markdown_dir
        assert selected_cities == ["Munich"]
        return [{"city_name": "Munich", "content": "Munich markdown content"}]

    monkeypatch.setattr(
        chat_followup_research,
        "load_markdown_documents",
        _fake_load_markdown_documents,
    )
    monkeypatch.setattr(
        chat_followup_research,
        "extract_markdown_excerpts",
        lambda *args, **kwargs: MarkdownResearchResult(
            excerpts=[
                MarkdownExcerpt(
                    quote="Munich markdown quote.",
                    city_name="Munich",
                    partial_answer="Munich markdown quote.",
                    source_chunk_ids=["chunk-fallback-1"],
                )
            ]
        ),
    )

    result = chat_followup_research.run_chat_followup_search(
        runs_dir=config.runs_dir,
        run_id="run-fallback",
        conversation_id="conversation-1",
        turn_index=2,
        question="Tell me more about Munich.",
        target_city="Munich",
        config=config,
        api_key="test-key",
    )

    assert loaded["called"]
    assert result.status == "success"
    bundle_dir = chat_followup_research.followup_bundle_dir(
        runs_dir=config.runs_dir,
        run_id="run-fallback",
        conversation_id="conversation-1",
        bundle_id=result.bundle_id,
    )
    context_bundle = json.loads((bundle_dir / "context_bundle.json").read_text(encoding="utf-8"))
    assert context_bundle["markdown"]["source_mode"] == "standard_chunking"
    assert context_bundle["markdown"]["selected_city_names"] == ["Munich"]
    assert not (bundle_dir / "markdown" / "retrieval.json").exists()


def test_run_chat_followup_search_persists_empty_successful_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)

    monkeypatch.setattr(
        chat_followup_research,
        "list_city_names",
        lambda _markdown_dir: ["Munich"],
    )
    monkeypatch.setattr(
        chat_followup_research,
        "refine_research_question",
        lambda **kwargs: ResearchQuestionRefinement(
            research_question="What does Munich report?",
            retrieval_queries=[],
        ),
    )
    monkeypatch.setattr(
        chat_followup_research,
        "load_markdown_documents",
        lambda *args, **kwargs: [{"city_name": "Munich", "content": "No answer here"}],
    )
    monkeypatch.setattr(
        chat_followup_research,
        "extract_markdown_excerpts",
        lambda *args, **kwargs: MarkdownResearchResult(excerpts=[]),
    )

    result = chat_followup_research.run_chat_followup_search(
        runs_dir=config.runs_dir,
        run_id="run-empty",
        conversation_id="conversation-1",
        turn_index=3,
        question="Tell me more about Munich.",
        target_city="Munich",
        config=config,
        api_key="test-key",
    )

    assert result.status == "success"
    assert result.excerpt_count == 0
    assert result.error_message is None

    bundle_dir = chat_followup_research.followup_bundle_dir(
        runs_dir=config.runs_dir,
        run_id="run-empty",
        conversation_id="conversation-1",
        bundle_id=result.bundle_id,
    )
    context_bundle = json.loads((bundle_dir / "context_bundle.json").read_text(encoding="utf-8"))
    references = json.loads((bundle_dir / "markdown" / "references.json").read_text(encoding="utf-8"))
    assert context_bundle["markdown"]["excerpts"] == []
    assert context_bundle["markdown"]["excerpt_count"] == 0
    assert context_bundle["markdown"]["inspected_city_names"] == []
    assert references["references"] == []


def test_run_chat_followup_search_persists_error_bundle_for_invalid_city(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)

    monkeypatch.setattr(
        chat_followup_research,
        "list_city_names",
        lambda _markdown_dir: ["Munich"],
    )
    monkeypatch.setattr(
        chat_followup_research,
        "refine_research_question",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Question refinement should not run for unavailable cities.")
        ),
    )

    result = chat_followup_research.run_chat_followup_search(
        runs_dir=config.runs_dir,
        run_id="run-error",
        conversation_id="conversation-1",
        turn_index=4,
        question="Tell me more about Atlantis.",
        target_city="Atlantis",
        config=config,
        api_key="test-key",
    )

    assert result.status == "error"
    assert result.error_code == chat_followup_research.CHAT_FOLLOWUP_CITY_UNAVAILABLE
    assert result.error_message == "Selected city is not available in markdown documents."

    bundle_dir = chat_followup_research.followup_bundle_dir(
        runs_dir=config.runs_dir,
        run_id="run-error",
        conversation_id="conversation-1",
        bundle_id=result.bundle_id,
    )
    context_bundle = json.loads((bundle_dir / "context_bundle.json").read_text(encoding="utf-8"))
    assert context_bundle["target_city"] == "Atlantis"
    assert context_bundle["markdown"]["status"] == "error"
    assert context_bundle["markdown"]["source_mode"] == "error"
    assert (
        context_bundle["markdown"]["error"]["code"]
        == chat_followup_research.CHAT_FOLLOWUP_CITY_UNAVAILABLE
    )
    assert (
        context_bundle["markdown"]["error"]["message"]
        == "Selected city is not available in markdown documents."
    )
