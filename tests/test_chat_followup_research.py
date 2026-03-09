import json
from pathlib import Path

import pytest

from backend.api.services import chat_followup_research
from backend.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.orchestrator.models import ResearchQuestionRefinement
from backend.modules.vector_store.models import RetrievedChunk
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)


def _build_test_config(tmp_path: Path, *, vector_store_enabled: bool) -> AppConfig:
    return AppConfig(
        orchestrator=OrchestratorConfig(
            model="test-model",
            context_bundle_name="context_bundle.json",
        ),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
        enable_sql=True,
        vector_store={
            "enabled": vector_store_enabled,
        },
    )


def test_run_chat_followup_search_uses_vector_store_retrieval_and_persists_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path, vector_store_enabled=True)
    retrieved_chunk = RetrievedChunk(
        city_name="Munich",
        raw_text="Munich retrieved chunk",
        source_path="documents/Munich.md",
        heading_path="Overview",
        block_type="paragraph",
        distance=0.12,
        chunk_id="chunk-1",
        metadata={"city_key": "munich"},
    )

    monkeypatch.setattr(
        chat_followup_research,
        "refine_research_question",
        lambda **kwargs: ResearchQuestionRefinement(
            research_question="What does Munich report about rooftop solar?",
            retrieval_queries=["Munich rooftop solar", "Munich solar targets"],
        ),
    )

    def _fake_retrieve_chunks_for_queries(
        *,
        queries: list[str],
        config: AppConfig,
        docs_dir: Path,
        selected_cities: list[str],
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        assert queries == [
            "What does Munich report about rooftop solar?",
            "Munich rooftop solar",
            "Munich solar targets",
        ]
        assert docs_dir == config.markdown_dir
        assert selected_cities == ["Munich"]
        return [retrieved_chunk], {"mode": "vector"}

    monkeypatch.setattr(
        chat_followup_research,
        "retrieve_chunks_for_queries",
        _fake_retrieve_chunks_for_queries,
    )
    monkeypatch.setattr(
        chat_followup_research,
        "as_markdown_documents",
        lambda chunks: [
            {
                "city_name": chunk.city_name,
                "content": chunk.raw_text,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ],
    )
    monkeypatch.setattr(
        chat_followup_research,
        "extract_markdown_excerpts",
        lambda *args, **kwargs: MarkdownResearchResult(
            excerpts=[
                MarkdownExcerpt(
                    quote="Munich plans rooftop solar expansion.",
                    city_name="Munich",
                    partial_answer="Munich plans rooftop solar expansion.",
                    source_chunk_ids=["chunk-1"],
                )
            ]
        ),
    )

    result = chat_followup_research.run_chat_followup_search(
        runs_dir=config.runs_dir,
        run_id="run-vector",
        conversation_id="conversation-1",
        turn_index=1,
        question="Tell me more about Munich solar.",
        target_city="Munich",
        config=config,
        api_key="test-key",
    )

    assert result.status == "success"
    assert result.excerpt_count == 1
    assert result.target_city == "Munich"
    assert result.total_tokens > 0
    assert not config.enable_sql

    bundle_dir = chat_followup_research.followup_bundle_dir(
        runs_dir=config.runs_dir,
        run_id="run-vector",
        conversation_id="conversation-1",
        bundle_id=result.bundle_id,
    )
    context_bundle = json.loads((bundle_dir / "context_bundle.json").read_text(encoding="utf-8"))
    references = json.loads((bundle_dir / "markdown" / "references.json").read_text(encoding="utf-8"))
    retrieval = json.loads((bundle_dir / "markdown" / "retrieval.json").read_text(encoding="utf-8"))

    assert context_bundle["source"] == "chat_followup"
    assert context_bundle["parent_run_id"] == "run-vector"
    assert context_bundle["conversation_id"] == "conversation-1"
    assert context_bundle["target_city"] == "Munich"
    assert context_bundle["research_question"] == "What does Munich report about rooftop solar?"
    assert context_bundle["retrieval_queries"] == [
        "What does Munich report about rooftop solar?",
        "Munich rooftop solar",
        "Munich solar targets",
    ]
    assert context_bundle["markdown"]["source_mode"] == "vector_store_retrieval"
    assert context_bundle["markdown"]["selected_city_names"] == ["Munich"]
    assert context_bundle["markdown"]["inspected_city_names"] == ["Munich"]
    assert context_bundle["markdown"]["excerpts"][0]["ref_id"] == "ref_1"
    assert references["references"][0]["ref_id"] == "ref_1"
    assert references["references"][0]["source_chunk_ids"] == ["chunk-1"]
    assert retrieval["selected_cities"] == ["Munich"]
    assert retrieval["chunks"][0]["chunk_id"] == "chunk-1"


def test_run_chat_followup_search_falls_back_to_standard_markdown_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path, vector_store_enabled=False)
    loaded = {"called": False}

    monkeypatch.setattr(
        chat_followup_research,
        "refine_research_question",
        lambda **kwargs: ResearchQuestionRefinement(
            research_question="What does Munich report?",
            retrieval_queries=["Munich report"],
        ),
    )
    monkeypatch.setattr(
        chat_followup_research,
        "retrieve_chunks_for_queries",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Vector retrieval should not run.")),
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
    assert not (bundle_dir / "markdown" / "retrieval.json").exists()


def test_run_chat_followup_search_persists_empty_successful_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path, vector_store_enabled=False)

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
    config = _build_test_config(tmp_path, vector_store_enabled=False)

    monkeypatch.setattr(
        chat_followup_research,
        "refine_research_question",
        lambda **kwargs: ResearchQuestionRefinement(
            research_question="What does Atlantis report?",
            retrieval_queries=["Atlantis report"],
        ),
    )

    def _failing_load_markdown_documents(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise ValueError("Selected city is not available in markdown documents.")

    monkeypatch.setattr(
        chat_followup_research,
        "load_markdown_documents",
        _failing_load_markdown_documents,
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
    assert "Selected city is not available" in (result.error_message or "")

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
    assert context_bundle["markdown"]["error"]["message"] == "Selected city is not available in markdown documents."


def test_run_chat_followup_search_fails_fast_for_unindexed_vector_store_city(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path, vector_store_enabled=True)

    monkeypatch.setattr(
        chat_followup_research,
        "list_indexed_city_names",
        lambda _config: ["Munich"],
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
        run_id="run-vector-error",
        conversation_id="conversation-1",
        turn_index=5,
        question="Tell me more about Berlin.",
        target_city="Berlin",
        config=config,
        api_key="test-key",
    )

    assert result.status == "error"
    assert result.error_code == chat_followup_research.CHAT_FOLLOWUP_CITY_UNAVAILABLE
    assert result.error_message == "Selected city is not available in the vector store index."

    bundle_dir = chat_followup_research.followup_bundle_dir(
        runs_dir=config.runs_dir,
        run_id="run-vector-error",
        conversation_id="conversation-1",
        bundle_id=result.bundle_id,
    )
    context_bundle = json.loads((bundle_dir / "context_bundle.json").read_text(encoding="utf-8"))
    assert context_bundle["target_city"] == "Berlin"
    assert (
        context_bundle["markdown"]["error"]["code"]
        == chat_followup_research.CHAT_FOLLOWUP_CITY_UNAVAILABLE
    )
    assert (
        context_bundle["markdown"]["error"]["message"]
        == "Selected city is not available in the vector store index."
    )
