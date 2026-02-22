import json

from agents.exceptions import MaxTurnsExceeded
from pytest import MonkeyPatch

from backend.modules.markdown_researcher import agent as markdown_agent
from backend.modules.markdown_researcher.agent import extract_markdown_excerpts
from backend.modules.markdown_researcher.models import MarkdownExcerpt, MarkdownResearchResult
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)


class _FakeRunResult:
    def __init__(self, final_output: MarkdownResearchResult) -> None:
        self.final_output = final_output
        self.raw_responses: list[object] = []


def _build_test_config() -> AppConfig:
    return AppConfig(
        orchestrator=OrchestratorConfig(model="test", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test"),
        markdown_researcher=MarkdownResearcherConfig(
            model="test",
            max_workers=2,
            request_backoff_base_seconds=0.1,
            request_backoff_max_seconds=0.1,
        ),
        writer=AgentConfig(model="test"),
    )


def test_markdown_returns_partial_success_with_city_failure_markers(
    monkeypatch: MonkeyPatch,
) -> None:
    config = _build_test_config()
    documents = [
        {
            "path": "A.md",
            "city_name": "A",
            "content": "A content",
        },
        {
            "path": "B.md",
            "city_name": "B",
            "content": "B content",
        },
    ]

    monkeypatch.setattr(markdown_agent, "build_markdown_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        city_name = payload["city_name"]
        if str(city_name).casefold() == "a":
            return _FakeRunResult(
                MarkdownResearchResult(
                    excerpts=[
                        MarkdownExcerpt(
                            quote="City A allocated EUR 1.2 million to retrofit public buildings in 2024.",
                            city_name="A",
                            partial_answer="City A allocated EUR 1.2 million to retrofit public buildings in 2024.",
                        )
                    ]
                )
            )
        raise MaxTurnsExceeded("max turns")

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extract_markdown_excerpts("question?", documents, config, api_key="test")

    assert result.status == "success"
    assert len(result.excerpts) == 1
    assert result.error is not None
    assert result.error.code == "MARKDOWN_PARTIAL_BATCH_FAILURE"
    assert result.error.details is not None
    assert "b#batch1: MARKDOWN_MAX_TURNS_EXCEEDED" in result.error.details


def test_markdown_returns_success_when_all_batches_hit_max_turns(
    monkeypatch: MonkeyPatch,
) -> None:
    config = _build_test_config()
    documents = [
        {
            "path": "OnlyCity.md",
            "city_name": "OnlyCity",
            "content": "content",
        }
    ]

    monkeypatch.setattr(markdown_agent, "build_markdown_agent", lambda *_args, **_kwargs: object())

    def _always_max_turns(_agent: object, _input_data: str, **_kwargs: object) -> _FakeRunResult:
        raise MaxTurnsExceeded("max turns")

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _always_max_turns)

    result = extract_markdown_excerpts("question?", documents, config, api_key="test")

    assert result.status == "success"
    assert result.excerpts == []
    assert result.error is not None
    assert result.error.code == "MARKDOWN_ALL_BATCHES_FAILED"
    assert result.error.details is not None
    assert "onlycity#batch1: MARKDOWN_MAX_TURNS_EXCEEDED" in result.error.details


def test_markdown_payload_batches_keep_city_chunk_integrity(
    monkeypatch: MonkeyPatch,
) -> None:
    config = _build_test_config()
    config.markdown_researcher.batch_max_chunks = 2
    config.markdown_researcher.batch_max_input_tokens = 20
    documents = [
        {
            "path": "A.md",
            "city_name": "A",
            "content": "alpha",
            "chunk_id": "a1",
        },
        {
            "path": "A.md",
            "city_name": "A",
            "content": "beta",
            "chunk_id": "a2",
        },
        {
            "path": "A.md",
            "city_name": "A",
            "content": "gamma",
            "chunk_id": "a3",
        },
        {
            "path": "B.md",
            "city_name": "B",
            "content": "delta",
            "chunk_id": "b1",
        },
    ]
    captured_payloads: list[dict] = []

    monkeypatch.setattr(markdown_agent, "build_markdown_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        captured_payloads.append(payload)
        return _FakeRunResult(MarkdownResearchResult(excerpts=[]))

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extract_markdown_excerpts("question?", documents, config, api_key="test")

    assert result.status == "success"
    assert result.excerpts == []
    assert len(captured_payloads) == 3

    seen_chunk_ids: list[str] = []
    for payload in captured_payloads:
        city_name = payload["city_name"]
        chunks = payload["chunks"]
        assert 1 <= len(chunks) <= 2
        for chunk in chunks:
            if str(city_name).casefold() == "a":
                assert chunk["chunk_id"].startswith("a")
            if str(city_name).casefold() == "b":
                assert chunk["chunk_id"].startswith("b")
            seen_chunk_ids.append(chunk["chunk_id"])

    assert seen_chunk_ids == ["a1", "a2", "a3", "b1"]
