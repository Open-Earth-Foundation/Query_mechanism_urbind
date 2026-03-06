import json

from agents.exceptions import MaxTurnsExceeded
from pytest import MonkeyPatch

from backend.modules.markdown_researcher import agent as markdown_agent
from backend.modules.markdown_researcher.agent import (
    build_markdown_agent,
    extract_markdown_excerpts,
)
from backend.modules.markdown_researcher.models import (
    MarkdownBatchFailure,
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.utils.config import (
    WriterConfig,
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
        writer=WriterConfig(model="test"),
    )


def test_markdown_returns_partial_success_with_city_failure_markers(
    monkeypatch: MonkeyPatch,
) -> None:
    config = _build_test_config()
    documents = [
        {
            "path": "A.md",
            "city_name": "A",
            "city_key": "a",
            "content": "A content",
            "chunk_id": "a-1",
        },
        {
            "path": "B.md",
            "city_name": "B",
            "city_key": "b",
            "content": "B content",
            "chunk_id": "b-1",
        },
    ]

    monkeypatch.setattr(markdown_agent, "build_markdown_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        city_name = payload["city_name"]
        chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in payload.get("chunks", [])]
        if str(city_name).casefold() == "a":
            accepted_chunk_ids = [chunk_ids[0]] if chunk_ids else []
            rejected_chunk_ids = chunk_ids[1:] if len(chunk_ids) > 1 else []
            return _FakeRunResult(
                MarkdownResearchResult(
                    accepted_chunk_ids=accepted_chunk_ids,
                    rejected_chunk_ids=rejected_chunk_ids,
                    excerpts=[
                        MarkdownExcerpt(
                            quote="City A allocated EUR 1.2 million to retrofit public buildings in 2024.",
                            city_name="A",
                            partial_answer="City A allocated EUR 1.2 million to retrofit public buildings in 2024.",
                            source_chunk_ids=accepted_chunk_ids,
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
    assert result.unresolved_chunk_ids == ["b-1"]


def test_markdown_returns_success_when_all_batches_hit_max_turns(
    monkeypatch: MonkeyPatch,
) -> None:
    config = _build_test_config()
    documents = [
        {
            "path": "OnlyCity.md",
            "city_name": "OnlyCity",
            "city_key": "onlycity",
            "content": "content",
            "chunk_id": "only-1",
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
    assert result.unresolved_chunk_ids == ["only-1"]


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
            "city_key": "a",
            "content": "alpha",
            "chunk_id": "a1",
        },
        {
            "path": "A.md",
            "city_name": "A",
            "city_key": "a",
            "content": "beta",
            "chunk_id": "a2",
        },
        {
            "path": "A.md",
            "city_name": "A",
            "city_key": "a",
            "content": "gamma",
            "chunk_id": "a3",
        },
        {
            "path": "B.md",
            "city_name": "B",
            "city_key": "b",
            "content": "delta",
            "chunk_id": "b1",
        },
    ]
    captured_payloads: list[dict] = []

    monkeypatch.setattr(markdown_agent, "build_markdown_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        captured_payloads.append(payload)
        chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in payload.get("chunks", [])]
        return _FakeRunResult(
            MarkdownResearchResult(
                accepted_chunk_ids=[],
                rejected_chunk_ids=chunk_ids,
                excerpts=[],
            )
        )

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


def test_markdown_uses_markdown_researcher_max_turns(
    monkeypatch: MonkeyPatch,
) -> None:
    config = _build_test_config()
    config.markdown_researcher.max_turns = 10
    config.retry.max_attempts = 5
    documents = [
        {
            "path": "OnlyCity.md",
            "city_name": "OnlyCity",
            "city_key": "onlycity",
            "content": "content",
            "chunk_id": "only-1",
        }
    ]
    observed_max_turns: list[int] = []

    monkeypatch.setattr(markdown_agent, "build_markdown_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(_agent: object, _input_data: str, **kwargs: object) -> _FakeRunResult:
        max_turns = kwargs.get("max_turns")
        assert isinstance(max_turns, int)
        observed_max_turns.append(max_turns)
        return _FakeRunResult(
            MarkdownResearchResult(
                accepted_chunk_ids=[],
                rejected_chunk_ids=["only-1"],
                excerpts=[],
            )
        )

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extract_markdown_excerpts("question?", documents, config, api_key="test")

    assert result.status == "success"
    assert observed_max_turns == [10]


def test_markdown_decision_validation_marks_invalid_batch_as_unresolved(
    monkeypatch: MonkeyPatch,
) -> None:
    config = _build_test_config()
    config.retry.max_attempts = 1
    documents = [
        {
            "path": "OnlyCity.md",
            "city_name": "OnlyCity",
            "city_key": "onlycity",
            "content": "content",
            "chunk_id": "only-1",
        }
    ]

    monkeypatch.setattr(markdown_agent, "build_markdown_agent", lambda *_args, **_kwargs: object())

    def _invalid_decision_output(
        _agent: object, _input_data: str, **_kwargs: object
    ) -> _FakeRunResult:
        return _FakeRunResult(
            MarkdownResearchResult(
                accepted_chunk_ids=[],
                rejected_chunk_ids=[],
                excerpts=[],
            )
        )

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _invalid_decision_output)

    result = extract_markdown_excerpts("question?", documents, config, api_key="test")

    assert result.status == "success"
    assert result.error is not None
    assert result.error.code == "MARKDOWN_ALL_BATCHES_FAILED"
    assert result.unresolved_chunk_ids == ["only-1"]


def test_markdown_batch_failure_schema_is_strict_for_agents_tooling() -> None:
    schema = MarkdownResearchResult.model_json_schema()
    batch_failures_schema = schema["properties"]["batch_failures"]
    items_schema = batch_failures_schema["items"]

    assert items_schema.get("additionalProperties") is not True


def test_build_markdown_agent_supports_strict_tool_schema() -> None:
    config = _build_test_config()
    agent = build_markdown_agent(config, api_key="test")
    assert agent is not None


def test_markdown_research_result_accepts_typed_batch_failures() -> None:
    result = MarkdownResearchResult(
        status="success",
        batch_failures=[
            MarkdownBatchFailure(
                city_name="aachen",
                batch_index=1,
                reason="UserError",
                unresolved_chunk_ids=["chunk-1"],
            )
        ],
    )

    assert len(result.batch_failures) == 1
    assert result.batch_failures[0].city_name == "aachen"
