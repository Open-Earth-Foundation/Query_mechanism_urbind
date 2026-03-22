import json

from agents.exceptions import MaxTurnsExceeded
from pytest import MonkeyPatch

from backend.modules.markdown_researcher import agent as markdown_agent
from backend.modules.markdown_researcher.agent import extract_markdown_excerpts
from backend.modules.markdown_researcher.models import MarkdownExcerpt, MarkdownResearchResult
from backend.utils.config import AppConfig
from tests.support import build_test_app_config


class _FakeRunResult:
    def __init__(self, final_output: MarkdownResearchResult) -> None:
        self.final_output = final_output
        self.raw_responses: list[object] = []


def _build_test_config() -> AppConfig:
    """Build the markdown researcher test config with required sections."""
    return build_test_app_config(
        markdown_researcher_overrides={
            "max_workers": 2,
            "request_backoff_base_seconds": 0.1,
            "request_backoff_max_seconds": 0.1,
        },
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
        },
        {
            "path": "B.md",
            "city_name": "B",
            "city_key": "b",
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
            "city_key": "onlycity",
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
        chunk_ids = [str(chunk["chunk_id"]) for chunk in payload["chunks"]]
        return _FakeRunResult(
            MarkdownResearchResult(
                accepted_chunk_ids=chunk_ids,
                rejected_chunk_ids=[],
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


def test_markdown_payload_capture_failed_only_records_failed_batches(
    monkeypatch: MonkeyPatch,
) -> None:
    config = _build_test_config()
    documents = [
        {
            "path": "A.md",
            "city_name": "A",
            "city_key": "a",
            "content": "alpha",
            "chunk_id": "a1",
        },
        {
            "path": "B.md",
            "city_name": "B",
            "city_key": "b",
            "content": "beta",
            "chunk_id": "b1",
        },
    ]
    captured_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(markdown_agent, "build_markdown_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        if payload["city_name"] == "a":
            return _FakeRunResult(
                MarkdownResearchResult(
                    accepted_chunk_ids=["a1"],
                    rejected_chunk_ids=[],
                    excerpts=[],
                )
            )
        raise MaxTurnsExceeded("max turns")

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extract_markdown_excerpts(
        "question?",
        documents,
        config,
        api_key="test",
        batch_payload_mode="failed_only",
        batch_payload_recorder=captured_payloads.append,
    )

    assert result.status == "success"
    assert len(captured_payloads) == 1
    failed_batch_payload = captured_payloads[0]
    assert failed_batch_payload["city_name"] == "b"
    attempts = failed_batch_payload["attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["outcome"] == "max_turns_exceeded"


def test_markdown_payload_capture_all_records_success_and_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    config = _build_test_config()
    documents = [
        {
            "path": "A.md",
            "city_name": "A",
            "city_key": "a",
            "content": "alpha",
            "chunk_id": "a1",
        },
        {
            "path": "B.md",
            "city_name": "B",
            "city_key": "b",
            "content": "beta",
            "chunk_id": "b1",
        },
    ]
    captured_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(markdown_agent, "build_markdown_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        if payload["city_name"] == "a":
            return _FakeRunResult(
                MarkdownResearchResult(
                    accepted_chunk_ids=["a1"],
                    rejected_chunk_ids=[],
                    excerpts=[],
                )
            )
        raise MaxTurnsExceeded("max turns")

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extract_markdown_excerpts(
        "question?",
        documents,
        config,
        api_key="test",
        batch_payload_mode="all",
        batch_payload_recorder=captured_payloads.append,
    )

    assert result.status == "success"
    assert [payload["city_name"] for payload in captured_payloads] == ["a", "b"]
