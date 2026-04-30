import json
import logging

from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
from pytest import MonkeyPatch

from backend.modules.markdown_researcher import agent as markdown_agent
from backend.modules.markdown_researcher.agent import extract_markdown_excerpts
from backend.modules.markdown_researcher.models import MarkdownExcerpt, MarkdownResearchResult
from tests.support import build_test_app_config


class _FakeRunResult:
    """Small test double that matches the runner result surface used by the agent."""

    def __init__(self, final_output: MarkdownResearchResult) -> None:
        self.final_output = final_output
        self.raw_responses: list[object] = []


def _build_config():
    """Build a small markdown config that forces one top-level batch in tests."""
    return build_test_app_config(
        markdown_researcher_overrides={
            "batch_max_chunks": 4,
            "batch_max_input_tokens": 100,
            "max_workers": 1,
            "request_backoff_base_seconds": 0.0,
            "request_backoff_max_seconds": 0.0,
        },
        retry_overrides={
            "max_attempts": 3,
            "backoff_base_seconds": 0.0,
            "backoff_max_seconds": 0.0,
        },
    )


def _build_documents() -> list[dict[str, object]]:
    """Build one deterministic four-chunk city batch for split-recovery tests."""
    return [
        {
            "path": "Riga.md",
            "city_name": "Riga",
            "city_key": "riga",
            "content": f"content-{idx}",
            "chunk_id": f"c{idx}",
            "chunk_index": idx,
        }
        for idx in range(1, 5)
    ]


def test_markdown_split_recovery_preserves_successful_child_results(
    monkeypatch: MonkeyPatch,
) -> None:
    """Successful child branches should survive sibling failures."""
    config = _build_config()
    documents = _build_documents()
    call_counts: dict[tuple[str, ...], int] = {}

    monkeypatch.setattr(
        markdown_agent,
        "build_markdown_agent",
        lambda *_args, **_kwargs: object(),
    )

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        chunk_ids = tuple(str(chunk["chunk_id"]) for chunk in payload["chunks"])
        call_counts[chunk_ids] = call_counts.get(chunk_ids, 0) + 1

        if chunk_ids == ("c1", "c2", "c3", "c4"):
            raise ModelBehaviorError("parent failure")
        if chunk_ids == ("c1", "c2"):
            return _FakeRunResult(
                MarkdownResearchResult(
                    excerpts=[
                        MarkdownExcerpt(
                            quote="Riga retrofits public buildings.",
                            city_name="Riga",
                            partial_answer="Riga retrofits public buildings.",
                            source_chunk_ids=["c1"],
                        )
                    ],
                    accepted_chunk_ids=["c1"],
                    rejected_chunk_ids=["c2"],
                )
            )
        if chunk_ids == ("c3", "c4"):
            raise ModelBehaviorError("child failure")
        if chunk_ids == ("c3",):
            return _FakeRunResult(
                MarkdownResearchResult(
                    excerpts=[],
                    rejected_chunk_ids=["c3"],
                )
            )
        if chunk_ids == ("c4",):
            raise ModelBehaviorError("leaf failure")
        raise AssertionError(f"Unexpected chunk ids: {chunk_ids}")

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extract_markdown_excerpts("question?", documents, config, api_key="test")

    assert result.status == "success"
    assert result.error is not None
    assert result.error.code == "MARKDOWN_PARTIAL_BATCH_FAILURE"
    assert result.accepted_chunk_ids == ["c1"]
    assert result.rejected_chunk_ids == ["c2", "c3"]
    assert result.unresolved_chunk_ids == ["c4"]
    assert len(result.excerpts) == 1
    assert result.excerpts[0].source_chunk_ids == ["c1"]
    assert len(result.batch_failures) == 1
    assert result.batch_failures[0].split_path == "2.2"
    assert result.batch_failures[0].unresolved_chunk_ids == ["c4"]
    assert call_counts == {
        ("c1", "c2", "c3", "c4"): 3,
        ("c1", "c2"): 1,
        ("c3", "c4"): 1,
        ("c3",): 1,
        ("c4",): 1,
    }


def test_markdown_split_children_get_one_attempt_each(
    monkeypatch: MonkeyPatch,
) -> None:
    """Split children should run once each after parent retries are exhausted."""
    config = _build_config()
    documents = _build_documents()
    call_counts: dict[tuple[str, ...], int] = {}

    monkeypatch.setattr(
        markdown_agent,
        "build_markdown_agent",
        lambda *_args, **_kwargs: object(),
    )

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        chunk_ids = tuple(str(chunk["chunk_id"]) for chunk in payload["chunks"])
        call_counts[chunk_ids] = call_counts.get(chunk_ids, 0) + 1

        if chunk_ids == ("c1", "c2", "c3", "c4"):
            raise ModelBehaviorError("parent failure")
        return _FakeRunResult(
            MarkdownResearchResult(
                excerpts=[],
                rejected_chunk_ids=list(chunk_ids),
            )
        )

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extract_markdown_excerpts("question?", documents, config, api_key="test")

    assert result.status == "success"
    assert result.error is None
    assert result.batch_failures == []
    assert result.unresolved_chunk_ids == []
    assert result.rejected_chunk_ids == ["c1", "c2", "c3", "c4"]
    assert call_counts == {
        ("c1", "c2", "c3", "c4"): 3,
        ("c1", "c2"): 1,
        ("c3", "c4"): 1,
    }


def test_markdown_splits_after_parent_max_turns_are_exhausted(
    monkeypatch: MonkeyPatch,
) -> None:
    """Parent batches should only split after using their full max-turn retry budget."""
    config = _build_config()
    documents = _build_documents()
    call_counts: dict[tuple[str, ...], int] = {}

    monkeypatch.setattr(
        markdown_agent,
        "build_markdown_agent",
        lambda *_args, **_kwargs: object(),
    )

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        chunk_ids = tuple(str(chunk["chunk_id"]) for chunk in payload["chunks"])
        call_counts[chunk_ids] = call_counts.get(chunk_ids, 0) + 1

        if chunk_ids == ("c1", "c2", "c3", "c4"):
            raise MaxTurnsExceeded("max turns")
        return _FakeRunResult(
            MarkdownResearchResult(
                excerpts=[],
                rejected_chunk_ids=list(chunk_ids),
            )
        )

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extract_markdown_excerpts("question?", documents, config, api_key="test")

    assert result.status == "success"
    assert result.error is None
    assert result.batch_failures == []
    assert result.unresolved_chunk_ids == []
    assert result.rejected_chunk_ids == ["c1", "c2", "c3", "c4"]
    assert call_counts == {
        ("c1", "c2", "c3", "c4"): 3,
        ("c1", "c2"): 1,
        ("c3", "c4"): 1,
    }


def test_markdown_non_retryable_failures_do_not_split(
    monkeypatch: MonkeyPatch,
) -> None:
    """Non-retryable failures should still stop without recursive splitting."""
    config = _build_config()
    documents = _build_documents()
    call_counts: dict[tuple[str, ...], int] = {}

    monkeypatch.setattr(
        markdown_agent,
        "build_markdown_agent",
        lambda *_args, **_kwargs: object(),
    )

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        chunk_ids = tuple(str(chunk["chunk_id"]) for chunk in payload["chunks"])
        call_counts[chunk_ids] = call_counts.get(chunk_ids, 0) + 1
        raise RuntimeError("boom")

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extract_markdown_excerpts("question?", documents, config, api_key="test")

    assert result.status == "success"
    assert result.error is not None
    assert result.error.code == "MARKDOWN_ALL_BATCHES_FAILED"
    assert result.unresolved_chunk_ids == ["c1", "c2", "c3", "c4"]
    assert len(result.batch_failures) == 1
    assert result.batch_failures[0].split_path is None
    assert call_counts == {("c1", "c2", "c3", "c4"): 1}


def test_markdown_logs_failure_points_for_split_recovery(
    monkeypatch: MonkeyPatch,
    caplog,
) -> None:
    """Failed parent and leaf batches should emit summary logs with split lineage."""
    config = _build_config()
    documents = _build_documents()

    monkeypatch.setattr(
        markdown_agent,
        "build_markdown_agent",
        lambda *_args, **_kwargs: object(),
    )

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = json.loads(input_data)
        chunk_ids = tuple(str(chunk["chunk_id"]) for chunk in payload["chunks"])
        if chunk_ids == ("c1", "c2", "c3", "c4"):
            raise ModelBehaviorError("parent failure")
        if chunk_ids == ("c1", "c2"):
            return _FakeRunResult(
                MarkdownResearchResult(
                    excerpts=[],
                    rejected_chunk_ids=["c1", "c2"],
                )
            )
        if chunk_ids == ("c3", "c4"):
            raise ModelBehaviorError("child failure")
        if chunk_ids == ("c3",):
            return _FakeRunResult(
                MarkdownResearchResult(
                    excerpts=[],
                    rejected_chunk_ids=["c3"],
                )
            )
        if chunk_ids == ("c4",):
            raise ModelBehaviorError("leaf failure")
        raise AssertionError(f"Unexpected chunk ids: {chunk_ids}")

    monkeypatch.setattr(markdown_agent, "run_agent_sync", _fake_run_agent_sync)
    caplog.set_level(logging.WARNING, logger=markdown_agent.__name__)

    extract_markdown_excerpts("question?", documents, config, api_key="test")

    messages = [record.message for record in caplog.records]
    assert any("split=root markdown batch failed" in message for message in messages)
    assert any("split=2.2 markdown batch failed" in message for message in messages)
