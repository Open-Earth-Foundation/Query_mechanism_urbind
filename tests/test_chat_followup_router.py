import json
from pathlib import Path

import pytest

from backend.modules.orchestrator import agent as orchestrator_agent
from backend.modules.orchestrator.models import ChatFollowupDecision
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)


class _FakeRunResult:
    def __init__(self, final_output: ChatFollowupDecision) -> None:
        self.final_output = final_output


def _markdown_researcher_config() -> MarkdownResearcherConfig:
    return MarkdownResearcherConfig(
        model="test-model",
        chunk_overlap_tokens=2000,
        batch_max_chunks=32,
        max_workers=8,
        request_backoff_base_seconds=0.5,
        request_backoff_max_seconds=2.0,
    )


def _build_test_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        orchestrator=OrchestratorConfig(
            model="test-model",
            context_bundle_name="context_bundle.json",
        ),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=_markdown_researcher_config(),
        writer=AgentConfig(model="test-model"),
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
        enable_sql=False,
    )


@pytest.mark.parametrize(
    ("decision", "expected_action"),
    [
        (
            ChatFollowupDecision(
                action="answer_from_context",
                reason="Existing excerpts already answer the request.",
                confidence=0.9,
            ),
            "answer_from_context",
        ),
        (
            ChatFollowupDecision(
                action="out_of_scope",
                reason="The message is unrelated to the report context.",
                confidence=0.95,
            ),
            "out_of_scope",
        ),
        (
            ChatFollowupDecision(
                action="search_single_city",
                reason="Fresh Munich context is required.",
                target_city="Munich",
                rewritten_question="What does Munich report?",
                confidence=0.8,
            ),
            "search_single_city",
        ),
        (
            ChatFollowupDecision(
                action="needs_city_clarification",
                reason="A new search is needed but the city is ambiguous.",
                confidence=0.7,
            ),
            "needs_city_clarification",
        ),
    ],
)
def test_route_chat_followup_returns_structured_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: ChatFollowupDecision,
    expected_action: str,
) -> None:
    config = _build_test_config(tmp_path)
    payload = {
        "user_message": "Tell me more about Munich.",
        "original_question": "Build doc",
        "history": [{"role": "user", "content": "Start here."}],
        "selected_run_ids": ["run-parent"],
        "selected_followup_bundle_ids": ["fup_chat_001_munich"],
        "contexts": [],
    }
    sentinel_agent = object()
    captured_input: dict[str, object] = {}

    monkeypatch.setattr(
        orchestrator_agent,
        "build_chat_followup_router_agent",
        lambda *_args, **_kwargs: sentinel_agent,
    )

    def _fake_run_agent_sync(
        agent: object,
        input_text: str,
        max_turns: int,
        log_llm_payload: bool,
    ) -> _FakeRunResult:
        assert agent is sentinel_agent
        assert max_turns == config.retry.max_attempts
        assert not log_llm_payload
        captured_input.update(json.loads(input_text))
        return _FakeRunResult(decision)

    monkeypatch.setattr(orchestrator_agent, "run_agent_sync", _fake_run_agent_sync)

    result = orchestrator_agent.route_chat_followup(
        payload=payload,
        config=config,
        api_key="test-key",
    )

    assert captured_input == payload
    assert result.action == expected_action
    assert result == decision


def test_route_chat_followup_rejects_unstructured_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    sentinel_agent = object()

    class _UnexpectedResult:
        final_output = {"action": "answer_from_context"}

    monkeypatch.setattr(
        orchestrator_agent,
        "build_chat_followup_router_agent",
        lambda *_args, **_kwargs: sentinel_agent,
    )
    monkeypatch.setattr(
        orchestrator_agent,
        "run_agent_sync",
        lambda *_args, **_kwargs: _UnexpectedResult(),
    )

    with pytest.raises(ValueError, match="structured decision"):
        orchestrator_agent.route_chat_followup(
            payload={"user_message": "Hello"},
            config=config,
            api_key="test-key",
        )
