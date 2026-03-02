from pathlib import Path

import pytest

from backend.api.services import context_chat
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    ChatConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)
from backend.utils.tokenization import count_tokens


def _catalog_entry(ref_id: str, token_repeats: int) -> dict[str, str]:
    return {
        "ref_id": ref_id,
        "city_name": "Munich",
        "quote": "evidence " * token_repeats,
        "partial_answer": "grounded partial answer",
    }


def test_fit_citation_catalog_to_budget_prunes_refs() -> None:
    citation_catalog = [
        _catalog_entry("ref_1", 8),
        _catalog_entry("ref_2", 180),
        _catalog_entry("ref_3", 8),
    ]
    prompt_header = context_chat._build_system_prompt_header(
        original_question="What is the policy status?",
        retry_missing_citation=False,
    )
    user_content = "Summarize the policy."
    fixed_tokens = context_chat._estimate_messages_tokens(
        [{"role": "user", "content": user_content}]
    )
    first_entry_budget = (
        count_tokens(context_chat._render_citation_catalog_block(citation_catalog[:1])) + 20
    )
    token_cap = (
        fixed_tokens
        + count_tokens(prompt_header)
        + context_chat.CHAT_PROMPT_TOKEN_BUFFER
        + first_entry_budget
    )

    fitted = context_chat._fit_citation_catalog_to_budget(
        citation_catalog=citation_catalog,
        prompt_header=prompt_header,
        history=[],
        user_content=user_content,
        token_cap=token_cap,
    )

    assert [item["ref_id"] for item in fitted] == ["ref_1"]


def test_render_citation_catalog_block_for_empty_entries() -> None:
    rendered = context_chat._render_citation_catalog_block([])
    assert "No citation entries fit within the prompt token budget" in rendered


def test_system_prompt_header_avoids_inline_allowed_ref_list() -> None:
    header = context_chat._build_system_prompt_header(
        original_question="What does Aachen do for rooftop solar?",
        retry_missing_citation=False,
    )
    assert "Allowed references for this turn:" not in header
    assert "present in that catalog" in header
    assert "calculator tool is available" in header


def test_generate_context_chat_reply_forwards_reasoning_effort(
    monkeypatch,
) -> None:
    captured_request_kwargs: dict[str, object] = {}

    class _DummyResponse:
        choices = [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]

    def _stub_run_chat_completion_with_tools(
        *,
        client: object,
        messages: list[dict[str, str]],
        request_kwargs: dict[str, object],
        max_tool_rounds: int,
    ) -> object:
        _ = client, messages, max_tool_rounds
        captured_request_kwargs.update(request_kwargs)
        return _DummyResponse()

    monkeypatch.setattr(
        context_chat,
        "_run_chat_completion_with_tools",
        _stub_run_chat_completion_with_tools,
    )

    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(model="openai/gpt-5.2", reasoning_effort="high"),
        runs_dir=Path("output"),
        markdown_dir=Path("documents"),
        enable_sql=False,
    )
    result = context_chat.generate_context_chat_reply(
        original_question="Question",
        contexts=[
            {
                "run_id": "run-1",
                "question": "Question",
                "final_document": "# Final",
                "context_bundle": {"markdown": {"status": "success", "excerpts": []}},
            }
        ],
        history=[],
        user_content="Answer briefly.",
        config=config,
        api_key_override="sk-or-v1-test",
        citation_catalog=[],
    )

    assert result == "ok"
    assert captured_request_kwargs["model"] == "openai/gpt-5.2"
    assert captured_request_kwargs["reasoning_effort"] == "high"


def test_generate_context_chat_reply_rejects_over_hard_token_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyResponse:
        choices = [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]

    def _stub_run_chat_completion_with_tools(
        *,
        client: object,
        messages: list[dict[str, str]],
        request_kwargs: dict[str, object],
        max_tool_rounds: int,
    ) -> object:
        _ = client, messages, request_kwargs, max_tool_rounds
        return _DummyResponse()

    monkeypatch.setattr(
        context_chat,
        "_run_chat_completion_with_tools",
        _stub_run_chat_completion_with_tools,
    )
    monkeypatch.setattr(
        context_chat,
        "_estimate_messages_tokens",
        lambda _messages: context_chat.MAX_CHAT_CONTEXT_TOTAL_TOKENS + 1,
    )

    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(model="openai/gpt-5.2", reasoning_effort="high"),
        runs_dir=Path("output"),
        markdown_dir=Path("documents"),
        enable_sql=False,
    )

    with pytest.raises(ValueError, match="Chat context exceeds token budget"):
        context_chat.generate_context_chat_reply(
            original_question="Question",
            contexts=[
                {
                    "run_id": "run-1",
                    "question": "Question",
                    "final_document": "# Final",
                    "context_bundle": {"markdown": {"status": "success", "excerpts": []}},
                }
            ],
            history=[],
            user_content="Answer briefly.",
            config=config,
            token_cap=context_chat.MAX_CHAT_CONTEXT_TOTAL_TOKENS + 30_000,
            api_key_override="sk-or-v1-test",
            citation_catalog=[],
        )
