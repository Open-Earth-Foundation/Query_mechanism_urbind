import json
from pathlib import Path

import pytest

from backend.api.services import context_chat
from backend.utils.config import (
    WriterConfig,
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
    from backend.utils.config import load_config
    config = load_config()
    prompt_token_buffer = config.chat.prompt_token_buffer
    first_entry_budget = (
        count_tokens(context_chat._render_citation_catalog_block(citation_catalog[:1])) + 20
    )
    token_cap = (
        fixed_tokens
        + count_tokens(prompt_header)
        + prompt_token_buffer
        + first_entry_budget
    )


    fitted = context_chat._fit_citation_catalog_to_budget(
        citation_catalog=citation_catalog,
        prompt_header=prompt_header,
        history=[],
        user_content=user_content,
        token_cap=token_cap,
        prompt_token_buffer=prompt_token_buffer,
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
        writer=WriterConfig(model="test-model"),
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


def test_generate_context_chat_reply_rejects_citation_path_over_token_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When using citations, a message that exceeds the hard token cap raises an error."""

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
    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=WriterConfig(model="test-model"),
        chat=ChatConfig(model="openai/gpt-5.2", reasoning_effort="high"),
        runs_dir=Path("output"),
        markdown_dir=Path("documents"),
        enable_sql=False,
    )

    monkeypatch.setattr(
        context_chat,
        "_estimate_messages_tokens",
        lambda _messages: config.chat.max_context_total_tokens + 1,
    )

    citation_catalog = [
        {"ref_id": "ref_1", "city_name": "Munich", "quote": "evidence", "partial_answer": "answer"}
    ]
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
            token_cap=config.chat.max_context_total_tokens + 30_000,
            api_key_override="sk-or-v1-test",
            citation_catalog=citation_catalog,
        )


def test_run_chat_completion_with_tools_preserves_unsupported_tool_behavior() -> None:
    class _DummyCompletions:
        def __init__(self) -> None:
            """Initialize counter and captured message buffer."""
            self.call_count = 0
            self.last_messages: list[dict[str, object]] = []

        def create(
            self,
            messages: list[dict[str, object]],
            **kwargs: object,
        ) -> object:
            """Return a tool-call response first, then a final response."""
            _ = kwargs
            self.call_count += 1
            self.last_messages = list(messages)
            if self.call_count == 1:
                tool_call = type(
                    "ToolCall",
                    (),
                    {
                        "id": "call-1",
                        "function": type(
                            "Function",
                            (),
                            {"name": "unknown_tool", "arguments": "{}"},
                        )(),
                    },
                )()
                message = type("Message", (), {"content": "", "tool_calls": [tool_call]})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()
            message = type("Message", (), {"content": "done", "tool_calls": []})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = _DummyCompletions()
    client = type(
        "DummyClient",
        (),
        {"chat": type("DummyChat", (), {"completions": completions})()},
    )()

    response = context_chat._run_chat_completion_with_tools(
        client=client,
        messages=[{"role": "user", "content": "test"}],
        request_kwargs={},
        max_tool_rounds=3,
    )

    assert completions.call_count == 2
    assert response.choices[0].message.content == "done"
    tool_messages = [
        message
        for message in completions.last_messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    assert tool_messages
    tool_payload = json.loads(str(tool_messages[-1]["content"]))
    assert tool_payload == {"error": "Unsupported tool: unknown_tool"}
