import json
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
    assert "calculator tools are available" in header


def test_chat_tool_definitions_include_all_calculator_tools() -> None:
    expected_names = {
        "sum_numbers",
        "subtract_numbers",
        "multiply_numbers",
        "divide_numbers",
    }
    actual_names: set[str] = set()
    for definition in context_chat.CHAT_TOOL_DEFINITIONS:
        function_definition = definition.get("function")
        if not isinstance(function_definition, dict):
            continue
        name = function_definition.get("name")
        if isinstance(name, str):
            actual_names.add(name)
    assert actual_names == expected_names


def test_normalize_subtract_numbers_args_returns_numeric_operands() -> None:
    result = context_chat._normalize_subtract_numbers_args(
        '{"minuend": 10, "subtrahend": "3.5"}'
    )
    assert result == (10.0, 3.5)


def test_normalize_divide_numbers_args_rejects_missing_divisor() -> None:
    with pytest.raises(ValueError, match="`divisor`"):
        context_chat._normalize_divide_numbers_args('{"dividend": 12}')


def test_normalize_multiply_numbers_args_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        context_chat._normalize_multiply_numbers_args("{bad-json")


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


def test_generate_context_chat_reply_routes_citation_overflow_to_map_reduce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized citation catalogs should switch into overflow map-reduce."""
    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(
            model="openai/gpt-5.2",
            reasoning_effort="high",
            max_context_total_tokens=240,
            min_prompt_token_cap=0,
            prompt_token_buffer=0,
        ),
        runs_dir=tmp_path / "output",
        markdown_dir=Path("documents"),
        enable_sql=False,
    )
    overflow_calls: list[dict[str, object]] = []

    monkeypatch.setattr(context_chat, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        context_chat,
        "_run_overflow_evidence_map_reduce",
        lambda **kwargs: overflow_calls.append(kwargs) or "overflow answer [ref_1]",
    )

    citation_catalog = [
        _catalog_entry("ref_1", 20),
        _catalog_entry("ref_2", 220),
    ]
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
        token_cap=config.chat.max_context_total_tokens,
        api_key_override="sk-or-v1-test",
        citation_catalog=citation_catalog,
        run_id="run-1",
    )

    assert result == "overflow answer [ref_1]"
    assert overflow_calls
    assert overflow_calls[0]["run_id"] == "run-1"


def test_load_or_build_evidence_cache_strips_prompt_noise_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached overflow evidence should keep only compact prompt fields and be reusable."""
    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(model="openai/gpt-5.2"),
        runs_dir=tmp_path / "output",
        markdown_dir=Path("documents"),
        enable_sql=False,
    )
    normalized_contexts = [
        context_chat.ChatContextSource(
            run_id="run-1",
            question="Question",
            final_document="# Final",
            context_bundle={
                "final": "/tmp/final.md",
                "markdown": {
                    "status": "success",
                    "error": {"code": "E"},
                    "excerpts": [
                        {
                            "ref_id": "ref_1",
                            "city_name": "Munich",
                            "quote": "evidence",
                            "partial_answer": "answer",
                            "source_chunk_ids": ["chunk-1"],
                        }
                    ],
                },
            },
        )
    ]
    normalized_citations = [
        {
            "ref_id": "ref_1",
            "city_name": "Munich",
            "quote": "evidence",
            "partial_answer": "answer",
        }
    ]

    payload = context_chat._load_or_build_evidence_cache(
        run_id="run-1",
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
        config=config,
    )

    cache_path = context_chat._chat_evidence_cache_path(config.runs_dir, "run-1")
    assert cache_path.exists()
    assert payload["evidence_count"] == 1
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    item = cached["chunks"][0]["items"][0]
    assert set(item.keys()) == {"ref_id", "city_name", "quote", "partial_answer"}
    assert "source_chunk_ids" not in item
    assert "final" not in cached

    monkeypatch.setattr(
        context_chat,
        "_write_json_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Cache should be reused.")),
    )

    reused = context_chat._load_or_build_evidence_cache(
        run_id="run-1",
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
        config=config,
    )
    assert reused["source_signature"] == payload["source_signature"]


def test_run_reduce_passes_batches_recursively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large partial-answer sets should be merged across multiple reduce batches."""
    merge_calls: list[str] = []

    monkeypatch.setattr(
        context_chat,
        "_resolve_prompt_budget",
        lambda **_kwargs: ([], 120),
    )
    monkeypatch.setattr(context_chat, "count_tokens", lambda value: len(value))

    def _stub_run_single_pass(
        *,
        client: object,
        messages: list[dict[str, object]],
        request_kwargs: dict[str, object],
        retry_settings: object,
        run_id: str | None,
        context_count: int,
    ) -> str:
        _ = client, request_kwargs, retry_settings, run_id, context_count
        merge_calls.append(str(messages[0]["content"]))
        return "Merged answer [ref_1]"

    monkeypatch.setattr(context_chat, "_run_single_pass", _stub_run_single_pass)

    result = context_chat._run_reduce_passes(
        partial_answers=[
            "A" * 24 + " [ref_1]",
            "B" * 24 + " [ref_2]",
            "C" * 24 + " [ref_3]",
            "D" * 24 + " [ref_4]",
        ],
        prompt_header="Header",
        bounded_history=[],
        user_content="Summarize.",
        effective_token_cap=400,
        prompt_token_buffer=0,
        client=object(),
        request_kwargs={},
        retry_settings=object(),
        run_id="run-1",
        context_count=1,
    )

    assert result == "Merged answer [ref_1]"
    assert len(merge_calls) == 3


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
