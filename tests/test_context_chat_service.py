import json
import logging
from pathlib import Path

import pytest

import backend.api.services.context_chat_evidence as context_chat_evidence
import backend.api.services.context_chat_execution as context_chat_execution
import backend.api.services.context_chat_io as context_chat_io
import backend.api.services.context_chat_planning as context_chat_planning
import backend.api.services.prompts.context_chat as context_chat_prompts
import backend.api.services.utils.context_chat as context_chat_utils
from backend.api.services import context_chat
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    ChatConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
    load_config,
)
from backend.utils.retry import RetrySettings
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
    prompt_header = context_chat_prompts.build_system_prompt_header(
        original_question="What is the policy status?",
        retry_missing_citation=False,
    )
    user_content = "Summarize the policy."
    fixed_tokens = context_chat_utils.estimate_messages_tokens(
        [{"role": "user", "content": user_content}]
    )
    config = load_config()
    prompt_token_buffer = config.chat.prompt_token_buffer
    first_entry_budget = (
        count_tokens(context_chat_prompts.render_citation_catalog_block(citation_catalog[:1])) + 20
    )
    token_cap = (
        fixed_tokens
        + count_tokens(prompt_header)
        + prompt_token_buffer
        + first_entry_budget
    )

    fitted = context_chat_planning._fit_citation_catalog_to_budget(
        citation_catalog=citation_catalog,
        prompt_header=prompt_header,
        history=[],
        user_content=user_content,
        token_cap=token_cap,
        prompt_token_buffer=prompt_token_buffer,
    )

    assert [item["ref_id"] for item in fitted] == ["ref_1"]


def test_fit_citation_catalog_to_budget_uses_cached_prefix_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    citation_catalog = [
        _catalog_entry("ref_1", 8),
        _catalog_entry("ref_2", 180),
        _catalog_entry("ref_3", 8),
    ]
    prompt_header = context_chat_prompts.build_system_prompt_header(
        original_question="What is the policy status?",
        retry_missing_citation=False,
    )
    user_content = "Summarize the policy."
    config = load_config()
    prompt_token_buffer = config.chat.prompt_token_buffer
    token_cache = context_chat.build_citation_catalog_token_cache(citation_catalog)
    token_cap = (
        context_chat_utils.estimate_messages_tokens([{"role": "user", "content": user_content}])
        + count_tokens(prompt_header)
        + prompt_token_buffer
        + token_cache.prefix_tokens[0]
    )

    monkeypatch.setattr(
        context_chat_prompts,
        "render_citation_catalog_block",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Cached prefix tokens should avoid catalog re-rendering.")
        ),
    )

    fitted = context_chat_planning._fit_citation_catalog_to_budget(
        citation_catalog=citation_catalog,
        prompt_header=prompt_header,
        history=[],
        user_content=user_content,
        token_cap=token_cap,
        prompt_token_buffer=prompt_token_buffer,
        citation_prefix_tokens=token_cache.prefix_tokens,
    )

    assert [item["ref_id"] for item in fitted] == ["ref_1"]


def test_render_citation_catalog_block_for_empty_entries() -> None:
    rendered = context_chat_prompts.render_citation_catalog_block([])
    assert "No citation entries fit within the prompt token budget" in rendered


def test_resolve_request_evidence_blocks_splits_oversized_chunk_once() -> None:
    """Oversized cached chunks should split into two fitting request blocks."""
    items = [
        {
            "ref_id": "ref_1",
            "city_name": "Munich",
            "quote": "evidence " * 80,
            "partial_answer": "answer",
        },
        {
            "ref_id": "ref_2",
            "city_name": "Porto",
            "quote": "evidence " * 80,
            "partial_answer": "answer",
        },
    ]
    cache_chunk = context_chat_evidence._make_evidence_cache_chunk(
        chunk_id="chunk_1",
        items=items,
    )
    block_budget = max(
        count_tokens(context_chat_prompts.render_evidence_items_block([items[0]])),
        count_tokens(context_chat_prompts.render_evidence_items_block([items[1]])),
    )

    blocks = context_chat_evidence._resolve_request_evidence_blocks(
        cache_chunks=[cache_chunk],
        block_budget=block_budget,
    )

    assert len(blocks) == 2
    assert all(count_tokens(block) <= block_budget for block in blocks)


def test_resolve_request_evidence_blocks_errors_when_chunk_cannot_split() -> None:
    """Single-item cached chunks should fail instead of trimming evidence text."""
    cache_chunk = context_chat_evidence._make_evidence_cache_chunk(
        chunk_id="chunk_1",
        items=[
            {
                "ref_id": "ref_1",
                "city_name": "Munich",
                "quote": "evidence " * 120,
                "partial_answer": "answer",
            }
        ],
    )
    token_count = cache_chunk["token_count"]
    assert isinstance(token_count, int)

    with pytest.raises(ValueError, match="cannot be split further"):
        context_chat_evidence._resolve_request_evidence_blocks(
            cache_chunks=[cache_chunk],
            block_budget=token_count - 1,
        )


def test_estimate_context_window_reports_full_catalog_tokens_before_split() -> None:
    citation_catalog = [
        _catalog_entry("ref_1", 8),
        _catalog_entry("ref_2", 180),
        _catalog_entry("ref_3", 8),
    ]
    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(
            model="openai/gpt-5.2",
            max_context_total_tokens=260,
            min_prompt_token_cap=0,
        ),
        runs_dir=Path("output"),
        markdown_dir=Path("documents"),
        enable_sql=False,
    )

    estimate = context_chat.estimate_context_window(
        original_question="What is the policy status?",
        contexts=[
            {
                "run_id": "run-1",
                "question": "Question",
                "final_document": "# Final",
                "context_bundle": {"markdown": {"status": "success", "excerpts": []}},
            }
        ],
        config=config,
        token_cap=config.chat.max_context_total_tokens,
        citation_catalog=citation_catalog,
    )

    assert estimate.context_window_kind == "citation_catalog"
    assert estimate.mode == "split"
    assert estimate.context_window_tokens == count_tokens(
        context_chat_prompts.render_citation_catalog_block(citation_catalog)
    )
    assert estimate.fitted_context_window_tokens is not None
    assert estimate.context_window_tokens > estimate.fitted_context_window_tokens
    assert estimate.fitted_citation_entry_count is not None
    assert estimate.fitted_citation_entry_count < estimate.citation_catalog_entry_count


def test_system_prompt_header_avoids_inline_allowed_ref_list() -> None:
    header = context_chat_prompts.build_system_prompt_header(
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
    result = context_chat_execution._normalize_subtract_numbers_args(
        '{"minuend": 10, "subtrahend": "3.5"}'
    )
    assert result == (10.0, 3.5)


def test_normalize_divide_numbers_args_rejects_missing_divisor() -> None:
    with pytest.raises(ValueError, match="`divisor`"):
        context_chat_execution._normalize_divide_numbers_args('{"dividend": 12}')


def test_normalize_multiply_numbers_args_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        context_chat_execution._normalize_multiply_numbers_args("{bad-json")


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
        context_chat_execution,
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


def test_generate_context_chat_reply_logs_prompt_window_metrics(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct chat execution logs the prompt-window metrics used for the request."""

    class _DummyResponse:
        choices = [type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()]

    monkeypatch.setattr(
        context_chat_execution,
        "_run_chat_completion_with_tools",
        lambda **_kwargs: _DummyResponse(),
    )

    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(model="openai/gpt-5.2"),
        runs_dir=Path("output"),
        markdown_dir=Path("documents"),
        enable_sql=False,
    )

    caplog.set_level(logging.INFO, logger="backend.api.services.context_chat")
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
        citation_catalog=[_catalog_entry("ref_1", 6)],
    )

    assert result == "ok"
    messages = [record.message for record in caplog.records]
    assert any("context_window_kind=citation_catalog" in message for message in messages)
    assert any("citation_entries=1 fitted_citation_entries=1" in message for message in messages)


def test_plan_context_chat_request_prefers_direct_mode_for_small_prompt() -> None:
    """Small grounded prompts should stay on the direct single-pass path."""
    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(
            model="openai/gpt-5.2",
            max_context_total_tokens=2_000,
            min_prompt_token_cap=0,
            prompt_token_buffer=0,
        ),
        runs_dir=Path("output"),
        markdown_dir=Path("documents"),
        enable_sql=False,
    )

    plan = context_chat.plan_context_chat_request(
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
        citation_catalog=[_catalog_entry("ref_1", 6)],
    )

    assert plan.mode == "direct"
    assert plan.estimated_prompt_tokens is not None
    assert plan.split_reason is None
    assert plan.context_window_kind == "citation_catalog"
    assert plan.citation_catalog_entry_count == 1
    assert plan.fitted_citation_entry_count == 1
    assert plan.fitted_citation_ref_ids == ["ref_1"]
    assert plan.context_block_tokens is not None
    assert plan.prompt_header_tokens is not None
    assert plan.history_tokens == context_chat_utils.estimate_messages_tokens([])
    assert plan.user_tokens is not None


def test_plan_context_chat_request_marks_large_catalog_for_split_mode() -> None:
    """Oversized citation catalogs should be routed to split mode before execution."""
    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(
            model="openai/gpt-5.2",
            max_context_total_tokens=260,
            min_prompt_token_cap=0,
            prompt_token_buffer=0,
        ),
        runs_dir=Path("output"),
        markdown_dir=Path("documents"),
        enable_sql=False,
    )

    plan = context_chat.plan_context_chat_request(
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
        citation_catalog=[
            _catalog_entry("ref_1", 18),
            _catalog_entry("ref_2", 220),
        ],
    )

    assert plan.mode == "split"
    assert plan.split_reason is not None
    assert plan.context_window_kind == "citation_catalog"
    assert plan.citation_catalog_entry_count == 2
    assert plan.fitted_citation_entry_count == 1
    assert plan.fitted_citation_ref_ids == ["ref_1"]
    assert plan.context_block_tokens is not None


def test_plan_context_chat_request_reports_serialized_context_window_metrics() -> None:
    """Context-only prompts should expose serialized-context window metrics."""
    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(
            model="openai/gpt-5.2",
            max_context_total_tokens=4_000,
            min_prompt_token_cap=0,
            prompt_token_buffer=0,
            multi_pass_threshold_tokens=4_000,
        ),
        runs_dir=Path("output"),
        markdown_dir=Path("documents"),
        enable_sql=False,
    )

    plan = context_chat.plan_context_chat_request(
        original_question="Question",
        contexts=[
            {
                "run_id": "run-1",
                "question": "Question",
                "final_document": "# Final\n" + ("evidence " * 40),
                "context_bundle": {"markdown": {"status": "success", "excerpts": []}},
            }
        ],
        history=[],
        user_content="Answer briefly.",
        config=config,
        citation_catalog=None,
    )

    assert plan.mode == "direct"
    assert plan.context_window_kind == "serialized_contexts"
    assert plan.context_tokens is not None
    assert plan.context_block_tokens == plan.context_tokens
    assert plan.citation_catalog_entry_count is None
    assert plan.fitted_citation_entry_count is None
    assert plan.fitted_citation_ref_ids is None


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

    monkeypatch.setattr(context_chat_execution, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        context_chat_evidence,
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
    config = load_config()
    config.runs_dir = tmp_path / "output"
    config.markdown_dir = Path("documents")
    config.enable_sql = False
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

    payload = context_chat_evidence._load_or_build_evidence_cache(
        run_id="run-1",
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
        config=config,
    )

    cache_path = context_chat_utils.chat_evidence_cache_path(config.runs_dir, "run-1")
    assert cache_path.exists()
    assert payload["evidence_count"] == 1
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["schema_version"] == context_chat.CHAT_EVIDENCE_CACHE_SCHEMA_VERSION
    assert cached["chunks"][0]["token_count"] == count_tokens(
        context_chat_prompts.render_evidence_items_block(cached["chunks"][0]["items"])
    )
    item = cached["chunks"][0]["items"][0]
    assert set(item.keys()) == {"ref_id", "city_name", "quote", "partial_answer"}
    assert "source_chunk_ids" not in item
    assert "final" not in cached

    monkeypatch.setattr(
        context_chat_io,
        "_write_json_object",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Cache should be reused.")),
    )

    reused = context_chat_evidence._load_or_build_evidence_cache(
        run_id="run-1",
        normalized_contexts=normalized_contexts,
        normalized_citations=normalized_citations,
        config=config,
    )
    assert reused["source_signature"] == payload["source_signature"]


def test_run_overflow_evidence_map_reduce_logs_split_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Overflow map-reduce should log explicit split-mode entry and phase summaries."""
    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test-model", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(model="openai/gpt-5.2", prompt_token_buffer=0),
        runs_dir=tmp_path / "output",
        markdown_dir=Path("documents"),
        enable_sql=False,
    )
    caplog.set_level(logging.INFO, logger=context_chat.__name__)
    monkeypatch.setattr(
        context_chat_evidence,
        "_load_or_build_evidence_cache",
        lambda **_kwargs: {
            "chunks": [
                {
                    "chunk_id": "chunk_1",
                    "items": [
                        {
                            "ref_id": "ref_1",
                            "city_name": "Munich",
                            "quote": "Evidence 1",
                            "partial_answer": "Answer 1",
                        }
                    ],
                },
                {
                    "chunk_id": "chunk_2",
                    "items": [
                        {
                            "ref_id": "ref_2",
                            "city_name": "Porto",
                            "quote": "Evidence 2",
                            "partial_answer": "Answer 2",
                        }
                    ],
                },
            ]
        },
    )
    monkeypatch.setattr(context_chat_evidence, "_resolve_prompt_budget", lambda **_kwargs: ([], 120))
    monkeypatch.setattr(
        context_chat_evidence,
        "_resolve_request_evidence_blocks",
        lambda **_kwargs: ["chunk 1", "chunk 2"],
    )

    responses = iter(
        [
            "Partial one [ref_1]",
            "Partial two [ref_2]",
            "Merged answer [ref_1] [ref_2]",
        ]
    )
    monkeypatch.setattr(
        context_chat_execution,
        "_run_single_pass",
        lambda **_kwargs: next(responses),
    )

    result = context_chat_evidence._run_overflow_evidence_map_reduce(
        prompt_header="Prompt header",
        normalized_contexts=[
            context_chat.ChatContextSource(
                run_id="run-1",
                question="Question",
                final_document="# Final",
                context_bundle={},
            )
        ],
        normalized_citations=[],
        bounded_history=[],
        user_content="Summarize.",
        effective_token_cap=220_000,
        config=config,
        client=object(),
        request_kwargs={},
        retry_settings=RetrySettings.bounded(
            max_attempts=1,
            backoff_base_seconds=0.0,
            backoff_max_seconds=0.0,
        ),
        run_id="run-1",
        context_count=1,
    )

    assert result == "Merged answer [ref_1] [ref_2]"
    messages = [record.message for record in caplog.records]
    assert any("Context chat split mode enabled" in message for message in messages)
    assert any("Context chat split mode map phase" in message for message in messages)
    assert any("Context chat split mode reduce phase" in message for message in messages)


def test_run_reduce_passes_batches_recursively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large partial-answer sets should be merged across multiple reduce batches."""
    merge_calls: list[str] = []

    monkeypatch.setattr(
        context_chat_evidence,
        "_resolve_prompt_budget",
        lambda **_kwargs: ([], 120),
    )
    monkeypatch.setattr(context_chat_evidence, "count_tokens", lambda value: len(value))

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

    monkeypatch.setattr(context_chat_execution, "_run_single_pass", _stub_run_single_pass)

    result = context_chat_evidence._run_reduce_passes(
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

    response = context_chat_execution._run_chat_completion_with_tools(
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
