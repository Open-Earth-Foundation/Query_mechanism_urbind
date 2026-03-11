"""Queued-job and overflow API chat integration tests."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import backend.api.services.utils.context_chat as context_chat_utils
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.services import context_chat
from backend.api.services.chat_followup_research import ChatFollowupSearchResult
from backend.modules.orchestrator.models import ChatFollowupDecision
from backend.utils.config import AppConfig
from backend.utils.paths import RunPaths
from tests.api_chat.support import (
    build_config,
    patch_api_config_loaders,
    poll_chat_job_terminal,
    poll_until_completed,
    write_followup_bundle,
    write_success_artifacts,
)


def test_chat_followup_queued_response_exposes_city_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return build_config(
            runs_dir=runs_dir,
            markdown_dir=markdown_dir,
            followup_search_enabled=True,
        ).model_copy(update={"enable_sql": True})

    def _stub_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
        analysis_mode: str = "aggregate",
        api_key_override: str | None = None,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert analysis_mode == "aggregate"
        assert api_key_override is None
        assert selected_cities is None
        assert config.enable_sql is True
        return write_success_artifacts(question, run_id, config, excerpts=[])

    def _stub_route_chat_followup(
        payload: dict[str, object],
        config: AppConfig,
        api_key: str,
        log_llm_payload: bool = False,
    ) -> ChatFollowupDecision:
        _ = payload, config, api_key, log_llm_payload
        return ChatFollowupDecision(
            action="search_single_city",
            reason="Need fresh context for Izmir.",
            target_city="Izmir",
            rewritten_question="What does Izmir report?",
        )

    def _stub_run_chat_followup_search(
        *,
        runs_dir: Path,
        run_id: str,
        conversation_id: str,
        turn_index: int,
        question: str,
        target_city: str,
        config: AppConfig,
        api_key: str,
        log_llm_payload: bool = False,
    ) -> ChatFollowupSearchResult:
        _ = question, config, api_key, log_llm_payload
        assert turn_index == 1
        assert config.enable_sql is True
        bundle_id = "fup_chat_001_izmir"
        write_followup_bundle(
            runs_dir=runs_dir,
            run_id=run_id,
            conversation_id=conversation_id,
            bundle_id=bundle_id,
            target_city=target_city,
            quote="Izmir follow-up quote.",
            partial_answer="Izmir follow-up answer.",
        )
        return ChatFollowupSearchResult(
            status="success",
            bundle_id=bundle_id,
            target_city=target_city,
            created_at=datetime.now(timezone.utc),
            excerpt_count=1,
            total_tokens=120,
            error_message=None,
        )

    def _stub_build_context_reply_plan(**_kwargs: object) -> context_chat.ContextChatPlan:
        return context_chat.ContextChatPlan(
            mode="split",
            context_ids=["run-chat-followup-queued", "fup_chat_001_izmir"],
            resolved_token_cap=220_000,
            effective_token_cap=220_000,
            estimated_prompt_tokens=221_000,
            context_tokens=221_000,
            split_reason="test split",
        )

    def _stub_generate_reply(
        original_question: str,
        contexts: list[dict[str, object]],
        history: list[dict[str, str]],
        user_content: str,
        config: AppConfig,
        token_cap: int = 0,
        api_key_override: str | None = None,
        citation_catalog: list[dict[str, str]] | None = None,
        citation_prefix_tokens: list[int] | None = None,
        retry_missing_citation: bool = False,
        run_id: str | None = None,
    ) -> str:
        _ = history, config, token_cap, api_key_override, retry_missing_citation
        assert original_question == "Build doc"
        assert user_content == "Tell me more about Izmir."
        assert run_id == "run-chat-followup-queued"
        assert isinstance(citation_catalog, list) and citation_catalog
        assert citation_prefix_tokens is None or isinstance(citation_prefix_tokens, list)
        assert contexts[-1]["run_id"] == "fup_chat_001_izmir"
        assert config.enable_sql is True
        return "Izmir plans district cooling expansion. [ref_1]"

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.chat_followup_flow.route_chat_followup", _stub_route_chat_followup)
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr(
        "backend.api.services.chat_send_service._build_and_log_context_reply_plan",
        _stub_build_context_reply_plan,
    )
    monkeypatch.setattr("backend.api.services.chat_reply_helpers.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-followup-queued"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-followup-queued")

        create_session = client.post(
            "/api/v1/runs/run-chat-followup-queued/chat/sessions",
            json={},
        )
        conversation_id = create_session.json()["conversation_id"]

        send_message = client.post(
            f"/api/v1/runs/run-chat-followup-queued/chat/sessions/{conversation_id}/messages",
            json={"content": "Tell me more about Izmir."},
        )
        assert send_message.status_code == 202
        payload = send_message.json()
        assert payload["mode"] == "queued"
        assert payload["routing"]["action"] == "search_single_city"
        assert payload["routing"]["target_city"] == "Izmir"
        assert payload["routing"]["bundle_id"] == "fup_chat_001_izmir"

        completed_job = poll_chat_job_terminal(
            client,
            "run-chat-followup-queued",
            conversation_id,
            str(payload["job"]["job_id"]),
        )
        assert completed_job["status"] == "completed"

        session_payload = client.get(
            f"/api/v1/runs/run-chat-followup-queued/chat/sessions/{conversation_id}"
        ).json()
        assert (
            session_payload["messages"][-1]["content"]
            == "Izmir plans district cooling expansion. [ref_1]"
        )
        assert session_payload["messages"][-1]["routing"]["target_city"] == "Izmir"


def test_chat_overflow_uses_evidence_map_reduce_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return build_config(
            runs_dir=runs_dir,
            markdown_dir=markdown_dir,
            max_context_total_tokens=600,
            min_prompt_token_cap=0,
            prompt_token_buffer=150,
            multi_pass_chunk_tokens=120,
        )

    def _stub_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
        analysis_mode: str = "aggregate",
        api_key_override: str | None = None,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert analysis_mode == "aggregate"
        assert api_key_override is None
        assert selected_cities is None
        excerpts = [
            {
                "ref_id": "ref_7",
                "city_name": "munich",
                "quote": "Munich evidence " * 120,
                "partial_answer": "Munich partial answer " * 20,
                "source_chunk_ids": ["chunk_hidden_1"],
            },
            {
                "ref_id": "ref_9",
                "city_name": "porto",
                "quote": "Porto evidence " * 120,
                "partial_answer": "Porto partial answer " * 20,
                "source_chunk_ids": ["chunk_hidden_2"],
            },
        ]
        return write_success_artifacts(question, run_id, config, excerpts=excerpts)

    responses = iter(
        [
            "Munich partial analysis. [ref_1]",
            "Porto partial analysis. [ref_2]",
            "Merged answer for both cities. [ref_1] [ref_2]",
            "Second Munich partial. [ref_1]",
            "Second Porto partial. [ref_2]",
            "Second merged answer. [ref_1] [ref_2]",
        ]
    )

    class _DummyResponse:
        def __init__(self, content: str) -> None:
            self.choices = [
                type("Choice", (), {"message": type("Message", (), {"content": content})()})()
            ]

    def _stub_run_chat_completion_with_tools(
        *,
        client: object,
        messages: list[dict[str, object]],
        request_kwargs: dict[str, object],
        max_tool_rounds: int,
    ) -> object:
        _ = client, messages, request_kwargs, max_tool_rounds
        return _DummyResponse(next(responses))

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.context_chat_execution.OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        "backend.api.services.context_chat_execution._run_chat_completion_with_tools",
        _stub_run_chat_completion_with_tools,
    )
    caplog.set_level(logging.INFO, logger="backend.api.routes.chat")

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-overflow"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-overflow")

        create_session = client.post(
            "/api/v1/runs/run-chat-overflow/chat/sessions",
            json={},
        )
        conversation_id = create_session.json()["conversation_id"]

        first_send = client.post(
            f"/api/v1/runs/run-chat-overflow/chat/sessions/{conversation_id}/messages",
            json={"content": "Compare the evidence."},
        )
        assert first_send.status_code == 202
        first_payload = first_send.json()
        assert first_payload["mode"] == "queued"
        assert first_payload["user_message"]["content"] == "Compare the evidence."
        pending_session = client.get(
            f"/api/v1/runs/run-chat-overflow/chat/sessions/{conversation_id}"
        )
        assert pending_session.status_code == 200
        assert pending_session.json()["pending_job"]["job_id"] == first_payload["job"]["job_id"]

        blocked_second_send = client.post(
            f"/api/v1/runs/run-chat-overflow/chat/sessions/{conversation_id}/messages",
            json={"content": "This should be blocked while pending."},
        )
        assert blocked_second_send.status_code == 409

        blocked_context_update = client.put(
            f"/api/v1/runs/run-chat-overflow/chat/sessions/{conversation_id}/contexts",
            json={"context_run_ids": ["run-chat-overflow"]},
        )
        assert blocked_context_update.status_code == 409

        first_job = poll_chat_job_terminal(
            client,
            "run-chat-overflow",
            conversation_id,
            str(first_payload["job"]["job_id"]),
        )
        assert first_job["status"] == "completed"

        first_session = client.get(
            f"/api/v1/runs/run-chat-overflow/chat/sessions/{conversation_id}"
        )
        assert first_session.status_code == 200
        first_session_payload = first_session.json()
        assert first_session_payload["pending_job"] is None
        assert first_session_payload["messages"][-1]["content"] == (
            "Merged answer for both cities. [ref_1] [ref_2]"
        )
        assert [
            citation["source_ref_id"]
            for citation in first_session_payload["messages"][-1]["citations"]
        ] == [
            "ref_7",
            "ref_9",
        ]

        cache_path = context_chat_utils.chat_evidence_cache_path(runs_dir, "run-chat-overflow")
        assert cache_path.exists()
        cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached_payload["schema_version"] == context_chat.CHAT_EVIDENCE_CACHE_SCHEMA_VERSION
        assert cached_payload["evidence_count"] == 2
        assert all(isinstance(chunk.get("token_count"), int) for chunk in cached_payload["chunks"])

        monkeypatch.setattr(
            "backend.api.services.context_chat_io._write_json_object",
            lambda path, payload: (_ for _ in ()).throw(
                AssertionError(f"Cache should be reused, not rewritten: {path}")
            ),
        )

        second_send = client.post(
            f"/api/v1/runs/run-chat-overflow/chat/sessions/{conversation_id}/messages",
            json={"content": "Summarize again."},
        )
        assert second_send.status_code == 202
        second_payload = second_send.json()
        assert second_payload["mode"] == "queued"

        second_job = poll_chat_job_terminal(
            client,
            "run-chat-overflow",
            conversation_id,
            str(second_payload["job"]["job_id"]),
        )
        assert second_job["status"] == "completed"

        second_session = client.get(
            f"/api/v1/runs/run-chat-overflow/chat/sessions/{conversation_id}"
        )
        assert second_session.status_code == 200
        assert second_session.json()["messages"][-1]["content"] == (
            "Second merged answer. [ref_1] [ref_2]"
        )

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    log_output = f"{log_output}\n{capsys.readouterr().err}"
    assert "Context chat request summary" in log_output
    assert "Context chat reply plan" in log_output
    assert "mode=split" in log_output
