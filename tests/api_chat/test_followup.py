"""Follow-up routing and bundle-management API chat integration tests."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.services.chat_followup_research import (
    CHAT_FOLLOWUP_CITY_UNAVAILABLE,
    ChatFollowupSearchResult,
)
from backend.modules.orchestrator.models import ChatFollowupDecision
from backend.utils.config import AppConfig
from backend.utils.paths import RunPaths
from tests.api_chat.support import (
    build_config,
    patch_api_config_loaders,
    poll_until_completed,
    write_followup_bundle,
    write_success_artifacts,
)


def test_chat_followup_search_attaches_bundle_and_exposes_references(
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
        return write_success_artifacts(question, run_id, config, excerpts=[])

    def _stub_route_chat_followup(
        payload: dict[str, object],
        config: AppConfig,
        api_key: str,
        log_llm_payload: bool = False,
    ) -> ChatFollowupDecision:
        _ = config, api_key, log_llm_payload
        assert payload["user_message"] == "Tell me more about Munich."
        return ChatFollowupDecision(
            action="search_single_city",
            reason="Current excerpts do not answer the Munich question.",
            target_city="Munich",
            rewritten_question="What does Munich report?",
            confidence=0.8,
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
        _ = config, api_key, log_llm_payload
        assert turn_index == 1
        assert question == "What does Munich report?"
        bundle_id = "fup_chat_001_munich"
        write_followup_bundle(
            runs_dir=runs_dir,
            run_id=run_id,
            conversation_id=conversation_id,
            bundle_id=bundle_id,
            target_city=target_city,
            quote="Munich plans rooftop solar expansion.",
            partial_answer="Munich plans rooftop solar expansion.",
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
        assert original_question == "Build doc"
        assert run_id == "run-chat-followup"
        assert isinstance(citation_catalog, list) and citation_catalog
        assert citation_prefix_tokens is None or isinstance(citation_prefix_tokens, list)
        assert contexts[-1]["run_id"] == "fup_chat_001_munich"
        return "Munich plans rooftop solar expansion. [ref_1]"

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.chat_followup_flow.route_chat_followup", _stub_route_chat_followup)
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr("backend.api.services.chat_reply_helpers.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-followup"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-followup")

        create_session = client.post("/api/v1/runs/run-chat-followup/chat/sessions", json={})
        conversation_id = create_session.json()["conversation_id"]
        send_message = client.post(
            f"/api/v1/runs/run-chat-followup/chat/sessions/{conversation_id}/messages",
            json={"content": "Tell me more about Munich."},
        )
        assert send_message.status_code == 200
        payload = send_message.json()
        assert payload["assistant_message"]["content"] == "Munich plans rooftop solar expansion. [ref_1]"
        assert payload["assistant_message"]["routing"]["action"] == "search_single_city"
        assert payload["assistant_message"]["citations"][0]["source_type"] == "followup_bundle"
        assert payload["assistant_message"]["citations"][0]["source_id"] == "fup_chat_001_munich"

        session_contexts = client.get(
            f"/api/v1/runs/run-chat-followup/chat/sessions/{conversation_id}/contexts"
        )
        assert session_contexts.status_code == 200
        contexts_payload = session_contexts.json()
        assert contexts_payload["followup_bundles"][0]["bundle_id"] == "fup_chat_001_munich"

        followup_refs = client.get(
            f"/api/v1/runs/run-chat-followup/chat/sessions/{conversation_id}/followups/fup_chat_001_munich/references",
            params={"ref_id": "ref_1", "include_quote": "true"},
        )
        assert followup_refs.status_code == 200
        refs_payload = followup_refs.json()
        assert refs_payload["references"][0]["quote"] == "Munich plans rooftop solar expansion."


def test_chat_followup_router_returns_out_of_scope_message(
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
            followup_search_enabled=True,
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
        return write_success_artifacts(question, run_id, config, excerpts=[])

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.route_chat_followup",
        lambda payload, config, api_key, log_llm_payload=False: ChatFollowupDecision(
            action="out_of_scope",
            reason="The message is unrelated to the report context.",
        ),
    )
    caplog.set_level(logging.INFO, logger="backend.api.routes.chat")

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-oos"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-oos")

        create_session = client.post("/api/v1/runs/run-chat-oos/chat/sessions", json={})
        conversation_id = create_session.json()["conversation_id"]
        send_message = client.post(
            f"/api/v1/runs/run-chat-oos/chat/sessions/{conversation_id}/messages",
            json={"content": "What's the weather tomorrow?"},
        )
        assert send_message.status_code == 200
        payload = send_message.json()
        assert "outside the current city-report context" in payload["assistant_message"]["content"]
        assert payload["assistant_message"]["routing"]["action"] == "out_of_scope"

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    log_output = f"{log_output}\n{capsys.readouterr().err}"
    assert "Context chat request summary" in log_output
    assert "Context chat router preflight" in log_output


def test_chat_followup_router_requests_city_clarification(
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
        return write_success_artifacts(question, run_id, config, excerpts=[])

    def _unexpected_reply(**_kwargs: object) -> str:
        raise AssertionError("Direct answer generation should not run for clarification prompts.")

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.services.chat_reply_helpers.generate_context_chat_reply",
        _unexpected_reply,
    )
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.route_chat_followup",
        lambda payload, config, api_key, log_llm_payload=False: ChatFollowupDecision(
            action="needs_city_clarification",
            reason="The user asked about more than one city.",
        ),
    )

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-clarify"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-clarify")

        create_session = client.post("/api/v1/runs/run-chat-clarify/chat/sessions", json={})
        conversation_id = create_session.json()["conversation_id"]
        send_message = client.post(
            f"/api/v1/runs/run-chat-clarify/chat/sessions/{conversation_id}/messages",
            json={"content": "Compare Munich and Berlin."},
        )
        assert send_message.status_code == 200
        payload = send_message.json()
        assert "one city at a time" in payload["assistant_message"]["content"]
        assert payload["assistant_message"]["routing"]["action"] == "needs_city_clarification"
        assert (
            payload["assistant_message"]["routing"]["pending_user_message"]
            == "Compare Munich and Berlin."
        )


def test_chat_followup_search_failure_returns_grounded_limitation(
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
        return write_success_artifacts(question, run_id, config, excerpts=[])

    def _unexpected_reply(**_kwargs: object) -> str:
        raise AssertionError("Direct answer generation should not run after failed follow-up search.")

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.services.chat_reply_helpers.generate_context_chat_reply",
        _unexpected_reply,
    )
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.route_chat_followup",
        lambda payload, config, api_key, log_llm_payload=False: ChatFollowupDecision(
            action="search_single_city",
            reason="Existing context does not cover Munich.",
            target_city="Munich",
            rewritten_question="What does Munich report?",
        ),
    )
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.run_chat_followup_search",
        lambda **kwargs: ChatFollowupSearchResult(
            status="error",
            bundle_id="fup_chat_001_munich",
            target_city="Munich",
            created_at=datetime.now(timezone.utc),
            excerpt_count=0,
            total_tokens=10,
            error_message="No excerpts available.",
        ),
    )

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-failed-followup"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-failed-followup")

        create_session = client.post(
            "/api/v1/runs/run-chat-failed-followup/chat/sessions",
            json={},
        )
        conversation_id = create_session.json()["conversation_id"]
        send_message = client.post(
            f"/api/v1/runs/run-chat-failed-followup/chat/sessions/{conversation_id}/messages",
            json={"content": "Tell me more about Munich."},
        )
        assert send_message.status_code == 200
        payload = send_message.json()
        assert "could not refresh the context for Munich" in payload["assistant_message"]["content"]
        assert payload["assistant_message"]["routing"]["action"] == "search_single_city"
        assert payload["assistant_message"]["routing"]["bundle_id"] == "fup_chat_001_munich"

        session_contexts = client.get(
            f"/api/v1/runs/run-chat-failed-followup/chat/sessions/{conversation_id}/contexts"
        )
        assert session_contexts.status_code == 200
        assert session_contexts.json()["followup_bundles"] == []


def test_chat_unavailable_followup_city_returns_city_choice_trigger(
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
        return write_success_artifacts(question, run_id, config, excerpts=[])

    def _unexpected_reply(**_kwargs: object) -> str:
        raise AssertionError("Direct answer generation should not run for unavailable cities.")

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.services.chat_reply_helpers.generate_context_chat_reply",
        _unexpected_reply,
    )
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.route_chat_followup",
        lambda payload, config, api_key, log_llm_payload=False: ChatFollowupDecision(
            action="search_single_city",
            reason="Existing context does not cover Atlantis.",
            target_city="Atlantis",
            rewritten_question="What does Atlantis report?",
        ),
    )
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.run_chat_followup_search",
        lambda **kwargs: ChatFollowupSearchResult(
            status="error",
            bundle_id="fup_chat_001_atlantis",
            target_city="Atlantis",
            created_at=datetime.now(timezone.utc),
            excerpt_count=0,
            total_tokens=10,
            error_code=CHAT_FOLLOWUP_CITY_UNAVAILABLE,
            error_message="Selected city is not available in the vector store index.",
        ),
    )

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-unavailable-city"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-unavailable-city")

        create_session = client.post(
            "/api/v1/runs/run-chat-unavailable-city/chat/sessions",
            json={},
        )
        conversation_id = create_session.json()["conversation_id"]
        send_message = client.post(
            f"/api/v1/runs/run-chat-unavailable-city/chat/sessions/{conversation_id}/messages",
            json={"content": "Tell me more about Atlantis."},
        )
        assert send_message.status_code == 200
        payload = send_message.json()
        assert payload["assistant_message"]["content"] == (
            "We don't have Atlantis in the current city list. Please choose another one."
        )
        assert payload["assistant_message"]["routing"]["action"] == "needs_city_clarification"
        assert (
            payload["assistant_message"]["routing"]["pending_user_message"]
            == "Tell me more about Atlantis."
        )

        session_contexts = client.get(
            f"/api/v1/runs/run-chat-unavailable-city/chat/sessions/{conversation_id}/contexts"
        )
        assert session_contexts.status_code == 200
        assert session_contexts.json()["followup_bundles"] == []


def test_chat_clarification_city_selection_triggers_direct_followup_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    route_calls: list[str] = []

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return build_config(
            runs_dir=runs_dir,
            markdown_dir=markdown_dir,
            followup_search_enabled=True,
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
        return write_success_artifacts(question, run_id, config, excerpts=[])

    def _stub_route_chat_followup(
        payload: dict[str, object],
        config: AppConfig,
        api_key: str,
        log_llm_payload: bool = False,
    ) -> ChatFollowupDecision:
        _ = config, api_key, log_llm_payload
        route_calls.append(str(payload["user_message"]))
        return ChatFollowupDecision(
            action="needs_city_clarification",
            reason="The follow-up question mentions multiple cities.",
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
        _ = config, api_key, log_llm_payload
        assert turn_index == 2
        assert question == "Compare Munich and Berlin on rooftop solar."
        assert target_city == "Munich"
        bundle_id = "fup_chat_002_munich"
        write_followup_bundle(
            runs_dir=runs_dir,
            run_id=run_id,
            conversation_id=conversation_id,
            bundle_id=bundle_id,
            target_city=target_city,
            quote="Munich plans rooftop solar expansion.",
            partial_answer="Munich plans rooftop solar expansion.",
        )
        return ChatFollowupSearchResult(
            status="success",
            bundle_id=bundle_id,
            target_city=target_city,
            created_at=datetime.now(timezone.utc),
            excerpt_count=1,
            total_tokens=120,
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
        _ = original_question, history, config, token_cap, api_key_override, retry_missing_citation
        assert run_id == "run-chat-city-choice"
        assert isinstance(citation_catalog, list) and citation_catalog
        assert citation_prefix_tokens is None or isinstance(citation_prefix_tokens, list)
        assert user_content == "Compare Munich and Berlin on rooftop solar."
        assert contexts[-1]["run_id"] == "fup_chat_002_munich"
        return "Munich plans rooftop solar expansion. [ref_1]"

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.chat_followup_flow.route_chat_followup", _stub_route_chat_followup)
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr("backend.api.services.chat_reply_helpers.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-city-choice"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-city-choice")

        create_session = client.post("/api/v1/runs/run-chat-city-choice/chat/sessions", json={})
        conversation_id = create_session.json()["conversation_id"]

        clarify_response = client.post(
            f"/api/v1/runs/run-chat-city-choice/chat/sessions/{conversation_id}/messages",
            json={"content": "Compare Munich and Berlin on rooftop solar."},
        )
        assert clarify_response.status_code == 200
        clarify_payload = clarify_response.json()
        assert clarify_payload["assistant_message"]["routing"]["action"] == "needs_city_clarification"
        assert (
            clarify_payload["assistant_message"]["routing"]["pending_user_message"]
            == "Compare Munich and Berlin on rooftop solar."
        )

        city_choice_response = client.post(
            f"/api/v1/runs/run-chat-city-choice/chat/sessions/{conversation_id}/messages",
            json={
                "content": "Compare Munich and Berlin on rooftop solar.",
                "clarification_city": "Munich",
            },
        )
        assert city_choice_response.status_code == 200
        payload = city_choice_response.json()
        assert payload["user_message"]["content"] == "Compare Munich and Berlin on rooftop solar."
        assert payload["assistant_message"]["content"] == "Munich plans rooftop solar expansion. [ref_1]"
        assert payload["assistant_message"]["routing"]["action"] == "search_single_city"
        assert payload["assistant_message"]["routing"]["target_city"] == "Munich"
        assert payload["assistant_message"]["citations"][0]["source_type"] == "followup_bundle"

    assert route_calls == ["Compare Munich and Berlin on rooftop solar."]


def test_chat_followup_bundles_are_pruned_to_configured_maximum(
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
            max_auto_followup_bundles=1,
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
        return write_success_artifacts(question, run_id, config, excerpts=[])

    def _stub_route_chat_followup(
        payload: dict[str, object],
        config: AppConfig,
        api_key: str,
        log_llm_payload: bool = False,
    ) -> ChatFollowupDecision:
        _ = config, api_key, log_llm_payload
        if payload["user_message"] == "Tell me more about Munich.":
            target_city = "Munich"
        else:
            target_city = "Berlin"
        return ChatFollowupDecision(
            action="search_single_city",
            reason=f"Need fresh context for {target_city}.",
            target_city=target_city,
            rewritten_question=f"What does {target_city} report?",
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
        bundle_id = f"fup_chat_{turn_index:03d}_{target_city.casefold()}"
        write_followup_bundle(
            runs_dir=runs_dir,
            run_id=run_id,
            conversation_id=conversation_id,
            bundle_id=bundle_id,
            target_city=target_city,
            quote=f"{target_city} follow-up quote.",
            partial_answer=f"{target_city} follow-up answer.",
        )
        return ChatFollowupSearchResult(
            status="success",
            bundle_id=bundle_id,
            target_city=target_city,
            created_at=datetime.now(timezone.utc),
            excerpt_count=1,
            total_tokens=100,
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
        _ = (
            original_question,
            history,
            config,
            token_cap,
            api_key_override,
            citation_prefix_tokens,
            retry_missing_citation,
        )
        assert run_id == "run-chat-prune-followups"
        assert isinstance(citation_catalog, list) and citation_catalog
        return f"{user_content} [ref_1]"

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.chat_followup_flow.route_chat_followup", _stub_route_chat_followup)
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr("backend.api.services.chat_reply_helpers.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-prune-followups"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-prune-followups")

        create_session = client.post(
            "/api/v1/runs/run-chat-prune-followups/chat/sessions",
            json={},
        )
        conversation_id = create_session.json()["conversation_id"]

        first_send = client.post(
            f"/api/v1/runs/run-chat-prune-followups/chat/sessions/{conversation_id}/messages",
            json={"content": "Tell me more about Munich."},
        )
        assert first_send.status_code == 200

        second_send = client.post(
            f"/api/v1/runs/run-chat-prune-followups/chat/sessions/{conversation_id}/messages",
            json={"content": "Tell me more about Berlin."},
        )
        assert second_send.status_code == 200
        second_payload = second_send.json()
        assert second_payload["assistant_message"]["routing"]["bundle_id"] == "fup_chat_002_berlin"

        session_contexts = client.get(
            f"/api/v1/runs/run-chat-prune-followups/chat/sessions/{conversation_id}/contexts"
        )
        assert session_contexts.status_code == 200
        contexts_payload = session_contexts.json()
        assert [bundle["bundle_id"] for bundle in contexts_payload["followup_bundles"]] == [
            "fup_chat_002_berlin"
        ]


def test_chat_followup_same_city_search_replaces_previous_bundle(
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
        return write_success_artifacts(question, run_id, config, excerpts=[])

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
        bundle_id = f"fup_chat_{turn_index:03d}_munich"
        write_followup_bundle(
            runs_dir=runs_dir,
            run_id=run_id,
            conversation_id=conversation_id,
            bundle_id=bundle_id,
            target_city=target_city,
            quote=f"{target_city} follow-up quote {turn_index}.",
            partial_answer=f"{target_city} follow-up answer {turn_index}.",
        )
        return ChatFollowupSearchResult(
            status="success",
            bundle_id=bundle_id,
            target_city=target_city,
            created_at=datetime.now(timezone.utc),
            excerpt_count=1,
            total_tokens=100,
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
        _ = (
            original_question,
            history,
            config,
            token_cap,
            api_key_override,
            citation_catalog,
            citation_prefix_tokens,
            retry_missing_citation,
        )
        assert run_id == "run-chat-replace-followup"
        assert contexts[-1]["run_id"].startswith("fup_chat_")
        return f"{user_content} [ref_1]"

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.route_chat_followup",
        lambda payload, config, api_key, log_llm_payload=False: ChatFollowupDecision(
            action="search_single_city",
            reason="Need fresh context for Munich.",
            target_city="Munich",
            rewritten_question="What does Munich report?",
        ),
    )
    monkeypatch.setattr(
        "backend.api.services.chat_followup_flow.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr("backend.api.services.chat_reply_helpers.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-replace-followup"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-replace-followup")

        create_session = client.post(
            "/api/v1/runs/run-chat-replace-followup/chat/sessions",
            json={},
        )
        conversation_id = create_session.json()["conversation_id"]

        first_send = client.post(
            f"/api/v1/runs/run-chat-replace-followup/chat/sessions/{conversation_id}/messages",
            json={"content": "Tell me more about Munich."},
        )
        assert first_send.status_code == 200

        second_send = client.post(
            f"/api/v1/runs/run-chat-replace-followup/chat/sessions/{conversation_id}/messages",
            json={"content": "Refresh Munich again."},
        )
        assert second_send.status_code == 200
        second_payload = second_send.json()
        assert second_payload["assistant_message"]["routing"]["bundle_id"] == "fup_chat_002_munich"

        session_contexts = client.get(
            f"/api/v1/runs/run-chat-replace-followup/chat/sessions/{conversation_id}/contexts"
        )
        assert session_contexts.status_code == 200
        contexts_payload = session_contexts.json()
        assert [bundle["bundle_id"] for bundle in contexts_payload["followup_bundles"]] == [
            "fup_chat_002_munich"
        ]
