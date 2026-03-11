"""Session-focused API chat integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.utils.config import AppConfig
from backend.utils.paths import RunPaths
from tests.api_chat.support import (
    build_config,
    patch_api_config_loaders,
    poll_until_completed,
    write_success_artifacts,
)


def test_chat_session_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return build_config(
            runs_dir=runs_dir,
            markdown_dir=markdown_dir,
            followup_search_enabled=False,
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
        return write_success_artifacts(question, run_id, config)

    def _stub_generate_reply(
        original_question: str,
        contexts: list[dict[str, object]],
        history: list[dict[str, str]],
        user_content: str,
        config: AppConfig,
        token_cap: int = 0,
        citation_catalog: list[dict[str, str]] | None = None,
        citation_prefix_tokens: list[int] | None = None,
        retry_missing_citation: bool = False,
        run_id: str | None = None,
    ) -> str:
        assert isinstance(original_question, str) and original_question
        assert isinstance(contexts, list) and contexts
        assert isinstance(contexts[0].get("final_document"), str)
        assert isinstance(contexts[0].get("context_bundle"), dict)
        assert isinstance(history, list)
        assert config.chat.model == "openai/gpt-5.2"
        assert token_cap == config.chat.max_context_total_tokens
        assert isinstance(citation_catalog, list)
        assert citation_prefix_tokens is None or isinstance(citation_prefix_tokens, list)
        assert isinstance(retry_missing_citation, bool)
        assert run_id == "run-chat"
        return f"Echo: {user_content}"

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.chat_reply_helpers.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat")

        create_session = client.post("/api/v1/runs/run-chat/chat/sessions", json={})
        assert create_session.status_code == 201
        session_payload = create_session.json()
        conversation_id = session_payload["conversation_id"]
        assert session_payload["messages"] == []

        send_message = client.post(
            f"/api/v1/runs/run-chat/chat/sessions/{conversation_id}/messages",
            json={"content": "What was the context?"},
        )
        assert send_message.status_code == 200
        send_payload = send_message.json()
        assert send_payload["assistant_message"]["content"] == "Echo: What was the context?"

        send_second_message = client.post(
            f"/api/v1/runs/run-chat/chat/sessions/{conversation_id}/messages",
            json={"content": "List only numeric targets in markdown table format."},
        )
        assert send_second_message.status_code == 200
        send_second_payload = send_second_message.json()
        assert (
            send_second_payload["assistant_message"]["content"]
            == "Echo: List only numeric targets in markdown table format."
        )

        fetch_session = client.get(f"/api/v1/runs/run-chat/chat/sessions/{conversation_id}")
        assert fetch_session.status_code == 200
        fetch_payload = fetch_session.json()
        assert len(fetch_payload["messages"]) == 4
        assert fetch_payload["messages"][0]["role"] == "user"
        assert fetch_payload["messages"][1]["role"] == "assistant"

        session_contexts = client.get(
            f"/api/v1/runs/run-chat/chat/sessions/{conversation_id}/contexts"
        )
        assert session_contexts.status_code == 200
        session_contexts_payload = session_contexts.json()
        assert session_contexts_payload["context_run_ids"] == ["run-chat"]
        assert session_contexts_payload["token_cap"] == 220_000
        assert session_contexts_payload["total_tokens"] > 0
        assert session_contexts_payload["prompt_context_tokens"] > 0
        assert session_contexts_payload["contexts"][0]["prompt_context_tokens"] > 0

        update_contexts = client.put(
            f"/api/v1/runs/run-chat/chat/sessions/{conversation_id}/contexts",
            json={"context_run_ids": ["run-chat"]},
        )
        assert update_contexts.status_code == 200

        list_sessions = client.get("/api/v1/runs/run-chat/chat/sessions")
        assert list_sessions.status_code == 200
        list_payload = list_sessions.json()
        assert conversation_id in list_payload["conversations"]

        catalog = client.get("/api/v1/chat/contexts")
        assert catalog.status_code == 200
        catalog_payload = catalog.json()
        assert catalog_payload["total"] == 1
        assert catalog_payload["contexts"][0]["run_id"] == "run-chat"
        assert catalog_payload["contexts"][0]["prompt_context_tokens"] > 0


def test_chat_requires_completed_run(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        run_store = client.app.state.run_store
        run_store.create_queued_run(question="Queued run", requested_run_id="run-queued")
        response = client.post("/api/v1/runs/run-queued/chat/sessions", json={})
        assert response.status_code == 409


def test_chat_unknown_run_returns_not_found(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        response = client.post("/api/v1/runs/missing/chat/sessions", json={})
        assert response.status_code == 404
