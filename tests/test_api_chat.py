import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.utils.config import (
    AgentConfig,
    AppConfig,
    ChatConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)
from app.utils.paths import RunPaths, create_run_paths


def _build_config(runs_dir: Path, markdown_dir: Path) -> AppConfig:
    return AppConfig(
        orchestrator=OrchestratorConfig(
            model="test-model", context_bundle_name="context_bundle.json"
        ),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(model="openai/gpt-5.2", max_history_messages=10),
        runs_dir=runs_dir,
        markdown_dir=markdown_dir,
        enable_sql=False,
    )


def _write_success_artifacts(question: str, run_id: str, config: AppConfig) -> RunPaths:
    paths = create_run_paths(config.runs_dir, run_id, config.orchestrator.context_bundle_name)
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.context_bundle.write_text(
        json.dumps(
            {
                "sql": None,
                "markdown": {"status": "success", "excerpts": []},
                "drafts": [],
                "final": str(paths.final_output),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.final_output.write_text("# Answer\nStub document", encoding="utf-8")
    run_log = {
        "run_id": run_id,
        "question": question,
        "status": "completed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "finish_reason": "completed (write)",
        "artifacts": {
            "context_bundle": str(paths.context_bundle),
            "final_output": str(paths.final_output),
        },
    }
    paths.run_log.write_text(
        json.dumps(run_log, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return paths


def _poll_until_completed(client: TestClient, run_id: str, timeout_seconds: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = client.get(f"/api/v1/runs/{run_id}/status").json()
        if payload["status"] == "completed":
            return
        time.sleep(0.02)
    raise AssertionError(f"Run `{run_id}` did not complete in time.")


def test_chat_session_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    def _stub_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
    ) -> RunPaths:
        assert run_id is not None
        return _write_success_artifacts(question, run_id, config)

    def _stub_generate_reply(
        original_question: str,
        contexts: list[dict[str, object]],
        history: list[dict[str, str]],
        user_content: str,
        config: AppConfig,
        token_cap: int = 300000,
    ) -> str:
        assert isinstance(original_question, str) and original_question
        assert isinstance(contexts, list) and contexts
        assert isinstance(contexts[0].get("final_document"), str)
        assert isinstance(contexts[0].get("context_bundle"), dict)
        assert isinstance(history, list)
        assert config.chat.model == "openai/gpt-5.2"
        assert token_cap == 300000
        return f"Echo: {user_content}"

    monkeypatch.setattr("app.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr("app.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("app.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat")

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

        fetch_session = client.get(f"/api/v1/runs/run-chat/chat/sessions/{conversation_id}")
        assert fetch_session.status_code == 200
        fetch_payload = fetch_session.json()
        assert len(fetch_payload["messages"]) == 2
        assert fetch_payload["messages"][0]["role"] == "user"
        assert fetch_payload["messages"][1]["role"] == "assistant"

        session_contexts = client.get(
            f"/api/v1/runs/run-chat/chat/sessions/{conversation_id}/contexts"
        )
        assert session_contexts.status_code == 200
        session_contexts_payload = session_contexts.json()
        assert session_contexts_payload["context_run_ids"] == ["run-chat"]
        assert session_contexts_payload["token_cap"] == 300000
        assert session_contexts_payload["total_tokens"] > 0

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


def test_chat_supports_header_api_key_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    captured_key: dict[str, str | None] = {"value": None}

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    def _stub_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
    ) -> RunPaths:
        assert run_id is not None
        return _write_success_artifacts(question, run_id, config)

    def _stub_generate_reply(
        original_question: str,
        contexts: list[dict[str, object]],
        history: list[dict[str, str]],
        user_content: str,
        config: AppConfig,
        token_cap: int = 300000,
        api_key_override: str | None = None,
    ) -> str:
        assert isinstance(original_question, str)
        assert isinstance(contexts, list) and contexts
        captured_key["value"] = api_key_override
        return "Header key response"

    monkeypatch.setattr("app.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr("app.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("app.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-header"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-header")

        create_session = client.post("/api/v1/runs/run-chat-header/chat/sessions", json={})
        assert create_session.status_code == 201
        conversation_id = create_session.json()["conversation_id"]

        send_message = client.post(
            f"/api/v1/runs/run-chat-header/chat/sessions/{conversation_id}/messages",
            json={"content": "Use my key"},
            headers={"X-OpenRouter-Api-Key": "sk-or-v1-user-key-123"},
        )
        assert send_message.status_code == 200
        assert send_message.json()["assistant_message"]["content"] == "Header key response"

    assert captured_key["value"] == "sk-or-v1-user-key-123"


def test_chat_context_update_rejects_unknown_context_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    def _stub_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
    ) -> RunPaths:
        assert run_id is not None
        return _write_success_artifacts(question, run_id, config)

    monkeypatch.setattr("app.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr("app.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat")

        create_session = client.post("/api/v1/runs/run-chat/chat/sessions", json={})
        assert create_session.status_code == 201
        conversation_id = create_session.json()["conversation_id"]

        update_contexts = client.put(
            f"/api/v1/runs/run-chat/chat/sessions/{conversation_id}/contexts",
            json={"context_run_ids": ["missing-run-id"]},
        )
        assert update_contexts.status_code == 400
