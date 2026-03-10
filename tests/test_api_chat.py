import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.services import context_chat
from backend.api.services.chat_followup_research import (
    CHAT_FOLLOWUP_CITY_UNAVAILABLE,
    ChatFollowupSearchResult,
    followup_bundle_dir,
)
from backend.modules.orchestrator.models import ChatFollowupDecision
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    ChatConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)
from backend.utils.paths import RunPaths, create_run_paths


def _build_config(
    runs_dir: Path,
    markdown_dir: Path,
    *,
    followup_search_enabled: bool = False,
    max_auto_followup_bundles: int = 3,
    max_context_total_tokens: int = 220_000,
    min_prompt_token_cap: int = 20_000,
    prompt_token_buffer: int = 2_000,
    multi_pass_chunk_tokens: int = 150_000,
) -> AppConfig:
    return AppConfig(
        orchestrator=OrchestratorConfig(
            model="test-model", context_bundle_name="context_bundle.json"
        ),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        chat=ChatConfig(
            model="openai/gpt-5.2",
            max_history_messages=10,
            max_context_total_tokens=max_context_total_tokens,
            min_prompt_token_cap=min_prompt_token_cap,
            prompt_token_buffer=prompt_token_buffer,
            multi_pass_chunk_tokens=multi_pass_chunk_tokens,
            followup_search_enabled=followup_search_enabled,
            max_auto_followup_bundles=max_auto_followup_bundles,
        ),
        runs_dir=runs_dir,
        markdown_dir=markdown_dir,
        enable_sql=False,
    )


def _write_success_artifacts(
    question: str,
    run_id: str,
    config: AppConfig,
    excerpts: list[dict[str, object]] | None = None,
) -> RunPaths:
    paths = create_run_paths(config.runs_dir, run_id, config.orchestrator.context_bundle_name)
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.context_bundle.write_text(
        json.dumps(
            {
                "sql": None,
                "markdown": {"status": "success", "excerpts": excerpts or []},
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


def _poll_chat_job_terminal(
    client: TestClient,
    run_id: str,
    conversation_id: str,
    job_id: str,
    timeout_seconds: float = 3.0,
) -> dict[str, object]:
    """Poll one chat job until it reaches a terminal state."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(
            f"/api/v1/runs/{run_id}/chat/sessions/{conversation_id}/jobs/{job_id}"
        )
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Chat job `{job_id}` did not complete in time.")


def _patch_api_config_loaders(
    monkeypatch: pytest.MonkeyPatch,
    stub_load_config: object,
) -> None:
    """Patch both async run and chat-route config loading for deterministic tests."""
    monkeypatch.setattr("backend.api.services.run_executor.load_config", stub_load_config)
    monkeypatch.setattr("backend.api.routes.chat.load_config", stub_load_config)


def _write_followup_bundle(
    *,
    runs_dir: Path,
    run_id: str,
    conversation_id: str,
    bundle_id: str,
    target_city: str,
    quote: str,
    partial_answer: str,
) -> None:
    bundle_dir = followup_bundle_dir(
        runs_dir=runs_dir,
        run_id=run_id,
        conversation_id=conversation_id,
        bundle_id=bundle_id,
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    context_bundle = {
        "bundle_id": bundle_id,
        "parent_run_id": run_id,
        "conversation_id": conversation_id,
        "source": "chat_followup",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_city": target_city,
        "research_question": f"What does {target_city} say?",
        "retrieval_queries": [f"{target_city} question"],
        "sql": None,
        "final": None,
        "analysis_mode": "aggregate",
        "markdown": {
            "status": "success",
            "selected_city_names": [target_city],
            "inspected_city_names": [target_city],
            "excerpt_count": 1,
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": target_city,
                    "quote": quote,
                    "partial_answer": partial_answer,
                    "source_chunk_ids": ["chunk-followup-1"],
                }
            ],
        },
    }
    (bundle_dir / "context_bundle.json").write_text(
        json.dumps(context_bundle, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    markdown_dir = bundle_dir / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / "excerpts.json").write_text(
        json.dumps(
            {
                "excerpts": context_bundle["markdown"]["excerpts"],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    (markdown_dir / "references.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "ref_id": "ref_1",
                        "excerpt_index": 0,
                        "city_name": target_city,
                        "quote": quote,
                        "partial_answer": partial_answer,
                        "source_chunk_ids": ["chunk-followup-1"],
                    }
                ]
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


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
        analysis_mode: str = "aggregate",
        api_key_override: str | None = None,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert analysis_mode == "aggregate"
        assert api_key_override is None
        assert selected_cities is None
        return _write_success_artifacts(question, run_id, config)

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

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

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


def test_chat_context_metrics_expose_prompt_context_tokens_for_excerpt_runs(
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
        analysis_mode: str = "aggregate",
        api_key_override: str | None = None,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert analysis_mode == "aggregate"
        assert api_key_override is None
        assert selected_cities is None
        return _write_success_artifacts(
            question,
            run_id,
            config,
            excerpts=[
                {
                    "ref_id": "ref_1",
                    "city_name": "Warsaw",
                    "quote": "Grounded evidence sentence. " * 12,
                    "partial_answer": "Grounded partial answer. " * 8,
                    "source_chunk_ids": ["chunk_1", "chunk_2"],
                }
            ],
        )

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-excerpts"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-excerpts")

        final_path = runs_dir / "run-chat-excerpts" / "final.md"
        final_path.write_text("# Answer\n" + ("Stored artifact text. " * 80), encoding="utf-8")

        catalog = client.get("/api/v1/chat/contexts")
        assert catalog.status_code == 200
        catalog_payload = catalog.json()
        summary = next(
            item for item in catalog_payload["contexts"] if item["run_id"] == "run-chat-excerpts"
        )
        assert summary["prompt_context_kind"] == "citation_catalog"
        assert summary["prompt_context_tokens"] > 0
        assert summary["prompt_context_tokens"] < summary["total_tokens"]

        create_session = client.post("/api/v1/runs/run-chat-excerpts/chat/sessions", json={})
        assert create_session.status_code == 201
        conversation_id = create_session.json()["conversation_id"]

        session_contexts = client.get(
            f"/api/v1/runs/run-chat-excerpts/chat/sessions/{conversation_id}/contexts"
        )
        assert session_contexts.status_code == 200
        session_payload = session_contexts.json()
        assert session_payload["prompt_context_kind"] == "citation_catalog"
        assert session_payload["prompt_context_tokens"] > 0
        assert session_payload["prompt_context_tokens"] < session_payload["total_tokens"]
        assert session_payload["is_capped"] is False


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
        analysis_mode: str = "aggregate",
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert analysis_mode == "aggregate"
        assert selected_cities is None
        return _write_success_artifacts(question, run_id, config)

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
        assert isinstance(original_question, str)
        assert isinstance(contexts, list) and contexts
        assert isinstance(citation_catalog, list)
        assert citation_prefix_tokens is None or isinstance(citation_prefix_tokens, list)
        assert isinstance(retry_missing_citation, bool)
        assert run_id == "run-chat-header"
        captured_key["value"] = api_key_override
        return "Header key response"

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

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
        analysis_mode: str = "aggregate",
        api_key_override: str | None = None,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert analysis_mode == "aggregate"
        assert api_key_override is None
        assert selected_cities is None
        return _write_success_artifacts(question, run_id, config)

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

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


def test_chat_builds_prompt_safe_citation_catalog_and_persists_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    captured_catalog: list[dict[str, str]] = []

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

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
                "city_name": "seville",
                "quote": "Seville plans 350 double charging points.",
                "partial_answer": "Seville plans 350 double charging points.",
                "source_chunk_ids": ["chunk_hidden_1"],
            }
        ]
        return _write_success_artifacts(question, run_id, config, excerpts=excerpts)

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
        assert isinstance(original_question, str)
        assert isinstance(contexts, list) and contexts
        assert isinstance(history, list)
        assert config.chat.model == "openai/gpt-5.2"
        assert token_cap == config.chat.max_context_total_tokens
        assert isinstance(citation_catalog, list) and citation_catalog
        assert citation_prefix_tokens is None or isinstance(citation_prefix_tokens, list)
        captured_catalog[:] = citation_catalog
        assert not retry_missing_citation
        assert run_id == "run-chat-citations"
        return "Seville plans charging expansion. [ref_1]"

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-citations"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-citations")

        create_session = client.post("/api/v1/runs/run-chat-citations/chat/sessions", json={})
        assert create_session.status_code == 201
        conversation_id = create_session.json()["conversation_id"]

        send_message = client.post(
            f"/api/v1/runs/run-chat-citations/chat/sessions/{conversation_id}/messages",
            json={"content": "Show evidence."},
        )
        assert send_message.status_code == 200
        payload = send_message.json()
        citations = payload["assistant_message"]["citations"]
        assert citations[0]["ref_id"] == "ref_1"
        assert citations[0]["city_name"] == "Seville"
        assert citations[0]["source_type"] == "run"
        assert citations[0]["source_id"] == "run-chat-citations"
        assert citations[0]["source_ref_id"] == "ref_7"

    assert captured_catalog
    assert set(captured_catalog[0].keys()) == {"ref_id", "city_name", "quote", "partial_answer"}
    assert captured_catalog[0]["ref_id"] == "ref_1"
    assert "source_type" not in captured_catalog[0]
    assert "source_id" not in captured_catalog[0]
    assert "source_ref_id" not in captured_catalog[0]
    assert "source_chunk_ids" not in captured_catalog[0]


def test_chat_retries_once_when_first_reply_has_no_valid_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    call_log: list[bool] = []

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

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
                "ref_id": "ref_1",
                "city_name": "porto",
                "quote": "Porto expects 35% electric and hybrid by 2030.",
                "partial_answer": "Porto expects 35% electric and hybrid by 2030.",
                "source_chunk_ids": ["chunk_hidden_2"],
            }
        ]
        return _write_success_artifacts(question, run_id, config, excerpts=excerpts)

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
        _ = citation_prefix_tokens
        assert isinstance(citation_catalog, list) and citation_catalog
        assert run_id == "run-chat-retry"
        call_log.append(retry_missing_citation)
        if retry_missing_citation:
            return "Porto targets 35% by 2030. [ref_1]"
        return "Porto targets 35% by 2030."

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-retry"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-retry")

        create_session = client.post("/api/v1/runs/run-chat-retry/chat/sessions", json={})
        assert create_session.status_code == 201
        conversation_id = create_session.json()["conversation_id"]

        send_message = client.post(
            f"/api/v1/runs/run-chat-retry/chat/sessions/{conversation_id}/messages",
            json={"content": "Show demand."},
        )
        assert send_message.status_code == 200
        payload = send_message.json()
        assert payload["assistant_message"]["content"] == "Porto targets 35% by 2030. [ref_1]"
        assert payload["assistant_message"]["citations"][0]["ref_id"] == "ref_1"

    assert call_log == [False, True]


def test_chat_update_contexts_keeps_parent_run_pinned(
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
        analysis_mode: str = "aggregate",
        api_key_override: str | None = None,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert analysis_mode == "aggregate"
        assert api_key_override is None
        assert selected_cities is None
        return _write_success_artifacts(question, run_id, config)

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Parent doc", "run_id": "run-parent"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-parent")

        start_other = client.post(
            "/api/v1/runs",
            json={"question": "Other doc", "run_id": "run-other"},
        )
        assert start_other.status_code == 202
        _poll_until_completed(client, "run-other")

        create_session = client.post("/api/v1/runs/run-parent/chat/sessions", json={})
        conversation_id = create_session.json()["conversation_id"]
        update_contexts = client.put(
            f"/api/v1/runs/run-parent/chat/sessions/{conversation_id}/contexts",
            json={"context_run_ids": ["run-other"]},
        )
        assert update_contexts.status_code == 200
        assert update_contexts.json()["context_run_ids"] == ["run-parent", "run-other"]


def test_chat_followup_search_attaches_bundle_and_exposes_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=[])

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
        _write_followup_bundle(
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

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.routes.chat.route_chat_followup", _stub_route_chat_followup)
    monkeypatch.setattr(
        "backend.api.routes.chat.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr("backend.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-followup"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-followup")

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


def test_chat_followup_queued_response_exposes_city_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=[])

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
        bundle_id = "fup_chat_001_izmir"
        _write_followup_bundle(
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
        return "Izmir plans district cooling expansion. [ref_1]"

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.routes.chat.route_chat_followup", _stub_route_chat_followup)
    monkeypatch.setattr(
        "backend.api.routes.chat.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr(
        "backend.api.routes.chat._build_context_reply_plan",
        _stub_build_context_reply_plan,
    )
    monkeypatch.setattr("backend.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-followup-queued"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-followup-queued")

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

        completed_job = _poll_chat_job_terminal(
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
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=[])

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.routes.chat.route_chat_followup",
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
        _poll_until_completed(client, "run-chat-oos")

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=[])

    def _unexpected_reply(**_kwargs: object) -> str:
        raise AssertionError("Direct answer generation should not run for clarification prompts.")

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.routes.chat.generate_context_chat_reply",
        _unexpected_reply,
    )
    monkeypatch.setattr(
        "backend.api.routes.chat.route_chat_followup",
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
        _poll_until_completed(client, "run-chat-clarify")

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=[])

    def _unexpected_reply(**_kwargs: object) -> str:
        raise AssertionError("Direct answer generation should not run after failed follow-up search.")

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.routes.chat.generate_context_chat_reply",
        _unexpected_reply,
    )
    monkeypatch.setattr(
        "backend.api.routes.chat.route_chat_followup",
        lambda payload, config, api_key, log_llm_payload=False: ChatFollowupDecision(
            action="search_single_city",
            reason="Existing context does not cover Munich.",
            target_city="Munich",
            rewritten_question="What does Munich report?",
        ),
    )
    monkeypatch.setattr(
        "backend.api.routes.chat.run_chat_followup_search",
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
        _poll_until_completed(client, "run-chat-failed-followup")

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=[])

    def _unexpected_reply(**_kwargs: object) -> str:
        raise AssertionError("Direct answer generation should not run for unavailable cities.")

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.routes.chat.generate_context_chat_reply",
        _unexpected_reply,
    )
    monkeypatch.setattr(
        "backend.api.routes.chat.route_chat_followup",
        lambda payload, config, api_key, log_llm_payload=False: ChatFollowupDecision(
            action="search_single_city",
            reason="Existing context does not cover Atlantis.",
            target_city="Atlantis",
            rewritten_question="What does Atlantis report?",
        ),
    )
    monkeypatch.setattr(
        "backend.api.routes.chat.run_chat_followup_search",
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
        _poll_until_completed(client, "run-chat-unavailable-city")

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    route_calls: list[str] = []

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=[])

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
        _write_followup_bundle(
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

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.routes.chat.route_chat_followup", _stub_route_chat_followup)
    monkeypatch.setattr(
        "backend.api.routes.chat.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr("backend.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-city-choice"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-city-choice")

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
                "content": "Focus only on Munich.",
                "clarification_city": "Munich",
                "clarification_question": "Compare Munich and Berlin on rooftop solar.",
            },
        )
        assert city_choice_response.status_code == 200
        payload = city_choice_response.json()
        assert payload["user_message"]["content"] == "Focus only on Munich."
        assert payload["assistant_message"]["content"] == "Munich plans rooftop solar expansion. [ref_1]"
        assert payload["assistant_message"]["routing"]["action"] == "search_single_city"
        assert payload["assistant_message"]["routing"]["target_city"] == "Munich"
        assert payload["assistant_message"]["citations"][0]["source_type"] == "followup_bundle"

    assert route_calls == ["Compare Munich and Berlin on rooftop solar."]


def test_chat_followup_bundles_are_pruned_to_configured_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=[])

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
        _write_followup_bundle(
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

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.routes.chat.route_chat_followup", _stub_route_chat_followup)
    monkeypatch.setattr(
        "backend.api.routes.chat.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr("backend.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-prune-followups"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-prune-followups")

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=[])

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
        _write_followup_bundle(
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

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr(
        "backend.api.routes.chat.route_chat_followup",
        lambda payload, config, api_key, log_llm_payload=False: ChatFollowupDecision(
            action="search_single_city",
            reason="Need fresh context for Munich.",
            target_city="Munich",
            rewritten_question="What does Munich report?",
        ),
    )
    monkeypatch.setattr(
        "backend.api.routes.chat.run_chat_followup_search",
        _stub_run_chat_followup_search,
    )
    monkeypatch.setattr("backend.api.routes.chat.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-replace-followup"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-replace-followup")

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
        return _build_config(
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
        return _write_success_artifacts(question, run_id, config, excerpts=excerpts)

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

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.context_chat.OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        "backend.api.services.context_chat._run_chat_completion_with_tools",
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
        _poll_until_completed(client, "run-chat-overflow")

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

        first_job = _poll_chat_job_terminal(
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

        cache_path = context_chat._chat_evidence_cache_path(runs_dir, "run-chat-overflow")
        assert cache_path.exists()
        cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached_payload["evidence_count"] == 2

        monkeypatch.setattr(
            "backend.api.services.context_chat._write_json_object",
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

        second_job = _poll_chat_job_terminal(
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


def test_chat_session_contexts_allow_extra_runs_when_selection_exceeds_direct_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(
            runs_dir=runs_dir,
            markdown_dir=markdown_dir,
            max_context_total_tokens=120,
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
        return _write_success_artifacts(question, run_id, config)

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.routes.chat.load_config", _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start_parent = client.post(
            "/api/v1/runs",
            json={"question": "Parent doc", "run_id": "run-parent-cap"},
        )
        assert start_parent.status_code == 202
        _poll_until_completed(client, "run-parent-cap")

        start_other = client.post(
            "/api/v1/runs",
            json={"question": "Other doc", "run_id": "run-other-cap"},
        )
        assert start_other.status_code == 202
        _poll_until_completed(client, "run-other-cap")

        base_final = runs_dir / "run-parent-cap" / "final.md"
        base_bundle = runs_dir / "run-parent-cap" / "context_bundle.json"
        base_final.write_text("# Answer\n" + ("overflow " * 200), encoding="utf-8")
        base_bundle.write_text(
            json.dumps(
                {
                    "sql": None,
                    "markdown": {
                        "status": "success",
                        "excerpts": [],
                    },
                    "final": str(base_final),
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

        create_session = client.post(
            "/api/v1/runs/run-parent-cap/chat/sessions",
            json={},
        )
        assert create_session.status_code == 201
        conversation_id = create_session.json()["conversation_id"]

        session_contexts = client.get(
            f"/api/v1/runs/run-parent-cap/chat/sessions/{conversation_id}/contexts"
        )
        assert session_contexts.status_code == 200
        contexts_payload = session_contexts.json()
        assert [context["run_id"] for context in contexts_payload["contexts"]] == ["run-parent-cap"]
        assert contexts_payload["is_capped"] is True
        assert contexts_payload["excluded_context_run_ids"] == []

        update_contexts = client.put(
            f"/api/v1/runs/run-parent-cap/chat/sessions/{conversation_id}/contexts",
            json={"context_run_ids": ["run-other-cap"]},
        )
        assert update_contexts.status_code == 200
        updated_payload = update_contexts.json()
        assert updated_payload["context_run_ids"] == ["run-parent-cap", "run-other-cap"]
        assert [context["run_id"] for context in updated_payload["contexts"]] == [
            "run-parent-cap",
            "run-other-cap",
        ]
        assert updated_payload["is_capped"] is True


def test_chat_contexts_lazy_backfill_bundle_cache_and_reuse_session_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        analysis_mode: str = "aggregate",
        api_key_override: str | None = None,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert analysis_mode == "aggregate"
        assert api_key_override is None
        assert selected_cities is None
        return _write_success_artifacts(
            question,
            run_id,
            config,
            excerpts=[
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich evidence sentence.",
                    "partial_answer": "Munich evidence sentence.",
                    "source_chunk_ids": ["chunk-hidden-1"],
                }
            ],
        )

    _patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-cache"},
        )
        assert start.status_code == 202
        _poll_until_completed(client, "run-chat-cache")

        create_session = client.post("/api/v1/runs/run-chat-cache/chat/sessions", json={})
        assert create_session.status_code == 201
        conversation_id = create_session.json()["conversation_id"]

        first_contexts = client.get(
            f"/api/v1/runs/run-chat-cache/chat/sessions/{conversation_id}/contexts"
        )
        assert first_contexts.status_code == 200
        first_payload = first_contexts.json()
        assert first_payload["prompt_context_kind"] == "citation_catalog"
        assert first_payload["prompt_context_tokens"] > 0

        context_bundle = json.loads(
            (runs_dir / "run-chat-cache" / "context_bundle.json").read_text(encoding="utf-8")
        )
        markdown_excerpts = json.loads(
            (runs_dir / "run-chat-cache" / "markdown" / "excerpts.json").read_text(
                encoding="utf-8"
            )
        )
        session_payload = json.loads(
            (runs_dir / "run-chat-cache" / "chat" / f"{conversation_id}.json").read_text(
                encoding="utf-8"
            )
        )

        assert context_bundle["prompt_context_kind"] == "citation_catalog"
        assert context_bundle["prompt_context_tokens"] == first_payload["contexts"][0]["prompt_context_tokens"]
        assert markdown_excerpts["prompt_context_kind"] == "citation_catalog"
        assert markdown_excerpts["prompt_context_tokens"] == context_bundle["prompt_context_tokens"]
        assert session_payload["prompt_context_cache"]["prompt_context_tokens"] == first_payload["prompt_context_tokens"]
        assert session_payload["prompt_context_cache"]["citation_prefix_tokens"]

        monkeypatch.setattr(
            "backend.api.routes.chat._build_session_prompt_context_cache",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("Session prompt cache should be reused.")
            ),
        )

        second_contexts = client.get(
            f"/api/v1/runs/run-chat-cache/chat/sessions/{conversation_id}/contexts"
        )
        assert second_contexts.status_code == 200
        assert second_contexts.json()["prompt_context_tokens"] == first_payload["prompt_context_tokens"]
