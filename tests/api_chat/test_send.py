"""Send-message API chat integration tests."""

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


def test_chat_supports_header_api_key_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    captured_key: dict[str, str | None] = {"value": None}

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return build_config(
            runs_dir=runs_dir,
            markdown_dir=markdown_dir,
            followup_search_enabled=False,
        ).model_copy(update={"enable_sql": True})

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
        assert config.enable_sql is True
        return write_success_artifacts(question, run_id, config)

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
        assert config.enable_sql is True
        captured_key["value"] = api_key_override
        return "Header key response"

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.chat_reply_helpers.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-header"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-header")

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


def test_chat_builds_prompt_safe_citation_catalog_and_persists_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    captured_catalog: list[dict[str, str]] = []

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
        excerpts = [
            {
                "ref_id": "ref_7",
                "city_name": "seville",
                "quote": "Seville plans 350 double charging points.",
                "partial_answer": "Seville plans 350 double charging points.",
                "source_chunk_ids": ["chunk_hidden_1"],
            }
        ]
        return write_success_artifacts(question, run_id, config, excerpts=excerpts)

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

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.chat_reply_helpers.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-citations"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-citations")

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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    call_log: list[bool] = []

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
        excerpts = [
            {
                "ref_id": "ref_1",
                "city_name": "porto",
                "quote": "Porto expects 35% electric and hybrid by 2030.",
                "partial_answer": "Porto expects 35% electric and hybrid by 2030.",
                "source_chunk_ids": ["chunk_hidden_2"],
            }
        ]
        return write_success_artifacts(question, run_id, config, excerpts=excerpts)

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

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)
    monkeypatch.setattr("backend.api.services.chat_reply_helpers.generate_context_chat_reply", _stub_generate_reply)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-retry"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-retry")

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
