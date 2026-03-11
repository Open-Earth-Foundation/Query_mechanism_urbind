"""Context-selection and prompt-cache API chat integration tests."""

from __future__ import annotations

import json
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


def test_chat_context_metrics_expose_prompt_context_tokens_for_excerpt_runs(
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
        return write_success_artifacts(
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

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-excerpts"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-excerpts")

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


def test_chat_context_update_rejects_unknown_context_run(
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

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

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
        conversation_id = create_session.json()["conversation_id"]

        update_contexts = client.put(
            f"/api/v1/runs/run-chat/chat/sessions/{conversation_id}/contexts",
            json={"context_run_ids": ["missing-run-id"]},
        )
        assert update_contexts.status_code == 400


def test_chat_update_contexts_keeps_parent_run_pinned(
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

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Parent doc", "run_id": "run-parent"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-parent")

        start_other = client.post(
            "/api/v1/runs",
            json={"question": "Other doc", "run_id": "run-other"},
        )
        assert start_other.status_code == 202
        poll_until_completed(client, "run-other")

        create_session = client.post("/api/v1/runs/run-parent/chat/sessions", json={})
        conversation_id = create_session.json()["conversation_id"]
        update_contexts = client.put(
            f"/api/v1/runs/run-parent/chat/sessions/{conversation_id}/contexts",
            json={"context_run_ids": ["run-other"]},
        )
        assert update_contexts.status_code == 200
        assert update_contexts.json()["context_run_ids"] == ["run-parent", "run-other"]


def test_chat_session_contexts_allow_extra_runs_when_selection_exceeds_direct_cap(
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
            followup_search_enabled=False,
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
        return write_success_artifacts(question, run_id, config)

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.routes.chat.load_config", _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start_parent = client.post(
            "/api/v1/runs",
            json={"question": "Parent doc", "run_id": "run-parent-cap"},
        )
        assert start_parent.status_code == 202
        poll_until_completed(client, "run-parent-cap")

        start_other = client.post(
            "/api/v1/runs",
            json={"question": "Other doc", "run_id": "run-other-cap"},
        )
        assert start_other.status_code == 202
        poll_until_completed(client, "run-other-cap")

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
        return write_success_artifacts(
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

    patch_api_config_loaders(monkeypatch, _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Build doc", "run_id": "run-chat-cache"},
        )
        assert start.status_code == 202
        poll_until_completed(client, "run-chat-cache")

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
            "backend.api.services.chat_session_helpers.build_session_prompt_context_cache",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("Session prompt cache should be reused.")
            ),
        )

        second_contexts = client.get(
            f"/api/v1/runs/run-chat-cache/chat/sessions/{conversation_id}/contexts"
        )
        assert second_contexts.status_code == 200
        assert second_contexts.json()["prompt_context_tokens"] == first_payload["prompt_context_tokens"]
