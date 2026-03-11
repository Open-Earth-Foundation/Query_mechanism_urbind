"""Shared helpers for API chat integration tests."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.services.chat_followup_research import followup_bundle_dir
from backend.utils.config import AppConfig
from backend.utils.paths import RunPaths, create_run_paths
from tests.support import build_test_app_config


def build_config(
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
    """Build the default API chat test configuration."""
    return build_test_app_config(
        assumptions_reviewer_model="openai/gpt-5.2",
        runs_dir=runs_dir,
        markdown_dir=markdown_dir,
        enable_sql=False,
        chat_overrides={
            "max_history_messages": 10,
            "max_context_total_tokens": max_context_total_tokens,
            "min_prompt_token_cap": min_prompt_token_cap,
            "prompt_token_buffer": prompt_token_buffer,
            "multi_pass_chunk_tokens": multi_pass_chunk_tokens,
            "followup_search_enabled": followup_search_enabled,
            "max_auto_followup_bundles": max_auto_followup_bundles,
        },
    )


def write_success_artifacts(
    question: str,
    run_id: str,
    config: AppConfig,
    excerpts: list[dict[str, object]] | None = None,
) -> RunPaths:
    """Write the completed-run artifacts used by API chat tests."""
    paths = create_run_paths(
        config.runs_dir,
        run_id,
        config.orchestrator.context_bundle_name,
    )
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


def poll_until_completed(
    client: TestClient,
    run_id: str,
    timeout_seconds: float = 3.0,
) -> None:
    """Poll a run until it reaches completed status."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = client.get(f"/api/v1/runs/{run_id}/status").json()
        if payload["status"] == "completed":
            return
        time.sleep(0.02)
    raise AssertionError(f"Run `{run_id}` did not complete in time.")


def poll_chat_job_terminal(
    client: TestClient,
    run_id: str,
    conversation_id: str,
    job_id: str,
    timeout_seconds: float = 3.0,
) -> dict[str, object]:
    """Poll a chat job until it reaches a terminal state."""
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


def patch_api_config_loaders(
    monkeypatch: pytest.MonkeyPatch,
    stub_load_config: object,
) -> None:
    """Patch config loaders used by the API chat execution paths."""
    monkeypatch.setattr("backend.api.services.run_executor.load_config", stub_load_config)
    monkeypatch.setattr("backend.api.routes.chat.load_config", stub_load_config)
    monkeypatch.setattr("backend.api.services.chat_split_flow.load_config", stub_load_config)


def write_followup_bundle(
    *,
    runs_dir: Path,
    run_id: str,
    conversation_id: str,
    bundle_id: str,
    target_city: str,
    quote: str,
    partial_answer: str,
) -> None:
    """Write a synthetic follow-up bundle and its markdown artifacts."""
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
            {"excerpts": context_bundle["markdown"]["excerpts"]},
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


__all__ = [
    "build_config",
    "patch_api_config_loaders",
    "poll_chat_job_terminal",
    "poll_until_completed",
    "write_followup_bundle",
    "write_success_artifacts",
]
