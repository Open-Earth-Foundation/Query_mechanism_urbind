import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.services.run_store import TERMINAL_STATUSES
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)
from backend.utils.paths import RunPaths, create_run_paths


def _build_config(runs_dir: Path, markdown_dir: Path) -> AppConfig:
    return AppConfig(
        orchestrator=OrchestratorConfig(
            model="test-model", context_bundle_name="context_bundle.json"
        ),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        runs_dir=runs_dir,
        markdown_dir=markdown_dir,
        enable_sql=False,
    )


def _write_success_artifacts(question: str, run_id: str, config: AppConfig) -> RunPaths:
    paths = create_run_paths(config.runs_dir, run_id, config.orchestrator.context_bundle_name)
    paths.base_dir.mkdir(parents=True, exist_ok=True)

    context_bundle = {
        "sql": None,
        "markdown": {"status": "success", "excerpts": []},
        "drafts": [],
        "final": str(paths.final_output),
    }
    paths.context_bundle.write_text(
        json.dumps(context_bundle, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    rendered_output = f"# Question\n{question}\n\n# Answer\nStub answer"
    paths.final_output.write_text(rendered_output, encoding="utf-8")

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
        json.dumps(run_log, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return paths


def _poll_until_terminal(
    client: TestClient,
    run_id: str,
    timeout_seconds: float = 3.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}/status")
        assert response.status_code == 200
        payload = response.json()
        last_payload = payload
        if payload["status"] in TERMINAL_STATUSES:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Run {run_id} did not reach terminal status: {last_payload}")


def test_api_run_lifecycle_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert config.enable_sql is False
        assert run_id is not None
        assert isinstance(log_llm_payload, bool)
        assert selected_cities is None
        return _write_success_artifacts(question=question, run_id=run_id, config=config)

    monkeypatch.setattr("backend.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=2)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "What are the key initiatives?", "run_id": "run-success"},
        )
        assert start.status_code == 202
        start_payload = start.json()
        assert start_payload["run_id"] == "run-success"

        terminal = _poll_until_terminal(client, "run-success")
        assert terminal["status"] == "completed"

        output_response = client.get("/api/v1/runs/run-success/output")
        assert output_response.status_code == 200
        output_payload = output_response.json()
        assert output_payload["status"] == "completed"
        assert "Stub answer" in output_payload["content"]

        context_response = client.get("/api/v1/runs/run-success/context")
        assert context_response.status_code == 200
        context_payload = context_response.json()
        assert context_payload["status"] == "completed"
        assert isinstance(context_payload["context_bundle"], dict)


def test_api_duplicate_run_id_returns_conflict(
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
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert selected_cities is None
        return _write_success_artifacts(question=question, run_id=run_id, config=config)

    monkeypatch.setattr("backend.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1)
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/runs",
            json={"question": "First", "run_id": "same-run"},
        )
        assert first.status_code == 202

        second = client.post(
            "/api/v1/runs",
            json={"question": "Second", "run_id": "same-run"},
        )
        assert second.status_code == 409


def test_api_status_not_found(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        response = client.get("/api/v1/runs/unknown/status")
        assert response.status_code == 404


def test_api_root_healthcheck(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"


def test_api_list_runs_reads_artifact_folders(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)
    _write_success_artifacts(
        question="Historic run from artifact folder",
        run_id="run-from-folder",
        config=config,
    )

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["runs"][0]["run_id"] == "run-from-folder"
        assert (
            payload["runs"][0]["question"] == "Historic run from artifact folder"
        )


def test_api_output_returns_conflict_while_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    started = threading.Event()
    release = threading.Event()

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    def _slow_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert selected_cities is None
        started.set()
        release.wait(timeout=2)
        return _write_success_artifacts(question=question, run_id=run_id, config=config)

    monkeypatch.setattr("backend.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _slow_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Running test", "run_id": "run-running"},
        )
        assert start.status_code == 202
        assert started.wait(timeout=1)

        output_response = client.get("/api/v1/runs/run-running/output")
        assert output_response.status_code == 409

        release.set()
        terminal = _poll_until_terminal(client, "run-running")
        assert terminal["status"] == "completed"


def test_api_failed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    def _failing_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert selected_cities is None
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr("backend.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr(
        "backend.api.services.run_executor.run_pipeline", _failing_run_pipeline
    )

    app = create_app(runs_dir=runs_dir, max_workers=1)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Failing test", "run_id": "run-failed"},
        )
        assert start.status_code == 202

        terminal = _poll_until_terminal(client, "run-failed")
        assert terminal["status"] == "failed"
        assert terminal["error"]["code"] == "RUN_EXECUTION_ERROR"

        output_response = client.get("/api/v1/runs/run-failed/output")
        assert output_response.status_code == 409

        context_response = client.get("/api/v1/runs/run-failed/context")
        assert context_response.status_code == 409


def test_api_run_filters_markdown_by_selected_cities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / "Berlin.md").write_text("# Berlin", encoding="utf-8")
    (markdown_dir / "Munich.md").write_text("# Munich", encoding="utf-8")
    captured_files: list[str] = []

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    def _stub_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert selected_cities == ["Berlin"]
        captured_files.extend(
            sorted(path.name for path in config.markdown_dir.rglob("*.md"))
        )
        return _write_success_artifacts(question=question, run_id=run_id, config=config)

    monkeypatch.setattr("backend.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={
                "question": "Only Berlin please",
                "run_id": "run-berlin",
                "cities": ["Berlin"],
            },
        )
        assert start.status_code == 202
        terminal = _poll_until_terminal(client, "run-berlin")
        assert terminal["status"] == "completed"
        listed_runs = client.get("/api/v1/runs")
        assert listed_runs.status_code == 200
        listed_ids = [item["run_id"] for item in listed_runs.json()["runs"]]
        assert listed_ids == ["run-berlin"]

    assert captured_files == ["Berlin.md"]


def test_api_list_runs_deduplicates_legacy_alias_records(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    state_dir = runs_dir / "_api_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    legacy_run_dir = runs_dir / "legacy-run_01"
    legacy_run_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = legacy_run_dir / "run.json"
    started_at = datetime.now(timezone.utc).isoformat()
    completed_at = datetime.now(timezone.utc).isoformat()

    run_log_path.write_text(
        json.dumps(
            {
                "run_id": "legacy-run_01",
                "question": "Legacy alias run",
                "status": "completed",
                "started_at": started_at,
                "completed_at": completed_at,
                "finish_reason": "completed (write)",
                "artifacts": {
                    "context_bundle": str(legacy_run_dir / "context_bundle.json"),
                    "final_output": str(legacy_run_dir / "final.md"),
                },
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    shared_payload = {
        "question": "Legacy alias run",
        "status": "completed",
        "started_at": started_at,
        "completed_at": completed_at,
        "finish_reason": "completed (write)",
        "error": None,
        "final_output_path": str(legacy_run_dir / "final.md"),
        "context_bundle_path": str(legacy_run_dir / "context_bundle.json"),
        "run_log_path": str(run_log_path),
    }
    (state_dir / "legacy-run.json").write_text(
        json.dumps({"run_id": "legacy-run", **shared_payload}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (state_dir / "legacy-run_01.json").write_text(
        json.dumps(
            {"run_id": "legacy-run_01", **shared_payload}, ensure_ascii=True, indent=2
        ),
        encoding="utf-8",
    )

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        listed_runs = client.get("/api/v1/runs")
        assert listed_runs.status_code == 200
        payload = listed_runs.json()
        assert payload["total"] == 1
        assert payload["runs"][0]["run_id"] == "legacy-run"


def test_api_run_supports_header_api_key_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    captured_api_key: dict[str, str | None] = {"value": None}

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    def _stub_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
        api_key_override: str | None = None,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert run_id is not None
        assert selected_cities is None
        captured_api_key["value"] = api_key_override
        return _write_success_artifacts(question=question, run_id=run_id, config=config)

    monkeypatch.setattr("backend.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr("backend.api.services.run_executor.run_pipeline", _stub_run_pipeline)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Header key test", "run_id": "run-header-key"},
            headers={"X-OpenRouter-Api-Key": "sk-or-v1-user-test-key"},
        )
        assert start.status_code == 202
        terminal = _poll_until_terminal(client, "run-header-key")
        assert terminal["status"] == "completed"

    assert captured_api_key["value"] == "sk-or-v1-user-test-key"


def test_api_key_error_is_reported_with_sanitized_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    def _stub_load_config(_path: Path | None = None) -> AppConfig:
        return _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    def _failing_run_pipeline(
        question: str,
        config: AppConfig,
        run_id: str | None = None,
        log_llm_payload: bool = True,
        selected_cities: list[str] | None = None,
    ) -> RunPaths:
        assert selected_cities is None
        raise RuntimeError(
            "Incorrect API key provided: sk-or-v1-abcdefghijklmnopqrstuv0123456789"
        )

    monkeypatch.setattr("backend.api.services.run_executor.load_config", _stub_load_config)
    monkeypatch.setattr(
        "backend.api.services.run_executor.run_pipeline", _failing_run_pipeline
    )

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        start = client.post(
            "/api/v1/runs",
            json={"question": "Key fail test", "run_id": "run-key-fail"},
        )
        assert start.status_code == 202
        terminal = _poll_until_terminal(client, "run-key-fail")
        assert terminal["status"] == "failed"
        assert terminal["error"]["code"] == "API_KEY_ERROR"
        assert "sk-or-v1-" not in terminal["error"]["message"]
