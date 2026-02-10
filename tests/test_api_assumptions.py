import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.models import AssumptionsPayload, MissingDataItem, RegenerationResult
from app.api.services.assumptions_review import discover_missing_data
from app.api.services.run_store import RunStore
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


def _create_completed_run(
    run_store: RunStore,
    config: AppConfig,
    run_id: str,
    question: str,
) -> None:
    record = run_store.create_queued_run(question=question, requested_run_id=run_id)
    paths = _write_success_artifacts(question, run_id, config)
    run_store.mark_terminal(
        run_id=record.run_id,
        status="completed",
        finish_reason="completed (write)",
        final_output_path=paths.final_output,
        context_bundle_path=paths.context_bundle,
        run_log_path=paths.run_log,
    )


def test_assumptions_discover_returns_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        _create_completed_run(
            run_store=client.app.state.run_store,
            config=config,
            run_id="run-assumptions",
            question="Build assumptions run",
        )

        monkeypatch.setattr(
            "app.api.routes.assumptions.discover_missing_data_for_run",
            lambda **_: {
                "run_id": "run-assumptions",
                "items": [
                    {
                        "city": "Berlin",
                        "missing_description": "Missing EV charger count",
                        "proposed_number": 1500,
                    }
                ],
                "grouped_by_city": {
                    "Berlin": [
                        {
                            "city": "Berlin",
                            "missing_description": "Missing EV charger count",
                            "proposed_number": 1500,
                        }
                    ]
                },
                "verification_summary": {
                    "first_pass_count": 1,
                    "second_pass_count": 0,
                    "merged_count": 1,
                    "added_in_verification": 0,
                },
            },
        )

        response = client.post("/api/v1/runs/run-assumptions/assumptions/discover")
        assert response.status_code == 200
        payload = response.json()
        assert payload["run_id"] == "run-assumptions"
        assert payload["items"][0]["city"] == "Berlin"
        assert payload["grouped_by_city"]["Berlin"][0]["proposed_number"] == 1500


def test_assumptions_discover_not_found(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        response = client.post("/api/v1/runs/missing/assumptions/discover")
        assert response.status_code == 404


def test_assumptions_discover_requires_completed_run(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        run_store = client.app.state.run_store
        run_store.create_queued_run(question="Queued run", requested_run_id="run-queued")
        response = client.post("/api/v1/runs/run-queued/assumptions/discover")
        assert response.status_code == 409


def test_assumptions_apply_regeneration_returns_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        _create_completed_run(
            run_store=client.app.state.run_store,
            config=config,
            run_id="run-assumptions",
            question="Build assumptions run",
        )
        monkeypatch.setattr(
            "app.api.routes.assumptions.apply_assumptions_and_regenerate",
            lambda **_: RegenerationResult(
                run_id="run-assumptions",
                revised_output_path="output/run-assumptions/assumptions/final_with_assumptions.md",
                revised_content="# Revised",
                assumptions_path="output/run-assumptions/assumptions/edited.json",
            ),
        )
        response = client.post(
            "/api/v1/runs/run-assumptions/assumptions/apply",
            json={
                "items": [
                    {
                        "city": "Berlin",
                        "missing_description": "Missing EV charger count",
                        "proposed_number": 1500,
                    }
                ]
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["run_id"] == "run-assumptions"
        assert payload["assumptions_path"].endswith("edited.json")


def test_discover_missing_data_runs_two_pass_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    config = _build_config(runs_dir=runs_dir, markdown_dir=markdown_dir)
    call_order: list[str] = []

    def _stub_run_pass(
        pass_name: str,
        question: str,
        final_document: str,
        context_bundle: dict[str, object],
        existing_items: list[MissingDataItem],
        config: AppConfig,
        api_key_override: str | None = None,
    ) -> list[MissingDataItem]:
        call_order.append(pass_name)
        if pass_name == "extract":
            return [
                MissingDataItem(
                    city="Berlin",
                    missing_description="Missing EV charger count",
                    proposed_number=1000,
                )
            ]
        return [
            MissingDataItem(
                city="Berlin",
                missing_description="Missing EV charger count",
                proposed_number=1200,
            ),
            MissingDataItem(
                city="Munich",
                missing_description="Missing charging points per district",
                proposed_number=None,
            ),
        ]

    monkeypatch.setattr(
        "app.api.services.assumptions_review._run_discovery_pass",
        _stub_run_pass,
    )
    payload = discover_missing_data(
        question="Question",
        final_document="# Answer",
        context_bundle={"markdown": {"status": "success"}},
        config=config,
    )
    assert call_order == ["extract", "verify"]
    assert payload["verification_summary"]["first_pass_count"] == 1
    assert payload["verification_summary"]["added_in_verification"] == 1
    assert len(payload["items"]) == 2


def test_assumptions_payload_requires_items() -> None:
    with pytest.raises(ValueError):
        AssumptionsPayload(items=[])
