import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.api.services import vector_store_warmup as warmup_module
from backend.api.services.vector_store_warmup import VectorStoreWarmup
from tests.support import build_test_app_config


def test_vector_store_warmup_skips_when_auto_update_disabled(tmp_path: Path) -> None:
    """Warm-up should not run unless vector auto-update is enabled."""
    config = build_test_app_config(runs_dir=tmp_path / "runs", markdown_dir=tmp_path)
    config.vector_store.enabled = True
    config.vector_store.auto_update_on_run = False
    config.vector_store.chroma_persist_path = tmp_path / "chroma"
    config.vector_store.index_manifest_path = tmp_path / "chroma" / "index_manifest.json"
    warmup = VectorStoreWarmup()

    warmup.start(config=config, docs_dir=tmp_path)

    snapshot = warmup.snapshot()
    assert snapshot["status"] == "skipped"
    assert snapshot["enabled"] is True
    assert snapshot["auto_update_on_run"] is False
    assert warmup.is_blocking_runs() is False
    status_path = config.vector_store.chroma_persist_path / "update_status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["update_mode"] == "local_process"


def test_vector_store_warmup_records_successful_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm-up should expose update stats after the background refresh completes."""
    config = build_test_app_config(runs_dir=tmp_path / "runs", markdown_dir=tmp_path)
    config.vector_store.enabled = True
    config.vector_store.auto_update_on_run = True
    calls: list[bool] = []

    def _fake_update_markdown_index(**kwargs):
        assert kwargs["config"] is config
        assert kwargs["docs_dir"] == tmp_path
        assert kwargs["selected_cities"] is None
        calls.append(bool(kwargs["dry_run"]))
        return SimpleNamespace(
            files_indexed=6,
            files_changed=1,
            files_unchanged=2,
            files_deleted=3,
            chunks_created=4,
            table_chunks=5,
            min_tokens=10,
            avg_tokens=20.5,
            max_tokens=30,
            dry_run=bool(kwargs["dry_run"]),
            update_mode="incremental_update",
            changed_files=[
                {
                    "source_path": "documents/Aachen.md",
                    "status": "modified",
                    "previous_chunk_count": 3,
                    "current_chunk_count": 4,
                    "removed_previous_chunk_count": 3,
                }
            ],
            deleted_files=[],
        )

    monkeypatch.setattr(
        warmup_module,
        "update_markdown_index",
        _fake_update_markdown_index,
    )
    warmup = VectorStoreWarmup()

    warmup.start(config=config, docs_dir=tmp_path)
    warmup.shutdown(wait=True)

    snapshot = warmup.snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["message"] == "Vector store is up to date."
    assert snapshot["stats"] == {
        "files_indexed": 6,
        "files_changed": 1,
        "files_unchanged": 2,
        "files_deleted": 3,
        "chunks_created": 4,
        "table_chunks": 5,
        "min_tokens": 10,
        "avg_tokens": 20.5,
        "max_tokens": 30,
        "dry_run": False,
        "update_mode": "incremental_update",
        "changed_files": [
            {
                "source_path": "documents/Aachen.md",
                "status": "modified",
                "previous_chunk_count": 3,
                "current_chunk_count": 4,
                "removed_previous_chunk_count": 3,
            }
        ],
        "deleted_files": [],
    }
    latest_artifact = snapshot["latest_artifact"]
    assert latest_artifact == "system/vector_store_warmup/latest.json"
    latest_path = tmp_path / "runs" / str(latest_artifact)
    assert latest_path.exists()
    timestamped_artifacts = [
        path for path in latest_path.parent.glob("*.json") if path.name != "latest.json"
    ]
    assert len(timestamped_artifacts) == 1
    artifact = json.loads(latest_path.read_text(encoding="utf-8"))
    assert artifact["event_type"] == "vector_store_update"
    assert artifact["trigger"] == "startup"
    assert artifact["status"] == "completed"
    assert artifact["stats"] == snapshot["stats"]
    assert artifact["vector_store_snapshot"]["auto_update"]["update_mode"] == "incremental_update"
    assert artifact["vector_store_snapshot"]["auto_update"]["stats"]["changed_files"] == [
        {
            "source_path": "documents/Aachen.md",
            "status": "modified",
            "previous_chunk_count": 3,
            "current_chunk_count": 4,
            "removed_previous_chunk_count": 3,
        }
    ]
    assert calls == [True, False]
    assert warmup.is_blocking_runs() is False


def test_vector_store_warmup_records_failed_update_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm-up failures should persist diagnostics outside user run folders."""
    config = build_test_app_config(runs_dir=tmp_path / "runs", markdown_dir=tmp_path)
    config.vector_store.enabled = True
    config.vector_store.auto_update_on_run = True

    def _fake_update_markdown_index(**kwargs):
        assert kwargs["config"] is config
        assert kwargs["dry_run"] is True
        raise RuntimeError("embedding provider unavailable")

    monkeypatch.setattr(
        warmup_module,
        "update_markdown_index",
        _fake_update_markdown_index,
    )
    warmup = VectorStoreWarmup()

    warmup.start(config=config, docs_dir=tmp_path)
    warmup.shutdown(wait=True)

    snapshot = warmup.snapshot()
    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "embedding provider unavailable"
    assert snapshot["latest_artifact"] == "system/vector_store_warmup/latest.json"
    latest_path = tmp_path / "runs" / "system" / "vector_store_warmup" / "latest.json"
    artifact = json.loads(latest_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "failed"
    assert artifact["error"] == "embedding provider unavailable"
    assert artifact["stats"] is None
    assert artifact["vector_store_snapshot"]["enabled"] is True
    assert warmup.is_blocking_runs() is True


def test_vector_store_warmup_kubernetes_mode_creates_job_when_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kubernetes mode should trigger a Job instead of updating in-process."""
    config = build_test_app_config(runs_dir=tmp_path / "runs", markdown_dir=tmp_path)
    config.vector_store.enabled = True
    config.vector_store.auto_update_on_run = True
    config.vector_store.update_mode = "kubernetes_job"
    created_jobs: list[str] = []

    def _fake_update_markdown_index(**kwargs):
        assert kwargs["dry_run"] is True
        return SimpleNamespace(
            files_indexed=1,
            files_changed=1,
            files_unchanged=0,
            files_deleted=0,
            chunks_created=2,
            table_chunks=0,
            min_tokens=1,
            avg_tokens=1.0,
            max_tokens=1,
            dry_run=True,
            update_mode="incremental_update",
            changed_files=[],
            deleted_files=[],
        )

    def _fake_create_vector_store_update_job(*, trigger: str) -> str:
        created_jobs.append(trigger)
        return "vector-update-job"

    monkeypatch.setattr(
        warmup_module,
        "update_markdown_index",
        _fake_update_markdown_index,
    )
    monkeypatch.setattr(
        warmup_module,
        "create_vector_store_update_job",
        _fake_create_vector_store_update_job,
    )
    warmup = VectorStoreWarmup()

    blocking_reason = warmup.ensure_ready_for_run(config=config, docs_dir=tmp_path)

    assert blocking_reason == "Vector store is stale; updater Job is running."
    assert created_jobs == ["run"]
    snapshot = warmup.snapshot()
    assert snapshot["status"] == "running"
    assert snapshot["update_mode"] == "kubernetes_job"
    assert snapshot["job_name"] == "vector-update-job"


def test_vector_store_warmup_marks_stale_running_status_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A killed updater Job should not leave the UI stuck in running forever."""
    config = build_test_app_config(runs_dir=tmp_path / "runs", markdown_dir=tmp_path)
    config.vector_store.enabled = True
    config.vector_store.auto_update_on_run = True
    config.vector_store.update_mode = "kubernetes_job"
    config.vector_store.chroma_persist_path = tmp_path / "chroma"
    config.vector_store.index_manifest_path = tmp_path / "chroma" / "index_manifest.json"
    old_started_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    status_path = config.vector_store.chroma_persist_path / "update_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "message": "Vector store is stale; updater Job is running.",
                "started_at": old_started_at,
                "job_name": "dead-job",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VECTOR_STORE_UPDATE_JOB_TIMEOUT_SECONDS", "1")
    warmup = VectorStoreWarmup()
    warmup._configure(config)

    snapshot = warmup.snapshot()

    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "Vector store updater Job timed out."


def test_vector_store_warmup_ignores_stale_status_file_from_other_update_mode(
    tmp_path: Path,
) -> None:
    """A persisted status from another update mode should not block the current runtime."""
    config = build_test_app_config(runs_dir=tmp_path / "runs", markdown_dir=tmp_path)
    config.vector_store.enabled = True
    config.vector_store.auto_update_on_run = True
    config.vector_store.update_mode = "local_process"
    config.vector_store.chroma_persist_path = tmp_path / "chroma"
    config.vector_store.index_manifest_path = tmp_path / "chroma" / "index_manifest.json"
    status_path = config.vector_store.chroma_persist_path / "update_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "update_mode": "kubernetes_job",
                "message": "Vector store is stale; updater Job is running.",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "job_name": "old-job",
            }
        ),
        encoding="utf-8",
    )
    warmup = VectorStoreWarmup()
    warmup._configure(config)

    snapshot = warmup.snapshot()

    assert snapshot["status"] == "pending"
    assert snapshot["job_name"] is None
    assert snapshot["message"] == "Vector store warm-up has not started."


def test_vector_store_warmup_reconciles_completed_job_before_blocking_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed external updater Job should clear stale in-memory running state."""
    config = build_test_app_config(runs_dir=tmp_path / "runs", markdown_dir=tmp_path)
    config.vector_store.enabled = True
    config.vector_store.auto_update_on_run = True
    config.vector_store.update_mode = "kubernetes_job"
    config.vector_store.chroma_persist_path = tmp_path / "chroma"
    config.vector_store.index_manifest_path = tmp_path / "chroma" / "index_manifest.json"
    status_path = config.vector_store.chroma_persist_path / "update_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "update_mode": "kubernetes_job",
                "message": "Vector store is up to date.",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    def _fake_update_markdown_index(**kwargs):
        assert kwargs["dry_run"] is True
        return SimpleNamespace(
            files_indexed=1,
            files_changed=0,
            files_unchanged=1,
            files_deleted=0,
            chunks_created=0,
            table_chunks=0,
            min_tokens=1,
            avg_tokens=1.0,
            max_tokens=1,
            dry_run=True,
            update_mode="incremental_update",
            changed_files=[],
            deleted_files=[],
        )

    monkeypatch.setattr(
        warmup_module,
        "update_markdown_index",
        _fake_update_markdown_index,
    )
    warmup = VectorStoreWarmup()
    warmup._configure(config)
    warmup._status = "running"
    warmup._message = "Vector store is stale; updater Job is running."

    blocking_reason = warmup.ensure_ready_for_run(config=config, docs_dir=tmp_path)

    assert blocking_reason is None
    snapshot = warmup.snapshot()
    assert snapshot["status"] == "completed"
    assert snapshot["message"] == "Vector store is up to date."
