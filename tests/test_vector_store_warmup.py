import json
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
    warmup = VectorStoreWarmup()

    warmup.start(config=config, docs_dir=tmp_path)

    snapshot = warmup.snapshot()
    assert snapshot["status"] == "skipped"
    assert snapshot["enabled"] is True
    assert snapshot["auto_update_on_run"] is False
    assert warmup.is_blocking_runs() is False


def test_vector_store_warmup_records_successful_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm-up should expose update stats after the background refresh completes."""
    config = build_test_app_config(runs_dir=tmp_path / "runs", markdown_dir=tmp_path)
    config.vector_store.enabled = True
    config.vector_store.auto_update_on_run = True

    def _fake_update_markdown_index(**kwargs):
        assert kwargs["config"] is config
        assert kwargs["docs_dir"] == tmp_path
        assert kwargs["selected_cities"] is None
        assert kwargs["dry_run"] is False
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
            dry_run=False,
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
    assert artifact["event_type"] == "vector_store_startup_warmup"
    assert artifact["trigger"] == "api_startup"
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
    assert warmup.is_blocking_runs() is False
