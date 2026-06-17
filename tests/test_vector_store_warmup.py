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
            files_changed=1,
            files_unchanged=2,
            files_deleted=3,
            chunks_created=4,
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
        "files_changed": 1,
        "files_unchanged": 2,
        "files_deleted": 3,
        "chunks_created": 4,
    }
    assert warmup.is_blocking_runs() is False
