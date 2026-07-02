"""Regression tests for shared pytest configuration helpers."""

from __future__ import annotations

from pathlib import Path

from tests.support import build_test_app_config


def test_build_test_app_config_disables_vector_store_by_default(tmp_path: Path) -> None:
    """Shared test configs stay isolated from real Chroma state unless opted in."""
    config = build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
    )

    assert config.vector_store.enabled is False
    assert config.vector_store.auto_update_on_run is False
    assert config.vector_store.chroma_persist_path == tmp_path / "output" / "_test_chroma"
    assert config.vector_store.index_manifest_path == (
        tmp_path / "output" / "_test_chroma" / "index_manifest.json"
    )
