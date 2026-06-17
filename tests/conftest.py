"""Pytest auth shims for shared-session protected API tests."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support import build_test_app_config

TEST_SESSION_SECRET = "0123456789abcdef0123456789abcdef"
TEST_API_CORS_ORIGINS = "http://127.0.0.1:3000,http://localhost:3000"

# backend.api.main creates a module-level FastAPI app during import. These
# defaults must exist before API test modules are collected in clean CI shells.
os.environ.setdefault("APP_SESSION_SECRET", TEST_SESSION_SECRET)
os.environ.setdefault("API_CORS_ORIGINS", TEST_API_CORS_ORIGINS)


def _allow_shared_session() -> dict[str, int | str]:
    """Return a synthetic shared-session payload for non-auth API tests."""
    return {
        "sub": "shared-gate",
        "iat": 0,
        "exp": 4_102_444_800,
        "v": 1,
    }


@pytest.fixture(autouse=True)
def shared_session_test_auth(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[None]:
    """Bypass auth and isolate default API startup config for integration tests."""
    monkeypatch.setenv("APP_SESSION_SECRET", TEST_SESSION_SECRET)
    monkeypatch.setenv("API_CORS_ORIGINS", TEST_API_CORS_ORIGINS)
    config = build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
        vector_store_overrides={"enabled": False, "auto_update_on_run": False},
    )
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        json.dumps(config.model_dump(mode="json"), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("VECTOR_STORE_ENABLED", "false")
    monkeypatch.setenv("VECTOR_STORE_AUTO_UPDATE_ON_RUN", "false")
    if "test_api_auth.py" not in request.node.nodeid:
        monkeypatch.setattr("backend.api.main.require_shared_session", _allow_shared_session)
    yield
