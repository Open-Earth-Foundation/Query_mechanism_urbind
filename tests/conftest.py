"""Pytest auth shims for shared-session protected API tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

TEST_SESSION_SECRET = "0123456789abcdef0123456789abcdef"

os.environ.setdefault("APP_SESSION_SECRET", TEST_SESSION_SECRET)
os.environ.setdefault(
    "API_CORS_ORIGINS",
    "http://127.0.0.1:3000,http://localhost:3000",
)


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
) -> Iterator[None]:
    """Bypass auth for non-auth tests while keeping dedicated auth coverage real."""
    if "test_api_auth.py" not in request.node.nodeid:
        monkeypatch.setattr("backend.api.main.require_shared_session", _allow_shared_session)
    yield
