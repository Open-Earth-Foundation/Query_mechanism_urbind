"""Authentication coverage for shared-session protected API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.auth import SESSION_COOKIE_NAME, create_session_token
from backend.api.main import create_app

TEST_SESSION_SECRET = "0123456789abcdef0123456789abcdef"
TEST_SESSION_TTL_SECONDS = 604_800
ALLOWED_ORIGIN = "http://localhost:3000"


def test_api_routes_require_shared_session_cookie(tmp_path: Path) -> None:
    """Return 401 when a protected API route is called without a session cookie."""
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        client.cookies.pop(SESSION_COOKIE_NAME, None)
        response = client.get("/api/v1/runs")
        assert response.status_code == 401
        assert "authentication is required" in response.json()["detail"].lower()


def test_api_routes_accept_valid_shared_session_cookie(tmp_path: Path) -> None:
    """Allow protected API access when the session cookie is valid."""
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    token = create_session_token(TEST_SESSION_SECRET, TEST_SESSION_TTL_SECONDS)
    with TestClient(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, token)
        response = client.get("/api/v1/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": [], "total": 0}


def test_api_routes_reject_invalid_shared_session_cookie(tmp_path: Path) -> None:
    """Reject session cookies whose signature does not match the configured secret."""
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, "invalid.signature")
        response = client.get("/api/v1/runs")

    assert response.status_code == 401
    assert "authentication is required" in response.json()["detail"].lower()


def test_api_routes_reject_expired_shared_session_cookie(tmp_path: Path) -> None:
    """Reject session cookies once their expiry timestamp is in the past."""
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    token = create_session_token(TEST_SESSION_SECRET, TEST_SESSION_TTL_SECONDS, issued_at=1)
    with TestClient(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, token)
        response = client.get("/api/v1/runs")

    assert response.status_code == 401
    assert "authentication is required" in response.json()["detail"].lower()


def test_api_rejects_cookie_authenticated_post_without_origin(tmp_path: Path) -> None:
    """Reject unsafe cookie-authenticated API requests without a trusted origin."""
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    token = create_session_token(TEST_SESSION_SECRET, TEST_SESSION_TTL_SECONDS)
    with TestClient(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, token)
        response = client.post("/api/v1/runs", json={})

    assert response.status_code == 403
    assert "origin is not allowed" in response.json()["detail"].lower()


def test_api_rejects_cookie_authenticated_post_from_untrusted_origin(
    tmp_path: Path,
) -> None:
    """Reject unsafe cookie-authenticated API requests from disallowed origins."""
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    token = create_session_token(TEST_SESSION_SECRET, TEST_SESSION_TTL_SECONDS)
    with TestClient(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, token)
        response = client.post(
            "/api/v1/runs",
            json={},
            headers={"Origin": "https://evil.example"},
        )

    assert response.status_code == 403
    assert "origin is not allowed" in response.json()["detail"].lower()


def test_api_allows_cookie_authenticated_post_from_trusted_origin(
    tmp_path: Path,
) -> None:
    """Allow unsafe cookie-authenticated API requests from configured origins."""
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    token = create_session_token(TEST_SESSION_SECRET, TEST_SESSION_TTL_SECONDS)
    with TestClient(app) as client:
        client.cookies.set(SESSION_COOKIE_NAME, token)
        response = client.post(
            "/api/v1/runs",
            json={},
            headers={"Origin": ALLOWED_ORIGIN},
        )

    assert response.status_code == 422


def test_public_health_endpoints_remain_unprotected(tmp_path: Path) -> None:
    """Keep root and health checks public for infrastructure probes."""
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        root_response = client.get("/")
        health_response = client.get("/healthz")

    assert root_response.status_code == 200
    assert root_response.json() == {
        "status": "ok",
        "service": "query-mechanism-backend",
    }
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
