"""Authentication coverage for Clerk-protected API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.main import create_app


def test_api_routes_require_clerk_authentication(tmp_path: Path) -> None:
    """Return 401 when a protected API route is called without a session token."""
    app = create_app(runs_dir=tmp_path / "output", max_workers=1)
    with TestClient(app) as client:
        response = client.get("/api/v1/runs", headers={"Authorization": ""})
        assert response.status_code == 401
        assert "session token" in response.json()["detail"].lower()


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
