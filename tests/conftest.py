"""Pytest auth shims for Clerk-protected API tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from clerk_backend_api import RequestState
from clerk_backend_api.security import (
    AuthErrorReason,
    AuthStatus,
    TokenVerificationErrorReason,
)
from fastapi.testclient import TestClient

TEST_AUTHORIZATION_HEADER = "Bearer test-session-token"
TEST_SESSION_TOKEN = "test-session-token"

os.environ.setdefault("CLERK_SECRET_KEY", "sk_test_placeholder")
os.environ.setdefault("CLERK_PUBLISHABLE_KEY", "pk_test_placeholder")
os.environ.setdefault(
    "CLERK_JWT_KEY",
    "-----BEGIN PUBLIC KEY-----\\nTESTKEY\\n-----END PUBLIC KEY-----",
)
os.environ.setdefault(
    "API_CORS_ORIGINS",
    "http://127.0.0.1:3000,http://localhost:3000",
)


def _fake_authenticate_request(request: object, options: object) -> RequestState:
    """Return a signed-in Clerk request state only for the shared test token."""
    del options
    authorization = getattr(request, "headers", {}).get("Authorization")
    if authorization == TEST_AUTHORIZATION_HEADER:
        return RequestState(
            status=AuthStatus.SIGNED_IN,
            token=TEST_SESSION_TOKEN,
            payload={
                "sub": "user_test_123",
                "sid": "sess_test_123",
                "azp": "http://127.0.0.1:3000",
            },
        )
    if authorization:
        return RequestState(
            status=AuthStatus.SIGNED_OUT,
            reason=TokenVerificationErrorReason.TOKEN_INVALID,
        )
    return RequestState(
        status=AuthStatus.SIGNED_OUT,
        reason=AuthErrorReason.SESSION_TOKEN_MISSING,
    )


def _patch_test_client_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the shared test Clerk session header into protected API calls."""
    original_request = TestClient.request

    def authenticated_request(
        self: TestClient,
        method: str,
        url: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        if url.startswith("/api/v1"):
            raw_headers = kwargs.get("headers")
            headers = dict(raw_headers) if raw_headers is not None else {}
            lower_headers = {str(key).lower(): value for key, value in headers.items()}
            if "authorization" not in lower_headers:
                headers["Authorization"] = TEST_AUTHORIZATION_HEADER
            kwargs["headers"] = headers
        return original_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(TestClient, "request", authenticated_request)


@pytest.fixture(autouse=True)
def clerk_test_auth(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Patch Clerk verification so existing tests can exercise protected routes."""
    monkeypatch.setattr(
        "backend.api.auth.authenticate_request",
        _fake_authenticate_request,
    )
    _patch_test_client_request(monkeypatch)
    yield
