"""Clerk-backed authentication helpers for protected API routes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from clerk_backend_api import AuthenticateRequestOptions, RequestState, authenticate_request
from fastapi import HTTPException, Request, status

CLERK_AUTH_SETTINGS_STATE_KEY = "clerk_auth_settings"


@dataclass(frozen=True)
class ClerkAuthSettings:
    """Resolved Clerk auth settings required by the FastAPI API."""

    publishable_key: str
    secret_key: str
    jwt_key: str
    authorized_parties: list[str]


def _normalize_pem_value(raw_value: str) -> str:
    """Normalize PEM env values that may use escaped newlines."""
    return raw_value.replace("\\n", "\n").strip()


def _resolve_authorized_parties(raw_origins: str) -> list[str]:
    """Parse explicit frontend origins used for Clerk authorized parties."""
    authorized_parties: list[str] = []
    for value in raw_origins.split(","):
        cleaned = value.strip()
        if not cleaned or cleaned in authorized_parties:
            continue
        authorized_parties.append(cleaned)
    return authorized_parties


def load_clerk_auth_settings() -> ClerkAuthSettings:
    """Load and validate the Clerk settings required for API auth."""
    publishable_key = os.getenv("CLERK_PUBLISHABLE_KEY", "").strip()
    secret_key = os.getenv("CLERK_SECRET_KEY", "").strip()
    jwt_key = _normalize_pem_value(os.getenv("CLERK_JWT_KEY", ""))
    raw_origins = os.getenv("API_CORS_ORIGINS", "").strip()
    authorized_parties = _resolve_authorized_parties(raw_origins)

    missing_variables = [
        name
        for name, value in (
            ("CLERK_PUBLISHABLE_KEY", publishable_key),
            ("CLERK_SECRET_KEY", secret_key),
            ("CLERK_JWT_KEY", jwt_key),
        )
        if not value
    ]
    if missing_variables:
        missing = ", ".join(missing_variables)
        raise RuntimeError(f"Missing Clerk auth environment variables: {missing}.")
    if not authorized_parties or "*" in authorized_parties:
        raise RuntimeError(
            "API_CORS_ORIGINS must list explicit frontend origins when Clerk auth is enabled."
        )

    return ClerkAuthSettings(
        publishable_key=publishable_key,
        secret_key=secret_key,
        jwt_key=jwt_key,
        authorized_parties=authorized_parties,
    )


def attach_clerk_auth_settings(target: Any, settings: ClerkAuthSettings) -> None:
    """Attach resolved Clerk auth settings to FastAPI app state."""
    setattr(target.state, CLERK_AUTH_SETTINGS_STATE_KEY, settings)


def get_clerk_auth_settings(request: Request) -> ClerkAuthSettings:
    """Return Clerk auth settings stored on the FastAPI app state."""
    settings = getattr(request.app.state, CLERK_AUTH_SETTINGS_STATE_KEY, None)
    if not isinstance(settings, ClerkAuthSettings):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clerk auth settings are not initialized.",
        )
    return settings


def _build_authenticate_request_options(
    settings: ClerkAuthSettings,
) -> AuthenticateRequestOptions:
    """Build Clerk request-auth options for session-token verification."""
    return AuthenticateRequestOptions(
        secret_key=settings.secret_key,
        jwt_key=settings.jwt_key,
        authorized_parties=settings.authorized_parties,
        accepts_token=["session_token"],
    )


def _build_unauthorized_error(state: RequestState) -> HTTPException:
    """Translate a Clerk request state into a FastAPI 401 response."""
    detail = state.message or "Authentication is required."
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def require_clerk_session(request: Request) -> dict[str, Any]:
    """Require a valid Clerk session token and return the verified payload."""
    settings = get_clerk_auth_settings(request)
    state = authenticate_request(
        request,
        _build_authenticate_request_options(settings),
    )
    if not state.is_signed_in or state.payload is None:
        raise _build_unauthorized_error(state)
    request.state.clerk_session = state.payload
    return state.payload


__all__ = [
    "CLERK_AUTH_SETTINGS_STATE_KEY",
    "ClerkAuthSettings",
    "attach_clerk_auth_settings",
    "authenticate_request",
    "get_clerk_auth_settings",
    "load_clerk_auth_settings",
    "require_clerk_session",
]
