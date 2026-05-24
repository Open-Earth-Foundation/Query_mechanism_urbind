"""Shared-password session helpers for protected API routes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import TypedDict
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, status

SESSION_COOKIE_NAME = "urbind_session"
SESSION_SUBJECT = "shared-gate"
SESSION_VERSION = 1
SHARED_SESSION_SETTINGS_STATE_KEY = "shared_session_settings"
MIN_SESSION_SECRET_LENGTH = 32
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class SharedSessionPayload(TypedDict):
    """Validated session payload stored inside the shared cookie."""

    sub: str
    iat: int
    exp: int
    v: int


@dataclass(frozen=True)
class SharedSessionSettings:
    """Resolved session settings required by the FastAPI API."""

    secret_key: bytes
    allowed_origins: frozenset[str]


def _base64url_encode_bytes(value: bytes) -> str:
    """Return a base64url string without padding."""
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _base64url_decode_bytes(value: str) -> bytes | None:
    """Decode a base64url string, returning ``None`` for invalid input."""
    padding = "=" * ((4 - len(value) % 4) % 4)
    try:
        return base64.urlsafe_b64decode(f"{value}{padding}")
    except (ValueError, TypeError):
        return None


def _normalize_secret_key(raw_value: str) -> bytes:
    """Normalize the shared session secret from the environment."""
    return raw_value.strip().encode("utf-8")


def _normalize_origin(raw_value: str) -> str | None:
    """Normalize an Origin or Referer value to scheme://host[:port]."""
    try:
        parsed = urlsplit(raw_value.strip())
    except ValueError:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _sign_payload_segment(payload_segment: str, secret_key: bytes) -> str:
    """Return the base64url HMAC signature for a payload segment."""
    digest = hmac.new(
        secret_key,
        payload_segment.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _base64url_encode_bytes(digest)


def load_shared_session_settings(allowed_origins: list[str]) -> SharedSessionSettings:
    """Load and validate the shared session settings required for API auth."""
    secret_key = _normalize_secret_key(os.getenv("APP_SESSION_SECRET", ""))
    if not secret_key:
        raise RuntimeError("Missing shared auth environment variable: APP_SESSION_SECRET.")
    if len(secret_key) < MIN_SESSION_SECRET_LENGTH:
        raise RuntimeError(
            f"APP_SESSION_SECRET must be at least {MIN_SESSION_SECRET_LENGTH} characters."
        )

    normalized_origins: set[str] = set()
    for raw_origin in allowed_origins:
        origin = _normalize_origin(raw_origin)
        if origin is None:
            raise RuntimeError("API_CORS_ORIGINS contains an invalid origin.")
        normalized_origins.add(origin)

    return SharedSessionSettings(
        secret_key=secret_key,
        allowed_origins=frozenset(normalized_origins),
    )


def attach_shared_session_settings(target: object, settings: SharedSessionSettings) -> None:
    """Attach resolved shared session settings to FastAPI app state."""
    setattr(target.state, SHARED_SESSION_SETTINGS_STATE_KEY, settings)


def get_shared_session_settings(request: Request) -> SharedSessionSettings:
    """Return shared session settings stored on the FastAPI app state."""
    settings = getattr(request.app.state, SHARED_SESSION_SETTINGS_STATE_KEY, None)
    if not isinstance(settings, SharedSessionSettings):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Shared session settings are not initialized.",
        )
    return settings


def create_session_token(
    secret_key: str | bytes,
    ttl_seconds: int,
    issued_at: int | None = None,
) -> str:
    """Create a signed shared-session token for tests or server-side issuance."""
    normalized_secret = (
        secret_key if isinstance(secret_key, bytes) else secret_key.encode("utf-8")
    )
    now_seconds = int(time.time()) if issued_at is None else issued_at
    payload: SharedSessionPayload = {
        "sub": SESSION_SUBJECT,
        "iat": now_seconds,
        "exp": now_seconds + ttl_seconds,
        "v": SESSION_VERSION,
    }
    payload_segment = _base64url_encode_bytes(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    signature_segment = _sign_payload_segment(payload_segment, normalized_secret)
    return f"{payload_segment}.{signature_segment}"


def parse_session_token(
    token: str,
    secret_key: bytes,
    now_seconds: int | None = None,
) -> SharedSessionPayload | None:
    """Return the validated session payload, or ``None`` when invalid."""
    parts = token.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None

    payload_segment, signature_segment = parts
    expected_signature = _sign_payload_segment(payload_segment, secret_key)
    if not hmac.compare_digest(signature_segment, expected_signature):
        return None

    payload_bytes = _base64url_decode_bytes(payload_segment)
    if payload_bytes is None:
        return None

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    subject = payload.get("sub")
    issued_at = payload.get("iat")
    expires_at = payload.get("exp")
    version = payload.get("v")
    current_time = int(time.time()) if now_seconds is None else now_seconds
    if (
        subject != SESSION_SUBJECT
        or version != SESSION_VERSION
        or not isinstance(issued_at, int)
        or not isinstance(expires_at, int)
        or issued_at > expires_at
        or expires_at < current_time
    ):
        return None

    return {
        "sub": subject,
        "iat": issued_at,
        "exp": expires_at,
        "v": version,
    }


def _get_request_origin(request: Request) -> str | None:
    """Return the normalized browser request origin, if present."""
    origin = _normalize_origin(request.headers.get("origin", ""))
    if origin:
        return origin
    return _normalize_origin(request.headers.get("referer", ""))


def _require_trusted_request_origin(
    request: Request,
    settings: SharedSessionSettings,
) -> None:
    """Reject cookie-authenticated unsafe requests from untrusted origins."""
    if request.method.upper() in SAFE_METHODS:
        return

    request_origin = _get_request_origin(request)
    if request_origin in settings.allowed_origins:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Request origin is not allowed.",
    )


def require_shared_session(request: Request) -> SharedSessionPayload:
    """Require a valid shared session cookie and return the verified payload."""
    settings = get_shared_session_settings(request)
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
        )

    payload = parse_session_token(token, settings.secret_key)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
        )

    _require_trusted_request_origin(request, settings)
    request.state.shared_session = payload
    return payload


__all__ = [
    "SESSION_COOKIE_NAME",
    "SharedSessionPayload",
    "SharedSessionSettings",
    "attach_shared_session_settings",
    "create_session_token",
    "get_shared_session_settings",
    "load_shared_session_settings",
    "parse_session_token",
    "require_shared_session",
]
