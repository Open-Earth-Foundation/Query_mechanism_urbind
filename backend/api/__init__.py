"""FastAPI application package for async run lifecycle endpoints."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazily expose the FastAPI app to avoid package import cycles."""
    if name == "app":
        from backend.api.main import app

        return app
    if name == "create_app":
        from backend.api.main import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "create_app"]
