"""Ingestion handler registry.

Each handler accepts an ``IngestionContext`` and returns an
``IngestionState``.  Handlers are registered under stable dotted names
(e.g. ``ingest.pdf_to_markdown``) which the manifest references via the
``handler:`` field.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from backend.modules.sources.manifest import IngestionConfig, SourceConfig
from backend.modules.sources.state import IngestionState


class IngestionContext:
    """Inputs handed to an ingestion handler."""

    def __init__(
        self,
        *,
        source: SourceConfig,
        ingestion: IngestionConfig,
        upstream_root: Path,
        project_root: Path,
        resolved_commit: str | None = None,
    ) -> None:
        self.source = source
        self.ingestion = ingestion
        self.upstream_root = upstream_root
        self.project_root = project_root
        self.resolved_commit = resolved_commit


HandlerFn = Callable[[IngestionContext], IngestionState]


class _Registry:
    def __init__(self) -> None:
        self._handlers: dict[str, HandlerFn] = {}

    def register(self, name: str, fn: HandlerFn) -> None:
        if name in self._handlers:
            raise ValueError(f"handler already registered: {name!r}")
        self._handlers[name] = fn

    def get(self, name: str) -> HandlerFn:
        if name not in self._handlers:
            raise KeyError(f"handler not registered: {name!r}")
        return self._handlers[name]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._handlers

    def names(self) -> list[str]:
        return sorted(self._handlers.keys())


REGISTRY = _Registry()


def register(name: str) -> Callable[[HandlerFn], HandlerFn]:
    """Decorator for registering an ingestion handler."""

    def _wrap(fn: HandlerFn) -> HandlerFn:
        REGISTRY.register(name, fn)
        return fn

    return _wrap


def get_handler(name: str) -> HandlerFn:
    return REGISTRY.get(name)


# Eagerly import handler modules so their @register decorators run.
# Add new handler imports here as they're built.
from backend.modules.sources.handlers import pdf_to_markdown as _pdf_to_markdown  # noqa: E402,F401
from backend.modules.sources.handlers import bnetza_etl as _bnetza_etl  # noqa: E402,F401
from backend.modules.sources.handlers import pdf_to_vector as _pdf_to_vector  # noqa: E402,F401
from backend.modules.sources.handlers import extract_web_allowlist as _extract_web_allowlist  # noqa: E402,F401
from backend.modules.sources.handlers import urban_audit_population_etl as _urban_audit_population_etl  # noqa: E402,F401


__all__ = [
    "HandlerFn",
    "IngestionContext",
    "REGISTRY",
    "get_handler",
    "register",
]
