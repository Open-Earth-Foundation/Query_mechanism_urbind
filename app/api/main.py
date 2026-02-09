"""FastAPI application entrypoint for async run lifecycle API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat_router, cities_router, runs_router
from app.api.services import ChatMemoryStore, RunExecutor, RunStore
from app.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def _resolve_runs_dir(runs_dir: Path | None) -> Path:
    """Resolve runs directory from explicit argument or environment."""
    if runs_dir is not None:
        return runs_dir
    return Path(os.getenv("RUNS_DIR", "output"))


def _resolve_worker_count(max_workers: int | None) -> int:
    """Resolve worker count from argument or API_RUN_WORKERS environment variable."""
    if max_workers is not None:
        return max(1, max_workers)
    raw = os.getenv("API_RUN_WORKERS", "2").strip()
    try:
        parsed = int(raw)
    except ValueError:
        return 2
    return max(1, parsed)


def _resolve_markdown_dir(markdown_dir: Path | None) -> Path:
    """Resolve markdown directory from explicit argument or environment."""
    if markdown_dir is not None:
        return markdown_dir
    return Path(os.getenv("MARKDOWN_DIR", "documents"))


def _resolve_config_path(config_path: Path | None) -> Path:
    """Resolve config path from explicit argument or environment."""
    if config_path is not None:
        return config_path
    return Path(os.getenv("LLM_CONFIG_PATH", "llm_config.yaml"))


def _resolve_city_groups_path(city_groups_path: Path | None) -> Path:
    """Resolve city groups JSON path."""
    if city_groups_path is not None:
        return city_groups_path
    env_path = os.getenv("CITY_GROUPS_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent / "assets" / "city_groups.json"


def _resolve_cors_origins() -> list[str]:
    """Resolve allowed CORS origins for browser frontend."""
    raw = os.getenv(
        "API_CORS_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1:3001,http://localhost:3001",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


def create_app(
    runs_dir: Path | None = None,
    max_workers: int | None = None,
    markdown_dir: Path | None = None,
    config_path: Path | None = None,
    city_groups_path: Path | None = None,
) -> FastAPI:
    """Create FastAPI app instance."""
    setup_logger()
    resolved_runs_dir = _resolve_runs_dir(runs_dir)
    resolved_workers = _resolve_worker_count(max_workers)
    resolved_markdown_dir = _resolve_markdown_dir(markdown_dir)
    resolved_config_path = _resolve_config_path(config_path)
    resolved_city_groups_path = _resolve_city_groups_path(city_groups_path)
    logger.info(
        "Initializing API app runs_dir=%s workers=%d markdown_dir=%s config_path=%s city_groups_path=%s",
        resolved_runs_dir,
        resolved_workers,
        resolved_markdown_dir,
        resolved_config_path,
        resolved_city_groups_path,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("API startup: initializing run store and worker pool")
        run_store = RunStore(resolved_runs_dir)
        chat_memory_store = ChatMemoryStore(resolved_runs_dir)
        run_executor = RunExecutor(run_store=run_store, max_workers=resolved_workers)
        app.state.run_store = run_store
        app.state.chat_memory_store = chat_memory_store
        app.state.run_executor = run_executor
        app.state.markdown_dir = resolved_markdown_dir
        app.state.config_path = resolved_config_path
        app.state.city_groups_path = resolved_city_groups_path
        logger.info("API startup complete")
        yield
        logger.info("API shutdown: stopping worker pool")
        run_executor.shutdown(wait=True)
        logger.info("API shutdown complete")

    app = FastAPI(
        title="Query Mechanism Backend API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(runs_router, prefix="/api/v1", tags=["runs"])
    app.include_router(cities_router, prefix="/api/v1", tags=["cities"])
    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        """Healthcheck endpoint."""
        return {"status": "ok"}

    return app


app = create_app()

__all__ = ["app", "create_app"]
