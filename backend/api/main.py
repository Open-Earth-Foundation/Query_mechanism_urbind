"""FastAPI application entrypoint for async run lifecycle API"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.auth import (
    SESSION_COOKIE_NAME,
    attach_shared_session_settings,
    load_shared_session_settings,
    require_shared_session,
)
from backend.api.routes import (
    assumptions_router,
    chat_router,
    cities_router,
    runs_router,
    system_router,
)
from backend.api.services.chat_jobs import ChatJobExecutor, ChatJobStore
from backend.api.services.chat_memory import ChatMemoryStore
from backend.api.services.feature_readiness import FeatureReadinessService
from backend.api.services.run_executor import RunExecutor
from backend.api.services.run_store import RunStore
from backend.api.services.vector_store_warmup import VectorStoreWarmup
from backend.api.services.chat_split_flow import build_chat_job_processor
from backend.utils.config import AppConfig, load_config, resolve_path_relative_to_config
from backend.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)
# this is responsible for how many runs we can run per instance
DEFAULT_API_RUN_WORKERS = 2
DEFAULT_API_CHAT_JOB_WORKERS = 1


def _resolve_allowed_origins() -> list[str]:
    """Parse explicit frontend origins for credentialed browser requests."""
    raw_origins = os.getenv("API_CORS_ORIGINS", "").strip()
    allowed_origins = [value.strip() for value in raw_origins.split(",") if value.strip()]
    if not allowed_origins or "*" in allowed_origins:
        raise RuntimeError(
            "API_CORS_ORIGINS must list explicit frontend origins when shared auth is enabled."
        )
    return allowed_origins


def _resolve_runs_dir(
    runs_dir: Path | None,
    *,
    config_path: Path,
    config: AppConfig,
) -> Path:
    """Resolve runs directory from explicit argument or loaded config."""
    if runs_dir is not None:
        return resolve_path_relative_to_config(config_path, runs_dir)
    return config.runs_dir


def _resolve_worker_count(max_workers: int | None) -> int:
    """Resolve worker count from explicit argument or hardcoded default."""
    if max_workers is None:
        return DEFAULT_API_RUN_WORKERS
    return max(1, max_workers)


def _resolve_chat_job_worker_count() -> int:
    """Resolve dedicated split-mode chat job workers."""
    env_value = os.getenv("API_CHAT_JOB_WORKERS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            logger.warning(
                "Invalid API_CHAT_JOB_WORKERS=%s; falling back to default.", env_value
            )
    return DEFAULT_API_CHAT_JOB_WORKERS


def _resolve_markdown_dir(
    markdown_dir: Path | None,
    *,
    config_path: Path,
    config: AppConfig,
) -> Path:
    """Resolve markdown directory from explicit argument or loaded config."""
    if markdown_dir is not None:
        return resolve_path_relative_to_config(config_path, markdown_dir)
    return config.markdown_dir


def _resolve_config_path(config_path: Path | None) -> Path:
    """Resolve config path from explicit argument or environment."""
    if config_path is not None:
        return config_path.expanduser().resolve()
    return Path(os.getenv("LLM_CONFIG_PATH", "llm_config.yaml")).expanduser().resolve()


def _resolve_city_groups_path(city_groups_path: Path | None, *, config_path: Path) -> Path:
    """Resolve city groups JSON path."""
    if city_groups_path is not None:
        return resolve_path_relative_to_config(config_path, city_groups_path)
    env_path = os.getenv("CITY_GROUPS_PATH")
    if env_path:
        return resolve_path_relative_to_config(config_path, Path(env_path))
    return Path(__file__).resolve().parent / "assets" / "city_groups.json"


def create_app(
    runs_dir: Path | None = None,
    max_workers: int | None = None,
    markdown_dir: Path | None = None,
    config_path: Path | None = None,
    city_groups_path: Path | None = None,
) -> FastAPI:
    """Create FastAPI app instance."""
    load_dotenv()
    setup_logger()
    allowed_origins = _resolve_allowed_origins()
    shared_session_settings = load_shared_session_settings(allowed_origins)
    resolved_config_path = _resolve_config_path(config_path)
    startup_config = load_config(resolved_config_path)
    resolved_runs_dir = _resolve_runs_dir(
        runs_dir,
        config_path=resolved_config_path,
        config=startup_config,
    )
    resolved_workers = _resolve_worker_count(max_workers)
    resolved_chat_job_workers = _resolve_chat_job_worker_count()
    resolved_markdown_dir = _resolve_markdown_dir(
        markdown_dir,
        config_path=resolved_config_path,
        config=startup_config,
    )
    resolved_city_groups_path = _resolve_city_groups_path(
        city_groups_path,
        config_path=resolved_config_path,
    )
    logger.info(
        "Initializing API app runs_dir=%s workers=%d chat_job_workers=%d markdown_dir=%s config_path=%s city_groups_path=%s",
        resolved_runs_dir,
        resolved_workers,
        resolved_chat_job_workers,
        resolved_markdown_dir,
        resolved_config_path,
        resolved_city_groups_path,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("API startup: initializing run store and worker pools")
        vector_store_warmup = VectorStoreWarmup()
        feature_readiness = FeatureReadinessService()
        try:
            mode = (
                "vector_store_retrieval"
                if startup_config.vector_store.enabled
                else "standard_chunking"
            )
            logger.info("API startup: markdown_source_mode=%s", mode)
            vector_store_warmup.start(
                config=startup_config,
                docs_dir=resolved_markdown_dir,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("API startup: could not load config for mode log: %s", e)
        run_store = RunStore(resolved_runs_dir)
        chat_memory_store = ChatMemoryStore(resolved_runs_dir)
        chat_job_store = ChatJobStore(resolved_runs_dir)
        run_executor = RunExecutor(run_store=run_store, max_workers=resolved_workers)
        chat_job_executor = ChatJobExecutor(
            job_store=chat_job_store,
            chat_memory_store=chat_memory_store,
            processor=build_chat_job_processor(
                run_store=run_store,
                chat_memory_store=chat_memory_store,
                config_path=resolved_config_path,
            ),
            max_workers=resolved_chat_job_workers,
        )
        chat_job_executor.reconcile_interrupted_jobs()
        app.state.run_store = run_store
        app.state.chat_memory_store = chat_memory_store
        app.state.chat_job_store = chat_job_store
        app.state.run_executor = run_executor
        app.state.chat_job_executor = chat_job_executor
        app.state.vector_store_warmup = vector_store_warmup
        app.state.feature_readiness = feature_readiness
        app.state.markdown_dir = resolved_markdown_dir
        app.state.config_path = resolved_config_path
        app.state.city_groups_path = resolved_city_groups_path
        logger.info("API startup complete")
        yield
        logger.info("API shutdown: stopping worker pools")
        chat_job_executor.shutdown(wait=True)
        run_executor.shutdown(wait=True)
        vector_store_warmup.shutdown(wait=False)
        logger.info("API shutdown complete")

    app = FastAPI(
        title="Query Mechanism Backend API",
        version="0.1.0",
        lifespan=lifespan,
    )
    attach_shared_session_settings(app, shared_session_settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(
        "CORS allow_origins=%s allow_credentials=%s shared_cookie_name=%s",
        allowed_origins,
        True,
        SESSION_COOKIE_NAME,
    )
    protected_api = APIRouter(
        prefix="/api/v1",
        dependencies=[Depends(require_shared_session)],
    )
    protected_api.include_router(runs_router, tags=["runs"])
    protected_api.include_router(cities_router, tags=["cities"])
    protected_api.include_router(chat_router, tags=["chat"])
    protected_api.include_router(assumptions_router, tags=["assumptions"])
    protected_api.include_router(system_router, tags=["system"])
    app.include_router(protected_api)

    @app.get("/")
    def root() -> dict[str, str]:
        """Root health endpoint used by default health checks."""
        return {"status": "ok", "service": "query-mechanism-backend"}

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        """Healthcheck endpoint."""
        return {"status": "ok"}

    return app


app = create_app()

__all__ = ["app", "create_app"]
