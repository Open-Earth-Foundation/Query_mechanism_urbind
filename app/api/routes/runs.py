"""Run lifecycle HTTP endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.api.models import (
    CreateRunRequest,
    CreateRunResponse,
    RunContextResponse,
    RunOutputResponse,
    RunStatusResponse,
)
from app.api.services import (
    DuplicateRunIdError,
    IN_PROGRESS_STATUSES,
    SUCCESS_STATUSES,
    RunExecutor,
    RunStore,
    StartRunCommand,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _resolve_api_key_override(raw: str | None) -> str | None:
    """Normalize optional API key header value."""
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    return cleaned


def _get_run_store(request: Request) -> RunStore:
    """Return run store from FastAPI app state."""
    run_store = getattr(request.app.state, "run_store", None)
    if not isinstance(run_store, RunStore):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Run store is not initialized.",
        )
    return run_store


def _get_run_executor(request: Request) -> RunExecutor:
    """Return run executor from FastAPI app state."""
    run_executor = getattr(request.app.state, "run_executor", None)
    if not isinstance(run_executor, RunExecutor):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Run executor is not initialized.",
        )
    return run_executor


@router.post(
    "/runs",
    response_model=CreateRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_run(
    payload: CreateRunRequest,
    request: Request,
    x_openrouter_api_key: str | None = Header(
        default=None, alias="X-OpenRouter-Api-Key"
    ),
) -> CreateRunResponse:
    """Queue a new asynchronous run."""
    logger.info(
        "API create_run received run_id=%s cities=%s config_path=%s markdown_path=%s api_key_override=%s",
        payload.run_id,
        len(payload.cities) if payload.cities else "all",
        payload.config_path,
        payload.markdown_path,
        x_openrouter_api_key is not None,
    )
    run_executor = _get_run_executor(request)
    api_key_override = _resolve_api_key_override(x_openrouter_api_key)
    try:
        record = run_executor.submit(
            StartRunCommand(
                question=payload.question,
                requested_run_id=payload.run_id,
                cities=payload.cities,
                config_path=payload.config_path,
                markdown_path=payload.markdown_path,
                log_llm_payload=payload.log_llm_payload,
                api_key=api_key_override,
            )
        )
    except DuplicateRunIdError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    logger.info("API create_run accepted run_id=%s", record.run_id)

    return CreateRunResponse(
        run_id=record.run_id,
        status=record.status,
        status_url=str(request.url_for("get_run_status", run_id=record.run_id)),
        output_url=str(request.url_for("get_run_output", run_id=record.run_id)),
        context_url=str(request.url_for("get_run_context", run_id=record.run_id)),
    )


@router.get(
    "/runs/{run_id}/status",
    name="get_run_status",
    response_model=RunStatusResponse,
)
def get_run_status(run_id: str, request: Request) -> RunStatusResponse:
    """Return run status for frontend polling."""
    run_store = _get_run_store(request)
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run `{run_id}` was not found.",
        )

    return RunStatusResponse(
        run_id=record.run_id,
        status=record.status,
        started_at=record.started_at,
        completed_at=record.completed_at,
        finish_reason=record.finish_reason,
        error=record.error,
    )


@router.get(
    "/runs/{run_id}/output",
    name="get_run_output",
    response_model=RunOutputResponse,
)
def get_run_output(run_id: str, request: Request) -> RunOutputResponse:
    """Return final output markdown for completed run."""
    run_store = _get_run_store(request)
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run `{run_id}` was not found.",
        )
    if record.status in IN_PROGRESS_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run `{run_id}` is still {record.status}.",
        )
    if record.status not in SUCCESS_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run `{run_id}` ended with status `{record.status}`.",
        )

    output_path = _resolve_output_path(record.final_output_path, run_store.runs_dir, run_id)
    if output_path is None or not output_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Final output is missing for run `{run_id}`.",
        )

    try:
        content = output_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read final output for run `{run_id}`: {exc}",
        ) from exc

    return RunOutputResponse(
        run_id=record.run_id,
        status=record.status,
        content=content,
        final_output_path=str(output_path),
    )


@router.get(
    "/runs/{run_id}/context",
    name="get_run_context",
    response_model=RunContextResponse,
)
def get_run_context(run_id: str, request: Request) -> RunContextResponse:
    """Return context bundle for completed run."""
    run_store = _get_run_store(request)
    record = run_store.get_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run `{run_id}` was not found.",
        )
    if record.status in IN_PROGRESS_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run `{run_id}` is still {record.status}.",
        )
    if record.status not in SUCCESS_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run `{run_id}` ended with status `{record.status}`.",
        )

    context_path = _resolve_context_path(
        record.context_bundle_path, run_store.runs_dir, run_id
    )
    if context_path is None or not context_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context bundle is missing for run `{run_id}`.",
        )

    try:
        context_bundle = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read context bundle for run `{run_id}`: {exc}",
        ) from exc

    if not isinstance(context_bundle, dict):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context bundle for run `{run_id}` is not a JSON object.",
        )

    return RunContextResponse(
        run_id=record.run_id,
        status=record.status,
        context_bundle=context_bundle,
        context_bundle_path=str(context_path),
    )


def _resolve_output_path(path: Path | None, runs_dir: Path, run_id: str) -> Path | None:
    """Resolve final output path with run directory fallback."""
    if path is not None:
        return path
    fallback = runs_dir / run_id / "final.md"
    if fallback.exists():
        return fallback
    return None


def _resolve_context_path(path: Path | None, runs_dir: Path, run_id: str) -> Path | None:
    """Resolve context bundle path with run directory fallback."""
    if path is not None:
        return path
    fallback = runs_dir / run_id / "context_bundle.json"
    if fallback.exists():
        return fallback
    return None


__all__ = ["router"]
