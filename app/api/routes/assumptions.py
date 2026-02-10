"""Missing-data assumptions discovery and regeneration endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request, status
from openai import APIStatusError, AuthenticationError

from app.api.models import AssumptionsPayload, RegenerationResult
from app.api.services.assumptions_review import (
    apply_assumptions_and_regenerate,
    discover_missing_data_for_run,
    load_latest_assumptions_payload,
)
from app.api.services.run_store import RunRecord, RunStore, SUCCESS_STATUSES
from app.utils.config import AppConfig, load_config

router = APIRouter()


def _resolve_api_key_override(raw: str | None) -> str | None:
    """Normalize optional API key override header."""
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


def _require_ready_run(run_id: str, request: Request) -> tuple[RunStore, RunRecord]:
    """Require run existence and success status before assumptions operations."""
    run_store = _get_run_store(request)
    run_record = run_store.get_run(run_id)
    if run_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run `{run_id}` was not found.",
        )
    if run_record.status not in SUCCESS_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Run `{run_id}` is not ready for assumptions review "
                f"(status: `{run_record.status}`)."
            ),
        )
    return run_store, run_record


def _load_api_config(request: Request) -> AppConfig:
    """Load API configuration using app-level configured path."""
    config_path = getattr(request.app.state, "config_path", Path("llm_config.yaml"))
    config = load_config(Path(config_path))
    config.enable_sql = False
    return config


def _raise_llm_http_error(exc: Exception) -> None:
    """Normalize LLM provider failures to HTTP errors."""
    if isinstance(exc, AuthenticationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Authentication failed for provided API key. "
                "Switch to a valid OpenRouter key and retry."
            ),
        ) from exc
    if isinstance(exc, APIStatusError):
        if exc.status_code in {401, 403}:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "API key rejected by provider (401/403). "
                    "Switch to another OpenRouter key and retry."
                ),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Assumptions reviewer provider error: {exc.status_code}",
        ) from exc
    if isinstance(exc, EnvironmentError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"{exc}. Provide your own OpenRouter key via UI "
                "or configure server OPENROUTER_API_KEY."
            ),
        ) from exc


@router.post(
    "/runs/{run_id}/assumptions/discover",
    name="discover_run_assumptions",
)
def discover_run_assumptions(
    run_id: str,
    request: Request,
    x_openrouter_api_key: str | None = Header(
        default=None, alias="X-OpenRouter-Api-Key"
    ),
) -> dict[str, object]:
    """Run two-pass missing-data discovery for a completed run."""
    run_store, run_record = _require_ready_run(run_id, request)
    api_key_override = _resolve_api_key_override(x_openrouter_api_key)
    config = _load_api_config(request)
    try:
        return discover_missing_data_for_run(
            run_store=run_store,
            run_record=run_record,
            config=config,
            api_key_override=api_key_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _raise_llm_http_error(exc)
        raise


@router.post(
    "/runs/{run_id}/assumptions/apply",
    name="apply_run_assumptions",
    response_model=RegenerationResult,
)
def apply_run_assumptions(
    run_id: str,
    payload: AssumptionsPayload,
    request: Request,
    x_openrouter_api_key: str | None = Header(
        default=None, alias="X-OpenRouter-Api-Key"
    ),
) -> RegenerationResult:
    """Apply edited assumptions and regenerate revised output for a completed run."""
    run_store, run_record = _require_ready_run(run_id, request)
    api_key_override = _resolve_api_key_override(x_openrouter_api_key)
    config = _load_api_config(request)
    try:
        return apply_assumptions_and_regenerate(
            run_store=run_store,
            run_record=run_record,
            payload=payload,
            config=config,
            api_key_override=api_key_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        _raise_llm_http_error(exc)
        raise


@router.get(
    "/runs/{run_id}/assumptions/latest",
    name="get_latest_run_assumptions",
)
def get_latest_run_assumptions(
    run_id: str,
    request: Request,
) -> dict[str, object]:
    """Return persisted assumptions artifacts for a run when present."""
    run_store, _ = _require_ready_run(run_id, request)
    payload = load_latest_assumptions_payload(run_store, run_id)
    if len(payload) <= 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No assumptions artifacts found for run `{run_id}`.",
        )
    return payload


__all__ = ["router"]
