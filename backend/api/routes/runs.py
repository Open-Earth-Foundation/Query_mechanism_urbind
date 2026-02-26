"""Run lifecycle HTTP endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, Request, status

from backend.api.models import (
    CreateRunRequest,
    CreateRunResponse,
    RunReferenceItem,
    RunReferenceListResponse,
    RunReferenceResponse,
    RunListResponse,
    RunContextResponse,
    RunOutputResponse,
    RunSummary,
    RunStatusResponse,
)
from backend.api.services import (
    DuplicateRunIdError,
    IN_PROGRESS_STATUSES,
    RunRecord,
    SUCCESS_STATUSES,
    RunExecutor,
    RunStore,
    StartRunCommand,
)
from backend.modules.orchestrator.utils.references import (
    build_markdown_references,
    is_valid_ref_id,
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


def _require_completed_run(
    run_id: str, request: Request
) -> tuple[RunStore, RunRecord]:
    """Return run store and run record for completed runs only."""
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
    return run_store, record


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


@router.get("/runs", response_model=RunListResponse)
def list_runs(request: Request) -> RunListResponse:
    """List runs with run_id and original question."""
    run_store = _get_run_store(request)
    records = run_store.list_runs()
    runs = [
        RunSummary(
            run_id=record.run_id,
            question=record.question,
        )
        for record in records
    ]
    return RunListResponse(runs=runs, total=len(runs))


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
    run_store, record = _require_completed_run(run_id, request)

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
    run_store, record = _require_completed_run(run_id, request)

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


@router.get(
    "/runs/{run_id}/references",
    name="list_run_references",
    response_model=RunReferenceListResponse,
    response_model_exclude_none=True,
)
def list_run_references(
    run_id: str,
    request: Request,
    ref_id: str | None = Query(
        default=None,
        description="Optional `ref_n` id filter. When omitted, all references are returned.",
    ),
    include_quote: bool = Query(
        default=False,
        description=(
            "When false (default), only lightweight citation fields are returned for "
            "inline label rendering. Set true to include quote payload for click-to-inspect UX."
        ),
    ),
) -> RunReferenceListResponse:
    """Return run-scoped references for document/chat citation rendering."""
    items = _resolve_run_reference_items(
        run_id=run_id,
        request=request,
        ref_id=ref_id,
        include_quote=include_quote,
    )
    return RunReferenceListResponse(
        run_id=run_id,
        reference_count=len(items),
        references=items,
    )


@router.get(
    "/runs/{run_id}/references/{ref_id}",
    name="get_run_reference",
    response_model=RunReferenceResponse,
)
def get_run_reference(
    run_id: str,
    ref_id: str,
    request: Request,
) -> RunReferenceResponse:
    """Compatibility alias for fetching one reference with quote payload."""
    items = _resolve_run_reference_items(
        run_id=run_id,
        request=request,
        ref_id=ref_id,
        include_quote=True,
    )
    item = items[0]
    return RunReferenceResponse(
        run_id=run_id,
        ref_id=item.ref_id,
        excerpt_index=item.excerpt_index,
        city_name=item.city_name,
        quote=item.quote or "",
        partial_answer=item.partial_answer or "",
        source_chunk_ids=item.source_chunk_ids or [],
    )


def _resolve_run_reference_items(
    run_id: str,
    request: Request,
    ref_id: str | None,
    include_quote: bool,
) -> list[RunReferenceItem]:
    """Resolve run references with optional `ref_id` filter and quote payload."""
    normalized_ref_id = (ref_id or "").strip()
    if normalized_ref_id and not is_valid_ref_id(normalized_ref_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ref_id must match format `ref_<positive_integer>`.",
        )

    run_store, record = _require_completed_run(run_id, request)
    run_dir = _resolve_run_dir(record, run_store.runs_dir, run_id)
    records = _load_reference_records(run_dir, run_id)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No markdown references available for run `{run_id}`.",
        )

    items: list[RunReferenceItem] = []
    for item in records:
        normalized = _build_run_reference_item(record=item, include_quote=include_quote)
        if not is_valid_ref_id(normalized.ref_id):
            continue
        items.append(normalized)
    items.sort(key=lambda item: (item.excerpt_index, item.ref_id))

    if normalized_ref_id:
        filtered = [item for item in items if item.ref_id == normalized_ref_id]
        if not filtered:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference `{normalized_ref_id}` was not found for run `{run_id}`.",
            )
        return filtered
    return items


def _build_run_reference_item(
    record: dict[str, object], include_quote: bool
) -> RunReferenceItem:
    """Normalize one raw reference record into API response shape."""
    item = RunReferenceItem(
        ref_id=str(record.get("ref_id", "")).strip(),
        excerpt_index=_parse_excerpt_index(record.get("excerpt_index")),
        city_name=str(record.get("city_name", "")).strip(),
    )
    if include_quote:
        item.quote = str(record.get("quote", ""))
        item.partial_answer = str(record.get("partial_answer", ""))
        item.source_chunk_ids = _normalize_source_chunk_ids(record.get("source_chunk_ids"))
    return item


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


def _resolve_run_dir(record: RunRecord, runs_dir: Path, run_id: str) -> Path:
    """Resolve run artifact directory from available run record paths."""
    if record.run_log_path is not None:
        return record.run_log_path.parent
    if record.context_bundle_path is not None:
        return record.context_bundle_path.parent
    if record.final_output_path is not None:
        return record.final_output_path.parent
    return runs_dir / run_id


def _parse_excerpt_index(value: object) -> int:
    """Parse excerpt index into a non-negative integer."""
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str):
        try:
            return max(int(value.strip()), 0)
        except ValueError:
            return 0
    return 0


def _normalize_source_chunk_ids(value: object) -> list[str]:
    """Normalize source chunk ids to string list."""
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def _coerce_reference_records(value: object) -> list[dict[str, object]]:
    """Coerce reference payload into a list of dict records."""
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def _load_reference_records(run_dir: Path, run_id: str) -> list[dict[str, object]]:
    """Load reference records from references artifact or fallback excerpts artifact."""
    references_path = run_dir / "markdown" / "references.json"
    if references_path.exists():
        try:
            payload = json.loads(references_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to parse references artifact at %s", references_path)
            return []
        if isinstance(payload, dict):
            records = _coerce_reference_records(payload.get("references"))
            if records:
                return records

    excerpts_path = run_dir / "markdown" / "excerpts.json"
    if not excerpts_path.exists():
        return []
    try:
        excerpts_payload = json.loads(excerpts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to parse excerpts artifact at %s", excerpts_path)
        return []
    if not isinstance(excerpts_payload, dict):
        return []
    raw_excerpts = excerpts_payload.get("excerpts")
    excerpt_records = _coerce_reference_records(raw_excerpts)
    if not excerpt_records:
        return []
    _enriched_excerpts, references_payload = build_markdown_references(
        run_id=run_id,
        excerpts=excerpt_records,
    )
    return _coerce_reference_records(references_payload.get("references"))


__all__ = ["router"]
