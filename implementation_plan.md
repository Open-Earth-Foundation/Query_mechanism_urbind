# FastAPI Backend Implementation Plan (Ticketized)

## Target Outcome
Build an async FastAPI backend inside `app/` that:

1. Starts a long-running answer generation job with a unique `run_id`.
2. Supports frontend ping/poll for status by `run_id`.
3. Exposes final answer and `context_bundle` as API endpoints after completion.
4. Persists backend artifacts (`run.json`, `final.md`, `context_bundle.json`) on server side.

The existing pipeline (`app/modules/orchestrator/module.py::run_pipeline`) remains the execution engine. The API layer orchestrates job lifecycle and exposes artifacts.

## API Contract (v1)

### `POST /api/v1/runs`
Purpose: trigger a run.

Request body:

```json
{
  "question": "What are main climate initiatives?",
  "run_id": "optional-client-id",
  "enable_sql": false,
  "markdown_path": "documents",
  "log_llm_payload": false
}
```

Response `202 Accepted`:

```json
{
  "run_id": "20260209_1201",
  "status": "queued",
  "status_url": "/api/v1/runs/20260209_1201/status",
  "output_url": "/api/v1/runs/20260209_1201/output",
  "context_url": "/api/v1/runs/20260209_1201/context"
}
```

### `GET /api/v1/runs/{run_id}/status`
Purpose: ping endpoint for frontend polling.

Response `200 OK`:

```json
{
  "run_id": "20260209_1201",
  "status": "running",
  "started_at": "2026-02-09T12:01:11Z",
  "completed_at": null,
  "finish_reason": null,
  "error": null
}
```

Status values:

- `queued`
- `running`
- `completed`
- `completed_with_gaps`
- `failed`
- `stopped`

### `GET /api/v1/runs/{run_id}/output`
Purpose: return final answer.

Response `200 OK` (completed states):

```json
{
  "run_id": "20260209_1201",
  "status": "completed",
  "content": "<final markdown output>",
  "final_output_path": "output/20260209_1201/final.md"
}
```

Response `409 Conflict` when run is still `queued` or `running`.

### `GET /api/v1/runs/{run_id}/context`
Purpose: return persisted context bundle.

Response `200 OK`:

```json
{
  "run_id": "20260209_1201",
  "status": "completed",
  "context_bundle": {},
  "context_bundle_path": "output/20260209_1201/context_bundle.json"
}
```

Response `409 Conflict` when run is still `queued` or `running`.

## Ticket Backlog

## BE-001: FastAPI App Scaffold in `app/`
Deep dive:

- Add `app/api/` package with `__init__.py`.
- Create `app/api/main.py` with FastAPI app factory and router registration.
- Create `app/api/routes/runs.py` for run lifecycle endpoints.
- Create `app/api/models.py` for request/response Pydantic models.
- Keep imports absolute (project rule), keep logic in services, routes thin.

Deliverables:

- `app/api/main.py`
- `app/api/routes/runs.py`
- `app/api/models.py`
- `app/api/services/__init__.py`

Acceptance criteria:

- `uvicorn app.api.main:app` starts successfully.
- OpenAPI docs show all run endpoints.

## BE-002: Run State Store and Persistence
Deep dive:

- Implement `app/api/services/run_store.py` as single source of truth for API job state.
- Track state transitions and timestamps for each `run_id`.
- Persist state to disk (for restart visibility), for example `output/<run_id>/api_state.json`.
- Reconcile with existing `output/<run_id>/run.json` written by `RunLogger`.
- Guarantee thread-safe access with `threading.Lock`.

Deliverables:

- In-memory + file-backed run state.
- State transition API (`queued -> running -> terminal`).

Acceptance criteria:

- Status endpoint shows deterministic state for every run.
- Duplicate `run_id` requests are rejected with clear error.

## BE-003: Background Execution Worker
Deep dive:

- Implement `app/api/services/run_executor.py`.
- Run pipeline in background thread or executor so HTTP request returns quickly.
- Reuse existing `run_pipeline()`; do not duplicate orchestration logic.
- On completion, capture:
  - terminal status (`completed`, `completed_with_gaps`, `failed`, `stopped`)
  - `final.md` path
  - `context_bundle.json` path
  - failure details if exception occurs

Deliverables:

- Non-blocking run submission from API.
- Reliable terminal transition and error capture.

Acceptance criteria:

- `POST /runs` responds in under 1 second for normal conditions.
- Long LLM execution no longer blocks request thread.

## BE-004: Trigger Endpoint (`POST /api/v1/runs`)
Deep dive:

- Validate input and normalize options to `AppConfig`.
- Generate/validate unique `run_id` (client-provided optional; server fallback using `build_run_id`).
- Queue execution via run executor.
- Return contract with status/output/context URLs.

Deliverables:

- Request model with strict validation.
- Idempotency/uniqueness rules defined and enforced.

Acceptance criteria:

- Returns `202` with generated `run_id`.
- Handles invalid payloads with `422`.

## BE-005: Ping Endpoint (`GET /api/v1/runs/{run_id}/status`)
Deep dive:

- Serve lightweight polling response for frontend.
- Include timestamps, finish reason, and optional error payload.
- Ensure response is fast and independent from artifact file size.

Deliverables:

- Stable status contract for frontend polling loop.

Acceptance criteria:

- Frontend can poll every 2-5 seconds without heavy server load.
- Unknown run returns `404`.

## BE-006: Output + Context Endpoints
Deep dive:

- `GET /output`: read `final.md` and return markdown content + path metadata.
- `GET /context`: read `context_bundle.json` and return parsed JSON + path metadata.
- Return `409` if run is not terminal.
- Return `500` with meaningful error if artifact is expected but missing.

Deliverables:

- Consistent artifact retrieval API.
- Explicit error semantics for frontend.

Acceptance criteria:

- Completed runs always provide both answer and context endpoints.
- Endpoint payloads match v1 contract exactly.

## BE-007: API Test Coverage (pytest)
Deep dive:

- Add tests under `tests/` for:
  - run trigger response shape
  - status polling transitions
  - output/context retrieval
  - unknown run and duplicate run behavior
  - failed run behavior
- Stub executor for fast deterministic tests (no real LLM calls in unit tests).

Deliverables:

- `tests/test_api_runs.py` with regression coverage.

Acceptance criteria:

- `pytest` passes locally.
- Critical path endpoints have deterministic tests.

## BE-008: Pre-Frontend Smoke Test Script
Deep dive:

- Add runnable script in `app/scripts/` that simulates frontend behavior:
  - trigger run
  - poll status until terminal
  - fetch output endpoint
  - fetch context endpoint
  - save responses locally for debugging
- Script becomes the integration gate before frontend starts.

Deliverables:

- `python -m app.scripts.test_async_backend_flow ...` command.

Acceptance criteria:

- Script exits non-zero when lifecycle contract is broken.
- Script writes JSON artifacts for debugging.

## BE-009: Documentation and Developer Workflow
Deep dive:

- Update `README.md` with FastAPI run commands and endpoint usage.
- Document polling flow for frontend engineers.
- Add Docker example for API mode.

Deliverables:

- README API section with local + Docker run and smoke test command.

Acceptance criteria:

- New contributor can run API and smoke test without guessing.

## Suggested Build Order

1. BE-001
2. BE-002
3. BE-003
4. BE-004
5. BE-005
6. BE-006
7. BE-007
8. BE-008
9. BE-009

## Notes

- Keep all backend implementation under `app/` as requested.
- Reuse existing artifact persistence from `output/<run_id>/`.
- Do not block frontend on synchronous LLM execution; status polling is the primary UX pattern.
