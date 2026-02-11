# Implementation Plan: Missing Data Assumptions by City

## 1. Goal

Add a post-generation flow that:

1. Detects missing quantitative inputs in the generated report or the context bundle even if they are not mentioned in the answer (for example EV charger counts).
2. Returns a structured list grouped by city.
3. Lets the user adjust proposed values.
4. Regenerates a revised final document using the adjusted assumptions and the original context bundle.

This flow should make missing data explicit in the final answer instead of silently skipping gaps.

## 2. User Flow (Target UX)

1. User runs the normal document build and gets `final.md` + `context_bundle.json`.
2. User opens a new page: `Assumptions Review`.
3. Frontend calls an API endpoint that runs a first "bigger model" pass over:
   - original question
   - final answer document
   - context bundle
4. Backend runs a second verification pass:
   - receives pass-1 output + same source inputs
   - checks for missed items and adds any missing rows
   - deduplicates and normalizes final list
5. API returns missing-data items grouped by city, each with:
   - city
   - missing description
   - proposed number
6. User edits proposed numbers.
7. User clicks `Regenerate`.
8. API regenerates a revised document using:
   - original context bundle
   - user-approved assumptions
9. Frontend shows the revised document and keeps the assumptions artifact visible for auditability.

## 3. Backend Design

### 3.1 Data Models (Pydantic)

Keep schemas minimal in `backend/api/models.py`:

- `MissingDataItem`
  - `city: str`
  - `missing_description: str`
  - `proposed_number: float | int | None`
- `AssumptionsPayload`
  - `items: list[MissingDataItem]` (user-adjusted values)
  - optional `rewrite_instructions: str | None`
- `RegenerationResult`
  - `run_id: str`
  - `revised_output_path: str`
  - `revised_content: str`
  - `assumptions_path: str`

Notes:

- No `id` field is used, to avoid LLM-generated identifiers.
- `discover` can return an inline payload (`run_id`, `items`, `grouped_by_city`, `verification_summary`) without adding extra wrapper models.

### 3.1.1 Schema Ownership (LLM vs API)

Use strict separation so LLM errors cannot break the API contract.

- LLM-facing schema (structured output from model):
  - `MissingDataItem` only.
  - Optional envelope for model output parsing: `{ "items": list[MissingDataItem] }`.
- API-facing schemas (frontend-backend contract clarity):
  - `AssumptionsPayload` for `POST /assumptions/apply` request validation.
  - `RegenerationResult` for `POST /assumptions/apply` response stability.
  - Inline typed payload for `discover` response (backend-generated).

LLM must never generate these fields:

- `run_id`
- `grouped_by_city`
- artifact paths (`assumptions_path`, `revised_output_path`, etc.)
- status/metadata fields

Backend responsibilities (deterministic, non-LLM):

- validate `MissingDataItem` output
- run pass-2 verification merge + deduplication
- build `grouped_by_city`
- attach run metadata and artifact paths in API responses

### 3.2 New API Endpoints

Add a dedicated router, for example `backend/api/routes/assumptions.py`:

- `POST /api/v1/runs/{run_id}/assumptions/discover`
  - Loads run question + final output + context bundle
  - Call #1: high-capability model extracts missing-data items
  - Call #2: high-capability model verifies coverage and appends missed items
  - Returns final structured items grouped by city + verification summary
- `POST /api/v1/runs/{run_id}/assumptions/apply`
  - Accepts `AssumptionsPayload` from edited UI values
  - Produces revised document content
  - Persists revised artifacts
  - Returns `RegenerationResult`
- Optional: `GET /api/v1/runs/{run_id}/assumptions/latest`
  - Returns last discovered/edited assumptions without re-running model

Register this router in `backend/api/main.py`.

### 3.3 Service Layer

Add `backend/api/services/assumptions_review.py` with focused functions:

- `discover_missing_data(...) -> list[MissingDataItem]`
- `group_missing_data_by_city(...) -> dict[str, list[MissingDataItem]]`
- `apply_assumptions_to_context(...) -> dict[str, object]`
- `rewrite_document_with_assumptions(...) -> str`

Implementation notes:

- Use one dedicated model config for this stage (new config section; see section 5).
- Keep prompt + parsing strict so output always matches Pydantic schema. Use function calling or structured output for llm models
- Discovery uses a 2-pass sequence (extract -> verify) before returning results.
- Fail fast with clear API errors when required artifacts are missing.

### 3.4 Artifact Persistence

Store outputs under each run:

- `output/<run_id>/assumptions/discovered.json`
- `output/<run_id>/assumptions/edited.json`
- `output/<run_id>/assumptions/revised_context_bundle.json`
- `output/<run_id>/assumptions/final_with_assumptions.md`

Also record paths in `run.json` artifacts so everything is traceable.

## 4. Frontend Design

### 4.1 New Assumptions Page

Add a new page/workspace that is reachable from the generated document panel, for example:

- route option: `frontend/src/app/runs/[runId]/assumptions/page.tsx`
- or integrated workspace component with mode switch in current page

Recommended behavior:

1. "Find Missing Data" button triggers discovery endpoint.
2. UI shows "Pass 1 + verification pass complete" status, then renders final grouped list.
3. Each row shows:
   - city (read-only)
   - missing description (read-only)
   - proposed number (editable input)
4. `Regenerate` button submits edited rows.
5. Display revised document preview and artifact metadata.

### 4.2 Frontend API Client Additions

Extend `frontend/src/lib/api.ts` with:

- types: `MissingDataItem`, `AssumptionsPayload`, `RegenerationResult`
- functions:
  - `discoverRunAssumptions(runId: string)`
  - `applyRunAssumptions(runId: string, payload: AssumptionsPayload)`

## 5. LLM/Config Changes

Add an assumptions model section in config:

- `assumptions_reviewer` in `llm_config.yaml`
  - model: larger/stronger reasoning model
  - temperature: low
  - max_output_tokens tuned for structured JSON output

Update `AppConfig` in `backend/utils/config.py` to load this section.

Prompt contract for discovery call:

- Input includes question + final answer + context bundle.
- Pass-1 output is structured JSON matching `MissingDataItem` only.
- Pass-2 verifies pass-1 coverage, adds missing rows, and removes duplicates (still `MissingDataItem` only).
- Focus on missing quantitative values needed for actionable city-level recommendations.
- Do not invent values without stating they are proposed assumptions.
- Do not output API metadata fields (`run_id`, grouped structures, paths).

## 6. Rewrite Strategy

For `assumptions/apply` after user clicks `Regenerate`:

1. Build an `assumptions` block from user-edited items.
2. Merge it into a revised context bundle (do not overwrite raw source data).
3. Call writer stage with extra instruction:
   - explicitly list missing data and applied assumptions
   - incorporate assumptions into calculations/recommendations
4. Save revised markdown as a separate artifact (`final_with_assumptions.md`), keeping original `final.md` unchanged.

## 7. Testing Plan

Backend tests (extend `tests/test_api_runs.py` or add `tests/test_api_assumptions.py`):

- discovery returns 404 when run missing
- discovery returns 409 when run not terminal/success
- discovery executes both passes and returns grouped items with valid schema
- discovery verification pass can add at least one item when pass-1 misses data
- apply endpoint accepts edited numbers and writes revised artifacts
- apply endpoint fails gracefully on invalid payload

Frontend tests (at minimum component behavior checks):

- grouped rendering by city
- number editing and validation
- successful regenerate flow updates preview
- error states for discovery/apply failures

## 8. Rollout Sequence

1. Backend models + service + endpoints + tests.
2. Frontend page + API client integration.
3. Config wiring for assumptions reviewer model.
4. Documentation update (README API endpoints + flow).
5. End-to-end smoke test with one real run and assumptions regeneration.

## 9. Definition of Done

- User can trigger missing-data discovery for a completed run.
- Missing items are shown grouped by city with editable proposed numbers.
- User can regenerate a revised document with those assumptions.
- Original output remains preserved.
- All new API paths are covered by tests and documented.
