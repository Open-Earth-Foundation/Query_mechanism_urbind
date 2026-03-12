# Query Input Rework

Status after current implementation:

- Implemented:
  - API request contract accepts `query_mode`, `query_2`, and `query_3`
  - backend run execution trims blank optional queries and forwards direct-query mode into `run_pipeline(...)`
  - `run_pipeline(...)` supports both `standard` and `dev` query flows
  - frontend dev mode exposes optional query 2 and query 3 inputs
  - run logs, run summaries, and `research_question.json` now use explicit query metadata instead of relying on "refined question" naming
- Deferred:
  - CLI remains unchanged

## Current state

- The public run flow now has two modes.
  - API request model: `backend/api/models.py`
  - API route and executor wiring: `backend/api/routes/runs.py`, `backend/api/services/run_executor.py`
  - Frontend request type and form submission: `frontend/src/lib/api.ts`, `frontend/src/app/page.tsx`
  - CLI entrypoint remains unchanged: `backend/scripts/run_pipeline.py`

- The main pipeline now supports both automatic and direct retrieval-query sources.
  - In `standard` mode, `backend/modules/orchestrator/module.py` still calls `refine_research_question(...)`.
  - That refinement step can:
    - lightly rewrite the main question
    - generate up to two additional retrieval queries
  - In `dev` mode, the pipeline uses the main question directly and appends only non-empty direct query inputs.
  - The resulting values are persisted as `canonical_research_query` plus `retrieval_queries`.

- The automatic refinement is implemented as an LLM call.
  - Refiner function and agent setup: `backend/modules/orchestrator/agent.py`
  - Output model: `backend/modules/orchestrator/models.py`
  - Prompt contract: `backend/prompts/orchestrator_research_question_system.md`

- Vector retrieval already supports multiple queries.
  - `backend/modules/vector_store/retriever.py` normalizes the query list, embeds all queries, then runs retrieval per city and per query.
  - This means the current behavior is not "one vector search for all keywords together".
  - It is one search for query 1, one for query 2, and one for query 3, then merge and dedupe.

- The reproducibility concern already exists in the codebase.
  - Benchmarks can bypass live LLM refinement and use fixed query overrides instead.
  - Files: `backend/benchmarks/runner.py`, `backend/benchmarks/prompts/retrieval_query_overrides.json`

- Follow-up chat research currently still uses the same automatic refinement approach.
  - File: `backend/api/services/chat_followup_research.py`
  - This is related behavior, but this rework is focused on the main run flow.

## Implemented behavior

- We are not changing the retriever logic.
  - The backend still receives up to three queries and runs retrieval exactly as it did before.

- We changed where those queries come from.
  - Standard mode:
    - keeps the current behavior where the backend refines the main question and generates the second and third retrieval queries
  - Dev mode:
    - input 1 is the user's main query, used directly
    - input 2 is an optional user-provided second query
    - input 3 is an optional user-provided third query
  - In practice:
    - the user-provided second and third inputs replace today's automatically generated retrieval variants

- The automatic LLM rewrite is no longer the only path through the system.
  - Standard mode keeps the automatic query-generation path.
  - Dev mode uses the primary user question as-is.
  - In dev mode, there is no LLM-based cleanup or rewrite before retrieval query construction.

- The "keywords" inputs are not a separate retrieval algorithm.
  - They map to the existing second and third retrieval queries.
  - The retriever still does one vector search per query.

- The frontend now has two more optional user inputs in dev mode.
  - Standard mode keeps those extra inputs hidden.
  - Dev mode shows the main query plus optional query 2 and query 3.
  - The UI guidance text keeps the fields open-ended rather than prescribing a fixed schema.

- A mode split is how both behaviors now coexist without changing retrieval.
  - Standard mode preserves the current generated-query behavior.
  - Dev mode switches to the direct-input behavior.
  - The existing frontend dev toggle controls both the visible inputs and the backend query-generation variant.
  - The backend contract uses an explicit mode field instead of inferring behavior from missing inputs.

## Implementation details

### 1. Backend core

- `run_pipeline(...)` now accepts three explicit query inputs.
  - Required: main user question
  - Optional: query 2
  - Optional: query 3

- The previous unconditional call to `refine_research_question(...)` is now conditional.
  - Standard mode:
    - keeps today's refiner behavior
    - continues generating retrieval queries from the main question
  - Dev mode:
    - uses the main question as the canonical research question
    - builds `retrieval_queries` from the direct user inputs

- The shared cleanup behavior remains in place without an LLM.
  - trim whitespace
  - drop empty values
  - dedupe case-insensitively
  - cap the final query list to the supported maximum

- Main files:
  - `backend/modules/orchestrator/module.py`
  - `backend/modules/orchestrator/agent.py`
  - `backend/modules/orchestrator/models.py`
  - `backend/prompts/orchestrator_research_question_system.md`

### 2. API contract

- The run creation request model now includes optional second and third queries.
- An explicit mode field distinguishes the standard and dev paths.
  - `standard`: preserve the current behavior where the backend refines or generates retrieval queries from the main question.
  - `dev`: use the provided query inputs directly.
- Those values are threaded through the route and background executor into `run_pipeline(...)`.

- Main files:
  - `backend/api/models.py`
  - `backend/api/routes/runs.py`
  - `backend/api/services/run_executor.py`

### 3. CLI entrypoint

- Keep the current CLI unchanged for this scope.
- Explicit decision: do not add `--query-mode`, `--query-2`, or `--query-3` to the CLI in this change.
- Rationale:
  - the `standard` vs `dev` split is primarily a frontend developer workflow
  - the frontend already has a visible mode toggle and different input behavior
  - the CLI does not need to mirror that toggle unless terminal users must explicitly exercise both backend paths
- Revisit CLI support only if we later have a concrete use case such as:
  - local debugging of direct-query retrieval behavior
  - scripted comparison of `standard` vs `dev`
  - benchmark runs that need deterministic direct-query inputs
- If that need appears later, prefer adding explicit CLI support as a separate, intentional follow-up rather than expanding the current scope now.

- Main file:
  - `backend/scripts/run_pipeline.py`

### 4. Frontend

- Keep the main question required.

- Dev mode now adds two additional user inputs for the extra retrieval queries.
  - Standard mode hides query 2 and query 3 and keeps the current generated-query behavior.
  - Dev mode shows query 2 and query 3 and uses only the non-empty values directly.
  - Query 2 and query 3 stay optional in both backend and frontend.
  - Empty or null query 2/query 3 values are ignored and do not trigger retrieval work on their own.
  - Help text explains these fields in plain language without implying a fixed schema.
    - Main question: the required question the user wants answered.
    - Query 2 and query 3: optional retrieval phrasings that can be anything the user thinks will help find better evidence for the main question in dev mode.
    - Examples should be illustrative only, not prescriptive.

- Use the existing dev mode to expose advanced controls when needed.
  - The toggle switches between the standard and dev query flows.
  - It can also expose any developer-facing query debugging controls.

- Make the mode behavior explicit in the request payload.
  - Standard mode sends `standard`.
  - Dev mode sends `dev`.

- Main files:
  - `frontend/src/app/page.tsx`
  - `frontend/src/lib/api.ts`
  - possible toggle-related touchpoints: `frontend/src/components/dev-mode-toggle.tsx`, `frontend/src/lib/frontend-mode.ts`

### 5. Logging and artifacts

- This area is now implemented for the main run flow.
- `research_question.json` records:
  - original question
  - canonical research query
  - query mode
  - retrieval query 1
  - retrieval query 2
  - retrieval query 3
- `run_logger` now records the same query metadata in structured run inputs and in the text run summary.
- `RunStore` still supports older stored runs that only have `initial_question` and `refined_question`, but new runs use the explicit query-field naming.

- Main files:
  - `backend/services/run_logger.py`
  - `backend/api/services/run_store.py`
  - `backend/modules/orchestrator/module.py`

### 6. Tests

- Pipeline and API regression tests now cover the new request fields and optional-query behavior.
- The new cases specifically verify:
  - dev mode bypasses refinement
  - blank optional queries are ignored
  - retrieval queries are trimmed before use
  - retrieval queries are deduped case-insensitively
  - retrieval queries are capped to the supported maximum of three
  - a run can proceed with only the main question plus one extra query
- Only add CLI tests if the runnable interface changes in a follow-up.
- Only touch follow-up research tests if that flow is explicitly brought into scope.

- Main files:
  - `tests/test_orchestrator.py`
  - `tests/test_api_runs.py`
  - `tests/test_chat_followup_research.py` only if follow-up behavior changes
