# Query Input Rework

## Current state

- The public run entrypoints currently accept a single user question.
  - API request model: `backend/api/models.py`
  - API route and executor wiring: `backend/api/routes/runs.py`, `backend/api/services/run_executor.py`
  - CLI entrypoint: `backend/scripts/run_pipeline.py`
  - Frontend request type and form submission: `frontend/src/lib/api.ts`, `frontend/src/app/page.tsx`

- The main pipeline does not use the user question as-is for retrieval.
  - `backend/modules/orchestrator/module.py` starts with the raw `question`, then calls `refine_research_question(...)`.
  - That refinement step can:
    - lightly rewrite the main question
    - generate up to two additional retrieval queries
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

## What we want to do

- We are not changing the retriever logic.
  - The backend should still receive up to three queries and run retrieval exactly as it does now.

- We are changing where those queries come from.
  - Standard mode:
    - keep the current behavior where the backend refines the main question and generates the second and third retrieval queries
  - Dev mode:
    - input 1 is the user's main query, used directly
    - input 2 is an optional user-provided second query
    - input 3 is an optional user-provided third query
  - In practice:
    - the user-provided second and third inputs replace today's automatically generated keyword-style and evidence-style retrieval queries

- The automatic LLM rewrite should no longer be the only path through the system.
  - Standard mode should keep the current automatic query-generation path.
  - Dev mode should use the primary user question as-is.
  - In dev mode, that means removing typo-fixing and city-name cleanup from the retrieval-input generation path.

- The "keywords" inputs are not a separate retrieval algorithm.
  - They map to the existing second and third retrieval queries.
  - The retriever still does one vector search per query.

- The frontend should add two more user inputs.
  - Standard mode should keep those extra inputs hidden.
  - Dev mode should show the main query plus optional query 2 and query 3.
  - The UI should include guidance text because this is harder for non-technical users.

- A mode split is the clearest way to support both behaviors without changing retrieval.
  - Standard mode should preserve the current generated-query behavior.
  - Dev mode should switch to the new direct-input behavior.
  - The existing frontend dev toggle should control both the visible inputs and the backend query-generation variant.
  - If both behaviors need to coexist in the same backend contract, use an explicit mode field instead of inferring behavior from missing inputs.

## Recommended implementation

### 1. Backend core

- Change `run_pipeline(...)` so the main run flow can accept three explicit query inputs.
  - Required: main user question
  - Optional: query 2
  - Optional: query 3

- Replace the current unconditional call to `refine_research_question(...)` in the main run path.
  - Standard mode:
    - keep today's refiner behavior
    - continue generating retrieval queries from the main question
  - Dev mode:
    - use the main question as the canonical research question
    - build `retrieval_queries` from the direct user inputs

- Keep the existing cleanup behavior that is still useful without an LLM.
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

- Extend the run creation request model with optional second and third queries.
- Add an explicit mode field so the backend can distinguish the standard and dev paths.
  - `standard`: preserve the current behavior where the backend refines or generates retrieval queries from the main question.
  - `dev`: use the provided query inputs directly.
- Thread those values through the route and background executor into `run_pipeline(...)`.

- Main files:
  - `backend/api/models.py`
  - `backend/api/routes/runs.py`
  - `backend/api/services/run_executor.py`

### 3. CLI entrypoint

- Extend the runnable script so local and benchmark-style testing can pass the extra queries directly.
- Expected shape:
  - `--question`
  - `--query-2`
  - `--query-3`
  ##########
    ###### what would this be used for in the CLI? Its mostly a frontend dev toggle no?
    ##########
  - optional `--query-mode standard|dev` 

- Update the top-level docstring and `argparse` help so the behavior is self-explanatory.

- Main file:
  - `backend/scripts/run_pipeline.py`

### 4. Frontend

- Keep the main question required.

- Add two additional user inputs for the extra retrieval queries.
  - Standard mode should hide query 2 and query 3 and keep the current generated-query behavior.
  - Dev mode should show query 2 and query 3 and use them directly.
  - Add help text that explains the role of each field in plain language.
    ########
    ######## those examples for query 2 and query 3 are for the auto mode only. For the dev mode they code be anything the user envisions to be useful depending on the main query
    ########
    - Main question: what answer the user wants
    - Query 2: topic and initiative keywords
    - Query 3: numbers, budgets, targets, timelines, evidence terms

- Use the existing dev mode to expose advanced controls when needed.
  - The toggle should switch between the standard and dev query flows.
  - It can also expose any developer-facing query debugging controls.

- Make the mode behavior explicit in the request payload.
  - Standard mode should send `standard`.
  - Dev mode should send `dev`.

- Main files:
  - `frontend/src/app/page.tsx`
  - `frontend/src/lib/api.ts`
  - possible toggle-related touchpoints: `frontend/src/components/dev-mode-toggle.tsx`, `frontend/src/lib/frontend-mode.ts`

### 5. Logging and artifacts

- Current persisted naming still assumes a "refined question".
- Those labels become misleading as soon as standard and dev modes coexist.
- Update artifacts and logs so they clearly distinguish:
  - original question
  - retrieval query 1
  - retrieval query 2
  - retrieval query 3
  - query source or mode, if we keep standard and dev side by side

- Main files:
  - `backend/services/run_logger.py`
  - `backend/api/services/run_store.py`
  - `backend/modules/orchestrator/module.py`

### 6. Tests

- Update pipeline tests that currently assume automatic refinement.
- Add API tests for the new request fields.
- Update CLI tests or script coverage if the runnable interface changes.
- Only touch follow-up research tests if that flow is explicitly brought into scope.

- Main files:
  - `tests/test_orchestrator.py`
  - `tests/test_api_runs.py`
  - `tests/test_chat_followup_research.py` only if follow-up behavior changes
 


############### Comments
Additionally we need to make sure that when the user only passes one extra query, that the the second extra query is beeing ignored and the backend is only doing the retrieval based on 
main and second query.
Meaning only the main query is mandatory.
The 2nd and 3rd queries are optional in backend AND frontend and if null (empty) should not lead to any retrieval
