# Query Input Rework

## Goal

Document the current query flow, capture what the Jira ticket plus Slack discussion are actually asking for, and outline how to implement the change without changing the retriever's internal behavior.

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
  - This is related behavior, but the ticket and Slack discussion are focused on the main run flow.

## What the ticket and Slack conversation are saying

- We are not changing the retriever logic.
  - The backend should still receive up to three queries and run retrieval exactly as it does now.

- We are changing where those queries come from.
  - Today:
    - query 1 is LLM-adjusted
    - query 2 is LLM-generated keyword-style retrieval input
    - query 3 is LLM-generated evidence-style retrieval input
  - Target:
    - query 1 is the user's main query, used directly
    - query 2 is an optional user-provided keyword query
    - query 3 is an optional user-provided evidence or metrics query

- The automatic LLM rewrite of the main user input should be removed from the main run path.
  - That includes typo-fixing and city-name cleanup for the primary pipeline question.

- The "keywords" inputs are not a separate retrieval algorithm.
  - They map to the existing second and third retrieval queries.
  - The retriever still does one vector search per query.

- The frontend should expose two additional input boxes.
  - The user provides the main question plus optional query 2 and query 3.
  - Because this is harder for non-technical users, the UI should add guidance text.

- An optional `auto` mode was discussed.
  - Manual/direct-input mode should be the default.
  - An optional `auto` or dev-only mode could preserve the current LLM-generated behavior.
  - That is an extension, not the core of the ticket.

## Recommended implementation

### 1. Backend core

- Change `run_pipeline(...)` so the main run flow can accept three explicit query inputs.
  - Required: main user question
  - Optional: query 2
  - Optional: query 3

- Replace the current unconditional call to `refine_research_question(...)` in the main run path.
  - Manual mode:
    - use the main question as the canonical research question
    - build `retrieval_queries` from the direct user inputs
  - Optional auto mode:
    - keep today's refiner, but call it only when the mode explicitly requests it

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
- If auto mode is included now, add an explicit mode field rather than inferring behavior.
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
  - optional `--query-mode manual|auto` if the toggle is included now

- Update the top-level docstring and `argparse` help so the behavior is self-explanatory.

- Main file:
  - `backend/scripts/run_pipeline.py`

### 4. Frontend

- Add two optional inputs near the main question.
- Keep the main question required.
- Add help text that explains the role of each field in plain language.
  - Main question: what answer the user wants
  - Query 2: topic and initiative keywords
  - Query 3: numbers, budgets, targets, timelines, evidence terms

- If auto mode is included, manual should stay the default.
- The existing frontend dev toggle is separate today, but it is a likely place to expose advanced behavior if product wants that.

- Main files:
  - `frontend/src/app/page.tsx`
  - `frontend/src/lib/api.ts`
  - possible toggle-related touchpoints: `frontend/src/components/dev-mode-toggle.tsx`, `frontend/src/lib/frontend-mode.ts`

### 5. Logging and artifacts

- Current persisted naming still assumes a "refined question".
- Once manual mode becomes default, those labels become misleading.
- Update artifacts and logs so they clearly distinguish:
  - original question
  - retrieval query 1
  - retrieval query 2
  - retrieval query 3
  - query source or mode, if we keep manual and auto side by side

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

## Recommended scope boundary

- In-scope for this rework:
  - main run pipeline
  - run API contract
  - run CLI contract
  - frontend run form

- Explicitly confirm whether chat follow-up search should stay on automatic refinement for now.
  - It is currently a separate flow and was not directly covered in the Slack implementation notes.

- If we want the lowest-risk rollout:
  - first ship manual/direct-input mode
  - then add optional auto mode as a follow-up if it is still wanted

That sequencing matches the stated goal: keep retrieval behavior the same, but stop making the main run outcome depend on hidden LLM-generated query variants by default.

###TODO app should be adding additional parameters in dev mode like showing us that option. For standard mode we keep current route of pregenerating queries
