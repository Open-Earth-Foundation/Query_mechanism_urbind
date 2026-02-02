## Document Builder: Implementation Plan

### Goals
- Build a multi-agent document builder that answers user questions by combining SQL data and Markdown sources.
- Use OpenAI Agents framework for orchestration with function-call based decisions and structured outputs.
- Persist artifacts from every run for evaluation of extraction quality.
- Output final answers as Markdown.
- Keep repo structure and coding conventions aligned with `AGENTS.md`.

### Non-goals (for now)
- Production deployments or hosted DBs (local DB only).
- UI or web API (CLI-only first pass).
- Advanced retrieval (vector DB) unless needed later.
- Multi-tenant auth or user permissioning.

### Assumptions
- Source data is in a local SQLite DB derived from SQLAlchemy models in `app/db_models/`.
- Markdown sources are local files on disk in a dedicated documents folder.
- Agents can read from disk and the local DB.
- Prompts and tool schemas are owned by this repo.

---

## Architecture overview

### Agents and responsibilities
1. Orchestrator
   - Decides whether more research is needed or if the Writer can produce the final answer.
   - Uses function calls for all decisions.
   - Calls both SQL and Markdown researchers by default.
2. SQL Researcher
   - Generates SQL queries based on the question and schema.
   - Executes queries against a local DB.
   - Saves results as JSON and returns structured summaries to the Orchestrator.
3. Markdown Researcher
   - Reads full Markdown files from the documents folder.
   - Extracts relevant facts with citations to file path and section.
   - Returns structured summaries to the Orchestrator.
4. Writer
   - Produces final Markdown output from structured inputs once Orchestrator approves.
   - Writes output to `output/<run_id>/final.md`.

### Interfaces and data contracts
- Every agent interaction is a tool/function call with a Pydantic model.
- No control flow decisions based on free-form text.
- Each result includes `status`, `run_id`, `created_at`, and optional `error`.

### Tool call contracts (draft)
- `decide_next_action(input: OrchestratorInput) -> OrchestratorDecision`
- `run_sql_research(input: SqlResearchInput) -> SqlResearchResult`
- `run_markdown_research(input: MarkdownResearchInput) -> MarkdownResearchResult`
- `compose_answer(input: WriterInput) -> WriterOutput`

### Data flow (high level)
1. Orchestrator receives user question.
2. Orchestrator runs both SQL Researcher and Markdown Researcher.
3. Researchers return structured outputs.
4. Run logger updates `context_bundle.json` with the latest model outputs.
5. Orchestrator decides if enough information exists.
6. Writer composes final Markdown and saves it.
7. Run artifacts (queries, JSON, excerpts, decisions, drafts, final output) are persisted.

### Error handling expectations
- All tools return `status: success|error`.
- On error, return a structured `error` object: `code`, `message`, `details`.
- Orchestrator logs errors, records them in the run log, and fails gracefully.

### Token budget policy (SQL extraction)
- Apply a token cap to the total SQL payload per run (all queries combined).
- Token limit is configured in `llm_config.yaml` as `sql_researcher.max_result_tokens` (default 100000).
- Use `tiktoken` to count tokens for SQL result payloads.
- SQL Researcher records `token_count` per query and `truncation_applied` in its output.
- Persist both full results and capped results for auditing.

### Configuration overview
- `llm_config.yaml` (models and per-agent settings)
  - `orchestrator.model`
  - `sql_researcher.model`
  - `markdown_researcher.model`
  - `writer.model`
  - `orchestrator.context_bundle_name`
  - `sql_researcher.max_result_tokens`
  - optional `temperature`, `max_output_tokens`
- `.env` (only `OPENROUTER_API_KEY`)
- `RUNS_DIR` (defaults to `output/`)
- `SOURCE_DB_PATH`
- `MARKDOWN_DIR` (defaults to `documents/`)
- `LOG_LEVEL`

---

## Database schema reference (from app/db_models)

Schema source: `app/db_models/` in the repo root. These SQLAlchemy models define the tables and foreign keys for the source DB.

Tables and relationships (FK -> target):
- City
  - `City.cityId` primary key
- CityAnnualStats
  - FK `cityId` -> `City.cityId`
  - Unique: `(cityId, year)`
- CityBudget
  - FK `cityId` -> `City.cityId`
- FundingSource
  - no FKs
- BudgetFunding
  - FK `budgetId` -> `CityBudget.budgetId`
  - FK `fundingSourceId` -> `FundingSource.fundingSourceId`
- ClimateCityContract
  - FK `cityId` -> `City.cityId`
  - Unique: `(cityId)`
- Sector
  - `Sector.sectorId` primary key
- EmissionRecord
  - FK `cityId` -> `City.cityId`
  - FK `sectorId` -> `Sector.sectorId`
  - Unique: `(cityId, year, sectorId, scope, ghgType)`
- Indicator
  - FK `cityId` -> `City.cityId`
  - FK `sectorId` -> `Sector.sectorId`
- IndicatorValue
  - FK `indicatorId` -> `Indicator.indicatorId`
  - Unique: `(indicatorId, year)`
- CityTarget
  - FK `cityId` -> `City.cityId`
  - FK `indicatorId` -> `Indicator.indicatorId`
- Initiative
  - FK `cityId` -> `City.cityId`
- Stakeholder
  - no FKs
- InitiativeStakeholder
  - FK `initiativeId` -> `Initiative.initiativeId`
  - FK `stakeholderId` -> `Stakeholder.stakeholderId`
  - Unique: `(initiativeId, stakeholderId)`
- InitiativeIndicator
  - FK `initiativeId` -> `Initiative.initiativeId`
  - FK `indicatorId` -> `Indicator.indicatorId`
  - Unique: `(initiativeId, indicatorId)`
- TefCategory
  - FK `parentId` -> `TefCategory.tefId` (self-referential)
- InitiativeTef
  - FK `initiativeId` -> `Initiative.initiativeId`
  - FK `tefId` -> `TefCategory.tefId`
  - Unique: `(initiativeId, tefId)`

---

## Data persistence plan

### Local databases
- Source DB (local): stores domain tables queried by SQL Researcher.
- Run log is a JSON file (no DB for run logs).

### Output layout
```
output/
  <run_id>/
    run.json
    context_bundle.json
    schema_summary.json
    sql/
      queries.json
      results.json
      results_full.json
    markdown/
      excerpts.json
    drafts/
      draft_01.md
      draft_02.md
    final.md
```

### Run id format
- Timestamp-based, minute-level: `YYYYMMDD_HHMM` (e.g., `20260128_0032`).
- If a folder already exists, append a numeric suffix (e.g., `_01`, `_02`).

### Run log (run.json) contents
- Run metadata: question, run_id, timestamps, status.
- Decisions: action type, reason, token counts, and next step.
- Paths to all saved artifacts, including `context_bundle.json`.
- Partial drafts produced during the run.

### Context bundle (context_bundle.json)
- Aggregates model outputs from SQL and Markdown researchers in one document for Orchestrator reading.
- Updated after each researcher completes to keep a single source of truth.

---

## Ticket plan (detailed, ordered)

### T1: Repository scaffolding and configuration
Goal: Establish app structure and configuration patterns that follow `AGENTS.md`.

Tasks
- Create base package layout under `app/` with `__init__.py` everywhere.
- Add `app/utils/logging_config.py` with standardized logger setup.
- Add `app/utils/config.py` for centralized config loading.
- Add `app/utils/paths.py` for output folder path construction.
- Add `app/models.py` for shared Pydantic models.
- Create `llm_config.yaml` with per-agent model config.
- Add `sql_researcher.max_result_tokens` and `orchestrator.context_bundle_name` to `llm_config.yaml`.
- Add `tiktoken` to `pyproject.toml` dependencies for token counting.
- Create `documents/` (markdown folder) for city files like `Munich.md` and point Markdown Researcher to it.

Files
- `app/__init__.py`
- `app/utils/__init__.py`
- `app/utils/logging_config.py`
- `app/utils/config.py`
- `app/utils/paths.py`
- `app/models.py`
- `llm_config.yaml`
- `documents/` (folder)
- `pyproject.toml`

Acceptance criteria
- Repo structure matches `AGENTS.md` hierarchy.
- Config loads models from `llm_config.yaml` and API key from `.env`.
- Documents folder exists and is used by Markdown Researcher.

---

### T2: Source DB client and schema access
Goal: Provide a clean API to query the source DB and expose schema for SQL Researcher.

Tasks
- Implement a DB client for SQLite using `sqlite3` and `pathlib.Path`.
- Provide `query_source_db` that only allows SELECT statements.
- Add schema introspection helper (tables + columns + FKs) to feed SQL Researcher prompt.
- Add a `schema_summary.json` generator based on `app/db_models/` or DB introspection.

Files
- `app/services/__init__.py`
- `app/services/db_client.py`
- `app/services/schema_registry.py`

Acceptance criteria
- Source DB queries work via a single client API.
- Non-SELECT SQL is blocked by validation.
- SQL Researcher receives schema + FK info.

---

### T3: Run artifact storage and run log (JSON)
Goal: Persist run artifacts and decisions as JSON in the output folder.

Tasks
- Implement a run logger that writes `run.json` at the end of a run.
- Capture paths to `queries.json`, `results.json`, `excerpts.json`, drafts, and final output.
- Record token counts and trimming decisions.
- Store partial drafts under `output/<run_id>/drafts/`.
- Maintain `context_bundle.json` as a consolidated view for Orchestrator reading (name from `orchestrator.context_bundle_name`).

Files
- `app/services/run_logger.py`
- `app/utils/paths.py`

Acceptance criteria
- All artifacts are discoverable via `run.json`.
- Partial drafts are saved in `drafts/`.
- `context_bundle.json` is updated after each research step.

---

### T4: Pydantic models for structured outputs
Goal: Define all structured I/O types as Pydantic models, used by tool calls.

Tasks
- Add models for:
  - SQL query plan and results (query text, rows, columns, row_count, elapsed_ms).
  - Markdown excerpts (file path, heading, snippet, relevance).
  - Orchestrator decision (action type, reason, next agent, confidence).
  - Writer output (output path, summary, draft_paths).
  - Run context (question, run id, timestamps).
- Add a shared `ErrorInfo` model with `code`, `message`, `details`.
- Include common fields: `status`, `run_id`, `created_at`, `error`.

Files
- `app/models.py`
- `app/modules/orchestrator/models.py`
- `app/modules/sql_researcher/models.py`
- `app/modules/markdown_researcher/models.py`
- `app/modules/writer/models.py`

Acceptance criteria
- All agent interactions use Pydantic models.
- No agent uses free-form text for decisions.
- Models enforce required fields and sensible defaults.

---

### T5: OpenAI Agents framework integration
Goal: Establish agent definitions and tool interfaces for function-call outputs.

Tasks
- Add a module for agent initialization and tool registration.
- Define tools for:
  - `decide_next_action`
  - `run_sql_research`
  - `run_markdown_research`
  - `compose_answer`
- Ensure each tool returns a Pydantic model instance or dict from `.model_dump()`.
- Add system prompts per agent under `prompts/`.
- Add a small adapter so each agent can be run in isolation for tests.

Files
- `app/services/agents.py`
- `app/modules/orchestrator/agent.py`
- `app/prompts/orchestrator_system.md`
- `app/prompts/sql_researcher_system.md`
- `app/prompts/markdown_researcher_system.md`
- `app/prompts/writer_system.md`

Acceptance criteria
- Agents are created through a single integration module.
- Orchestrator can call other agents via tool functions.
- Each agent can be invoked via a single function call in tests.

---

### T6: SQL Researcher agent implementation
Goal: Generate SQL, execute it locally, and return structured results.

Tasks
- Accept question + schema summary + run_id.
- Generate SQL queries via tool call output (list of query objects).
- Execute queries with SELECT-only validation.
- Serialize results to JSON and persist under `output/<run_id>/sql/`.
- Use `tiktoken` to cap SQL result payloads at `sql_researcher.max_result_tokens`.
- Add max row limit and record row counts.
- Record query timing for traceability.

Files
- `app/modules/sql_researcher/__init__.py`
- `app/modules/sql_researcher/agent.py`
- `app/modules/sql_researcher/services.py`
- `app/modules/sql_researcher/models.py`

Acceptance criteria
- SQL Researcher returns structured output with query text and result rows.
- Query results saved as JSON for every run.
- Queries are logged and captured for audit.
- Token counts and truncation flags are recorded.

---

### T7: Markdown Researcher agent implementation
Goal: Read Markdown files and return structured excerpts with citations.

Tasks
- Accept question + list of Markdown file paths from `documents/`.
- Read files using `pathlib.Path` (full file content).
- Extract relevant excerpts and cite source file + heading.
- Persist excerpts JSON under `output/<run_id>/markdown/`.
- Add file size guardrails (max MB, max files) and clear errors.
- Normalize headings and section references for consistent citations.

Files
- `app/modules/markdown_researcher/__init__.py`
- `app/modules/markdown_researcher/agent.py`
- `app/modules/markdown_researcher/services.py`
- `app/modules/markdown_researcher/models.py`

Acceptance criteria
- Researcher reads full Markdown files (no partial chunking).
- Returns structured excerpts with citations.
- Excerpts include file path and heading for traceability.

---

### T8: Writer agent implementation
Goal: Produce final Markdown using structured inputs and persist output.

Tasks
- Accept aggregated SQL + Markdown results and run_id.
- Compose final response as Markdown.
- Save to `output/<run_id>/final.md`.
- Save partial drafts to `output/<run_id>/drafts/` during composition.
- Return `WriterOutput` metadata via tool call.

Files
- `app/modules/writer/__init__.py`
- `app/modules/writer/agent.py`
- `app/modules/writer/models.py`

Acceptance criteria
- Final output is saved as Markdown.
- Writer only runs when orchestrator approves.
- Partial drafts are saved and referenced in `run.json`.

---

### T9: Orchestrator workflow and run manager
Goal: Coordinate multi-agent flow and persist all decisions.

Tasks
- Implement orchestration loop:
  - Start run and persist run metadata.
  - Call SQL and Markdown researchers by default.
  - Reassess if enough information exists via a decision tool call.
  - Trigger Writer when ready.
- Read `context_bundle.json` as the single document of model outputs.
- Persist decisions to `output/<run_id>/run.json`.
- Allow configurable max iterations to avoid loops.
- Capture final status and error summary if any.

Files
- `app/modules/orchestrator/__init__.py`
- `app/modules/orchestrator/agent.py`
- `app/modules/orchestrator/module.py`

Acceptance criteria
- Orchestrator uses function calls for decisions.
- Run artifacts exist for each step.
- Context bundle usage and SQL truncation decisions are logged.

---

### T10: CLI runner script
Goal: Provide a runnable entry point that executes a full run.

Tasks
- Add `app/scripts/run_pipeline.py` with required top-level docstring.
- Use `argparse` for question, run id (optional), and config overrides.
- Set up logging and invoke orchestrator module.
- Allow `--markdown-path` and `--db-path` overrides for local testing.

Files
- `app/scripts/__init__.py`
- `app/scripts/run_pipeline.py`

Acceptance criteria
- Script follows standalone rules in `AGENTS.md`.
- Running `python -m app.scripts.run_pipeline --question "..."` produces output artifacts and final output.

---

### T11: Tests
Goal: Add basic coverage for critical logic.

Tasks
- Add unit tests for:
  - DB client queries and run logging.
  - Pydantic model validation.
  - Orchestrator decision flow (mocked agents).
- Use `pytest`.
- Add a small fixture SQLite DB for SQL Researcher tests.

Files
- `tests/test_db_client.py`
- `tests/test_models.py`
- `tests/test_orchestrator.py`
- `tests/fixtures/source.db`

Acceptance criteria
- `pytest` passes locally.
- At least one test covers run artifact persistence.
- Tests are deterministic and do not rely on network access.

---

### T12: README update
Goal: Document install/run/test workflows and Docker usage.

Tasks
- Expand `README.md` with:
  - Project description.
  - Install (pip or uv).
  - Run instructions (CLI and Docker).
  - Required env vars and `.env.example`.
  - Testing workflow.
- Document file structure under `output/` and DB locations.
- Add an end-to-end example run.

Files
- `README.md`
- `.env.example`

Acceptance criteria
- README satisfies `AGENTS.md` documentation requirements.
- README includes an end-to-end example run.

---

## Verification questions
All verification questions resolved. Proceeding with implementation.

---

## Suggested execution order
T1 -> T2 -> T3 -> T4 -> T5 -> T6 -> T7 -> T8 -> T9 -> T10 -> T11 -> T12
