# Query Mechanism Urbind

Multi-agent document builder that answers user questions by combining SQL data (optional; disabled by default) and Markdown sources. It orchestrates a SQL Researcher, Markdown Researcher, and Writer with OpenAI Agents, and logs every run artifact for inspection.

## Requirements

- Python 3.10+
- Node.js 20+ (frontend)
- Local SQLite source DB derived from `app/db_models/` (required only when SQL is enabled)
- `OPENROUTER_API_KEY` in environment

## Install

### pip

```
pip install -e .
```

If you prefer a frozen requirements file:

```
pip install -r requirements.txt
```

### uv

```
uv pip install -e .
```

## Configuration

- `llm_config.yaml` stores model names and settings.
- `.env` can define `OPENROUTER_API_KEY` (do not commit it).
- Optional environment overrides:
- `RUNS_DIR`
- `ENABLE_SQL` (set to true to enable SQL by default)
- `SOURCE_DB_PATH`
- `DATABASE_URL` (used as source DB and by `test_db_connection`)
- `MARKDOWN_DIR`
- `LOG_LEVEL`
- `OPENROUTER_BASE_URL` (optional override)
- `API_RUN_WORKERS` (optional FastAPI background worker count; default: 2)
- `API_CORS_ORIGINS` (optional comma-separated origins for frontend; default includes localhost:3000/3001)
- `LLM_CONFIG_PATH` (optional API config file path; default: `llm_config.yaml`)
- `CITY_GROUPS_PATH` (optional city groups JSON path; default: `app/api/assets/city_groups.json`)

Example `.env.example` is provided.

`.env` is loaded automatically via `python-dotenv` when running scripts.

## API key setup (important)

You have two supported options:

1. Backend default key (server-side):
- Put key in root `.env`:
  - `OPENROUTER_API_KEY=...`
- Use this when the deployment should use one shared server key.

2. User-provided key (frontend, per browser):
- In the app UI, use `OpenRouter API Key (Optional)` input and click `Use This Key`.
- This key is stored in browser `localStorage` and sent in header `X-OpenRouter-Api-Key`.
- Use this when users should pay with their own key instead of a shared backend key.

If key authentication fails:
- runs finish with `error.code = API_KEY_ERROR`
- chat endpoints return `401` with a key-specific message
- UI shows the error so the user can switch key and retry.

Default output directory is `output/` (override with `RUNS_DIR`).
Schema summary for SQL generation is derived from `app/db_models/`.

Example `llm_config.yaml`:

```
orchestrator:
  model: "moonshotai/kimi-k2.5"
  context_bundle_name: "context_bundle.json"
  context_window_tokens: 256000
  input_token_reserve: 2000
sql_researcher:
  model: "moonshotai/kimi-k2.5"
  max_result_tokens: 100000
  context_window_tokens: 256000
  input_token_reserve: 2000
markdown_researcher:
  model: "openai/gpt-5-mini"
  context_window_tokens: 400000
  input_token_reserve: 2000
  max_chunk_tokens: 120000
  chunk_overlap_tokens: 200
  max_workers: 2
  max_retries: 2
  retry_base_seconds: 0.8
  retry_max_seconds: 6.0
writer:
  model: "moonshotai/kimi-k2.5"
  context_window_tokens: 256000
  input_token_reserve: 2000
chat:
  model: "openai/gpt-5.2"
  context_window_tokens: 400000
  input_token_reserve: 20000
  max_history_messages: 24
assumptions_reviewer:
  model: "openai/gpt-5.2"
  temperature: 0.1
  max_output_tokens: 8000
openrouter_base_url: "https://openrouter.ai/api/v1"
enable_sql: false
```

## Run (local)

```
python -m app.scripts.run_pipeline --question "What initiatives exist for Munich?" \
  --markdown-path documents
```

Disable LLM payload logging:

```
python -m app.scripts.run_pipeline --question "What initiatives exist for Munich?" \
  --markdown-path documents \
  --no-log-llm-payload
```

Enable SQL (SQLite):

```
python -m app.scripts.run_pipeline --enable-sql --question "What initiatives exist for Munich?" \
  --db-path path/to/source.db \
  --markdown-path documents
```

## End-to-end batch queries

When `--question` is provided, it overrides `--questions-file` and only the CLI question(s) are executed.

```
python -m app.scripts.run_e2e_queries
python -m app.scripts.run_e2e_queries --questions-file assets/e2e_questions.txt
python -m app.scripts.run_e2e_queries --question "What initiatives exist for Munich?" --no-log-llm-payload
```

## Run API (local)

Start FastAPI backend:

```
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

SQL is force-disabled in the API execution path for now.

Core endpoints:

- `POST /api/v1/runs`
- `GET /api/v1/runs/{run_id}/status`
- `GET /api/v1/runs/{run_id}/output`
- `GET /api/v1/runs/{run_id}/context`
- `GET /api/v1/cities` (city names from markdown filenames in `MARKDOWN_DIR`, without `.md`)
- `GET /api/v1/city-groups` (predefined city groups filtered to currently available markdown cities)
- `GET /api/v1/chat/contexts` (catalog of completed run contexts with token counts)
- `GET /api/v1/runs/{run_id}/chat/sessions`
- `POST /api/v1/runs/{run_id}/chat/sessions`
- `GET /api/v1/runs/{run_id}/chat/sessions/{conversation_id}`
- `GET /api/v1/runs/{run_id}/chat/sessions/{conversation_id}/contexts`
- `PUT /api/v1/runs/{run_id}/chat/sessions/{conversation_id}/contexts`
- `POST /api/v1/runs/{run_id}/chat/sessions/{conversation_id}/messages`
- `POST /api/v1/runs/{run_id}/assumptions/discover` (two-pass missing-data extraction + verification)
- `POST /api/v1/runs/{run_id}/assumptions/apply` (apply edited assumptions and regenerate document)
- `GET /api/v1/runs/{run_id}/assumptions/latest` (load latest assumptions artifacts for a run)

`POST /api/v1/runs` accepts optional city filtering:

```json
{
  "question": "Build a report for selected cities",
  "cities": ["Munich", "Berlin"]
}
```

Optional header for user-owned key (without backend default key):

```
X-OpenRouter-Api-Key: sk-or-v1-...
```

Frontend scope options map directly to this:
- `all`: omit `cities` in payload (backend processes all markdown cities)
- `group`: send cities from a predefined group from `/api/v1/city-groups`
- `manual`: send explicit city list selected one-by-one

Context chat notes:
- Run outputs are persisted under `output/<run_id>/final.md` and `output/<run_id>/context_bundle.json`.
- Chat sessions persist under `output/<run_id>/chat/<conversation_id>.json`.
- Context manager supports selecting multiple completed run contexts.
- Selected context payload is capped at `300000` tokens to keep prompts within practical limits for the 400k model setup.

Run API in Docker:

```
docker build -f Dockerfile.backend -t query-mechanism-backend .
docker run -it --rm -p 8000:8000 \
  --env-file .env \
  -e RUNS_DIR=/data/output \
  -e MARKDOWN_DIR=/data/documents \
  -e LLM_CONFIG_PATH=/data/config/llm_config.yaml \
  -e CITY_GROUPS_PATH=/data/config/city_groups.json \
  -v ${PWD}/documents:/data/documents \
  -v ${PWD}/output:/data/output \
  -v ${PWD}/llm_config.yaml:/data/config/llm_config.yaml:ro \
  -v ${PWD}/app/api/assets/city_groups.json:/data/config/city_groups.json:ro \
  query-mechanism-backend
```

## Run frontend (shadcn/Next.js)

```
cd frontend
npm install
npm run dev
```

Optional frontend env:

```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Frontend supports three city scope modes in the build form: all cities, predefined group, and manual selection.
Clicking `Open Context Chat` switches to a dedicated chat workspace (not a chat modal), and `Manage Contexts` opens a popup for multi-context selection.
Clicking `Assumptions Review` opens a dedicated workspace where:
- `Find Missing Data` runs two LLM passes (extract + verification).
- Missing items are grouped by city with editable `proposed_number`.
- `Regenerate` writes a revised assumptions-based document artifact.

Example file is available at `frontend/.env.example`.

Run frontend in Docker:

```
docker build -f frontend/Dockerfile -t query-mechanism-frontend ./frontend
docker run -it --rm -p 3000:3000 \
  -e NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 \
  query-mechanism-frontend
```

## Docker Compose (backend + frontend)

Use the included `docker-compose.yml` to run both services together with persisted data directories:

- Host `./documents` -> container `/data/documents` (markdown sources)
- Host `./output` -> container `/data/output` (run artifacts, final docs, context bundles, chat memory)
- Host `./llm_config.yaml` -> container `/data/config/llm_config.yaml`
- Host `./app/api/assets/city_groups.json` -> container `/data/config/city_groups.json`

Commands:

```
docker compose up --build
docker compose down
```

After startup:
- Frontend: `http://localhost:3000`
- Backend API docs: `http://localhost:8000/docs`

## Test DB connection

```
python -m app.scripts.test_db_connection
```

Artifacts are written under `output/<run_id>/`:

- `run.json`
- `run.log`
- `context_bundle.json`
- `schema_summary.json` (when SQL is enabled)
- `city_list.json` (when SQL is enabled)
- `sql/queries.json`, `sql/results.json`, `sql/results_full.json` (when SQL is enabled)
- `markdown/excerpts.json`
- `drafts/draft_01.md`
- `final.md`
- `chat/<conversation_id>.json` (created when context chat sessions are used)
- `assumptions/discovered.json` (two-pass extraction output)
- `assumptions/edited.json` (user-edited assumptions payload)
- `assumptions/revised_context_bundle.json` (context + assumptions merge)
- `assumptions/final_with_assumptions.md` (regenerated document)

## Run (Docker)

Build:

```
docker build -t query-mechanism-urbind .
```

Run:

```
docker run -it --rm \
  -v ${PWD}:/app \
  --env-file .env \
  query-mechanism-urbind \
  --enable-sql \
  --question "What initiatives exist for Munich?" \
  --db-path /app/path/to/source.db \
  --markdown-path /app/documents
```

## Tests

```
pytest
```

## Async API smoke test (pre-frontend)

Use this script to validate the async backend lifecycle contract before frontend integration:

- `POST /api/v1/runs`
- `GET /api/v1/runs/{run_id}/status`
- `GET /api/v1/runs/{run_id}/output`
- `GET /api/v1/runs/{run_id}/context`

```
python -m app.scripts.test_async_backend_flow --question "What are main climate initiatives?"
python -m app.scripts.test_async_backend_flow \
  --base-url http://127.0.0.1:8000 \
  --question "What initiatives exist for Munich?" \
  --cities Munich,Berlin \
  --exercise-chat \
  --poll-interval-seconds 3 \
  --max-wait-seconds 1200
```

Smoke-test artifacts are written to `output/api_smoke_tests/<run_id>/`.

## Token analysis utilities

```
python -m app.scripts.analyze_run_tokens --run-log output/<run_id>/run.log
python -m app.scripts.calculate_tokens --documents-dir documents --recursive
python -m app.scripts.temp_analyze --run-log output/<run_id>/run.log
```

## Common workflows

- Update model names in `llm_config.yaml`.
- Place markdown sources in `documents/` (e.g., `documents/Munich.md`).
- Inspect per-run artifacts under `output/<run_id>/`.
