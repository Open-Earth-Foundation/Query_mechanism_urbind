# Query Mechanism Urbind

Multi-agent document builder that answers user questions by combining SQL data and Markdown sources. It orchestrates a SQL Researcher, Markdown Researcher, and Writer with OpenAI Agents, and logs every run artifact for inspection.

## Requirements
- Python 3.10+ (see `pyproject.toml`)
- Local SQLite source DB derived from `app/db_models/`
- `OPENROUTER_API_KEY` in environment

## Install

This project uses `uv` for dependency management. If you don't have it installed, [install uv](https://docs.astral.sh/uv/getting-started/installation/).

### Development setup

```bash
uv sync
```

This installs both production and dev dependencies (including pytest).

### Add dependencies

Production:
```bash
uv add package-name
```

Development (e.g., testing tools):
```bash
uv add --dev package-name
```

### Sync environment

After pulling changes with updated dependencies:
```bash
uv sync
```

## Configuration

- `llm_config.yaml` stores model names and settings.
- `.env` should define `OPENROUTER_API_KEY` (do not commit it).
- Optional environment overrides:
- `RUNS_DIR`
- `DATABASE_URL` (used as source DB)
- `MARKDOWN_DIR`
- `LOG_LEVEL`
- `DATABASE_URL` (for `test_db_connection` script)
- `OPENROUTER_BASE_URL` (optional override)

Example `.env.example` is provided.

`.env` is loaded automatically via `python-dotenv` when running scripts.

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
openrouter_base_url: "https://openrouter.ai/api/v1"
```

## Run (local)

```
python -m app.scripts.run_pipeline --question "What initiatives exist for Munich?" \
  --db-path path/to/source.db \
  --markdown-path documents
```

Using Postgres (via DATABASE_URL):
```
python -m app.scripts.run_pipeline --question "..." \
  --db-url "postgresql+psycopg://user:pass@localhost:5432/dbname" \
  --markdown-path documents
```

## End-to-end batch queries

```
python -m app.scripts.run_e2e_queries
python -m app.scripts.run_e2e_queries --questions-file assets/e2e_questions.txt
python -m app.scripts.run_e2e_queries --db-url "postgresql+psycopg://user:pass@localhost:5432/dbname"
```

## Test DB connection

```
python -m app.scripts.test_db_connection
```

Artifacts are written under `output/<run_id>/`:
- `run.json`
- `run.log`
- `context_bundle.json`
- `schema_summary.json`
- `city_list.json`
- `sql/queries.json`, `sql/results.json`, `sql/results_full.json`
- `markdown/excerpts.json`
- `drafts/draft_01.md`
- `final.md`

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
  --question "What initiatives exist for Munich?" \
  --db-path /app/path/to/source.db \
  --markdown-path /app/documents
```

## Tests

```
pytest
```

## Common workflows

- Update model names in `llm_config.yaml`.
- Place markdown sources in `documents/` (e.g., `documents/Munich.md`).
- Inspect per-run artifacts under `output/<run_id>/`.
