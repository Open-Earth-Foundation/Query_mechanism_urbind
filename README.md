# Query Mechanism Urbind

Multi-agent document builder that answers user questions by combining SQL data (optional; disabled by default) and Markdown sources. It orchestrates a SQL Researcher, Markdown Researcher, and Writer with OpenAI Agents, and logs every run artifact for inspection.

## Requirements

- Python 3.11.x
- Local SQLite source DB derived from `app/db_models/` (required only when SQL is enabled)
- `OPENROUTER_API_KEY` in environment

## Install

We use `uv` for dependency management with `pyproject.toml` as the single source of truth. Install dependencies with:

```bash
uv sync
```

To add a new production dependency:

```bash
uv add package-name
```

To add a development dependency (e.g., pytest):

```bash
uv add --dev package-name
```

The `uv.lock` file is committed to ensure reproducible builds.

## Configuration

- `llm_config.yaml` stores model names and settings.
- Copy `.env.example` to `.env` and fill in values for your environment.
- `.env` is loaded automatically via `python-dotenv` in the scripts.
- Do not commit `.env`.

Environment variables (`.env`):

- `OPENROUTER_API_KEY` (required): API key used for all LLM calls via OpenRouter.
- `ENABLE_SQL` (optional, default `false`): enables SQL mode by default for pipeline runs.
- `DATABASE_URL` (optional): Postgres source database URL. When set, it is used instead of SQLite (`SOURCE_DB_PATH`). Also used by `python -m app.scripts.test_db_connection`.
- `SOURCE_DB_PATH` (optional, default `data/source.db`): SQLite source DB path used when `DATABASE_URL` is not set.
- `MARKDOWN_DIR` (optional, default `documents`): default directory scanned for markdown files.
- `RUNS_DIR` (optional, default `output`): base directory for run artifacts.
- `LOG_LEVEL` (optional, default `INFO`): logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `OPENROUTER_BASE_URL` (optional, default `https://openrouter.ai/api/v1`): custom OpenRouter-compatible base URL.

CLI flags override `.env` values for a given run (for example `--db-path`, `--db-url`, `--markdown-path`, `--enable-sql`).

Example `.env.example` is provided.

Default output directory is `output/` (unless overridden by `RUNS_DIR`).
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
enable_sql: false
```

### How token and size limits are applied

Shared input-budget logic (used by orchestrator, SQL researcher, markdown researcher, and writer):

```text
if max_input_tokens is set:
    effective_max_input_tokens = max_input_tokens
elif context_window_tokens is set:
    effective_max_input_tokens = max(context_window_tokens - input_token_reserve - max_output_tokens, 0)
else:
    effective_max_input_tokens = None
```

`max_output_tokens` is treated as `0` when omitted.

What each key controls:

- `context_window_tokens`: Provides the model context-window assumption used for budget calculation.
- `input_token_reserve`: Safety margin kept free for system/tool overhead; subtracted from the available input budget.
- `max_output_tokens`: Output cap (when set) and also subtracted from input budget.
- `max_input_tokens`: Hard override for input budget; if set, it takes precedence over the formula above.
- `sql_researcher.max_result_tokens`: Hard cap for SQL rows included in the capped SQL bundle passed downstream.
- `markdown_researcher.max_chunk_tokens`: Hard cap for each markdown chunk size.
- `markdown_researcher.chunk_overlap_tokens`: Token overlap between neighboring chunks.

How this influences runtime behavior:

- SQL results are token-capped in `cap_results()`. Once the cap is hit, remaining rows are excluded from the capped payload.
- SQL truncation is recorded as `truncation_applied` in the SQL bundle and `truncated` in SQL result items.
- Full SQL output is still written to `output/<run_id>/sql/results_full.json`; capped SQL is written to `output/<run_id>/sql/results.json`.
- Markdown content is chunked first, then batched by token budget. Chunks larger than the current batch budget are skipped.
- Oversized markdown files are skipped when they exceed `max_file_bytes`.

Visibility and warnings:

- Markdown budget/file skips emit warnings in logs.
- SQL token-cap truncation currently does not emit a dedicated warning line; detect it via `truncation_applied`/`truncated` and by comparing `sql/results.json` vs `sql/results_full.json`.

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
