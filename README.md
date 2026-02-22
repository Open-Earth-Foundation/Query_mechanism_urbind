# Query Mechanism Urbind

Multi-agent document builder that answers user questions by combining SQL data (optional; disabled by default) and Markdown sources. It orchestrates a SQL Researcher, Markdown Researcher, and Writer with OpenAI Agents, and logs every run artifact for inspection.

## Requirements

- Python 3.11+
- Node.js 20+ (frontend)
- Local SQLite source DB derived from `backend/db_models/` (required only when SQL is enabled)
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
- `DATABASE_URL` (optional): Postgres source database URL. When set, it is used instead of SQLite (`SOURCE_DB_PATH`). Also used by `python -m backend.scripts.test_db_connection`.
- `SOURCE_DB_PATH` (optional, default `data/source.db`): SQLite source DB path used when `DATABASE_URL` is not set.
- `MARKDOWN_DIR` (optional, default `documents`): default directory scanned for markdown files.
- `RUNS_DIR` (optional, default `output`): base directory for run artifacts.
- `LOG_LEVEL` (optional, default `INFO`): logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `OPENROUTER_BASE_URL` (optional, default `https://openrouter.ai/api/v1`): custom OpenRouter-compatible base URL.
- `API_RUN_WORKERS` (optional, default `2`): FastAPI async background worker count.
- `CHAT_PROMPT_TOKEN_CAP` (optional, default `250000`): token cap for context chat prompt assembly.
- `CHAT_PROVIDER_TIMEOUT_SECONDS` (optional, default `50`): provider timeout for context chat.
- `API_CORS_ORIGINS` (optional): comma-separated frontend origins for API CORS.
- `LLM_CONFIG_PATH` (optional, default `llm_config.yaml`): API config file path.
- `CITY_GROUPS_PATH` (optional, default `backend/api/assets/city_groups.json`): city groups catalog JSON path.
- `VECTOR_STORE_ENABLED` (optional, default `false`): enables local Chroma markdown indexing flows.
- `CHROMA_PERSIST_PATH` (optional, default `.chroma`): local Chroma persistence directory.
- `CHROMA_COLLECTION_NAME` (optional, default `markdown_chunks`): Chroma collection used for markdown chunks.
- `EMBEDDING_MODEL` (optional, default `text-embedding-3-large`): embedding model for vector index build/update.
- `EMBEDDING_CHUNK_TOKENS` (optional, default `800`): chunk token budget for markdown packing.
- `EMBEDDING_CHUNK_OVERLAP_TOKENS` (optional, default `80`): chunk overlap token budget.
- `TABLE_ROW_GROUP_MAX_ROWS` (optional, default `25`): max rows per split group for oversized markdown tables.
- `MARKDOWN_BATCH_MAX_CHUNKS` (optional, default `4`): hard cap on number of markdown chunks sent in one markdown researcher request.
- `MARKDOWN_BATCH_MAX_INPUT_TOKENS` (optional): explicit token budget per markdown researcher batch. If unset, an adaptive budget is derived from `EMBEDDING_CHUNK_TOKENS`, `MARKDOWN_BATCH_MAX_CHUNKS`, and `MARKDOWN_BATCH_OVERHEAD_TOKENS`.
- `MARKDOWN_BATCH_OVERHEAD_TOKENS` (optional, default `600`): reserved prompt/payload overhead used by adaptive markdown batching.
- `VECTOR_STORE_RETRIEVAL_MAX_DISTANCE` (optional, default `1.0`): Chroma distance cutoff; only chunks with `distance <= cutoff` are kept.
- `VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY_QUERY` (optional, default `60`): max candidates fetched per city/query before distance filtering/top-up.
- `VECTOR_STORE_RETRIEVAL_FALLBACK_MIN_CHUNKS_PER_CITY_QUERY` (optional, default `20`): minimum returned per city/query (top-up target when too few pass `VECTOR_STORE_RETRIEVAL_MAX_DISTANCE`).
- `VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY` (optional, default `300`): final per-city cap after query merge and neighbor expansion.
- `VECTOR_STORE_CONTEXT_WINDOW_CHUNKS` (optional, default `0`): number of neighboring chunks to include around each retrieved chunk.
- `VECTOR_STORE_TABLE_CONTEXT_WINDOW_CHUNKS` (optional, default `1`): neighbor chunk window for table chunks.
- `VECTOR_STORE_AUTO_UPDATE_ON_RUN` (optional, default `false`): if `true`, run an incremental index update before retrieval.
- `INDEX_MANIFEST_PATH` (optional, default `.chroma/index_manifest.json`): JSON manifest path for incremental updates.

CLI flags override `.env` values for a given run (for example `--db-path`, `--db-url`, `--markdown-path`, `--enable-sql`).
Use `--city` (repeatable) to load markdown only for selected city files. City filters are normalized case-insensitively to backend `city_key` values (for example `Munich`, `MUNICH`, and `munich` all resolve to `munich`).

Example `.env.example` is provided. Vector-store retriever defaults in `.env.example` match `backend/benchmarks/config` (base.env + mode_vector.env); use those files as the reference when tuning.

Default output directory is `output/` (unless overridden by `RUNS_DIR`).
Schema summary for SQL generation is derived from `backend/db_models/`.

### Vector retrieval sizing and thresholds

When vector retrieval is enabled, retrieval runs per city and per query (original + refined variants), then merges and expands context.

- For each (city × query), the retriever:
  - fetches up to `VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY_QUERY` candidates from Chroma (ranked by increasing distance);
  - if `VECTOR_STORE_RETRIEVAL_MAX_DISTANCE` is set, it first keeps only candidates with `distance <= cutoff`;
  - if fewer than `VECTOR_STORE_RETRIEVAL_FALLBACK_MIN_CHUNKS_PER_CITY_QUERY` pass the cutoff, it **tops up** with the next-best candidates (above the cutoff) until it reaches the fallback minimum (or runs out of candidates).
- After per-(city × query) retrieval:
  - results are merged across queries within a city (dedupe by `chunk_id`, keep the smallest distance);
  - neighbor chunks are added by `chunk_index` window (same file/city);
  - optionally, `VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY` caps the final chunks per city after merge + neighbor expansion.
- `VECTOR_STORE_RETRIEVAL_MAX_DISTANCE` is the strictness control:
  - smaller value = stricter matching, fewer chunks;
  - larger value = higher recall, more chunks.

Important distinction between the “max” knobs:

- `VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY_QUERY` controls the **candidate pool size per (city × query)** _before_ distance filtering/top-up.
  - If this is too small, you may not have enough candidates to top up to the fallback minimum.
- `VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY` controls the **final per-city cap** _after_ query-merge and neighbor expansion.
  - Use it as a latency/cost guardrail; setting it too low can drop context neighbors or even primary hits with weaker distances.

Distance scale note:

- Do not assume distance is always in `[0, 1]`. It depends on collection metric and embedding characteristics.
- `0` means identical vectors; values above `0` are increasingly dissimilar.
- A cutoff of `0` is the strictest setting and usually returns very few (often zero) chunks, not all chunks.

How to estimate chunk counts:

- **Minimum (practical):**
  - per (city × query): approximately `min(VECTOR_STORE_RETRIEVAL_FALLBACK_MIN_CHUNKS_PER_CITY_QUERY, available_chunks_total)`, before merge/neighbor expansion.
- **Maximum (practical):**
  - per (city × query): `VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY_QUERY`, before merge/neighbor expansion;
  - bounded by `VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY` if set;
  - otherwise depends on distance cutoff + number of merged queries + neighbor expansion windows.

Recommended tuning workflow:

1. Start recall-friendly:
   - leave `VECTOR_STORE_RETRIEVAL_MAX_DISTANCE` empty, or set a permissive value;
   - set `VECTOR_STORE_RETRIEVAL_FALLBACK_MIN_CHUNKS_PER_CITY_QUERY` to a meaningful fallback (for example 20-40).
2. Run and inspect `output/<run_id>/markdown/retrieval.json` for returned distances and counts.
3. Set/tighten `VECTOR_STORE_RETRIEVAL_MAX_DISTANCE` based on observed distance distribution.
4. Add `VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY` only if latency/cost grows too much.

## API key setup (important)

You have two supported options:

1. Backend default key (server-side):

- Put key in root `.env`:
  - `OPENROUTER_API_KEY=...`
- Use this when deployment should use one shared server key.

2. User-provided key (frontend, per browser):

- In the app UI, use `OpenRouter API Key (Optional)` and click `Use This Key`.
- This key is stored in browser `localStorage` and sent as `X-OpenRouter-Api-Key`.
- Use this when users should provide their own key instead of a shared backend key.

If key authentication fails:

- runs finish with `error.code = API_KEY_ERROR`
- chat endpoints return `401` with a key-specific message
- UI surfaces the error so users can switch key and retry.

Example `llm_config.yaml`:

```
orchestrator:
  model: "moonshotai/kimi-k2.5"
  temperature: 0.0
  context_bundle_name: "context_bundle.json"
  context_window_tokens: 256000
  input_token_reserve: 2000
sql_researcher:
  model: "moonshotai/kimi-k2.5"
  temperature: 0.0
  max_result_tokens: 100000
  context_window_tokens: 256000
  input_token_reserve: 2000
markdown_researcher:
  model: "openai/gpt-5-mini"
  temperature: 0.0
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
  temperature: 0.0
  context_window_tokens: 256000
  input_token_reserve: 2000
chat:
  model: "openai/gpt-5.2"
  temperature: 0.0
  context_window_tokens: 400000
  input_token_reserve: 20000
  max_history_messages: 24
assumptions_reviewer:
  model: "openai/gpt-5.2"
  temperature: 0.0
  max_output_tokens: 8000
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
python -m backend.scripts.run_pipeline --question "What initiatives exist for Munich?" \
  --markdown-path documents
```

Limit to selected cities only:

```
python -m backend.scripts.run_pipeline --question "What initiatives exist for Munich and Leipzig?" \
  --markdown-path documents \
  --city Munich \
  --city Leipzig
```

Disable LLM payload logging:

```
python -m backend.scripts.run_pipeline --question "What initiatives exist for Munich?" \
  --markdown-path documents \
  --no-log-llm-payload
```

Enable SQL (SQLite):

```
python -m backend.scripts.run_pipeline --enable-sql --question "What initiatives exist for Munich?" \
  --db-path path/to/source.db \
  --markdown-path documents
```

## Happy-path workflow (no SQL)

High-level flow from user input to final output text when `ENABLE_SQL=false`:

```mermaid
flowchart TD
    A[User question + CLI/config input] --> B[run_pipeline in orchestrator module]
    B --> C[Refine question into research_question]
    C --> D[Load markdown documents]
    D --> E[Markdown extractor: extract_markdown_excerpts]
    E --> F[Store markdown bundle in context_bundle.json<br/>(excerpts + excerpt_count)]
    F --> G[Writer: write_markdown]
    G --> H[Write final.md and finalize run<br/>(writer includes evidence preface)]
```

What each stage does:

- Orchestrator receives the input question and creates a research-oriented question.
- Extractor input source is configurable:
  - default path: markdown files are token-chunked directly from disk;
  - vector path (`VECTOR_STORE_ENABLED=true`): per-city, distance-thresholded chunks are retrieved from Chroma using explicit query embeddings.
- Markdown researcher returns evidence excerpts selected from whichever chunk source was used.
- `markdown_chunk_count` tracks how many chunk inputs were processed; `excerpt_count` (also logged as `markdown_excerpt_count` in run metadata) tracks how many evidence snippets were extracted from those chunks.
- Context bundle is updated with extracted evidence for downstream writing.
- Orchestrator hands the prepared context bundle directly to the writer.
- Writer uses the context bundle and writes final output text to `output/<run_id>/final.md`. The response starts with an evidence preface (based on `excerpt_count`); when `excerpt_count=0`, it returns a "no evidence found" response.

## End-to-end batch queries

When `--question` is provided, it overrides `--questions-file` and only the CLI question(s) are executed.

```
python -m backend.scripts.run_e2e_queries
python -m backend.scripts.run_e2e_queries --questions-file assets/e2e_questions.txt
python -m backend.scripts.run_e2e_queries --question "What initiatives exist for Munich?" --no-log-llm-payload
python -m backend.scripts.run_e2e_queries --question "What initiatives exist for Munich and Leipzig?" --markdown-path documents --city Munich --city Leipzig
```

## Retrieval strategy benchmark

Use this benchmark to compare standard markdown chunking (`VECTOR_STORE_ENABLED=false`) against vector-store retrieval (`VECTOR_STORE_ENABLED=true`) without changing normal runtime behavior.

Configuration and prompts are intentionally separated under `backend/benchmarks/`. The benchmark config is the **reference for vector-store settings** (paths, retriever knobs); `.env.example` and local dev should align with it.

- `backend/benchmarks/prompts/retrieval_questions.txt`: benchmark question set.
- `backend/benchmarks/config/base.env`: shared benchmark env (Chroma paths, auto-update off).
- `backend/benchmarks/config/mode_standard.env`: standard-mode overrides.
- `backend/benchmarks/config/mode_vector.env`: vector-mode overrides (retrieval knobs).

Command example:

```
python -m backend.scripts.run_retrieval_benchmark --city Munich --city Leipzig --city Mannheim
```

Useful flags:

- `--questions-file backend/benchmarks/prompts/retrieval_questions.txt`
- `--repetitions 2`
- `--mode vector_store` — run only vector retrieval (no standard chunking). The benchmark runs every question in the questions file; `--repetitions N` runs each question N times per mode (total runs = questions × repetitions × modes).

**Vector-only reproducibility (same query and same revised retrieval queries):** To run the vector strategy multiple times with the exact same question and the exact same refined retrieval queries (e.g. to check outcome stability):

1. Run the pipeline once to get a run with the desired question and cities, e.g. `python -m backend.scripts.run_pipeline --question "What does Aachen do for PV rooftop?" --city Aachen --markdown-path documents`. Note the run id and open `output/<run_id>/research_question.json`.
2. Create a one-line questions file (e.g. `my_questions.txt`) containing exactly the `original_question` from that run.
3. Create a query-overrides JSON (e.g. `my_overrides.json`) with one key: the same `original_question` string; value: `{"canonical_research_query": "<from research_question.json>", "retrieval_queries": [<from research_question.json>]}`. You can copy these fields from `research_question.json`.
4. Run the benchmark in vector-only mode with fixed queries and several repetitions:

   ```
   python -m backend.scripts.run_retrieval_benchmark --questions-file my_questions.txt --query-overrides my_overrides.json --mode vector_store --repetitions 5 --city Aachen
   ```

   Each run will use the same refined question and retrieval queries; only retrieval, extraction, and writing are re-executed. Compare `output/benchmarks/<benchmark_id>/runs/vector_store/*/final.md` (and optionally `retrieval.json`, `excerpts.json`) across repetitions.

Benchmark behavior notes:

- The benchmark runs all questions from the questions file (not a single query repeated N times). Run IDs use `rNN` = repetition and `qNN` = question index (e.g. `vector_store_r01_q02` = repetition 1, second question). For identical queries across all runs, use a one-line questions file.
- The script always loads benchmark env files from `backend/benchmarks/config/`.
- The benchmark is runtime-only; it does not build/update the vector index.
- Vector mode uses the existing default Chroma store/collection unless overridden in your main environment.
- The benchmark also runs LLM-as-judge scoring (`openai/gpt-5.2`) per matched standard-vs-vector run pair.

Outputs are written to `output/benchmarks/<benchmark_id>/`:

- `benchmark_report.json`: machine-readable benchmark results.
- `benchmark_report.md`: human-readable summary with runtime/tokens and judge score summaries.
- `runs/<mode>/<run_id>/...`: original pipeline artifacts for each benchmark run.

Standalone judge command for any two outputs:

```
python -m backend.scripts.judge_final_outputs \
  --left-final output/<run_a>/final.md \
  --right-final output/<run_b>/final.md \
  --question "Compare charging and retrofit initiatives..."
```

## Run API (local)

Start FastAPI backend:

```
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

SQL is force-disabled in the API execution path for now.

Core endpoints:

- `GET /` (root health endpoint)
- `POST /api/v1/runs`
- `GET /api/v1/runs` (list discovered runs as `run_id` + `question`)
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
- `POST /api/v1/runs/{run_id}/assumptions/apply` (apply edited assumptions and regenerate document; ephemeral by default)
- `GET /api/v1/runs/{run_id}/assumptions/latest` (load latest assumptions artifacts for a run; only when persisted)

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
- Chat first tries full-context prompting; when token budget is exceeded, it switches to excerpt pooling across all selected runs.
- Prompt budget defaults to `250000` tokens (`CHAT_PROMPT_TOKEN_CAP`) and switches to pooled excerpts if full context exceeds this budget.

Run API in Docker:

```
docker build -f backend/Dockerfile -t query-mechanism-backend .
docker run -it --rm -p 8000:8000 \
  --env-file .env \
  -e RUNS_DIR=/data/output \
  -e MARKDOWN_DIR=/data/documents \
  -e LLM_CONFIG_PATH=/data/config/llm_config.yaml \
  -e CITY_GROUPS_PATH=/data/config/city_groups.json \
  -v ${PWD}/documents:/data/documents \
  -v ${PWD}/output:/data/output \
  -v ${PWD}/llm_config.yaml:/data/config/llm_config.yaml:ro \
  -v ${PWD}/backend/api/assets/city_groups.json:/data/config/city_groups.json:ro \
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
- Missing items are grouped by city with editable `proposed_number` (number or free-form text).
- `Regenerate` returns revised content without persisting assumptions by default.
  The `Load Existing Run` picker reads `run_id` + `question` from `GET /api/v1/runs`, then loads selected run artifacts through the standard run endpoints.

Example file is available at `frontend/.env.example`.

Run frontend in Docker:

```
docker build -f frontend/Dockerfile \
  --build-arg NEXT_PUBLIC_API_BASE_URL=https://query-mechanism-api.openearth.dev \
  -t query-mechanism-frontend ./frontend
docker run -it --rm -p 3000:3000 \
  query-mechanism-frontend
```

## Docker Compose (backend + frontend)

Use the included `docker-compose.yml` to run both services together with persisted data directories:

- Host `./documents` -> container `/data/documents` (markdown sources)
- Host `./output` -> container `/data/output` (run artifacts, final docs, context bundles, chat memory)
- Host `./llm_config.yaml` -> container `/data/config/llm_config.yaml`
- Host `./backend/api/assets/city_groups.json` -> container `/data/config/city_groups.json`

Commands:

```
docker compose up --build
docker compose down
```

After startup:

- Frontend: `http://localhost:3000`
- Backend API docs: `http://localhost:8000/docs`

## Manual EKS deployment

For manual GHCR + EKS deployment without GitHub Actions, use `urbind-query-mechanism.md`.
It includes exact build/push commands and `kubectl` apply steps for the manifests in `k8s/`.

## GitHub Actions deployment

Automated development workflow is available at `.github/workflows/develop.yml`.
It runs tests for PRs targeting `main` and for pushes to `main`; image build and EKS deploy run only on `main` branch runs (push/manual dispatch).

Required repository secrets:

- `AWS_ACCESS_KEY_ID_EKS_DEV_USER`
- `AWS_SECRET_ACCESS_KEY_EKS_DEV_USER`
- `EKS_DEV_NAME`
- `OPENROUTER_API_KEY`

Optional repository variables:

- `EKS_DEV_REGION` (default `us-east-1`)
- `FRONTEND_API_BASE_URL` (default `https://urbind-query-mechanism-api.openearth.dev`)

## Test DB connection

```
python -m backend.scripts.test_db_connection
```

Artifacts are written under `output/<run_id>/`:

- `run.json`: machine-readable run metadata (status, timestamps, artifacts, decisions).
- `run.log`: detailed runtime logs, including per-agent `LLM_USAGE` lines.
- `run_summary.txt`: human-readable consolidated report. Header includes `Started`, `Completed`, and explicit `Total runtime` in seconds, plus `LLM Usage` totals/per-agent. It also captures an input snapshot (`initial question`, `refined question`, `selected cities` planned/found, markdown dir/file/chunk/excerpt counts) and a `MARKDOWN_FAILURE_SUMMARY` aggregated from batch failures.
- `context_bundle.json`: payload passed between agents (`sql`, `markdown`, `research_question`, final path).
- `research_question.json`: orchestrator-refined research payload. Includes:
  - `original_question`: raw user question.
  - `canonical_research_query`: canonical refined question.
  - `retrieval_queries`: retrieval-ready query list where index `0` is always `canonical_research_query`.
- `schema_summary.json` (when SQL is enabled): schema digest derived from `backend/db_models/`.
- `city_list.json` (when SQL is enabled): city names fetched from the source DB.
- `sql/queries.json` (when SQL is enabled): SQL plan generated by the SQL researcher.
- `sql/results_full.json` (when SQL is enabled): uncapped SQL execution results.
- `sql/results.json` (when SQL is enabled): token-capped SQL results sent downstream.
- `markdown/excerpts.json`: markdown researcher evidence bundle. Includes `excerpts` (items with `quote`, `city_name`, `partial_answer`), `inspected_cities` (normalized backend city keys present in inspected markdown inputs), and `excerpt_count` (count of extracted excerpts).
- `markdown/retrieval.json` (when `VECTOR_STORE_ENABLED=true`): vector retrieval inputs and results summary. Includes the final retrieval query list, optional city filter, retrieval tuning metadata (cutoffs/caps), and per-chunk summaries (`chunk_id`, `city_name`, `city_key`, `source_path`, `heading_path`, `block_type`, `distance`).
- `markdown/batches.json`: markdown batching plan used for the markdown researcher calls. Includes per-city batch indices, estimated tokens, and chunk ordering fields (`path`, `chunk_index`, `chunk_id`), making it easy to inspect how chunks were grouped into LLM requests.
- `final.md`: final delivered markdown output. Content format is:
  1. `# Question` heading with the original user question,
  2. generated markdown answer body from the writer,
  3. footer line `Finish reason: ...`.

`markdown/excerpts.json` excerpt entries include:

- `quote`: verbatim extracted supporting text from markdown.
- `city_name`: city identifier for the excerpt.
- `partial_answer`: concise fact grounded in the quote.
- `inspected_cities` (bundle-level): normalized backend city keys inspected by markdown extraction.
- `excerpt_count` (bundle-level): number of extracted excerpts included in the bundle.

- `chat/<conversation_id>.json` (created when context chat sessions are used)
- `assumptions/discovered.json` (two-pass extraction output; only when `persist_artifacts=true`)
- `assumptions/edited.json` (user-edited assumptions payload; only when `persist_artifacts=true`)
- `assumptions/revised_context_bundle.json` (context + assumptions merge; only when `persist_artifacts=true`)
- `assumptions/final_with_assumptions.md` (regenerated document; only when `persist_artifacts=true`)

Count semantics:

- `markdown_chunk_count` (run input snapshot): number of markdown chunks sent to the markdown researcher.
- `excerpt_count` (markdown bundle): number of extracted evidence snippets returned by the markdown researcher.
- `markdown_excerpt_count` (run input snapshot): mirrors `excerpt_count` so summary and run metadata can show chunk count and excerpt count side by side.

## Docker images (manual)

This repository ships two service images (no single root Dockerfile image):

- Backend image: `backend/Dockerfile`
- Frontend image: `frontend/Dockerfile`

Build commands:

```
docker build -f backend/Dockerfile -t query-mechanism-backend .
docker build -f frontend/Dockerfile -t query-mechanism-frontend ./frontend
```

For local multi-service runs, prefer Docker Compose:

```
docker compose up --build
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
python -m backend.scripts.test_async_backend_flow --question "What are main climate initiatives?"
python -m backend.scripts.test_async_backend_flow \
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
python -m backend.scripts.analyze_run_tokens --run-log output/<run_id>/run.log
python -m backend.scripts.calculate_tokens --documents-dir documents --recursive
python -m backend.scripts.temp_analyze --run-log output/<run_id>/run.log
```

## Vector store indexing utilities

Build markdown index from scratch:

```
python -m backend.scripts.build_markdown_index --docs-dir documents
```

Analyze retrieval distance distributions (to help choose `VECTOR_STORE_RETRIEVAL_MAX_DISTANCE`):

```
python -m backend.scripts.analyze_retrieval_distances --runs-dir output
python -m backend.scripts.analyze_retrieval_distances --city Munich --city Leipzig --thresholds "0.5,1.0,2.0" --show-per-run
```

How to use the output:

- Start with `VECTOR_STORE_RETRIEVAL_MAX_DISTANCE` empty (no distance filtering) and run a few representative queries.
- Run the analysis script and look at the overall/per-city percentiles.
- Pick a cutoff that keeps the bulk of “good” chunks (often somewhere around the p90–p99 region for your corpus), then iterate.

Dry-run build that also writes chunks to JSON for inspection (no embeddings, no Chroma writes):

```
python -m backend.scripts.build_markdown_index --docs-dir documents --dry-run --write-chunks-json output/vector_index_dryrun/chunks.json
```

Incrementally update existing index:

```
python -m backend.scripts.update_markdown_index --docs-dir documents
```

Inspect indexed chunks:

```
python -m backend.scripts.inspect_markdown_index --city Munich --limit 20
python -m backend.scripts.inspect_markdown_index --where block_type=table --limit 20
python -m backend.scripts.inspect_markdown_index --show-id <chunk_id>
```

Run chunking benchmark (manual/long-running, not part of default test loop):

```
python -m backend.scripts.benchmark_chunking_strategy --docs-dir documents --sample-size 25 --seed 42
```

Benchmark outputs are written under `output/chunk_benchmarks/<timestamp>/`:

- `benchmark.json`: full machine-readable metrics, counts, per-file stats, sampled docs.
- `report.md`: human-readable summary with final score, metric breakdown, and sampled-document list.

Metrics reported:

- `final_accuracy_score`: **overall scalar score** in \[0, 1\], combining the individual metrics below using fixed weights.
- `caption_linkage_rate`: **caption attachment quality** – fraction of source tables with `Table ...` captions whose caption text is attached as `table_title` on at least one table chunk.
- `table_header_valid_rate`: **table structure quality** – fraction of table chunks whose `raw_text` parses as a valid markdown header row followed by a separator row.
- `table_detection_rate`: **table recall proxy** – detected table chunks divided by the number of source tables, capped at 1.0 (can exceed 1.0 before capping when large tables are split into multiple chunks).
- `heading_alignment_rate`: **section alignment quality** – fraction of chunks where `heading_path` matches the heading stack implied by the source at `start_line`.
- `token_budget_compliance_rate`: **chunk-size budget compliance** – fraction of chunks whose `token_count` is within the configured chunk token budget.

## Common workflows

- Update model names in `llm_config.yaml`.
- Place markdown sources in `documents/` (e.g., `documents/Munich.md`).
- Inspect per-run artifacts under `output/<run_id>/`.
