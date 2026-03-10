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
- Markdown researcher batching knobs are configured in `llm_config.yaml` under `markdown_researcher` (`batch_max_chunks`, `batch_max_input_tokens`, `batch_overhead_tokens`).
- Retry policy is centralized in top-level `retry` in `llm_config.yaml` (`max_attempts`, `backoff_base_seconds`, `backoff_max_seconds`) and is shared across LLM calls, agent max-turn limits, chat tool-call loop limits, and vector retrieval operations.
- Optional `markdown_researcher.reasoning_effort` can be set for Grok reasoning control (for example `"none"`), but this is model/provider-specific and may fail on unsupported models.
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
- `API_CHAT_JOB_WORKERS` (optional, default `1`): dedicated worker count for async split-mode chat jobs.
- `LOG_LEVEL` (optional, default `INFO`): logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `OPENROUTER_BASE_URL` (optional, default `https://openrouter.ai/api/v1`): custom OpenRouter-compatible base URL.
- `CHAT_PROMPT_TOKEN_CAP` (optional, default `250000`): token cap for context chat prompt assembly.
- `CHAT_PROVIDER_TIMEOUT_SECONDS` (optional, default `50`): provider timeout for context chat.
- `LLM_CONFIG_PATH` (optional, default `llm_config.yaml`): API config file path.
- `CITY_GROUPS_PATH` (optional, default `backend/api/assets/city_groups.json`): city groups catalog JSON path.
- `VECTOR_STORE_ENABLED` (optional, default `false`): enables local Chroma markdown indexing flows.
- `ANONYMIZED_TELEMETRY` (optional, default `FALSE`): disables Chroma anonymized telemetry when set to `FALSE`.
- `CHROMA_PERSIST_PATH` (optional, default `.chroma`): local Chroma persistence directory.
- `CHROMA_COLLECTION_NAME` (optional, default `markdown_chunks`): Chroma collection used for markdown chunks.
- Vector-store embedding and retrieval tuning is configured in `llm_config.yaml` under `vector_store.*` (for example `embedding_model`, `embedding_max_input_tokens`, `retrieval_max_distance`, `auto_update_on_run`, `index_manifest_path`).

CLI flags override `.env` values for a given run (for example `--db-path`, `--db-url`, `--markdown-path`, `--enable-sql`).
Use `--city` (repeatable) to load markdown only for selected city files. City filters are normalized case-insensitively to backend `city_key` values (for example `Munich`, `MUNICH`, and `munich` all resolve to `munich`).

Example `.env.example` is provided. Use `llm_config.yaml` as the source of truth for vector-store and markdown batching tuning.

Default output directory is `output/` (unless overridden by `RUNS_DIR`).
Schema summary for SQL generation is derived from `backend/db_models/`.

### Vector retrieval sizing and thresholds

When vector retrieval is enabled, retrieval runs per city and per query (original + refined variants), then merges and expands context.

- For each (city × query), the retriever:
  - fetches up to `vector_store.retrieval_max_chunks_per_city_query` candidates from Chroma (ranked by increasing distance);
  - if `vector_store.retrieval_max_distance` is set, it first keeps only candidates with `distance <= cutoff`;
  - if fewer than `vector_store.retrieval_fallback_min_chunks_per_city_query` pass the cutoff, it **tops up** with the next-best candidates (above the cutoff) until it reaches the fallback minimum (or runs out of candidates).
- After per-(city × query) retrieval:
  - results are merged across queries within a city (dedupe by `chunk_id`, keep the smallest distance);
  - neighbor chunks are added by `chunk_index` window (same file/city);
  - optionally, `vector_store.retrieval_max_chunks_per_city` caps the final chunks per city after merge + neighbor expansion.
- `vector_store.retrieval_max_distance` is the strictness control:
  - smaller value = stricter matching, fewer chunks;
  - larger value = higher recall, more chunks.

Important distinction between the “max” knobs:

- `vector_store.retrieval_max_chunks_per_city_query` controls the **candidate pool size per (city × query)** _before_ distance filtering/top-up.
  - If this is too small, you may not have enough candidates to top up to the fallback minimum.
- `vector_store.retrieval_max_chunks_per_city` controls the **final per-city cap** _after_ query-merge and neighbor expansion.
  - Use it as a latency/cost guardrail; setting it too low can drop context neighbors or even primary hits with weaker distances.

Distance scale note:

- Do not assume distance is always in `[0, 1]`. It depends on collection metric and embedding characteristics.
- `0` means identical vectors; values above `0` are increasingly dissimilar.
- A cutoff of `0` is the strictest setting and usually returns very few (often zero) chunks, not all chunks.

How to estimate chunk counts:

- **Minimum (practical):**
  - per (city × query): approximately `min(vector_store.retrieval_fallback_min_chunks_per_city_query, available_chunks_total)`, before merge/neighbor expansion.
- **Maximum (practical):**
  - per (city × query): `vector_store.retrieval_max_chunks_per_city_query`, before merge/neighbor expansion;
  - bounded by `vector_store.retrieval_max_chunks_per_city` if set;
  - otherwise depends on distance cutoff + number of merged queries + neighbor expansion windows.

Recommended tuning workflow:

1. Start recall-friendly:
   - leave `vector_store.retrieval_max_distance` empty, or set a permissive value;
   - set `vector_store.retrieval_fallback_min_chunks_per_city_query` to a meaningful fallback (for example 20-40).
2. Run and inspect `output/<run_id>/markdown/retrieval.json` for returned distances and counts.
3. Set/tighten `vector_store.retrieval_max_distance` based on observed distance distribution.
4. Add `vector_store.retrieval_max_chunks_per_city` only if latency/cost grows too much.

## API key setup (important)

Current UI flow uses the backend default key:

- Put key in root `.env`:
  - `OPENROUTER_API_KEY=...`
- Use this when deployment should use one shared server key.

If key authentication fails:

- runs finish with `error.code = API_KEY_ERROR`
- chat endpoints return `401` with a key-specific message
- UI surfaces the error so backend credentials can be fixed and the run retried.

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
  # Optional Grok-only setting. Unsupported models/providers may reject requests.
  # reasoning_effort: "none"
  context_window_tokens: 400000
  input_token_reserve: 2000
  max_chunk_tokens: 120000
  chunk_overlap_tokens: 200
  batch_max_chunks: 32
  batch_max_input_tokens: null
  batch_overhead_tokens: 600
  max_workers: 8
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
  followup_search_enabled: false
  max_auto_followup_bundles: 3
assumptions_reviewer:
  model: "openai/gpt-5.2"
  temperature: 0.0
  max_output_tokens: 8000
retry:
  max_attempts: 5
  backoff_base_seconds: 0.8
  backoff_max_seconds: 8.0
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
- `markdown_researcher.batch_max_chunks`: Hard cap on chunk count per markdown researcher request batch.
- `markdown_researcher.batch_max_input_tokens`: Optional explicit token budget per markdown researcher request batch.
- `markdown_researcher.batch_overhead_tokens`: Reserved prompt/payload overhead used when adaptive markdown batch token budget is calculated.
- `markdown_researcher.reasoning_effort`: Optional reasoning effort hint for Grok-compatible models (for example `none`, `low`, `medium`, `high`); avoid setting this for models/providers that do not support reasoning controls.

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

Configuration and prompts are intentionally separated under `backend/benchmarks/`. Benchmark env files select runtime mode (`standard_chunking` vs `vector_store`), while vector-store tuning remains in `llm_config.yaml` (`vector_store.*`).

- `backend/benchmarks/prompts/retrieval_questions.txt`: benchmark question set.
- `backend/benchmarks/config/base.env`: shared benchmark env.
- `backend/benchmarks/config/mode_standard.env`: standard-mode toggle.
- `backend/benchmarks/config/mode_vector.env`: vector-mode toggle.

Command example:

```
python -m backend.scripts.run_retrieval_benchmark --city Munich --city Leipzig --city Mannheim
```

Useful flags:

- `--questions-file backend/benchmarks/prompts/retrieval_questions.txt`
- `--repetitions 2`
- `--mode vector_store` — run only vector retrieval (no standard chunking).
- `--markdown-option 16:8 --markdown-option 32:4 --markdown-option 32:8` — run explicit markdown benchmark options (`batch_max_chunks:max_workers`).
- The benchmark runs every question in the questions file; `--repetitions N` runs each question N times per mode and markdown option (total runs = questions × repetitions × modes × markdown_options).

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

- The benchmark runs all questions from the questions file (not a single query repeated N times).
- Run IDs include repetition/question indices and markdown benchmark option, for example `vector_store_b32_w8_r01_q02_...`.
- Default markdown benchmark options are `16:8`, `32:4`, and `32:8`.
- For identical queries across all runs, use a one-line questions file.
- The script always loads benchmark env files from `backend/benchmarks/config/`.
- The benchmark is runtime-only; it does not build/update the vector index.
- Vector mode uses the existing default Chroma store/collection unless overridden in your main environment.
- The benchmark also runs LLM-as-judge scoring (`openai/gpt-5.2`) per matched standard-vs-vector run pair within the same markdown option.
- The benchmark report includes speed metrics (`runtime`, `tokens/sec`) and LLM issue counters (rate limits, retries exhausted, max-turns, and non-working calls).
- Individual run failures are recorded and counted (instead of aborting the full matrix); summaries include success rate and failed run count.

Outputs are written to `output/benchmarks/<benchmark_id>/`:

- `benchmark_report.json`: machine-readable benchmark results.
- `benchmark_report.md`: human-readable summary with runtime/tokens/sec, judge score summaries, and LLM issue counters.
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
- `GET /api/v1/runs` (list discovered runs as `run_id` + `question`; refreshed from `RUNS_DIR/*/run.json` artifact folders on each request, plus currently queued/running in-memory runs)
- `GET /api/v1/runs/{run_id}/status`
- `GET /api/v1/runs/{run_id}/output`
- `GET /api/v1/runs/{run_id}/context`
- `GET /api/v1/runs/{run_id}/references` (canonical citation endpoint; supports optional query params `ref_id` and `include_quote`)
- `GET /api/v1/runs/{run_id}/references/{ref_id}` (compatibility alias for one reference with quote payload)
- `GET /api/v1/cities` (city names from markdown filenames in `MARKDOWN_DIR`, without `.md`)
- `GET /api/v1/city-groups` (predefined city groups filtered to currently available markdown cities)
- `GET /api/v1/chat/contexts` (catalog of completed run contexts with token counts)
- `GET /api/v1/runs/{run_id}/chat/sessions`
- `POST /api/v1/runs/{run_id}/chat/sessions`
- `GET /api/v1/runs/{run_id}/chat/sessions/{conversation_id}`
- `GET /api/v1/runs/{run_id}/chat/sessions/{conversation_id}/jobs/{job_id}`
- `GET /api/v1/runs/{run_id}/chat/sessions/{conversation_id}/contexts`
- `PUT /api/v1/runs/{run_id}/chat/sessions/{conversation_id}/contexts`
- `POST /api/v1/runs/{run_id}/chat/sessions/{conversation_id}/messages` (`200` with `mode="completed"` for direct replies, `202` with `mode="queued"` for split-mode jobs)
- `POST /api/v1/runs/{run_id}/assumptions/discover` (two-pass missing-data extraction + verification)
- `POST /api/v1/runs/{run_id}/assumptions/apply` (apply edited assumptions and regenerate document; ephemeral by default)
- `GET /api/v1/runs/{run_id}/assumptions/latest` (load latest assumptions artifacts for a run; only when persisted)

`POST /api/v1/runs` accepts optional city filtering:

```json
{
  "question": "Build a report for selected cities",
  "cities": ["Munich", "Berlin"],
  "analysis_mode": "aggregate"
}
```

`analysis_mode` values:
- `aggregate` (default): one integrated synthesis across selected cities.
- `city_by_city`: one city section at a time with similarities/comparison at the end.

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
- Split-mode chat jobs persist under `output/<run_id>/chat_jobs/<conversation_id>/<job_id>.json`.
- Context manager supports selecting multiple completed run contexts; manual selections may exceed the direct prompt cap and rely on overflow handling when needed.
- Chat builds a deterministic synthetic citation catalog from selected context bundles and requires assistant citations in `[ref_n]` format.
- Chat prompt citation context contains only `ref_id`, `city_name`, `quote`, and `partial_answer` (no chunk ids and no internal source ids).
- Chat context APIs use `prompt_context_tokens` as the canonical context-size metric for UI warnings, token-cap decisions, and direct-vs-split planning; raw stored totals remain diagnostic only.
- Assistant messages persist citation metadata (`source_type`, `source_id`, `source_ref_id`) for deterministic click-to-quote resolution in frontend.
- When a turn is predicted to use split/map-reduce overflow mode, the API persists the user message, returns `202 Accepted`, and the frontend polls the chat-job status endpoint until the final assistant message is attached to the session.
- Only one split-mode chat job may be active per session at a time; sending another message or changing contexts while that job is pending returns `409`.
- Prompt budget defaults to `250000` tokens (`CHAT_PROMPT_TOKEN_CAP`) and switches to the overflow map-reduce path described below when direct chat would exceed the effective budget.
- `include_quote=false` on `/references` is the default for lightweight city-label rendering; quote payload is fetched on click using `include_quote=true`.

## Context chat overflow handling

This chapter describes the full runtime path used when a chat turn is too large for a normal single-pass prompt.

### Goal

The system should still answer grounded chat questions even when the selected run context is too large to fit in one prompt. Instead of failing or sending the raw run artifacts unchanged, chat switches to an evidence-only map-reduce flow that keeps citations deterministic and frontend click-to-quote behavior intact.

### Normal direct path

For every chat turn, the backend first tries the direct path:

1. Resolve chat context sources.
2. Keep the parent/base run pinned.
3. Include all manually selected run contexts, even when the combined selection exceeds the direct prompt cap.
4. Add auto-generated follow-up bundles only while they fit after the pinned base run and any manually selected runs.
5. Build a synthetic chat citation catalog from excerpt evidence across all included sources.
6. Try to answer in one direct chat completion call.

If that direct prompt fits, chat stays on the fast path and no overflow artifact is created.
If it does not fit, the API now queues a split-mode chat job and returns immediately so the frontend can poll instead of waiting on a long-running HTTP request.

### When overflow is triggered

The overflow path is used only when the direct prompt would exceed the effective chat budget after normal history trimming.

Two main cases trigger it:

- Citation-backed direct chat cannot fit the full normalized citation catalog inside the prompt budget.
- Full serialized run context is too large for the direct prompt threshold or still exceeds the effective token cap after history is trimmed.

This keeps the direct path fast for normal cases and activates the more expensive flow only when needed.

### Lazy chat artifact

On the first overflowed turn for a run, the backend builds a cached compact evidence artifact at:

`output/<run_id>/chat_cache/evidence_chunks.json`

This is a lazy artifact:

- It is created only if chat actually overflows.
- It is a cache, not a new source of truth.
- It is safe to reuse because completed run artifacts are treated as immutable.
- It is rebuilt only when the active context selection or extracted evidence changes.

The cache stores a source signature plus compact evidence chunks. The source signature is derived from the active context ids and normalized evidence items so the backend can tell whether an existing cache is still valid for the current chat source set.

### What is kept and what is stripped

Overflow mode is evidence-only by design. The prompt payload keeps only the fields the LLM actually needs for grounded answering:

- `ref_id`
- `city_name`
- `quote`
- `partial_answer`

The overflow prompt intentionally strips prompt-noise and backend-only data, including:

- raw `context_bundle` JSON serialization
- full `final_document`
- artifact paths such as `final`
- status/debug/error metadata
- retrieval bookkeeping such as `source_chunk_ids`

This reduction is the main reason large contexts can still be answered reliably.

### Evidence cache structure

The cached artifact is organized as token-bounded chunks of evidence items. At a high level it looks like this:

```json
{
  "source_signature": "...",
  "evidence_count": 42,
  "chunks": [
    {
      "chunk_id": "chunk_1",
      "ref_ids": ["ref_1", "ref_2"],
      "items": [
        {
          "ref_id": "ref_1",
          "city_name": "Munich",
          "quote": "...",
          "partial_answer": "..."
        }
      ]
    }
  ]
}
```

The backend may trim oversized `quote` or `partial_answer` fields so a single evidence item can still fit inside a chunk budget. This trimming happens only inside the overflow prompt cache and does not mutate the original run artifacts.

### Map step

Once the compact evidence cache exists, the backend flattens the cached items and groups them into token-bounded request blocks.

Each map pass:

- receives one evidence block
- is told which chunk number it is processing
- is instructed to use only evidence from that block
- must cite factual claims using only `[ref_n]` values present in that block

This means the model never sees the full raw run payload during overflow mode. It sees only compact evidence records and produces partial grounded analyses per chunk.

### Reduce step

After all map passes finish, the backend merges the partial grounded analyses into a final answer.

The reduce prompt:

- uses only facts and citations that already appear in the partial map outputs
- preserves valid `[ref_n]` citations on factual claims
- merges duplicate statements
- resolves contradictions by preferring the later corrected grounded summary when appropriate

If the reduce prompt itself would become too large, the backend reduces recursively in batches until only one final answer remains. This is how the system handles very large evidence sets without assuming that a single reduce pass will fit.

### Citation preservation

Overflow mode does not break frontend citation behavior.

The chat layer still uses the same synthetic `ref_n` scheme. After the final answer is generated, the backend resolves each synthetic citation back to its original source metadata:

- `source_type`
- `source_id`
- `source_ref_id`

Because of that mapping, the frontend can still render compact city labels and fetch the original quote on click, even when the answer came from overflow map-reduce instead of the direct prompt path.

### Base-run pinning and token caps

Overflow handling works together with pinned-base context selection.

The parent/base run is always treated as mandatory:

- it stays included even if it alone exceeds `chat.max_context_total_tokens`
- manually selected extra runs remain included even when they push the selection above the direct prompt cap
- auto-added follow-up bundles are trimmed after the base run and manual contexts

This avoids a failure mode where the main report disappears from the chat context simply because additional sources were selected.

When the base run alone exceeds the configured token cap:

- the contexts response still includes the base run
- `is_capped` becomes `true`
- the UI shows that the selection exceeds the direct prompt cap and overflow handling will be used when needed

### Empty-evidence case

If overflow mode finds no usable compact evidence items, the backend still uses the LLM to answer. It does not guess missing facts. Instead, it asks the model to explain briefly that the current saved context does not provide extractable grounded evidence for the question.

### Why this design exists

This design keeps three things true at once:

- normal chat stays fast when the prompt fits
- very large run contexts still remain answerable
- citations remain deterministic and clickable in the UI

In practice, the most important optimization is not splitting the raw run JSON into arbitrary pieces. It is stripping the prompt down to compact evidence records first, then map-reducing over those records while preserving citation ids end-to-end.

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
NEXT_PUBLIC_FRONTEND_MODE=standard
```

Frontend supports three city scope modes in the build form: all cities, predefined group, and manual selection.
Frontend also supports two answer modes: `Aggregate Mode` and `City-by-City Mode` (sent as `analysis_mode` in run requests).
Clicking `Chat About the Answer` switches to a dedicated chat workspace (not a chat modal).
Document and chat citations render as compact city labels; clicking a label loads and shows only the source quote.
When `chat.followup_search_enabled` is `true`, the chat router may run a synchronous one-city markdown-only follow-up search, attach the resulting follow-up bundle to the session, and keep citations clickable for both base runs and chat-owned follow-up bundles.
Follow-up search stays conservative: it never launches a multi-city refresh, and failed follow-up searches return a limitation message instead of a guessed answer.
When chat needs a single city before searching, the backend now sends clarification metadata and the frontend opens a city-picker popup that can trigger the one-city follow-up search directly.
When a direct chat prompt would overflow, the backend now falls back to an evidence-only map-reduce flow built from compact excerpt evidence and caches that stripped chat artifact under `output/<run_id>/chat_cache/evidence_chunks.json`.
The parent/base run stays pinned in chat context selection, manual run selections may exceed the direct prompt cap, and auto-added follow-up bundles are still trimmed first.
The `Load Previous Answer` picker reads `run_id` + `question` from `GET /api/v1/runs`, then loads selected run artifacts through the standard run endpoints.
`NEXT_PUBLIC_FRONTEND_MODE` sets the default frontend surface, and the page header always exposes a persistent browser toggle between `standard` and `dev`.

Dev-mode frontend features:

- `Assumptions Review` workspace: `Find Missing Data` runs two LLM passes (extract + verification), missing items are editable by city, and `Regenerate` returns revised content without persisting assumptions by default.
- `Manage Contexts` in chat workspace: switching/combining multiple completed run contexts with token-cap enforcement.
- Chat token metrics in UI (`prompt_context_tokens`, `token_cap`, and per-context context-token counts).
- Read-only `run_id` display with a copy action for quick run identification.
- Frontend user-owned OpenRouter key controls: `OpenRouter API Key (Optional)`, `Use This Key`, and `Clear`; the override stays in memory for the current tab and is not stored in localStorage.

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

- `run.json`: machine-readable run metadata (status, timestamps, artifacts, decisions), including `inputs.analysis_mode` and `artifacts.error_log` when available.
- `run.log`: detailed runtime logs, including per-agent `LLM_USAGE` lines, chat prompt-window diagnostics (`Context chat reply plan`, `Context chat direct request`, with fitted source ids and token-component counts), and writer city-citation coverage checkpoints (`WRITER_CITATION_COVERAGE`, with `coverage_ratio` such as `33/33`).
- `error_log.txt`: extracted error-focused log view from `run.log` (`ERROR`, `CRITICAL`, and exhausted retry events).
- `run_summary.txt`: human-readable consolidated report. Header includes `Started`, `Completed`, and explicit `Total runtime` in seconds, plus `LLM Usage` totals/per-agent. It also captures an input snapshot (`initial question`, `refined question`, `selected cities` planned/found, markdown dir/file/chunk/excerpt counts) and a `MARKDOWN_FAILURE_SUMMARY` aggregated from batch failures.
- `context_bundle.json`: payload passed between agents (`sql`, `markdown`, `research_question`, `analysis_mode`, final path).
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
- `markdown/references.json`: run-local citation map generated from markdown excerpts. Includes sequential `ref_n` entries with `excerpt_index`, `city_name`, `quote`, `partial_answer`, and `source_chunk_ids`.
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
- `ref_id`: sequential run-local citation id (`ref_1`, `ref_2`, ...), used by writer output and frontend reference lookups.
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

The build now fails fast on embedding failures and exits non-zero before any collection reset or manifest write.

Analyze retrieval distance distributions (to help choose `vector_store.retrieval_max_distance`):

```
python -m backend.scripts.analyze_retrieval_distances --runs-dir output
python -m backend.scripts.analyze_retrieval_distances --city Munich --city Leipzig --thresholds "0.5,1.0,2.0" --show-per-run
```

How to use the output:

- Start with `vector_store.retrieval_max_distance` empty (no distance filtering) and run a few representative queries.
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

The update now fails fast on embedding failures and exits non-zero before any delete/upsert/manifest-write commit.

Check manifest and Chroma DB status:

```
python -m backend.scripts.check_vector_index
python -m backend.scripts.check_vector_index --no-show-files
```

**Building the vector index on Kubernetes:** The backend and the one-off build Job share the same PVC mounted once at `/data` (no subPath). Both use the same `securityContext` (runAsUser 0, DAC_READ_SEARCH) so the Job can write `/data/chroma` and the backend can read it. Apply the Job from the repo root (see `k8s/backend-build-vector-index-job.yml` header for full steps):

```bash
kubectl scale deployment urbind-query-mechanism-backend --replicas=0
kubectl apply -f k8s/backend-build-vector-index-job.yml
kubectl logs job/urbind-query-mechanism-build-vector-index -f
kubectl scale deployment urbind-query-mechanism-backend --replicas=1
```

Scaling down the backend to 0 ensures no concurrent reads/writes to the vector index.
Paths on the PVC are `/data/output` (run artifacts) and `/data/chroma` (vector index and manifest). Restart the backend after the Job completes so it picks up the new index.
The Job manifest includes disruption resilience for long runs (`karpenter.sh/do-not-disrupt: "true"`, `backoffLimit: 3`, and `podFailurePolicy` that ignores `DisruptionTarget` pod failures).

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
