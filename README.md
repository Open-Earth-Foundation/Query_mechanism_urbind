# Query Mechanism Urbind

Multi-agent document builder that answers user questions from Markdown sources. It orchestrates retrieval-query preparation, markdown extraction, and writing with OpenAI Agents, and logs every run artifact for inspection.

## Requirements

- Python 3.11+
- Node.js 20+ (frontend)
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
- Initiative extraction knobs are configured in `llm_config.yaml` under `initiative_extractor`
  (`max_segment_tokens`, `segment_overlap_lines`, `max_workers`,
  `prior_initiatives_max_tokens`, `action_heavy_initiative_threshold`,
  `action_heavy_max_followup_calls`). The default extraction budget sends up to
  `20,000` source segment tokens and `10,000` prior-initiative tokens. This
  artifact-first extractor uses bounded ordered segments and does not perform TEF
  classification or database writes. When
  `prior_initiatives_max_tokens` is greater than zero, segments are processed in order so each LLM
  call can see a token-capped list of already extracted canonical initiatives and avoid duplicates.
  Initiative extraction does not split segments again based on how many initiatives the model returns;
  if a segment returns more than the action-heavy threshold, follow-up calls reuse the same source
  segment and show only initiatives already extracted from that segment until the model stops.
  `semantic_dedupe_enabled` defaults to on and runs the only persisted merge pass over
  candidate records so initiatives are merged when they describe the same real-world action.
- TEF mapping knobs are configured in `llm_config.yaml` under `tef_mapper`
  (`max_workers`, `review_confidence_threshold`, `close_alternative_delta`,
  `min_transition_confidence`, `numeric_unit_classifier_enabled`). The mapper is JSON-only: it
  reads initiative extraction artifacts, runs sector, category, Transition Element, and optional
  numeric-unit classification passes with stage-scoped prompts and four catalog JSON files, and
  writes mapping artifacts without database writes or an LLM review pass. TEF sector,
  subcategory, and subsubcategory catalog cards include prompt-ready routing definitions,
  positive use signals, and avoid rules so the category router can compare sibling branches.
- Retry policy is centralized in top-level `retry` in `llm_config.yaml` (`max_attempts`, `backoff_base_seconds`, `backoff_max_seconds`) and is shared across retry/backoff behavior for LLM calls and related operations.
- Agent turn limits are configured per agent via `max_turns` (for example `markdown_researcher.max_turns`, `writer.max_turns`).
- Writer prompt sizing and fallback batching are configured in `llm_config.yaml` under `writer` (`multi_pass_threshold_tokens`, `multi_pass_chunk_tokens`).
- Optional `markdown_researcher.reasoning_effort` can be set for Grok reasoning control (for example `"none"`), but this is model/provider-specific and may fail on unsupported models.
- Copy `.env.example` to `.env` and fill in values for your environment.
- `.env` is loaded automatically via `python-dotenv` in the scripts.
- Do not commit `.env`.

Environment variables (`.env`):

- `OPENROUTER_API_KEY` (required): API key used for all LLM calls via OpenRouter.
- `APP_SHARED_PASSWORD_HASH` (required for Docker Compose and deployed frontend runtime): bcrypt hash of the shared login password used by the login page.
- `APP_SESSION_SECRET` (required): shared HMAC secret used to sign and verify the `urbind_session` cookie. Use the same value in backend `.env`, frontend runtime env, Docker, and Kubernetes.
- `MARKDOWN_DIR` (optional, default `documents`): default directory scanned for top-level city markdown files. Runtime markdown discovery ignores subfolders under this directory.
- `RUNS_DIR` (optional, default `output`): base directory for run artifacts.
- `LOG_LEVEL` (optional, default `INFO`): logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `OPENROUTER_BASE_URL` (optional, default `https://openrouter.ai/api/v1`): override for OpenRouter-compatible backends.
- `API_CORS_ORIGINS` (optional): comma-separated allowed frontend origins for the API. When omitted, the API allows wildcard CORS without credentials.
- `API_CHAT_JOB_WORKERS` (optional, default `1`): number of background workers for split-mode chat jobs.
- `LLM_CONFIG_PATH` (optional, default `llm_config.yaml`): API config file path.
- `CITY_GROUPS_PATH` (optional, default `backend/api/assets/city_groups.json`): city groups catalog JSON path.
- `ENRICHMENT_ENABLED` (optional, default `false`): enables gap analysis and assumptions enrichment.
- `WRITER_ENABLED` (optional, default `true`): enables the final writer stage. When `false`, the pipeline still runs retrieval and markdown extraction, writes run artifacts, and emits a lightweight `final.md` noting that the writer was skipped.
- `WEB_RESEARCH_ENABLED` (optional, default `false`): enables web research when enrichment is enabled and the required API keys are present.
- `SERPER_API_KEY` (optional, required when web research is enabled): Serper.dev key for web search.
- `FIRECRAWL_API_KEY` (optional, required when web research is enabled): Firecrawl key for rendered-page scraping.
- `VECTOR_STORE_ENABLED` (optional, default `false`): enables local Chroma markdown indexing flows.
- `VECTOR_STORE_AUTO_UPDATE_ON_RUN` (optional, default from `llm_config.yaml`): when `true`, every vector-backed run performs a full-corpus freshness check and refreshes the shared index automatically when needed; when `false`, the same full-corpus check still runs and blocks stale runs until the maintenance workflow is executed.
- `VECTOR_STORE_UPDATE_MODE` (optional, default `local_process`): used only when automatic updates are enabled. Keep `local_process` for local development; the deployed maintenance workflow does not rely on automatic updater Jobs.
- `ANONYMIZED_TELEMETRY` (optional, default `FALSE`): disables Chroma anonymized telemetry when set to `FALSE`.
- `CHROMA_PERSIST_PATH` (optional, default `.chroma`): local Chroma persistence directory.
- `CHROMA_COLLECTION_NAME` (optional, default `markdown_chunks`): Chroma collection used for markdown chunks.
- `EXTERNAL_SOURCE_SEARCH_ENABLED` (optional, default `true`): enables governed external Markdown library enrichment when `sources.yaml` is available.
- `EXTERNAL_SOURCE_DIR` (optional, default `documents/source_library`): directory containing `sources.yaml` and Markdown files whose stems match `source_id`.

Chat prompt sizing, follow-up router history and excerpt caps, retry backoff, provider timeouts, and vector-store retrieval tuning all come from `llm_config.yaml`.
When a run requests provider-backed features, the API now validates them before queueing work: missing `OPENROUTER_API_KEY` blocks the run immediately, and web-research runs fail fast if `SERPER_API_KEY` or `FIRECRAWL_API_KEY` is missing or rejected by the upstream provider.
CLI flags override `.env` values for a given run (for example `--markdown-path`).
Use `--city` (repeatable) to load markdown only for selected city files. City filters are normalized case-insensitively to backend `city_key` values (for example `Munich`, `MUNICH`, and `munich` all resolve to `munich`).

Example `.env.example` is provided. Use `llm_config.yaml` as the source of truth for vector-store and markdown batching tuning; deployment env vars may override operational toggles such as `WRITER_ENABLED`, `VECTOR_STORE_ENABLED`, `VECTOR_STORE_AUTO_UPDATE_ON_RUN`, and `VECTOR_STORE_UPDATE_MODE`. For local retrieval-only testing, set `WRITER_ENABLED=false` in your local `.env`.

Default output directory is `output/` (unless overridden by `RUNS_DIR`).

### Vector retrieval sizing and thresholds

When vector retrieval is enabled, retrieval runs per city and per user-provided query (required question plus optional question 2 and 3), then merges and expands context.

- For each (city x query), the retriever:
  - fetches up to `vector_store.retrieval_max_chunks_per_city_query` candidates from Chroma (ranked by increasing distance);
  - if `vector_store.retrieval_max_distance` is set, it first keeps only candidates with `distance <= cutoff`;
  - if fewer than `vector_store.retrieval_fallback_min_chunks_per_city_query` pass the cutoff, it **tops up** with the next-best candidates (above the cutoff) until it reaches the fallback minimum (or runs out of candidates).
- After per-(city x query) retrieval:
  - results are merged across queries within a city (dedupe by `chunk_id`, keep the smallest distance as chunk distance metadata);
  - neighbor chunks are added by `chunk_index` window (same file/city);
  - optionally, `vector_store.retrieval_max_chunks_per_city` caps the final chunks per city after merge + neighbor expansion.
- Retrieval artifacts now persist both layers explicitly:
  - `seed_chunks[]` = unique direct hits before neighbor expansion and before per-city caps;
  - `chunks[]` = the final delivered context after neighbor expansion and caps.
- Strict Stage A benchmark metrics must use `seed_chunks[]`, not `chunks[]`.
- Every serialized retrieval chunk now includes `chunk_index` plus `provenance`
  (`origin`, `selection_mode`, `seed_rank`, `seed_query_ids`, `expanded_from_chunk_ids`).
- Retrieval artifacts now label the active distance metric explicitly as `cosine_distance`.
- Per-query retrieval stats are split truthfully into `distance_qualified_chunks`, `fallback_top_up_chunks`, and `seed_chunks_selected`; run-level retrieval metrics use the same naming.
- `vector_store.retrieval_max_distance` is the strictness control:
  - smaller value = stricter matching, fewer chunks;
  - larger value = higher recall, more chunks.

Important distinction between the "max" knobs:

- `vector_store.retrieval_max_chunks_per_city_query` controls the **candidate pool size per (city x query)** _before_ distance filtering/top-up.
  - If this is too small, you may not have enough candidates to top up to the fallback minimum.
- `vector_store.retrieval_max_chunks_per_city` controls the **final per-city cap** _after_ query-merge and neighbor expansion.
  - Use it as a latency/cost guardrail; setting it too low can drop context neighbors or even primary hits with weaker distances.

Distance scale note:

- The shared Chroma collection now uses cosine distance (`1 - cosine similarity`), not the old default L2 metric.
- Do not assume distance is always in `[0, 1]`. It depends on collection metric and embedding characteristics.
- `0` means identical vectors; values above `0` are increasingly dissimilar.
- A cutoff of `0` is the strictest setting and usually returns very few (often zero) chunks, not all chunks.

Recommended tuning workflow:

1. Start recall-friendly:
   - leave `vector_store.retrieval_max_distance` empty, or set a permissive value;
   - set `vector_store.retrieval_fallback_min_chunks_per_city_query` to a meaningful fallback (for example 20-40).
2. Run and inspect `output/<run_id>/stage_files/003_retrieval/retrieval.json` for returned distances and counts.
3. Set/tighten `vector_store.retrieval_max_distance` based on observed distance distribution.
4. Add `vector_store.retrieval_max_chunks_per_city` only if latency/cost grows too much.

## Shared password gate (required)

The frontend and backend now use one shared password gate with a signed `urbind_session` cookie:

- one shared password for the entire app
- one shared session secret used to sign and verify the cookie
- no separate users, OAuth providers, or third-party auth service

Configure it like this:

1. For local no-Docker runs, set backend values in the root `.env` and frontend values in `frontend/.env.local`. This duplicate local setup is only needed because FastAPI and Next.js are started as separate processes from separate directories.
2. For Docker Compose, the root `.env` is enough; Compose passes the shared auth values into both containers.
3. For deployed dev/prod, do not use repo `.env` files. Set `APP_SHARED_PASSWORD_HASH` and `APP_SESSION_SECRET` through GitHub Secrets/Kubernetes secrets.
4. Leave `APP_SESSION_COOKIE_DOMAIN` unset locally. Set it to `.openearth.dev` in production so the cookie reaches both frontend and backend subdomains.
5. Keep `API_CORS_ORIGINS` explicit. Backend `/api/v1/*` requires the shared session cookie; backend `/` and `/healthz` stay public for probes.

Users type the shared password into the login page, but the runtime stores only its bcrypt hash in `APP_SHARED_PASSWORD_HASH`. Generate `APP_SESSION_SECRET`; users never type this value. It only signs and verifies the session cookie. The secret must be at least 32 characters; use a 64-character hex value from the commands below.

Generate a bcrypt hash from `frontend/` after `npm install`:

```bash
node -e "const bcrypt = require('bcryptjs'); bcrypt.hash(process.argv[1], 10).then((hash) => console.log(hash));" "your-shared-password"
```

In the root `.env` used by Docker Compose, escape each `$` in the bcrypt hash as `$$` (for example `$$2b$$12$$...`). For `frontend/.env.local`, escape each `$` as `\$` instead. GitHub Secrets and Kubernetes secrets should store the raw hash.

Generate a session secret on Windows PowerShell:

```powershell
-join ((1..64) | ForEach-Object { "{0:x}" -f (Get-Random -Maximum 16) })
```

Generate a session secret on macOS/Linux:

```bash
openssl rand -hex 32
```

Rotating `APP_SESSION_SECRET` logs out all active browser sessions.

For cookie-authenticated unsafe API methods (`POST`, `PUT`, `PATCH`, and `DELETE`), the backend also requires the browser `Origin` or `Referer` to match `API_CORS_ORIGINS`. This prevents sibling subdomains or unrelated sites from using the shared cookie for state-changing requests.

The frontend applies a basic in-memory failed-login throttle per client address and per frontend pod. It also keeps a fixed global failed-login bucket of 50 attempts per window so spoofed forwarding headers cannot fully bypass the app-level throttle. For production, keep an ingress or WAF rate limit on `/api/auth/login` as the durable control.

## API key setup (important)

Current UI flow uses the backend default key:

- Put key in root `.env`:
  - `OPENROUTER_API_KEY=...`
- Use this when deployment should use one shared server key.

If key authentication fails:

- runs finish with `error.code = API_KEY_ERROR`
- chat endpoints return `401` with a key-specific message
- UI surfaces the error so backend credentials can be fixed and the run retried.

See the checked-in example config at [`llm_config.yaml`](llm_config.yaml).

### How token and size limits are applied

Shared input-budget logic (used by orchestrator, markdown researcher, and writer):

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
- `markdown_researcher.max_chunk_tokens`: Hard cap for each markdown chunk size.
- `markdown_researcher.chunk_overlap_tokens`: Token overlap between neighboring chunks.
- `markdown_researcher.max_turns`: Max LLM turns per markdown batch extraction call.
- `markdown_researcher.batch_max_chunks`: Hard cap on chunk count per markdown researcher request batch.
- `markdown_researcher.batch_max_input_tokens`: Optional explicit token budget per markdown researcher request batch.
- `markdown_researcher.batch_overhead_tokens`: Reserved prompt/payload overhead used when adaptive markdown batch token budget is calculated.
- `markdown_researcher.reasoning_effort`: Optional reasoning effort hint for Grok-compatible models (for example `none`, `low`, `medium`, `high`); avoid setting this for models/providers that do not support reasoning controls.
- `writer.multi_pass_threshold_tokens`: token threshold where writer switches from one-shot writing to multi-pass batching over accepted evidence excerpts.
- `writer.multi_pass_chunk_tokens`: target token cap per writer batch when multi-pass batching is used.

How this influences runtime behavior:

- Markdown content is chunked first, then batched by token budget. Chunks larger than the current batch budget are skipped.
- Each original markdown batch gets `retry.max_attempts` total tries.
- If an original markdown batch still fails with a retryable markdown extraction error, only that failing batch is recursively halved up to two split rounds.
- Each split child batch gets exactly one try; successful child branches are kept immediately, and only final failed leaves remain unresolved.
- Oversized markdown files are skipped when they exceed `max_file_bytes`.

Visibility and warnings:

- Markdown budget/file skips emit warnings in logs.

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

Extract city initiatives into inspectable JSONL artifacts without TEF classification:

```
python -m backend.scripts.extract_initiatives --markdown-path documents --city Krakow
```

The extractor writes to `output/initiative_extraction/<run_id>/` with source manifests, line-aware
segments, raw per-segment extractions, candidate records, semantic duplicate groups, final
deduplicated initiatives, review items, and a summary. `03_deduped/initiatives.jsonl` contains only
the agreed canonical v1 initiative shape from `docs/plan.md`; generated ids and quote-only audit
citations are kept separately in `03_deduped/initiative_records.jsonl` for downstream mapping. Use
`--run-id`, `--output-dir`, `--max-workers`, and `--log-llm-payload` to override run naming,
artifact location, concurrency, and
payload logging. `--max-workers` only affects extraction when prior-initiative context is disabled in
config. The canonical `city` field and structured source references are assigned from source
segment metadata, not inferred by the LLM. The LLM returns only `source_quote` for citation text.

Run initiative extraction and TEF mapping as one artifact pipeline:

```
python -m backend.scripts.map_initiatives_to_tef \
  --markdown-path documents \
  --city Krakow
```

By default, `map_initiatives_to_tef` first runs the initiative extractor and then maps the
resulting `03_deduped/initiative_records.jsonl` to TEF targets. Omit `--city` to process all
top-level Markdown city files discovered under `--markdown-path`; repeat `--city` to process a selected
list. Use `--extraction-output-dir`, `--extraction-run-id`, and `--extraction-max-workers` to
control the extraction stage separately from TEF mapper `--output-dir`, `--run-id`, and
`--max-workers`.

Run mapping only against an existing extraction artifact when needed:

```
python -m backend.scripts.map_initiatives_to_tef \
  --mapping-only \
  --extraction-run-dir output/initiative_extraction/five_cities_20260421_002 \
  --city Krakow
```

The TEF mapper reads `03_deduped/initiative_records.jsonl` from an extraction run so it can use
pipeline-generated record ids while keeping the canonical extraction file clean. It writes to
`output/tef_mapping/<run_id>/` with source manifests, input initiative rows,
sector routes, recursive category routes, Transition Element mapping outputs, final mappings,
manual-review items, numeric facts, TEF-grouped initiatives, metric rollups, and a summary. It
loads only the active stage prompt and the catalog slice for that pass. Router prompts ask the
model to follow the initiative's main causal shift, not to make a smaller supporting component
primary only because it is explicitly named. Category routing continues through child categories
before Transition Element matching, so parent categories that also have direct Transition
Elements do not stop the route early. If the selected TEF leaf has no Transition Elements, the
final mapping uses `target_type: "subcategory"` and the selected TEF path as `target_id`.
The same subcategory fallback is used when the transition mapper finds no exact Transition
Element match. A successful initiative mapping never drops the initiative solely because the TEF
catalog has no precise Transition Element for it. If a category route returns a descendant of the
current direct-child candidate, the mapper normalizes it to that direct child for the current pass
and emits a manual-review item. Sector paths are assigned from the TEF catalog after the model
selects a sector key, so the sector router does not generate path fields. `source_quote` is copied
through mapper input rows, final mappings, numeric facts, and TEF-grouped initiatives for
search-back traceability, but sector/category/Transition Element mapper LLM passes do not receive
it as classification evidence. When `tef_mapper.numeric_unit_classifier_enabled` is true, numeric
facts use a constrained Pydantic LLM classifier to choose only the supported metric types, units,
and aggregation methods before default rollups are written; invalid or failed classifications fall
back to rule-based unit inference and are marked for review.

Run the full Krakow TEF benchmark against the curated CCC source-truth mappings:

```
python -m backend.scripts.benchmark_krakow_tef_mapping --max-workers 3
```

The benchmark converts `backend/benchmarks/tef_mapping/krakow_source_truth/all_correct_initiatives_mapped_to_tef.json`
into mapper-ready initiative records, runs the TEF mapper, and compares final mappings
against source truth. Outputs are written under
`output/tef_benchmarks/krakow_tef_mapping/<benchmark_id>/`, including
`00_inputs/initiatives.jsonl`, the standard `01_tef_mapping/` mapper artifacts,
`02_comparison/tef_benchmark_issues.json`, `02_comparison/tef_benchmark_report.md`,
and `benchmark_summary.json`. Use `--limit N` for a smoke check before a full run.

Run the governed external-source benchmark and writer scenario for Krakow:

```
python -m backend.scripts.benchmark_external_source_pipeline --run-id krakow_external_smoke
```

The benchmark reads `backend/benchmarks/external_sources/krakow_external_source_benchmark.json`,
loads `documents/source_library/sources.yaml`, runs the external-source researcher with controlled
Markdown search tools, resolves external evidence into the enrichment bundle, and optionally runs
the writer. Outputs are written under
`output/external_source_benchmarks/krakow/<run_id>/`, including `benchmark_summary.json`,
`context_bundle.json`, `writer_answer.md`, and
`stage_files/008_enrichment/external_source_search_audit.json`.
Use `--skip-writer` for extraction-only validation.

Design notes and example workflows for this stage live in `docs/plan.md`,
`docs/example.md`, and `docs/tool_implementation.md`.

### Krakow TEF source-of-truth assets

The curated source-of-truth files for the Krakow CCC manual-scan baseline live in
`backend/benchmarks/tef_mapping/krakow_source_truth/`. This folder intentionally contains
exactly two JSON files:

- `all_correct_initiatives.json`: 58 Krakow CCC manual-scan source-of-truth initiatives
  from `documents/Krakow.md`, Module B-2.2 outline of individual activities/actions/measures,
  source lines `2590-4551`.
- `all_correct_initiatives_mapped_to_tef.json`: the same 58 initiatives mapped to the
  TEF framework.

The initiative file comes from source run id `krakow_20260420` at
`output/tef_mapping/krakow_20260420`. The mapped file comes from TEF mapping run id
`krakow_manual_assets_tef_20260421_002` at
`output/tef_mapping/krakow_manual_assets_tef_20260421_002`, using the mapper input
adapted from `all_correct_initiatives.json`. The baseline is narrower than the later
broader 98-record automated Krakow subset.

Count scope: `Krakow=58`, with local-code prefixes `BIC=11`, `E=17`, `TR=16`,
`GOZ=4`, and `I=10`. In the mapped file, all 58 initiatives have TEF mappings:
78 final mapping rows, including 58 primary mapping rows. Those rows include 52
Transition Element mappings and 26 TEF subcategory mappings. The mapped file also
contains 100 review items, 66 mapping rows marked as needing review, 42 unique target
ids, and 21 unique target paths.

The benchmark audit records B-2.2 local-code coverage as complete, but does not treat
the extraction as pixel-perfect for every source quantity or damaged Markdown section.
Use `output/tef_mapping/krakow_20260420/extraction_benchmark_audit.md` for those
precision caveats.

TEF target fallback policy: use Transition Element mappings returned by the transition
mapper. When the catalog has no Transition Elements or the transition mapper returns no
exact Transition Element match, use the selected TEF subcategory path as the final
target. Empty TEF targets are not allowed.

The prompt-ready category guidance lives directly in `tef_mapping/catalog/subcategories.json`
and `tef_mapping/catalog/subsubcategories.json`. Each category record carries its own
`description` and `card_text`, including Routing Definition, Use This Category When, and
Avoid This Category When sections. Transition Element candidate descriptions used by the
mapper live in `tef_mapping/catalog/transition_elements.json`.

Build or refresh numeric TEF rollups for an existing mapping run without LLM calls:

```
python -m backend.scripts.rollup_tef_numeric_facts \
  --tef-run-dir output/tef_mapping/three_cities_tef_balanced_smoke_20260421_001 \
  --extraction-run-dir output/initiative_extraction/three_cities_20260421_001
```

The rollup reads generated ids from `initiative_records.jsonl`, reads numbers only from the clean
canonical `initiative` object inside each record, and writes `07_numeric_facts/` and
`08_tef_groups/` artifacts.

## Happy-path workflow

High-level flow from user input to final output text:

```mermaid
flowchart TD
    A[User question + CLI/config input] --> B[run_pipeline in orchestrator module]
    B --> C[Prepare retrieval queries]
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
- Writer builds a writer-specific minimal bundle from the accepted markdown excerpts, selected-city metadata, and writer-visible enrichment artifacts before prompting the model; markdown audit fields and non-writer enrichment bookkeeping such as generic status or notes are not sent to the writer.
- When the writer bundle exceeds `writer.multi_pass_threshold_tokens`, the writer splits accepted evidence into multiple batches, writes batch drafts, and then combines those drafts into one final answer. If a post-batching payload still exceeds the configured writer input budget, the run now fails explicitly instead of silently reverting to one-shot writing.
- Writer writes final output text to `output/<run_id>/final.md`. The response starts with an evidence preface (based on `excerpt_count`); when `excerpt_count=0`, it returns a "no evidence found" response.

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

**Vector-only reproducibility (same query and same optional retrieval queries):** To run the vector strategy multiple times with the exact same question and optional retrieval queries (e.g. to check outcome stability):

1. Run the pipeline once to get a run with the desired question and cities, e.g. `python -m backend.scripts.run_pipeline --question "What does Aachen do for PV rooftop?" --city Aachen --markdown-path documents`. Note the run id and open `output/<run_id>/stage_files/002_query_preparation/research_question.json`.
2. Create a one-line questions file (e.g. `my_questions.txt`) containing exactly the `original_question` from that run.
3. Create a query-overrides JSON (e.g. `my_overrides.json`) with one key: the same `original_question` string; value: `{"canonical_research_query": "<original_question>", "retrieval_queries": ["<optional query 2>", "<optional query 3>"]}`. The override format still uses the legacy `canonical_research_query` key, but it must mirror `original_question`; do not provide a rewritten query there.
4. Run the benchmark in vector-only mode with fixed queries and several repetitions:

   ```
   python -m backend.scripts.run_retrieval_benchmark --questions-file my_questions.txt --query-overrides my_overrides.json --mode vector_store --repetitions 5 --city Aachen
   ```

   Each run will use the same verbatim question plus optional retrieval queries; only retrieval, extraction, and writing are re-executed. Compare `output/benchmarks/<benchmark_id>/runs/vector_store/*/final.md` (and optionally `retrieval.json`, `accepted_excerpts.json`) across repetitions.

Benchmark behavior notes:

- The benchmark runs all questions from the questions file (not a single query repeated N times).
- Run IDs include repetition/question indices and markdown benchmark option, for example `vector_store_b32_w8_r01_q02_...`.
- Default markdown benchmark options are `16:8`, `32:4`, and `32:8`.
- For identical queries across all runs, use a one-line questions file.
- The script always loads benchmark env files from `backend/benchmarks/config/`.
- The benchmark is runtime-only; it does not build/update the vector index.
- Vector mode uses the existing default Chroma store/collection unless overridden in your main environment.
- The benchmark also runs LLM-as-judge scoring (`openai/gpt-5.4-mini`) per matched standard-vs-vector run pair within the same markdown option.
- The benchmark report includes speed metrics (`runtime`, `tokens/sec`) and LLM issue counters (rate limits, retries exhausted, max-turns, and non-working calls).
- Individual run failures are recorded and counted (instead of aborting the full matrix); summaries include success rate and failed run count.

### Speed / Chunk / Worker benchmark

On March 30, 2026 we ran a stress comparison for the broad aggregate question
`Aggregate and compare all EV charging and building retrofit initiatives with quantified targets and budgets.`
under `output/benchmarks/full_retrieval_20260330_live/`.

Standard chunking results for that question:

- `b16_w8` (`batch_max_chunks=16`, `max_workers=8`): completed in about 17.6 minutes, used about 13.12M tokens, and writer citation coverage stopped at `94/102`.
- `b32_w4` (`batch_max_chunks=32`, `max_workers=4`): completed in about 28.4 minutes, used about 13.14M tokens, and writer citation coverage reached `102/102`.
- `b32_w8` (`batch_max_chunks=32`, `max_workers=8`): completed in about 14.6 minutes, used about 13.13M tokens, and writer citation coverage stopped at `40/102`.

Interpretation:

- `b32_w4` gave the best citation coverage in this broad aggregate benchmark.
- `b32_w8` was much faster than `b32_w4`, but completeness was materially worse in that run.
- `b16_w8` sat between them on speed and coverage.

Current recommendation:

- Keep the normal runtime configuration as it is today in `llm_config.yaml`: `batch_max_chunks=32` and `max_workers=8`.
- We are not promoting `b32_w4` to the default from this benchmark alone, because normal runtime speed is also important and this benchmark is a heavy stress case rather than the only production workload.
- Use `b32_w4` as a benchmark reference point when testing completeness on very broad aggregate questions.

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

### Gold recall benchmark

Use this benchmark to measure where information is lost across retrieval,
markdown extraction, and final writing for a manually curated gold dataset.

Command example:

```
python -m backend.scripts.benchmark_recall --gold-file tests/fixtures/benchmark_gold.json
```

Useful flags:

- `--benchmark-id`: override the default UTC timestamp benchmark id.
- `--output-dir output/benchmarks/recall`: change the benchmark output root.
- `--config llm_config.yaml`: use a specific runtime config for live cases.
- `--case-id <case_id>`: repeatable case filter.
- `--log-llm-payload` / `--no-log-llm-payload`: toggle full LLM payload logging.

Behavior notes:

- This benchmark is not pairwise standard-vs-vector judging. It scores a single
  run against gold chunks and gold facts.
- Official Stage A metrics are strict seed retrieval:
  `retrieval_recall`, `retrieval_precision`, and `mrr` are computed from
  `retrieval.json.seed_chunks[]`.
- `delivery_recall` and `delivery_precision` are supplemental metrics computed
  from the final delivered `retrieval.json.chunks[]`.
- The gold file schema is `{"version": 1, "cases": [...]}` with `case_id`,
  `question`, `gold_chunk_ids`, `gold_facts`, `gold_city`, and optional
  `selected_cities`, `gold_chunk_texts`, and `gold_chunk_alternatives`.
- `gold_chunk_alternatives` stores accepted equivalent runtime chunks explicitly
  as `{chunk_id, chunk_text}` objects, so the fixture keeps both the chunk
  number/id and the chunk text in JSON.
- `gold_chunk_texts` should store the canonical chunk text for each gold slot.
  The scorer still supports containment fallback, but the fixture data should
  keep the actual chunk text in JSON.
- Stage B and Stage C fact verification use an LLM fact judge configured under
  `benchmark_fact_judge` in `llm_config.yaml` (model, temperature,
  `max_output_tokens`, `reasoning_effort`) to handle paraphrases.

Outputs are written to `output/benchmarks/recall/<benchmark_id>/`:

- `benchmark_report.json`: machine-readable per-case metrics, fact judgements,
  and loss waterfalls.
- `benchmark_report.md`: concise human-readable summary.
- `runs/<case_id>/...`: live pipeline artifacts for each benchmark case.

### Writer numeric benchmark

Use this benchmark to check whether the final writer output preserves specific
manual baseline numbers for Krakow, the Poland group, the Balkans & Eastern
Mediterranean group, and the optional frozen 102-city corpus snapshot.

Command example:

```
python -m backend.scripts.benchmark_writer_numbers
```

Useful flags:

- `--mode ccc_only|full_pipeline|both`: benchmark the CCC-only pipeline, the
  full enrichment pipeline, or both. The fixture default is `ccc_only`.
- `--include-optional-cases`: include fixture cases marked optional, including
  the expensive all-cities run.
- `--run-id <benchmark_id>`: override the default UTC timestamp benchmark id.
- `--output-dir output/benchmarks/writer_numeric`: change the benchmark output
  root.
- `--benchmark-file backend/benchmarks/writer_numeric/writer_numeric_benchmark.json`:
  use a different frozen benchmark fixture.

Behavior notes:

- The fixture schema is `{"version": 1, "default_mode": "...", "cases": [...]}`.
- Every case stores an explicit `selected_cities` list. The all-cities case is
  frozen in the fixture and is not resolved dynamically at runtime.
- Cases can be marked optional in the fixture and require
  `--include-optional-cases` to run.
- The all-cities case is optional by default because it can take a long time
  and consume a large number of LLM tokens.
- Every baseline metric stores manual `components[]` so the benchmark remains
  baseline-driven instead of recomputing the totals from the corpus at runtime.
- The benchmark is report-only: mismatches do not fail the run; they are
  surfaced in the report.
- Numeric extraction from `final.md` is driven by the separate
  `benchmark_number_extractor` config block in `llm_config.yaml`. The
  extractor receives metric ids, labels, and units, while baseline expected
  values are used only by the deterministic comparison step after extraction.

Outputs are written to `output/benchmarks/writer_numeric/<benchmark_id>/`:

- `benchmark_summary.json`: machine-readable benchmark report with per-metric
  comparisons.
- `benchmark_report.md`: concise human-readable diff report.
- `runs/<case_id>__<mode>/final.md`
- `runs/<case_id>__<mode>/context_bundle.json`
- `runs/<case_id>__<mode>/extracted_numbers.json`

## Run API (local)

Start FastAPI backend:

```
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Core endpoints:

- `GET /` (root health endpoint)
- `POST /api/v1/runs`
- `GET /api/v1/runs` (list discovered runs as `run_id` + `question` + `status` + `picker_timestamp`; returns successful runs by default, accepts `include_all=true` for dev/debug views that also include queued, running, failed, and stopped runs, and supports optional `search` across run ids, compact picker dates/times, question text, and selected city names, refreshed from `RUNS_DIR/*/api_state.json` artifact folders on each request, plus currently queued/running in-memory runs)
- `GET /api/v1/runs/{run_id}/status`
- `GET /api/v1/runs/{run_id}/diagnostics` (developer-focused run warnings, retry summaries, writer citation coverage, and run-local artifact labels without exposing host filesystem paths)
- `GET /api/v1/runs/{run_id}/output`
- `GET /api/v1/runs/{run_id}/export/docx` (Word export of `final.md`; inline `[ref_n]` citation tags are omitted from the exported document)
- `GET /api/v1/runs/{run_id}/export/writer-context` (developer-focused JSON download of the exact writer-safe context bundle, including the accepted excerpts used by the writer)
- `GET /api/v1/runs/{run_id}/export/writer-context.md` (Markdown variant of the writer-safe context export)
- `GET /api/v1/runs/{run_id}/context`
- `GET /api/v1/runs/{run_id}/references` (canonical citation endpoint; supports optional query params `ref_id` and `include_quote`)
- `GET /api/v1/runs/{run_id}/references/{ref_id}` (compatibility alias for one reference with quote payload)
- `GET /api/v1/cities` (city names from top-level markdown filenames in `MARKDOWN_DIR`, without `.md`)
- `GET /api/v1/cities/{city_name}/markdown` (concatenated raw CCC markdown for one normalized city name, including contributing source paths)
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
- Prompt budget defaults to `chat.max_context_total_tokens` from `llm_config.yaml` and switches to the overflow map-reduce path described below when direct chat would exceed the effective budget.
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
- It is also rebuilt when the cache schema changes, so older trim-based cache files are not reused by newer overflow logic.

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
  "schema_version": 2,
  "source_signature": "...",
  "evidence_count": 42,
  "chunks": [
    {
      "chunk_id": "chunk_1",
      "ref_ids": ["ref_1", "ref_2"],
      "token_count": 1234,
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

`token_count` stores the rendered prompt-token cost of each cached chunk. This lets split-mode decide quickly whether a chunk already fits the current map-pass budget.

### Map step

Once the compact evidence cache exists, the backend uses those cached chunks directly for the map step.

If a cached chunk is larger than the current map-pass budget because the request budget shrank further, the backend tries one half split of that chunk:

- left half of the chunk items
- right half of the chunk items

If both halves fit, they are used as two map blocks for that request.
If either half still does not fit, the overflow job fails instead of trimming `city_name`, `quote`, or `partial_answer`.

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

In the current implementation, that means:

- build and cache full evidence-item chunks once
- store per-chunk token counts in the cache
- reuse those chunks across overflowed turns
- split one oversized chunk in half once when a later request has a tighter budget
- fail clearly if even a half split cannot fit

Run API locally:

```bash
python -m uvicorn backend.api.main:app --reload
```

The API will start on `http://localhost:8000` with auto-reload enabled for development.

(Run from the project root, not from the backend directory.)

Run API in Docker:

```bash
docker build -f backend/Dockerfile -t query-mechanism-backend .
docker run -it --rm -p 8000:8000 \
  --env-file .env \
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
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_LOCAL_API_PORT=8000
NEXT_PUBLIC_FRONTEND_MODE=standard
APP_SHARED_PASSWORD_HASH=replace_with_bcrypt_hash_of_shared_password
APP_SESSION_SECRET=change_me_to_a_32_char_min_random_secret
APP_SESSION_COOKIE_DOMAIN=
APP_SESSION_TTL_SECONDS=604800
APP_LOGIN_RATE_LIMIT_MAX_ATTEMPTS=5
APP_LOGIN_RATE_LIMIT_WINDOW_SECONDS=900
```

`NEXT_PUBLIC_API_BASE_URL` should be set for deployed environments.
If it is omitted, the frontend falls back to a local backend URL built from `NEXT_PUBLIC_LOCAL_API_PORT`.
`APP_SHARED_PASSWORD_HASH` and `APP_SESSION_SECRET` are required for `npm run dev`.
Keep frontend and backend on the same host label locally: `localhost` with `localhost`, or `127.0.0.1` with `127.0.0.1`.

Frontend supports three city scope modes in the build form: all cities, predefined group, and manual selection.
Frontend also supports two answer modes: `Aggregate Mode` and `City-by-City Mode` (sent as `analysis_mode` in run requests).
Clicking `Chat About the Answer` opens a dedicated chat workspace and keeps the generated writer document available in the left rail for cross-reference.
When the rail is in `Writer Doc` mode on desktop, a drag handle between the rail and the main workspace lets users resize the document view without leaving chat.
Document and chat citations render as compact city labels; clicking a label loads and shows only the source quote.
When `chat.followup_search_enabled` is `true`, the chat router may run a synchronous one-city markdown-only follow-up search, attach the resulting follow-up bundle to the session, and keep citations clickable for both base runs and chat-owned follow-up bundles.
The follow-up router is intentionally lightweight: it sees a bounded payload with recent history plus compact context summaries, and each source summary may include only a capped subset of excerpts. This trim exists to keep the routing step cheap and stable; the router is only deciding whether the current context is clearly sufficient, whether a fresh one-city search is needed, whether the request is out of scope, or whether city clarification is required.
This means the router is deliberately conservative. If the summarized excerpts are not clearly enough to answer from context, it should prefer a fresh one-city search instead of assuming the answer is absent.
This trim applies only to the routing step. When chat actually answers from context, the answering path still uses the full loaded chat sources rather than the router's reduced summary payload.
Follow-up search stays conservative: it never launches a multi-city refresh, and failed follow-up searches return a limitation message instead of a guessed answer.
When chat needs a single city before searching, the backend sends clarification metadata and the frontend opens a city-picker popup that resubmits the original question with the selected city directly into one-city follow-up search.
When a direct chat prompt would overflow, the backend now falls back to an evidence-only map-reduce flow built from compact excerpt evidence and caches that stripped chat artifact under `output/<run_id>/chat_cache/evidence_chunks.json`.
The parent/base run stays pinned in chat context selection, manual run selections may exceed the direct prompt cap, and auto-added follow-up bundles are still trimmed first.
The `Load Previous Answer` picker reads `run_id`, `question`, and `picker_timestamp` from `GET /api/v1/runs`, renders rows as `MMDD-HHMM | question preview` inside a searchable popup list, filters runs by run id, date/time, question text, or city as you type, and then loads selected run artifacts through the standard run endpoints.
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
  --build-arg NEXT_PUBLIC_API_BASE_URL=https://urbind-query-mechanism-api.openearth.dev \
  -t query-mechanism-frontend ./frontend
docker run -it --rm -p 3000:3000 \
  -e APP_SHARED_PASSWORD_HASH=replace_with_bcrypt_hash \
  -e APP_SESSION_SECRET=change_me_to_a_32_char_min_secret \
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

Local no-Docker config split:

- root `.env`: backend settings, including `OPENROUTER_API_KEY`, `APP_SESSION_SECRET`, `APP_SHARED_PASSWORD_HASH`, and explicit `API_CORS_ORIGINS`
- `frontend/.env.local`: frontend runtime settings, including `APP_SHARED_PASSWORD_HASH`, the same `APP_SESSION_SECRET`, and `NEXT_PUBLIC_*`

This split is only for local no-Docker runs. Docker Compose reads the root `.env` and injects values into both containers. Deployed dev/prod uses GitHub Secrets and Kubernetes secrets instead of repo `.env` files.

Supported modes:

- Local: frontend at `http://localhost:3000`, backend at `http://localhost:8000`, `APP_SESSION_COOKIE_DOMAIN` unset.
- Dev deployment: frontend at `https://urbind-query-mechanism.openearth.dev`, backend at `https://urbind-query-mechanism-api.openearth.dev`, `APP_SESSION_COOKIE_DOMAIN=.openearth.dev`.

## Manual EKS deployment

For manual GHCR + EKS deployment without GitHub Actions, use the checked-in manifests in `k8s/`.
The manifests currently expect these image names:

- Backend: `ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:dev`
- Frontend: `ghcr.io/open-earth-foundation/query_mechanism_urbind-frontend:dev`

Build, push, create the backend secret, then apply the manifests:

```bash
docker build -f backend/Dockerfile -t ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:dev .
docker build -f frontend/Dockerfile \
  --build-arg NEXT_PUBLIC_API_BASE_URL=https://urbind-query-mechanism-api.openearth.dev \
  -t ghcr.io/open-earth-foundation/query_mechanism_urbind-frontend:dev ./frontend
docker push ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:dev
docker push ghcr.io/open-earth-foundation/query_mechanism_urbind-frontend:dev

kubectl create secret generic urbind-query-mechanism-backend-secrets \
  --from-literal=OPENROUTER_API_KEY=<openrouter-key> \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/backend-pvc.yml
kubectl apply -f k8s/backend-configmap.yml
kubectl apply -f k8s/backend-deployment.yml
kubectl apply -f k8s/backend-service.yml
kubectl apply -f k8s/frontend-deployment.yml
kubectl apply -f k8s/frontend-service.yml
```

Add `SERPER_API_KEY` and `FIRECRAWL_API_KEY` to the secret only when web research is enabled.

The backend ConfigMap sets `WRITER_ENABLED=true`, `VECTOR_STORE_ENABLED=true`, `VECTOR_STORE_AUTO_UPDATE_ON_RUN=false`, and `VECTOR_STORE_UPDATE_MODE=local_process` for the deployed environment. The backend still checks whether the vector store is stale, but it does not try to update it automatically in Kubernetes. When the store is stale, runs are blocked and the UI instructs the operator to run the maintenance workflow documented below.

## GitHub Actions deployment

Automated development workflow is available at `.github/workflows/develop.yml`.
It runs tests for PRs targeting `main` and for pushes to `main`; image build and EKS deploy run only on `main` branch runs (push/manual dispatch).
Dependency lockfile changes to `uv.lock` also trigger the test workflow so Dependabot lockfile PRs are verified.

CORS and auth notes

- Restrict `API_CORS_ORIGINS` in `k8s/backend-configmap.yml` to the deployed frontend origin(s) only, for example `https://urbind-query-mechanism.openearth.dev`.
- Remove local-only origins such as `http://localhost:3000`, `http://127.0.0.1:3000`, and private LAN hosts from the production ConfigMap.
- Backend startup now fails closed when shared-cookie auth is enabled without explicit `API_CORS_ORIGINS`.
- Keep localhost-friendly CORS settings only in local Docker/dev configuration.
- Ensure the deployed frontend sets `NEXT_PUBLIC_API_BASE_URL` explicitly so backend host mismatches are not confused with CORS failures.
- In local development, keep frontend and backend on the same host label (`localhost` with `localhost`, or `127.0.0.1` with `127.0.0.1`) so the host-scoped cookie reaches both services.

Required repository secrets:

- `AWS_ACCESS_KEY_ID_EKS_DEV_USER`
- `AWS_SECRET_ACCESS_KEY_EKS_DEV_USER`
- `EKS_DEV_NAME`
- `OPENROUTER_API_KEY`
- `APP_SHARED_PASSWORD_HASH`
- `APP_SESSION_SECRET`

Optional repository variables:

- `EKS_DEV_REGION` (default `us-east-1`)
- `FRONTEND_API_BASE_URL` (default `https://urbind-query-mechanism-api.openearth.dev`)

Artifacts are written under `output/<run_id>/`:

- `api_state.json`: machine-readable run metadata used for run discovery, terminal-state hydration, diagnostics, and benchmarks. It keeps status, timestamps, inputs, decisions, and compact metrics, while `manifest.json` remains the canonical artifact registry and the only artifact locator.
- `summary.jsonl`: append-only stage timeline. Each JSON line has an `event_index`, `event_type`, run id, timestamp, stable `stage_number`, and compact payload for one completed stage checkpoint. Fresh runs write `001_input_snapshot` before later numbered stage events. Detailed decisions are stored in `api_state.json` and, when they belong to an existing stage, in that stage's `stages/NNN_*.json` detail file.
- `manifest.json`: canonical run artifact registry. It records generated files and stable aliases such as `context_bundle`, `final_output`, `retrieval`, `markdown_excerpts`, `source_chunk_index`, `run_summary`, and `error_log`.
- `stages/NNN_<stage>.json`: numbered stage-detail artifacts with structured `inputs`, `outputs`, and `metrics` for query preparation, retrieval, markdown batching/extraction, enrichment, assumptions, context handoffs, writer, and finalize checkpoints.
- `stage_files/<numbered_stage>/...`: larger or reusable stage files. For example, `stage_files/005_markdown_batching/source_chunk_index.json` provides chunk id to source-path hints for downstream source lookup.
- `stage_files/001_input_snapshot/`: reproducibility snapshots for reruns and benchmark comparisons. Includes `execution_snapshot.json` (argv, cwd, config path, requested vs resolved run id, human-readable invocation command when available), `code_snapshot.json` (git commit, branch, dirty flag, changed files), `config_snapshot.json` (resolved full config plus file hash), `vector_store_snapshot.json` (resolved vector-store settings plus manifest summary/hash and, when auto-update runs, `auto_update` diagnostics with trigger, counts, changed files, deleted files, and chunk impacts), and `documents_snapshot.json` (compact corpus summary, full markdown file manifest, and stable snapshot hash).
- `run.log`: detailed runtime logs, including per-agent `LLM_USAGE` lines, chat prompt-window diagnostics (`Context chat reply plan`, `Context chat direct request`, with fitted source ids and token-component counts), retry reason lines (`RETRY_EVENT`/`RETRY_EXHAUSTED` with plain-text fields such as `reason`, `http_status`, `rate_limited`, and markdown split lineage when applicable), and writer city-citation coverage checkpoints (`WRITER_CITATION_COVERAGE`, with `coverage_ratio` such as `33/33`).
- `error_log.txt`: extracted error-focused log view from `run.log` (`ERROR`, `CRITICAL`, and exhausted retry events).
- `progress.json`: live progress state for API polling and frontend display. Steps keep user-facing labels, but also include canonical `stage_name` and `stage_number` fields so progress entries can be matched to `stages/NNN_<stage>.json`.
- `run_summary.txt`: compact human-readable run index. Header includes `Started`, `Completed`, and explicit `Total runtime` in seconds, plus `LLM Usage` totals/per-agent. It captures the input summary, artifact links, decisions, and a `MARKDOWN_FAILURE_SUMMARY` (`none` when no markdown failures occurred); large payloads stay in their canonical JSON/Markdown artifacts instead of being duplicated here.
- `context_bundle.json`: payload passed between agents (`markdown`, `original_question`, `research_question`, `query_mode`, `retrieval_queries`, `analysis_mode`, final path). When enrichment runs, `enrichment` carries evidence augmentation (`field_manifest`, `gap_manifest`, `enriched_fields`, external/web/freshness evidence), while top-level `assumptions` carries model estimates, non-estimable outputs, saturation warnings, and assumptions metadata.
- `stage_files/002_query_preparation/research_question.json`: run query metadata payload. Includes:
  - `original_question`: raw user question.
  - `query_mode`: `standard` or `dev`.
  - `canonical_research_query`: legacy field that mirrors the trimmed `original_question`.
  - `retrieval_queries`: retrieval-ready query list where index `0` is always the trimmed `original_question`.
  - `retrieval_query_1` / `retrieval_query_2` / `retrieval_query_3`: explicit query slots written for easier inspection and reproducibility.
- `system/vector_store_warmup/latest.json`: latest API startup vector-store warm-up diagnostics, written outside user run folders under `output/system/`. It records trigger, status, timing, compact stats, changed/deleted file details, and the post-update vector-store snapshot; timestamped history is kept beside it as `system/vector_store_warmup/<timestamp>.json`.
- `stage_files/006_markdown_extraction/accepted_excerpts.json`: accepted markdown evidence bundle. Includes `excerpts` (items with `ref_id`, `quote`, `city_name`, `city_key`, `partial_answer`, and `source_chunk_ids`) and `excerpt_count` (count of accepted excerpts). Accepted excerpt entries may also include a stage-only `retrieval` block with source-chunk cosine-distance diagnostics; these diagnostics are intentionally not persisted into `context_bundle.json`. Run-level city scope lives on the root `context_bundle.json` and in stage inputs/outputs instead of inside the markdown-only payload. Stage B extraction recall uses the union of `excerpts[].source_chunk_ids`, and `/references` API responses are derived from this artifact.
- `stage_files/006_markdown_extraction/city_summary.json`: city-level extraction observability artifact. Includes per-city batch counts, chunk decision counts, excerpt counts, status/error rollups, and top-level `cities_with_excerpts` / `cities_without_excerpts` / `cities_with_failures` lists for quick verification.
- `stage_files/006_markdown_extraction/rejected_chunks.json`: rejected markdown decision artifact with rejected chunk IDs, rejected-per-city grouping, status, counts, and lean per-chunk retrieval diagnostics (`distance`, `selection_mode`, optional `seed_rank`). Full chunk text for rejected IDs is still available in `stage_files/003_retrieval/retrieval.json` when vector retrieval is enabled.
- `stage_files/006_markdown_extraction/decision_audit.json`: run-level reconciliation counters and diagnostics (`retrieved_total`, accepted/rejected/unresolved totals, invariant status, mismatch details, and accepted-vs-rejected retrieval summaries such as cosine-distance distributions and selection-mode counts).
- `stage_files/003_retrieval/retrieval.json` (when `VECTOR_STORE_ENABLED=true`): vector retrieval inputs and results summary. Includes the final retrieval query list, optional city filter, retrieval tuning metadata (cutoffs/caps), strict direct-hit `seed_chunks[]`, final delivered `chunks[]`, `meta.seed_retrieved_total_chunks`, `meta.neighbor_expanded_total_chunks`, and per-chunk summaries (`chunk_id`, `chunk_text`, `chunk_index`, `city_name`, `city_key`, `source_path`, `heading_path`, `block_type`, `distance`, `provenance`).
- `stage_files/005_markdown_batching/batches.json`: markdown batching plan used for the markdown researcher calls. Includes per-city batch indices, estimated tokens, and chunk ordering fields (`path`, `chunk_index`, `chunk_id`), making it easy to inspect how chunks were grouped into LLM requests.
- `stage_files/007_markdown_context_handoff/context_bundle_after_markdown.json`: immutable full context snapshot after the markdown pipeline finishes.
- `stage_files/008_enrichment/enrichment_bundle.json`: canonical full enrichment payload. Includes `field_manifest`, `gap_manifest`, `enriched_fields`, web/external/freshness evidence, and enrichment metadata. Duplicate projection files such as standalone field/gap manifests are not written.
- `stage_files/008_enrichment/external_source_search_audit.json` (when external source search runs): source-search trace with searched city-fields, candidates, validated/rejected claims, no-evidence records, resolutions, tool calls, and metrics.
- `stage_files/008_enrichment/web_research_audit.json` (when web research has trace outputs): web-search trace fields that are not already represented as canonical enrichment payload data, such as search batches and benchmark findings.
- `stage_files/009_enrichment_context_handoff/context_bundle_after_enrichment.json`: immutable full context snapshot after enrichment completes.
- `stage_files/010_assumptions/assumptions_bundle.json`: assumptions-only payload with model estimates, non-estimable records, saturation warning, and assumptions metadata.
- `stage_files/010_assumptions/assumptions_stage.json`: assumptions stage detail support artifact with flags, outputs, and metrics.
- `stage_files/011_assumptions_context_handoff/context_bundle_after_assumptions.json`: immutable full context snapshot after assumptions complete.
- `final.md`: final delivered markdown output. Content format is:
  1. `# Question` heading with the original user question,
  2. generated markdown answer body from the writer.

`stage_files/006_markdown_extraction/accepted_excerpts.json` excerpt entries include:

- `quote`: verbatim extracted supporting text from markdown.
- `city_name`: city identifier for the excerpt.
- `city_key`: normalized backend city key for the excerpt.
- `partial_answer`: concise fact grounded in the quote.
- `source_chunk_ids`: chunk ids backing the excerpt, used for Stage B extraction recall and citation tracing.
- `ref_id`: sequential run-local citation id (`ref_1`, `ref_2`, ...), used by writer output and frontend reference lookups.
- `excerpt_count` (bundle-level): number of accepted excerpts included in the bundle.
- Decision bookkeeping such as accepted/rejected/unresolved totals, invariant status, and `batch_failures` lives in `decision_audit.json`.

- `chat/<conversation_id>.json` (created when context chat sessions are used)
- `stage_files/assumptions/discovered.json` (two-pass extraction output; only when `persist_artifacts=true`)
- `stage_files/assumptions/edited.json` (user-edited assumptions payload; only when `persist_artifacts=true`)
- `stage_files/assumptions/revised_context_bundle.json` (context + assumptions merge; only when `persist_artifacts=true`)
- `stage_files/assumptions/final_with_assumptions.md` (regenerated document; only when `persist_artifacts=true`)

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

Frontend build verification:

```
cd frontend
npm ci
npm run build
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
python -m backend.scripts.calculate_tokens --documents-dir documents
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
python -m backend.scripts.update_vector_store --docs-dir documents
```

The update now fails fast on embedding failures and exits non-zero before any delete/upsert/manifest-write commit.
It also forces a full rebuild when index-shaping vector-store settings change, so maintenance rebuilds do not silently keep vectors built with stale chunking or embedding settings.

What triggers an incremental refresh:

- A top-level markdown file under `documents/*.md` is added.
- A top-level markdown file under `documents/*.md` is removed.
- The SHA-256 file hash of a top-level markdown file changes, for example because you edited text or changed a number in the file.

What triggers a full rebuild instead of an incremental refresh:

- The manifest is missing the persisted index-settings metadata needed to prove the current vectors are still compatible.
- `vector_store.embedding_model` changes.
- `vector_store.embedding_max_input_tokens` changes.
- `vector_store.embedding_chunk_tokens` changes.
- `vector_store.embedding_chunk_overlap_tokens` changes.
- `vector_store.table_row_group_max_rows` changes.

Vector-store freshness behavior:

- It only scans top-level `documents/*.md` files.
- Any content edit causes the entire changed file to be rechunked and re-embedded.
- Config-driven rebuild detection covers index-shaping settings such as embedding model, embedding input limit, chunk size, chunk overlap, and table row grouping. Pure code changes in the chunking/indexing implementation still require an intentional rebuild if the manifest metadata alone cannot detect them.
- Vector-backed runs always perform full-corpus freshness checks against the shared manifest, even when retrieval itself is limited to selected cities such as `--city Munich`.
- With `VECTOR_STORE_AUTO_UPDATE_ON_RUN=true`, the API and local pipeline runs can perform the full-corpus update in-process. This is appropriate for local development.
- With `VECTOR_STORE_AUTO_UPDATE_ON_RUN=false`, the API and local pipeline runs still perform dry-run freshness checks before vector-backed runs. If the index is stale, new runs are blocked and the frontend shows a compact maintenance banner with the manual refresh command. `/healthz` still reports the pod as healthy.
- Shared status is written next to the vector index as `update_status.json`. Startup/run diagnostics are also persisted under `output/system/vector_store_warmup/` as both `latest.json` and timestamped history files.
- Kubernetes deployments should keep a single backend replica or add a stronger distributed lock before multiple replicas can check or rebuild the same Chroma path.

Check manifest and Chroma DB status:

```
python -m backend.scripts.check_vector_index
python -m backend.scripts.check_vector_index --no-show-files
```

**Updating the vector index on Kubernetes:** The backend PVC is `ReadWriteOnce`, so the deployed maintenance flow does not use automatic updater Jobs. Instead, scale the backend deployment down, run the manual rebuild Job on the freed PVC, then scale the backend back up. Paths on the PVC are `/data/output` (run artifacts), `/data/chroma` (vector index and manifest), and `/data/chroma/update_status.json` (shared update status).

For a manual rebuild or recovery run, apply `k8s/backend-build-vector-index-job.yml`. That Job runs the same `python -m backend.scripts.update_vector_store` entrypoint used by API-triggered and local maintenance flows. The checked-in manifest is pinned to `:dev`; if you want exact parity with the currently deployed backend image, replace that image tag before applying the Job.

For the maintenance flow, you can use the one-off helper script from the repo root instead of running each `kubectl` step manually:

```bash
bash scripts/update_vector_store_maintenance.sh
```

You can also run it from inside the `scripts/` directory with:

```bash
bash update_vector_store_maintenance.sh
```

The script resolves the repo root automatically, scales `deployment/urbind-query-mechanism-backend` to `0`, deletes any previous `urbind-query-mechanism-build-vector-index` Job, applies `k8s/backend-build-vector-index-job.yml`, prints live Job/Pod status while waiting, fails early on stuck `Pending` / `ContainerCreating` pods, surfaces `FailedAttachVolume` clearly, then scales the backend back to `1` and waits for rollout readiness. Override namespace, deployment, job name, replicas, polling, or timeouts with flags such as `--namespace`, `--deployment`, `--job-name`, `--replicas`, `--status-poll`, `--pending-timeout`, `--job-timeout`, and `--rollout-timeout`.

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
