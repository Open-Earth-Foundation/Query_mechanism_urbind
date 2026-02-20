## Implementation Future (High-ROI): Retrieval Accuracy + Runtime Performance

This document captures the **highest-leverage** improvements to reach a more production-ready retrieval pipeline (focused on **accuracy and performance**, not security).

It is written against the current implementation:

- Vector retrieval path: `backend/modules/vector_store/retriever.py` + `backend/modules/orchestrator/module.py::_run_initial_markdown()`
- Markdown extraction: `backend/modules/markdown_researcher/agent.py::extract_markdown_excerpts()`
- Run artifacts: `output/<run_id>/markdown/retrieval.json`, `run.json`, `run_summary.txt`

---

## Context: what works today (baseline)

- **Vector store retrieval is integrated** behind `VECTOR_STORE_ENABLED`.
- The orchestrator:
  - refines the research question, collects up to **3 retrieval queries**
  - (optional) auto-updates the index on run
  - retrieves chunks via `retrieve_chunks_for_queries(...)`
  - writes `markdown/retrieval.json`
  - passes retrieved chunks to the markdown researcher (batched by city)

---

## Status: `implementation-plan-retriever.md` (Iteration 2) vs current code

This repo has implemented the core “Retriever + orchestrator integration” plan.

- **A) Config additions**: **Done**
  - `backend/utils/config.py::VectorStoreConfig` includes: fallback min, per-city/query max, per-city max, max distance, context windows, `auto_update_on_run`, manifest path.
  - Env var wiring exists for all planned knobs in `backend/utils/config.py::load_config(...)`.
  - `.env.example` and `README.md` document the env vars and tuning behavior.
- **B) Retriever module (`backend/modules/vector_store/retriever.py`)**: **Done**
  - Query embedding is computed in code, and Chroma is queried by embedding.
  - Adapter exists: `as_markdown_documents(...)`.
- **C) Chroma query-by-embedding API**: **Done**
  - `backend/modules/vector_store/chroma_store.py::query_by_embedding(...)` exists and is used by the retriever.
- **D) Orchestrator integration (flag-gated)**: **Done**
  - `backend/modules/orchestrator/module.py::_run_initial_markdown()` switches between disk chunking vs vector retrieval behind `config.vector_store.enabled`.
  - Optional `auto_update_on_run` triggers `update_markdown_index(...)` before retrieval.
- **E) Run artifacts & observability**: **Mostly done**
  - `output/<run_id>/markdown/retrieval.json` is written and recorded as an artifact when vector retrieval is enabled.
  - Run metadata records `markdown_source_mode` (`standard_chunking` vs `vector_store_retrieval`).
  - **Still missing (nice-to-have)**: surface retrieval knob values (fallback/min/max/cutoffs) directly in `run.json` inputs (some of this is present inside `markdown/retrieval.json` already).
- **Testing**: **Done**
  - `tests/test_vector_store_retriever.py` covers adapter + distance cutoff/top-up + neighbor expansion behavior.
  - `tests/test_orchestrator.py` covers the orchestrator toggle path and verifies `retrieval.json` creation.

---

## Prerequisite: build the vector store properly for *all cities*

To keep retrieval accuracy stable (and to keep performance predictable), the vector store must be **complete and consistent** across the entire markdown corpus.

**What “properly created” means**

- Every city markdown file in `MARKDOWN_DIR` is indexed into the configured Chroma collection.
- The manifest (default `.chroma/index_manifest.json`) accurately tracks which chunk IDs belong to which file hash.
- A full build/update cycle produces stable counts (no silent drops, no partial indexing).

**Recommended workflow**

- One-time full build (all cities):
  - `python -m backend.scripts.build_markdown_index --docs-dir documents`
- Incremental updates as documents change:
  - `python -m backend.scripts.update_markdown_index --docs-dir documents`
  - optionally enable `VECTOR_STORE_AUTO_UPDATE_ON_RUN=true` for local iteration
- Spot-check indexing sanity:
  - `python -m backend.scripts.inspect_markdown_index --city Munich --limit 20`

**Acceptance criteria**

- The Chroma collection is non-empty and contains entries for all expected cities.
- The manifest includes every indexed file, and deletions/renames are reflected after update.
- A representative retrieval run produces a reasonable `output/<run_id>/markdown/retrieval.json` size (no “everything everywhere” explosions) and shows plausible distances.

---

## Top priorities (big gains)

### 1) Global city gating (avoid “query every city”)

**Problem**

When `selected_cities` is not provided, the retriever resolves cities by scanning markdown files and then runs retrieval **for each city × each query**. This creates:

- **Latency blowups**: too many Chroma queries (and neighbor expansions).
- **Lower accuracy**: “forced coverage” across irrelevant cities adds noise and distracts the markdown agent.

**Proposed change**

Add a two-stage retrieval strategy:

- **Stage A (global pass)**: query the collection *without* city filter and retrieve a moderate candidate set (e.g. 200–500).
  - Derive `top_cities` from the distribution of best-scoring hits.
- **Stage B (focused pass)**: run the existing city-scoped retrieval only for `top_cities` (e.g. 5–15).

**Config knobs (suggested)**

- `VECTOR_STORE_RETRIEVAL_TOP_CITIES`: max cities selected by gating (default: 10)
- `VECTOR_STORE_RETRIEVAL_GLOBAL_CANDIDATES`: global candidates retrieved before gating (default: 300)

**Acceptance criteria**

- For runs without `selected_cities`, retrieval issues **O(top_cities × queries)** queries, not **O(all_cities × queries)**.
- `markdown/retrieval.json` records `gated_top_cities` and the gating method.
- Wall time for retrieval decreases significantly on a corpus with many cities.

---

### 2) Fix neighbor expansion (remove per-neighbor Chroma `get()` calls)

**Problem**

`_expand_neighbors()` currently issues Chroma `get()` requests per neighbor index per seed row. This is a classic “N×window×network/database call” pattern and becomes a dominant cost as soon as results scale.

**Proposed change (high ROI)**

Switch neighbor expansion to **batched fetches**:

- Group seeds by `(city_name, source_path)`.
- For each group, fetch all candidate rows needed for neighbor indices in **one** query (or a small bounded number of queries).
  - Example: compute the set of required `chunk_index` values and fetch them with a single `where` that matches the file + city and then filter indices in memory.
- Keep behavior identical: add neighbors with the same distance as the seed (current behavior).

**Config knobs**

No new knobs required; reuse:

- `VECTOR_STORE_CONTEXT_WINDOW_CHUNKS`
- `VECTOR_STORE_TABLE_CONTEXT_WINDOW_CHUNKS`

**Acceptance criteria**

- Neighbor expansion performs **bounded** Chroma calls per file group (not per neighbor).
- Retrieval latency becomes stable as chunk count grows.

---

### 3) Add reranking (largest accuracy jump after embeddings)

**Problem**

Final ordering today is distance-only (plus optional max distance + per-city caps). This often fails on:

- “needle-in-haystack” facts (numbers, IDs, specific terms)
- tables where semantic similarity is insufficient
- multi-facet questions where several near-matches exist

**Proposed change**

Add an optional reranking stage:

- Retrieve a broader candidate set (e.g. 100–300 globally or per gated city).
- Rerank candidates against the user question (or refined question) and keep the best N (e.g. 20–60).

Two practical implementation options:

- **Lightweight model rerank**: a smaller LLM scores (query, chunk) pairs (batched), returning a relevance score.
- **Cross-encoder rerank**: if you later adopt a local reranker model, this can be cheaper and deterministic.

**Config knobs (suggested)**

- `VECTOR_STORE_RERANK_ENABLED` (default: false)
- `VECTOR_STORE_RERANK_MAX_CANDIDATES` (default: 200)
- `VECTOR_STORE_RERANK_KEEP` (default: 40)
- `VECTOR_STORE_RERANK_MODEL` (if using LLM rerank)

**Acceptance criteria**

- On a small evaluation set, reranking improves retrieval quality metrics (see “Evaluation harness” below).
- Downstream markdown extraction sees fewer, higher-quality chunks (reducing LLM calls/cost).

---

### 4) Hybrid retrieval (lexical + vector) for better recall

**Problem**

Pure embedding similarity misses exact-match signals (especially for tables and numeric facts).

**Proposed change**

Add a hybrid candidate generator:

- Combine vector retrieval candidates with lexical candidates (BM25/keyword search over `raw_text` or `embedding_text`).
- Merge + dedupe by `chunk_id`.
- Optionally rerank the combined set (preferred).

**Implementation note**

Start minimal: lexical search can run over an in-memory index or a lightweight on-disk index built alongside the manifest (avoid large new dependencies unless needed).

**Acceptance criteria**

- Improved recall on exact-term queries (numbers, codes, named entities).
- Stable latency (lexical index must be bounded and cacheable).

---

### 5) Remove filesystem scan dependency from vector retrieval

**Problem**

The retriever currently uses `docs_dir.rglob("*.md")` to enumerate cities when `selected_cities` is missing. This adds avoidable I/O and couples retrieval behavior to filesystem state rather than index state.

**Proposed change**

Use the index’s canonical sources:

- Prefer **manifest** (`.chroma/index_manifest.json`) to list indexed cities/files.
- Optionally cross-check collection metadata counts (diagnostics only).

**Acceptance criteria**

- Retrieval runs without scanning the documents folder when `VECTOR_STORE_ENABLED=true`.
- If manifest is missing or stale, fail fast with a clear error (unless auto-update is enabled).

---

### 6) Batch query embeddings and reuse embedding provider

**Problem**

Query embedding currently creates a new client per query and embeds one query at a time. The indexer already has a batching provider (`OpenAIEmbeddingProvider.embed_texts`).

**Proposed change**

- Reuse a single embedding provider instance in the retriever.
- Embed all normalized retrieval queries in one batch call.

**Acceptance criteria**

- Query embedding time scales with number of queries sublinearly (batched).
- Fewer outbound embedding calls and lower latency variance.

---

### 7) Add a global retrieval budget (cap total chunks / total tokens)

**Problem**

Even with per-city caps, the combination of:

- fallback minimum chunks per city/query
- multiple queries (up to 3)
- neighbor expansion

can create a large chunk set, increasing markdown batching and LLM costs.

**Proposed change**

Add a **global cap** and deterministic selection policy:

- `retrieval_max_total_chunks` (hard cap)
- Optionally `retrieval_max_total_tokens` (budget-based cap)
- Selection policy example:
  - sort by (score/distance) and keep best
  - optionally enforce a minimum diversity across cities/files

**Acceptance criteria**

- Total chunks sent to markdown agent is bounded and logged.
- Markdown stage latency and cost become predictable.

---

### 8) Retrieval evaluation harness (make accuracy improvements measurable)

**Problem**

There is no repeatable metric loop to validate retrieval changes (rerank/hybrid/gating) beyond manual inspection of `retrieval.json`.

**Proposed change**

Add a small offline evaluation harness:

- A tiny “gold set” fixture: question → expected city/file/heading (or expected excerpt quote substring).
- Metrics:
  - Recall@K (does a relevant chunk appear in top K?)
  - MRR / rank of first relevant hit
  - Optional: city-level accuracy (is correct city in top gated cities?)

**Acceptance criteria**

- CI-friendly tests can catch regressions (fast, deterministic; no network calls).
- Each retrieval feature (gating, rerank, hybrid) shows measurable improvement or stays neutral.

---

## Suggested execution order (fastest ROI first)

1. **Neighbor expansion batching** (speed win, low risk, behavior-preserving).
2. **Global city gating** (speed + quality win, moderate change).
3. **Global retrieval budget** (predictability win, moderate change).
4. **Reranking** (biggest accuracy win, needs careful cost control).
5. **Hybrid retrieval** (accuracy/recall win, may require extra indexing work).
6. **Evaluation harness** (enables safe iteration across all of the above).

---

## Notes on artifacts & observability (keep runs inspectable)

When adding the above, extend `output/<run_id>/markdown/retrieval.json` to include:

- `mode`: `vector_store_retrieval`
- `gating`: input/output and chosen top cities/files (if enabled)
- `rerank`: configuration and before/after rank summary (if enabled)
- `budgets`: configured caps + applied totals

This keeps behavior transparent and makes it easy to debug quality issues from run outputs alone.

