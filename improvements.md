# Retrieval Improvement Notes

## Changes Already Made On This Branch

This branch already made several vector-store and artifact-observability changes before the remaining retrieval-quality analysis below.

### Vector Store Stability And Rebuild Semantics

- Normalized vector-store manifest source paths to stable `documents/<city>.md` style paths so local, Docker / Docker Compose, and Kubernetes mount paths do not cause false staleness.
- Kept selected-city runs from changing persisted vector-store update scope. Selected cities now scope retrieval, not the shared full-corpus vector-store freshness check or persisted update.
- Added protection against overwriting a non-empty manifest with an empty rebuild when no markdown files are discovered.
- Improved update-status recovery so stale/interrupted `.chroma/update_status.json` states do not leave the store permanently marked as running.
- Kept startup vector-store warmup and active-run vector-store readiness aligned around the same full-corpus staleness semantics.

### Cosine Distance Migration

- Switched Chroma collection creation to cosine distance space.
- Added `distance_metric: "cosine_distance"` to persisted vector-store index settings so older L2-backed indexes trigger the existing full-rebuild path once.
- Added runtime validation so an existing non-cosine collection fails loudly instead of silently producing incomparable distances.
- Updated retrieval artifacts and stage metrics to label the metric as `cosine_distance`.

### Retrieval And Markdown Artifact Logging

- Split misleading retrieval counts into explicit fields:
  - `distance_qualified_total_chunks`
  - `fallback_top_up_total_chunks`
  - `neighbor_expanded_total_chunks`
- Replaced ambiguous per-query `qualified_chunks` with:
  - `distance_qualified_chunks`
  - `fallback_top_up_chunks`
  - `seed_chunks_selected`
- Added cosine-distance min / p50 / p95 / max metrics to retrieval stage summaries.
- Added accepted-vs-rejected cosine-distance summaries and selection-mode counts to `stage_files/006_markdown_extraction/decision_audit.json`.
- Kept rejected-only details in `rejected_chunks.json`, including `content`, `source_chunk_ids`, chunk metadata, and lean retrieval diagnostics.
- Added lean retrieval diagnostics to `accepted_excerpts.json` while keeping those diagnostics out of `context_bundle.json` and writer context.
- Removed misleading per-excerpt distance percentiles from `accepted_excerpts.json`.
- Aligned accepted and rejected artifact shape around `source_chunk_ids` plus `retrieval.source_chunks[]`.

### Documentation And Tests

- Updated README sections that describe cosine distance, retrieval metrics, accepted excerpts, rejected chunks, and decision-audit artifacts.
- Added focused regression coverage for cosine index settings, retrieval metadata naming, artifact diagnostics, and context-bundle non-propagation.
- Validated the focused vector-store and orchestrator test slices during implementation.

## Scope

This note summarizes retrieval-quality issues observed in these fresh run artifacts:

- `output/01_retrieval_ev_charging_8cities_01`
- `output/retrieval_fleet_electrification_8cities_02`
- `output/retrieval_building_retrofit_8cities_02`

It focuses on why Stage 003 retrieval still sends many weak chunks to Stage 006, and what changes are most likely to reduce irrelevant chunks without losing the good evidence.

## Current Issues

### 1. Accepted And Rejected Distances Are Too Close

Cosine distance gives a slightly better distribution for accepted chunks, but it is not a clean separator.

| Run | Retrieved chunks | Accepted chunk decisions | Rejected chunk decisions | Accepted distance p50 | Rejected distance p50 | Accepted p95 | Rejected p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EV charging | 590 | 60 | 530 | 0.549871 | 0.588492 | 0.603729 | 0.618663 |
| Fleet electrification | 593 | 75 | 518 | 0.547021 | 0.582392 | 0.617402 | 0.628151 |
| Building retrofit | 634 | 117 | 517 | 0.521858 | 0.556082 | 0.569545 | 0.589593 |

Observed pattern:

- Accepted chunks are usually closer on average.
- Rejected chunks still overlap heavily with accepted chunks.
- Distance alone is not enough to decide whether a chunk answers the question.

### 2. Retrieval Pool Is Too Broad

All three runs returned the maximum direct seed pool:

| Run | Seed chunks | Distance-qualified | Fallback top-up | Neighbor chunks |
| --- | ---: | ---: | ---: | ---: |
| EV charging | 480 | 480 | 0 | 110 |
| Fleet electrification | 480 | 480 | 0 | 113 |
| Building retrofit | 480 | 480 | 0 | 154 |

Interpretation:

- `retrieval_max_distance = 1.0` is permissive under cosine distance.
- `retrieval_max_chunks_per_city_query = 60` is effectively the active limiter.
- With 8 cities and 1 query, the system retrieves 8 x 60 = 480 direct chunks before neighbor expansion.
- The markdown extractor then has to reject most of the pool.

### 3. Markdown Extractor May Still Be Somewhat Permissive

Most chunks are rejected, so the extractor is not blindly accepting everything. However, some accepted excerpts are still topic-relevant rather than answer-bearing.

Example issue:

- Question asks for concrete EV charging initiatives, targets, budgets, and timelines.
- A chunk saying only "accelerated expansion of e-charging infrastructure" is topically relevant.
- It is not strong evidence for targets, budgets, timelines, or quantified commitments unless the chunk also states those details.

Interpretation:

- Retrieval is the bigger volume problem.
- Markdown acceptance criteria can still be tightened for "explicitly stated" and "concrete" questions.

### 4. Current Query Expansion Is Manual And Not Exposed In Local CLI

The pipeline already supports up to three retrieval queries internally:

- `query_1`: the original question
- `query_2`: optional direct retrieval query
- `query_3`: optional direct retrieval query

Current code paths:

- `backend.modules.orchestrator.module.run_pipeline(...)` accepts `query_2` and `query_3`.
- API request models expose `query_2` and `query_3`.
- benchmark overrides support fixed `retrieval_queries`.
- `backend.scripts.run_pipeline` currently does not expose `--query-2` or `--query-3`.

This means local CLI testing cannot easily use query expansion without changing the CLI or calling a lower-level path.

## Possible Improvements

### Option 1. Improve Query Specificity

The goal is to retrieve answer-shaped evidence, not just topical neighbors. There are two useful strategies.

### Option 1A. Single Enriched Query With Adaptive Focus Terms

Instead of adding more retrieval queries, keep one query per city but enrich the original query with adaptive focus terms.

Example:

```text
Question: What concrete EV charging initiatives, targets, budgets, and timelines are explicitly stated?
Retrieval focus: EV charging infrastructure; charging points; public chargers; targets; budgets; funding; timelines; milestones; number of chargers; implementation year.
```

This keeps the retrieval volume stable:

- 1 query x 8 cities x `retrieval_max_chunks_per_city_query`
- no automatic 2x or 3x expansion
- same retrieval pool size, but a more specific embedding

Potential focus terms for the current questions:

- `EV charging infrastructure targets budgets timelines deployment milestones number of chargers funding`
- `charging points public chargers fast chargers rollout by year budget EUR investment target KPI`
- `building retrofit renovation targets energy savings budgets timelines municipal buildings`
- `energy efficiency renovation rate CO2 reduction building upgrades funding target by year`
- `municipal fleet electrification commitments electric buses vehicles procurement milestones funding`
- `fleet replacement target electric vehicles by year budget charging depot infrastructure`

Important design constraint:

- Focus terms must adapt to the user question.
- A fixed global keyword list would help these three tests but may hurt unrelated questions.
- The focus terms should reflect both the topic and the requested evidence type, for example budgets, dates, counts, timelines, targets, or milestones.

Pros:

- Keeps retrieved chunk volume stable.
- Easy to test against the current artifact metrics.
- Likely helps these benchmark-style questions because they ask for concrete answer shapes.
- Avoids multiplying weak semantic matches across multiple query embeddings.

Cons:

- Keyword stuffing can distort the embedding if the focus text is too long.
- It nudges retrieval but cannot enforce that chunks contain numbers, budgets, or dates.
- It is less transparent than separate queries because there is only one combined distance score.
- Too-specific terms can reduce recall when cities use different wording.

Recommended implementation:

- Add an optional `retrieval_focus` / `retrieval_keywords` input and include it in query preparation.
- Keep the original question unchanged for the markdown extractor and writer.
- Build the actual retrieval query as original question plus the focus terms.
- Persist both the original question and the enriched retrieval query in `stage_files/002_query_preparation/research_question.json`.

### Option 1B. Multiple Retrieval Queries

Use query 2 and query 3 to run additional direct retrieval searches.

Example:

- query 1: original question
- query 2: answer-shape query focused on targets, budgets, dates, counts
- query 3: domain-specific query focused on the requested technology or policy area

Pros:

- Better recall when relevant chunks use very different wording.
- Existing orchestrator and API already support `query_2` and `query_3`.
- Benchmark overrides already use fixed optional retrieval queries.

Cons:

- More queries increase the retrieved pool unless per-query limits are reduced.
- With current settings, 3 queries could retrieve up to 3 x 480 direct city-query hits before dedupe.
- It may increase Stage 006 cost and noise unless paired with lower caps, prefiltering, or reranking.

Recommended implementation:

- Add CLI flags `--query-2` and `--query-3` to `backend.scripts.run_pipeline`.
- Keep query 1 as the original question.
- Use manual query 2 and query 3 first for controlled experiments.
- Add an optional lightweight query-expansion helper later that derives query 2 and query 3 from the question when automatic expansion is desired.
- Persist all generated/manual queries in `stage_files/002_query_preparation/research_question.json` as today.

Recommended order:

- Test Option 1A first because it keeps volume stable.
- Use Option 1B when recall is still weak after query enrichment.
- Pair Option 1B with lower per-query caps or reranking.

### Option 2. Reduce The Initial Pool

Lower the amount of material sent to Stage 006.

Candidate settings to test:

- `retrieval_max_chunks_per_city_query`: reduce from `60` to `30` or `40`.
- `retrieval_max_distance`: reduce from `1.0` to a more realistic cosine cutoff, for example around `0.60` for the current three runs.
- `table_context_window_chunks`: consider whether table neighbor expansion is adding useful context or mostly extra rejection load.

Why this helps:

- The current setup always retrieves 480 direct chunks across 8 cities.
- Reducing the pool immediately lowers Stage 006 cost and noise.

Risk:

- A blunt cap may drop useful evidence, especially for cities whose relevant chunks rank lower.
- Distance overlap means a stricter threshold will help only partially.

Recommended test:

- First try a single enriched query with the current pool size.
- Then try the same enriched query with `retrieval_max_chunks_per_city_query = 30` or `40`.
- Compare accepted count, rejected count, and final answer quality.
- Only then tune `retrieval_max_distance`.

### Option 3. Add A Deterministic Answer-Shape Prefilter

After vector retrieval and before markdown extraction, apply a cheap filter that checks whether chunks contain signs matching the question's expected evidence type.

For the current benchmark-style questions, useful signals are:

- numbers
- years
- percentages
- currency / budget terms
- target / commitment / milestone / timeline words
- domain terms such as charging, retrofit, fleet, electric buses, renovation

The filter should be adaptive:

- Parse the user question for requested evidence types.
- If the question asks for budgets, targets, timelines, quantified savings, or milestones, require at least one matching concreteness signal.
- If the question is qualitative, do not require numeric signals.

Why this helps:

- The Leipzig "accelerated expansion" example is semantically relevant but weakly answer-bearing.
- A concreteness prefilter could keep it only if the same chunk also contains a target, date, budget, count, or similar requested detail.

Risk:

- Over-filtering can remove evidence stated without numbers.
- The filter should be conservative and observable, with rejected-by-prefilter counts and sample reasons.

### Option 4. Add A Reranking Step

Retrieve broadly, then rerank chunks against the question before Stage 006.

Possible rerankers:

- heuristic reranker using keyword and answer-shape features
- embedding reranker using multiple query embeddings and weighted scores
- LLM reranker for top N chunks per city
- dedicated cross-encoder reranker if adding a dependency or hosted model is acceptable

Recommended first version:

- Build a simple heuristic score:
  - cosine distance score
  - domain term match
  - requested evidence-type match
  - number/year/currency/percentage presence
  - heading relevance
- Keep top K chunks per city after scoring.
- Log the score components in Stage 003 for inspection.

Why this helps:

- Raw vector distance measures topic closeness.
- Reranking can measure whether the chunk has the answer form the question asks for.

Risk:

- Heuristic scoring needs careful logging to avoid becoming opaque.
- A model-based reranker adds latency and cost.

### Option 5. Tighten Markdown Acceptance Criteria

Stage 006 should reject chunks that are merely topical when the question asks for concrete details.

Acceptance rule for these questions:

- Accept only if the quote directly supports at least one requested evidence type.
- For "targets, budgets, timelines, quantified savings, milestones", the quote should include the concrete target/budget/timeline/saving/milestone or a clear commitment.
- Generic program descriptions should be rejected as insufficient unless they include the requested concrete detail.

Why this helps:

- It improves precision of accepted excerpts.
- It makes final answers less likely to cite vague initiative language.

Risk:

- It does not reduce Stage 006 input volume by itself.
- It should be paired with retrieval improvements.

## Recommended Path

1. Move run-level accepted-vs-rejected retrieval summaries to `decision_audit.json`.
2. Add a single-query focus-term path so local runs can test one enriched retrieval query without increasing retrieved chunk volume.
3. Run the same three questions with adaptive focus terms and the current `retrieval_max_chunks_per_city_query = 60`.
4. Reduce `retrieval_max_chunks_per_city_query` from `60` to `30` or `40` if the enriched query keeps recall stable.
5. Add `--query-2` and `--query-3` to the local CLI only when broader multi-query recall testing is needed.
6. Add a deterministic answer-shape prefilter if enriched queries plus smaller pools still send too many weak chunks.
7. Tighten markdown acceptance criteria after retrieval volume is under better control.

## Suggested Next Experiment

Run EV charging with one enriched retrieval query:

```bash
python -m backend.scripts.run_pipeline \
  --run-id retrieval_ev_charging_8cities_enriched_query_test \
  --question "What concrete EV charging initiatives, targets, budgets, and timelines are explicitly stated?" \
  --retrieval-focus "EV charging infrastructure; charging points; public chargers; targets; budgets; funding; timelines; milestones; number of chargers; implementation year" \
  --city Aachen --city Amsterdam --city Aarhus --city Leipzig --city Mannheim --city Munich --city Krakow --city Copenhagen \
  --no-log-llm-payload
```

This requires adding a `--retrieval-focus` or equivalent field first. The implementation should persist both the original question and the enriched retrieval query so the artifact makes clear what was embedded.

## Success Metrics

Track these across the same three benchmark-style runs:

- retrieved total chunks
- seed chunks
- neighbor chunks
- accepted chunk decisions
- rejected chunk decisions
- acceptance ratio
- accepted-vs-rejected cosine distance p50 and p95
- accepted-vs-rejected selection mode counts
- final answer citation quality

The target is not perfect distance separation. The better target is fewer Stage 006 inputs with equal or better accepted evidence and better final citation quality.
