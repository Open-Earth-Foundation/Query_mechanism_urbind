# Gold Benchmark Report

Date: March 27, 2026
Repo state: current local workspace, including the locally modified `tests/fixtures/benchmark_gold.json`
Benchmark type: gold recall benchmark across all current fixture questions
Judge model: `openai/gpt-5.4-mini`

## Executive Summary

I ran the full gold benchmark against all 9 current fixture questions.

- A single monolithic live run was attempted first under `output/benchmarks/recall/full_live_20260327_codex/`.
- That run aborted on the 4th case with a `MemoryError` while loading the vector-store manifest during retrieval setup.
- I then reran the benchmark one case per process under `output/benchmarks/recall/per_case_live_20260327/` to avoid process-level accumulation and finish the matrix.
- Result: 7 of 9 cases completed and produced benchmark reports.
- 2 of 9 cases failed before scoring because vector retrieval exhausted on `Error loading hnsw index`.
- Later on March 27, 2026, I reran the suite again under the current `gpt-5.4-mini` config and that full live run completed all 9 cases. See the comparison chapter at the end of this document.

This means the earlier benchmarked config on March 27, 2026 did not support a clean end-to-end "full live benchmark in one command". The later rerun under the current `gpt-5.4-mini` config improved that operationally, but with mixed quality effects.

## What Was Run

### Attempt 1: full live run

Command:

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.benchmark_recall `
  --gold-file tests/fixtures/benchmark_gold.json `
  --benchmark-id full_live_20260327_codex `
  --run-live
```

Outcome:

- `vehicle_targets_cross_ccc`: completed
- `capex_cross_ccc`: completed
- `charging_targets_germany_seven_cities`: completed
- `dresden_charging_pilots_and_retrofits`: benchmark process aborted before retrieval setup completed

Observed blocker:

- `MemoryError` while reading the vector-store manifest inside `backend.modules.vector_store.manifest.load_manifest`

Interpretation:

- The benchmark runner currently has a process-level reliability issue for multi-case full-live execution.

### Attempt 2: isolated per-case live runs

Each case was rerun in a fresh Python process under:

- `output/benchmarks/recall/per_case_live_20260327/<case_id>/`

This completed 7 benchmark reports and isolated 2 hard failures.

### Additional diagnostic

For the 2 failed cases, I reran the pipeline with:

```powershell
$env:VECTOR_STORE_ENABLED='false'
.\.venv\Scripts\python.exe -m backend.scripts.benchmark_recall ...
```

Both cases then completed pipeline generation, but benchmark scoring still failed because the runner unconditionally requires `markdown/retrieval.json`, and standard-chunking live runs do not currently write that artifact.

Interpretation:

- Heidelberg and Mannheim are not impossible benchmark questions.
- The current live blocker is vector retrieval infrastructure.
- The current gold benchmark runner also cannot score standard-chunking live runs.

## Fixture Notes Verified Before Running

A subagent reviewed the current fixture and found:

- The fixture currently contains 9 cases.
- All 9 cases now include `gold_chunk_texts`, so text/excerpt fallback is now part of normal scoring, not an edge case.
- `vehicle_targets_cross_ccc` still points to a stale cached path, so default non-live benchmarking is operationally brittle unless `--run-live` is used.
- `charging_targets_germany_seven_cities` was materially expanded into a multi-city, 7-fact, 7-chunk case and is now one of the hardest cases in the fixture.
- `mannheim_transport_electrification_capex` also became more brittle due to long and noisy added fallback texts.

## Current Execution Matrix

| Case | Status | Duration (s) | Key result |
| --- | --- | ---: | --- |
| `vehicle_targets_cross_ccc` | completed | 374.1 | strong retrieval/extraction, writer missed 1 fact |
| `capex_cross_ccc` | completed | 277.4 | repeated markdown decision-payload retries; final recall still 0.75 |
| `quantified_charging_targets_munster` | completed | 105.1 | good retrieval/citations, weak fact retention on 2 facts |
| `charging_targets_germany_seven_cities` | completed | 234.2 | worst current live case by a wide margin |
| `dresden_charging_pilots_and_retrofits` | completed | 91.2 | good delivery, 2 facts still lost by stage C |
| `heidelberg_transport_electrification_capex` | failed | 19.7 | vector retrieval exhausted on HNSW index load |
| `mannheim_transport_electrification_capex` | failed | 17.8 | vector retrieval exhausted on HNSW index load |
| `krakow_vehicle_targets` | completed | 79.1 | weak stage A, but final answer still hit all facts |
| `krakow_warszawa_transport_electrification` | completed | 208.3 | perfect retrieval/citations, writer still lost 3 facts |

Completion rate for current live scoring:

- 7 / 9 fully scored cases = 77.8%

## Aggregate Metrics Across the 7 Completed Current Live Cases

These means are computed only across the 7 cases that produced full benchmark reports.

| Metric | Mean |
| --- | ---: |
| Retrieval recall | 0.727 |
| Retrieval precision | 0.055 |
| MRR | 0.377 |
| Delivery recall | 0.791 |
| Extraction recall | 0.755 |
| Fact extraction rate | 0.710 |
| End-to-end fact recall | 0.606 |
| Citation coverage | 0.719 |

Interpretation:

- Retrieval and excerpt delivery are acceptable but not strong enough for a gold benchmark intended to protect detailed quantitative facts.
- The largest drop is from stage B / delivered evidence to stage C / final answer.
- Citation coverage is decent, but citation presence is not the main failure mode. The bigger issue is factual compression and omission in the final answer.
- Retrieval precision is low, but that is less diagnostic here because the benchmark compares a small gold set against a broader candidate context set.

## Per-Case Results

| Case | Retrieval recall | Extraction recall | Fact recall | Citation coverage | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `vehicle_targets_cross_ccc` | 1.000 | 1.000 | 0.750 | 0.750 | missed Copenhagen 22% passenger-car EV projection in final answer |
| `capex_cross_ccc` | 0.750 | 0.750 | 0.750 | 0.750 | missed Aachen EUR 715.1M climate-friendly drives/fuels CAPEX |
| `quantified_charging_targets_munster` | 0.800 | 1.000 | 0.500 | 1.000 | missed rented-bus conflict fact and EUR 21.3M bus-electrification budget |
| `charging_targets_germany_seven_cities` | 0.286 | 0.286 | 0.143 | 0.286 | missed most Dresden, Munich, and Munster charging facts |
| `dresden_charging_pilots_and_retrofits` | 0.750 | 0.750 | 0.600 | 0.750 | missed Vonovia > EUR 2M and 8-to-16 charging-point expansion facts |
| `krakow_vehicle_targets` | 0.500 | 0.500 | 1.000 | 0.500 | stage C recovered all facts despite weak stage A coverage |
| `krakow_warszawa_transport_electrification` | 1.000 | 1.000 | 0.500 | 1.000 | writer lost 3 detailed charging/progress/investment facts |

## Strongest Current Failure Patterns

### 1. The benchmark runner is not stable as one full live process

Observed in the first run:

- after 3 completed cases, the full run aborted on the next case with `MemoryError`
- failure point: vector-store manifest loading

Impact:

- a single command does not currently complete the full matrix
- the benchmark must be run case-by-case to get near-complete current results

### 2. Heidelberg and Mannheim are blocked by vector-store retrieval infrastructure

Current live per-case failures:

- `heidelberg_transport_electrification_capex`
- `mannheim_transport_electrification_capex`

Exact repeated failure pattern in both `run.log` files:

- `vector_retrieval.query_by_embedding`
- `Error loading hnsw index`
- retries exhausted after 3 attempts

Impact:

- these 2 cases never reach benchmark scoring in current vector mode

Why this looks like infrastructure and not a bad gold case:

- when `VECTOR_STORE_ENABLED=false`, both cases progressed through markdown extraction and writing
- both standard-mode diagnostic runs produced `final.md`, `excerpts.json`, and `references.json`
- the benchmark still failed only because `backend.benchmarks.gold_recall.runner` requires `markdown/retrieval.json`, which standard mode does not emit
- there are recent historical single-case runs already in the repo that succeeded for both cases:
  - `output/benchmarks/recall/verify_heidelberg_textfallback_20260326/benchmark_report.md`
  - `output/benchmarks/recall/verify_mannheim_textfallback_20260327/benchmark_report.md`

Historical reference values already present in the repo:

- Heidelberg: retrieval recall `1.000`, fact recall `1.000`, citation coverage `1.000`
- Mannheim: retrieval recall `1.000`, fact recall `0.857`, citation coverage `0.857`

Conclusion:

- the current March 27, 2026 blocker is a live vector-store/index issue, not just difficult benchmark content

### 3. `charging_targets_germany_seven_cities` is the main benchmark-quality failure

Current metrics:

- retrieval recall `0.286`
- extraction recall `0.286`
- fact recall `0.143`
- citation coverage `0.286`

Loss profile:

- only 2 of 7 gold chunks were seed hits
- 5 of 7 gold chunks were missed entirely
- most misses cluster around Dresden, Munich, and Munster quantified charging evidence

Why this case matters:

- it is the most expanded case in the current fixture
- it now requires selective omission of unsupported cities and correct retention of multiple quantified charging items across four cities
- this looks like the clearest current benchmark gap in retrieval breadth plus final-answer retention

### 4. The writer is a bigger bottleneck than citations in several successful cases

Examples:

- `krakow_warszawa_transport_electrification`: retrieval recall `1.000`, extraction recall `1.000`, citation coverage `1.000`, but fact recall only `0.500`
- `vehicle_targets_cross_ccc`: retrieval/extraction `1.000`, but fact recall `0.750`
- `quantified_charging_targets_munster`: delivery/extraction `1.000`, citation coverage `1.000`, but fact recall `0.500`

Interpretation:

- the system is often finding the evidence and citing it
- the final answer still drops or compresses away specific quantitative facts

### 5. Markdown extraction is still flaky on structured outputs

Best example:

- `capex_cross_ccc`

Observed in `run.log`:

- repeated malformed decision payloads
- repeated `decision_invariant_failed` retries
- 1 exhausted retry sequence on `markdown.batch_extraction`

Even though the case eventually completed, this is still a current reliability risk because it consumes extra calls and can turn into a hard failure when the retries do not recover.

## Two Important Structural Caveats

### Standard-chunking live runs are currently not benchmark-scoreable

Observed in both standard-mode diagnostics:

- the live pipeline completed
- benchmark scoring failed on:

```text
ValueError: Expected a JSON object at .../markdown/retrieval.json
```

Reason:

- the gold benchmark runner unconditionally expects `markdown/retrieval.json`
- standard-chunking runs currently do not write that artifact

Impact:

- there is no clean fallback path when vector-store live scoring is broken

### Cached-mode defaults are still brittle

The fixture still contains stale and partial cached-run references.

Impact:

- a default non-live benchmark can fail for operational reasons unrelated to model quality
- `--run-live` is the safer mode for current-state verification

## Recommended Next Fixes

1. Fix the vector-store reliability issue first.
   - Investigate the city-specific HNSW index load failures for Heidelberg and Mannheim.
   - Investigate why Aarhus intermittently throws the same error but recovers after retries.

2. Fix the monolithic full-live runner reliability.
   - The benchmark should not die mid-suite on manifest loading with `MemoryError`.
   - A full 9-case run should complete in one process.

3. Make the benchmark runner tolerant of standard mode.
   - Either persist a benchmark-compatible `retrieval.json` in standard mode or make stage A scoring optional when the artifact is unavailable.

4. Improve retention on multi-city quantitative questions.
   - The strongest current product-quality gap is `charging_targets_germany_seven_cities`.
   - The second major gap is writer retention in `krakow_warszawa_transport_electrification`.

5. Reduce markdown extraction schema flakiness.
   - `capex_cross_ccc` shows the current structured-output contract is still fragile under batch pressure.

## Artifact Locations

Current live per-case outputs:

- `output/benchmarks/recall/per_case_live_20260327/`

Standard-mode diagnostics for the 2 blocked cases:

- `output/benchmarks/recall/per_case_standard_diag_20260327/`

Initial failed full-live attempt:

- `output/benchmarks/recall/full_live_20260327_codex/`

Historical successful reference runs for the currently blocked cases:

- `output/benchmarks/recall/verify_heidelberg_textfallback_20260326/`
- `output/benchmarks/recall/verify_mannheim_textfallback_20260327/`

## Comparison: March 27 Rerun With `gpt-5.4-mini` Config

After the first benchmark documented above, I reran the full suite after the model config change now present in `llm_config.yaml`.

Important comparison caveat:

- the current config is not a markdown-only change
- `orchestrator`, `sql_researcher`, `markdown_researcher`, `writer`, `chat`, and `assumptions_reviewer` are all now set to `openai/gpt-5.4-mini`
- this chapter therefore compares the earlier benchmarked config against the current full `gpt-5.4-mini` config state, not an isolated markdown-agent A/B test

New run artifact:

- `output/benchmarks/recall/full_live_20260327_gpt54mini_config/`

### Headline Outcome

The biggest improvement is stability.

- The new full live run completed all 9 cases in one process.
- The earlier full live attempt failed mid-suite and required isolated per-case reruns.
- Heidelberg and Mannheim, which previously failed as current live cases, both scored successfully in the new run.

The quality result is mixed.

- On the 7 cases that were already fully scored in the earlier run, mean retrieval recall improved from `0.727` to `0.762` (`+0.036`).
- On that same 7-case overlap, mean extraction recall fell from `0.755` to `0.667` (`-0.088`).
- Mean end-to-end fact recall fell from `0.606` to `0.570` (`-0.036`).
- Mean citation coverage fell from `0.719` to `0.667` (`-0.052`).

In short:

- suite stability got much better
- answer quality did not improve consistently, and regressed on some important cases

### New Full-Live Aggregate Summary

Across all 9 cases in the new run:

| Metric | New full-live mean |
| --- | ---: |
| Retrieval recall | 0.799 |
| Retrieval precision | 0.068 |
| MRR | 0.466 |
| Delivery recall | 0.821 |
| Extraction recall | 0.678 |
| Fact extraction rate | 0.692 |
| End-to-end fact recall | 0.602 |
| Citation coverage | 0.662 |

Interpretation:

- the new config finally gives a complete 9-case live report
- that top-line fact recall (`0.602`) looks similar to the earlier 7-case partial average (`0.606`), but that hides important case-by-case shifts
- the added Heidelberg success lifts coverage of the suite, while Mannheim remains weak and some previously stronger cases regressed

### Case-by-Case Delta Against the Earlier Run

| Case | Earlier state | New state | Comparison |
| --- | --- | --- | --- |
| `vehicle_targets_cross_ccc` | retrieval `1.000`, extraction `1.000`, fact `0.750`, citation `0.750` | retrieval `1.000`, extraction `0.750`, fact `1.000`, citation `0.750` | better final answer, weaker stage B |
| `capex_cross_ccc` | retrieval `0.750`, extraction `0.750`, fact `0.750`, citation `0.750` | retrieval `0.750`, extraction `0.750`, fact `0.500`, citation `0.750` | factual regression |
| `quantified_charging_targets_munster` | retrieval `0.800`, extraction `1.000`, fact `0.500`, citation `1.000` | retrieval `0.800`, extraction `0.800`, fact `0.750`, citation `0.800` | better fact recall, weaker extraction/citation |
| `charging_targets_germany_seven_cities` | retrieval `0.286`, extraction `0.286`, fact `0.143`, citation `0.286` | retrieval `0.286`, extraction `0.286`, fact `0.143`, citation `0.286` | unchanged and still the weakest benchmark-quality case |
| `dresden_charging_pilots_and_retrofits` | retrieval `0.750`, extraction `0.750`, fact `0.600`, citation `0.750` | retrieval `1.000`, extraction `0.750`, fact `0.600`, citation `0.750` | better stage A only |
| `krakow_vehicle_targets` | retrieval `0.500`, extraction `0.500`, fact `1.000`, citation `0.500` | retrieval `0.500`, extraction `0.500`, fact `1.000`, citation `0.500` | unchanged |
| `krakow_warszawa_transport_electrification` | retrieval `1.000`, extraction `1.000`, fact `0.500`, citation `1.000` | retrieval `1.000`, extraction `0.833`, fact `0.000`, citation `0.833` | severe regression |
| `heidelberg_transport_electrification_capex` | failed in current live run | retrieval `1.000`, extraction `1.000`, fact `1.000`, citation `1.000` | major stability and quality improvement |
| `mannheim_transport_electrification_capex` | failed in current live run | retrieval `0.857`, extraction `0.429`, fact `0.429`, citation `0.286` | major stability improvement, but weak answer quality |

### Most Important Changes

#### 1. The current `gpt-5.4-mini` config removed the full-suite live blocker

This is the clearest win.

- the earlier full live run died mid-suite with `MemoryError`
- the new full live run completed all 9 cases
- the earlier live Heidelberg and Mannheim failures did not recur

That means the current config is much more usable for actual benchmark operations.

#### 2. Heidelberg improved from "blocked" to "fully passing"

New Heidelberg metrics:

- retrieval recall `1.000`
- extraction recall `1.000`
- fact recall `1.000`
- citation coverage `1.000`

This is the strongest single-case improvement in the rerun.

#### 3. Mannheim improved from "blocked" to "scored", but not to a good quality level

New Mannheim metrics:

- retrieval recall `0.857`
- extraction recall `0.429`
- fact recall `0.429`
- citation coverage `0.286`

Interpretation:

- the infrastructure problem was solved for this case
- the answer-quality problem was not

Compared with the historical reference run already present in the repo (`verify_mannheim_textfallback_20260327`), the current rerun is materially weaker.

#### 4. `krakow_warszawa_transport_electrification` regressed sharply

This is the biggest quality regression in the overlapping cases.

Earlier:

- fact recall `0.500`

New:

- fact recall `0.000`

Observed stage-C misses in the new run:

- Krakow 74% electric-bus share / ~452 buses
- Krakow 150 new public charging stations in the first five years
- Krakow five pantograph bus-charging stations
- Warszawa already operating more than 160 electric buses plus 12 additional buses contracted in 2023
- Warszawa target of 202 electric buses by 2025
- Warszawa charging infrastructure on 48 loops and PLN 979.84M bus-electrification package

Interpretation:

- the current config can still retrieve and excerpt much of the evidence
- the final answer is where this case now collapses

#### 5. `capex_cross_ccc` also regressed

Earlier missing fact:

- Aachen EUR 715.1M climate-friendly drives and fuels CAPEX

New missing facts:

- Aachen EUR 715.1M climate-friendly drives and fuels CAPEX
- Aachen EUR 49.5M charging-infrastructure CAPEX for about 1,600 new charging points

Interpretation:

- the new config is dropping more Aachen quantitative CAPEX detail in the final answer than before

#### 6. Some cases improved in final-answer quality

Not all changes were negative.

- `vehicle_targets_cross_ccc` improved from fact recall `0.750` to `1.000`
- `quantified_charging_targets_munster` improved from fact recall `0.500` to `0.750`

So the rerun is not a simple across-the-board degradation. It is a more stable but more mixed-quality profile.

### Efficiency Notes

On the 7 overlapping cases, mean total token use from `LLM_USAGE_SUMMARY` shifted from about `109,074` tokens per case to about `103,023` tokens per case (`-6,051` mean delta), but the distribution was uneven:

- large decrease on `capex_cross_ccc`
- decrease on `krakow_warszawa_transport_electrification`
- increase on `charging_targets_germany_seven_cities`
- increase on `krakow_vehicle_targets`

So the new config is not simply more expensive or simply cheaper. The main operational gain is stability, not a clear uniform cost improvement.

### Comparison Conclusion

Compared with the earlier benchmarked state:

- better: full-suite live stability, Heidelberg, and current live coverage of Mannheim
- roughly the same: `charging_targets_germany_seven_cities`, `krakow_vehicle_targets`
- worse: `capex_cross_ccc`, `krakow_warszawa_transport_electrification`, overall overlap-case extraction recall, citation coverage, and fact recall

My read is:

- if the primary goal is to get a full live benchmark to complete reliably, the current `gpt-5.4-mini` config is an improvement
- if the primary goal is best answer quality on the already-working overlap cases, the result is mixed and likely worse overall
- because the current config changed multiple agents at once, this rerun does not isolate the markdown agent as the sole cause of either the stability improvement or the quality regressions
