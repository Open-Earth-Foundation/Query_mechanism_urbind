# Logging Update Plan

## Purpose

This note captures findings from inspecting `output/smoke_aachen`, `output/smoke_aachen_01`, and `output/smoke_aachen_02` after the artifact unification work.

The goal is to improve:

- overview of what is logged, where, and why
- discoverability of metrics, evidence, and stage outputs
- benchmark and regression usefulness
- consistency between runtime artifacts and diagnostic artifacts

## Scope Of Inspection

Reviewed artifacts included:

- `output/smoke_aachen/manifest.json`
- `output/smoke_aachen/summary.jsonl`
- `output/smoke_aachen/run.json`
- `output/smoke_aachen_01/api_state.json`
- `output/smoke_aachen_02/api_state.json`
- `output/smoke_aachen/context_bundle.json`
- `output/smoke_aachen/stages/*.json`
- `output/smoke_aachen/stage_files/**/*`
- `output/smoke_aachen/external_sources/*`

---

## Smoke Run Comparison Status

Compared runs:

- `output/smoke_aachen`
- `output/smoke_aachen_01`
- `output/smoke_aachen_02`

Observed improvements in `smoke_aachen_02`:

- retrieval metrics stayed identical across all three runs: 63 total chunks, 60 seed chunks, 3 neighbor chunks, and the same distance min/p50/p95/max values
- `run.log` now uses the resolved run id consistently, for example `run_id=smoke_aachen_02`
- `writer_citation_coverage` now logs `confirmed_city_count = 1` and `required_city_count = 1` instead of `null`
- `run.json` is gone and `api_state.json` is the run state file
- the duplicate top-level `external_sources/` folder is gone in `smoke_aachen_02`
- `summary.jsonl`, `stages/*.json`, `api_state.json`, and `manifest.json` now agree on status and compact metrics

Remaining observations from `smoke_aachen_02`:

- `stage_files/` still used unnumbered folders in that run (`retrieval`, `markdown_extraction`, `query_preparation`, `enrichment`), while `stages/` used numbered files. This has now been fixed in code for future runs.
- LLM behavior is still not deterministic across runs. Retrieval was identical, but markdown acceptance and enrichment gap analysis differed:
  - `smoke_aachen`: 26 LLM calls, 15 excerpts, 8 accepted chunks, external evidence found
  - `smoke_aachen_01`: 15 LLM calls, 12 excerpts, 8 accepted chunks, one enrichment gap
  - `smoke_aachen_02`: 3 LLM calls, 14 excerpts, 12 accepted chunks, zero enrichment gaps
- The lower LLM call count in `smoke_aachen_02` is not automatically an accuracy improvement. It happened because enrichment found zero city gaps, so external-source loops did not run.

Latest implementation updates for future runs:

- `001_input_snapshot` is now written once after input and reproducibility snapshots are resolved; markdown discovery no longer refreshes or overwrites that stage artifact
- `summary.jsonl` no longer gets an early stale stage-001 event before requested city scope is known
- `summary.jsonl` now writes `001_input_snapshot` before `002_query_preparation` in fresh runs, instead of emitting query preparation first as a side effect of input normalization
- `documents_snapshot.json` keeps the full file manifest but adds a compact summary with total file count, selected-city files, source-library file count, and used markdown files
- enrichment metadata now separates classified non-estimable field count from produced non-estimable output count
- enrichment stage and decision metrics now distinguish `city_gap_count` from `gap_field_count` so city-row counts are not mixed with actual blank/stale/bundled field counts
- enrichment stage metrics now surface unresolved external-source city fields, external-source max-turn events, the expanded-hit finalization path used when the normal external-source agent does not finish cleanly, and external-source token count when available
- `run_summary.txt` is now a compact run index that points to canonical payload artifacts instead of duplicating the full context bundle, markdown excerpts, and final answer
- `run_summary.txt` reports markdown failures as `none` when no markdown failures occurred
- `progress.json` keeps user-facing labels but now includes canonical `stage_name` and `stage_number` fields for each step

---

## Finding 1: Enrichment data is duplicated in multiple places

Status: partially implemented

### Observation

The enrichment result is represented in several overlapping places:

- `context_bundle.json` contains the final `enrichment` block
- `stage_files/008_enrichment/enrichment_bundle.json` contains the full enrichment bundle in new runs
- `stage_files/008_enrichment/*` contains split enrichment artifacts in new runs
- in older runs, `external_sources/external_evidence.json` duplicated enrichment-stage evidence artifacts

Current model:

- the top-level `external_sources/` copy is gone
- the raw external-source audit now lives under `stage_files/008_enrichment/external_source_search_audit.json`
- final accepted outputs are split into clearly named files such as:
  - `external_source_validated_claims.json`
  - `external_source_resolutions.json`
  - `external_source_no_evidence.json`
- runtime `context_bundle.json` still contains the final enrichment block, but that is intentional because it is the downstream-ready canonical context output rather than a logging artifact

This makes it harder to know which file is the canonical source for enrichment evidence.

### Why this is a problem

- human review becomes slower because multiple files must be compared
- future MLflow mirroring becomes less clean
- downstream usage risks drifting toward whichever duplicate file is easiest to reach

### Proposed solution

- make `stage_files/008_enrichment/*.json` the canonical enrichment audit artifact set
- remove the parallel `external_sources/` artifact copy
- keep `context_bundle.json` as the canonical downstream-ready output that aggregates the relevant stage results
- avoid treating `context_bundle.json` as the observability source of truth; that role belongs to the stage artifacts

---

## Finding 2: `api_state.json` still overlaps too much with `manifest.json`

Status: implemented

### Observation

`api_state.json` still contains a large `artifacts` map that repeats much of the alias registry already present in `manifest.json`.

### Why this is a problem

- there are still two discovery surfaces
- changes to artifact layout can require updates in two places
- it is not obvious which file is authoritative for artifact lookup

### Proposed solution

Implemented model:

- remove `run.json` entirely
- use `api_state.json` as the single run-metadata file for API state, diagnostics, benchmarks, and failure recovery
- keep `manifest.json` as the source of truth for artifact alias resolution
- keep the rich run-level metadata in `api_state.json`:
  - status and timestamps
  - error and finish reason
  - inputs summary
  - decisions
  - compact metrics such as `llm_usage`, `retry_summary`, and writer diagnostics
- do not duplicate artifact paths or the full artifact registry in `api_state.json`
- merge API status updates into existing `api_state.json` so richer run metadata is never overwritten by a thinner API snapshot

Result:

- `run.json` no longer exists
- artifact discovery is centered on `manifest.json`
- run state discovery is centered on `api_state.json`
- the old duplicated artifact-map pattern is removed while preserving all important metadata

---

## Finding 3: Stage `008_enrichment_web_search_assumptions` is misleading

Status: implemented

### Observation

The stage name includes `web_search`, but in this run web research was disabled:

- `web_finding_count = 0`
- `freshness_result_count = 0`

At the same time, external source search did run:

- `external_evidence_count = 1`
- `external_no_evidence_count = 1`

### Why this is a problem

- the stage name suggests a substep ran when it did not
- users cannot infer actual executed behavior from the file name

### Proposed solution

Implemented model:

- rename stage `008` to `enrichment`
- log execution flags directly on the stage detail, including:
  - `enrichment_enabled`
  - `use_split_gap_flow`
  - `web_research_enabled`
  - `web_research_executed`
  - `external_source_search_enabled`
  - `external_source_search_executed`
  - `assumptions_enabled`
  - `assumptions_executed`
- write dedicated substage artifacts under `stage_files/008_enrichment/`:
  - `gap_analysis_stage.json`
  - `external_source_search_stage.json`
  - `web_research_stage.json`
  - `assumptions_stage.json`
- keep the raw external-source search audit in the same numbered stage folder:
  - `external_source_search_audit.json`
- log the narrowing funnel for the external-source stage:
  - searched city-fields
  - candidates
  - validated claims
  - rejected claims
  - unused candidates
  - unresolved searched city-fields

Result:

- the stage name no longer implies that web search always ran
- reviewers can now distinguish disabled, skipped, and executed substages
- each substage shows its raw outputs plus a compact summary of what it added

---

## Finding 4: Stage numbering is confusing because event numbering and stage numbering are mixed

Status: implemented

### Observation

Earlier runs had no `stages/009_*.json`, but event index 9 existed in `summary.jsonl` as a `decision_recorded` event.

That mixed global event numbering with stage numbering.

### Why this is a problem

- reviewers expect a stable stage timeline
- non-stage decision events make `summary.jsonl` harder to read and benchmark-parse

### Proposed solution

Implemented model:

- keep `summary.jsonl` event ordering via `event_index`
- add a fixed canonical `stage_number` per logical stage
- use that same `stage_number` in both `summary.jsonl` and `stages/*.json`
- keep `summary.jsonl` as a stage-level timeline only
- store stage-scoped decisions in the relevant `stages/NNN_*.json` detail file
- keep all decisions available through `api_state.json` and `run_summary.txt`

Result:

- a given logical stage now keeps the same number across runs
- stage filenames and summary stage entries are aligned
- summary ordering stays readable while decision details remain discoverable

---

## Finding 5: The context bundle handoff snapshots are not modeled as explicit stage outputs

Status: implemented

### Observation

The root `context_bundle.json` is the canonical downstream-ready runtime artifact and keeps changing during the run.

At the same time, stage details should describe immutable outputs after each stage is complete.

In the current model before this update:

- `stages/007_context_bundle.json` reflected the context bundle before enrichment
- the final `context_bundle.json` included enrichment content later
- stage 007 pointed at a file that continued to change after the stage had completed

So the current flow is:

- stage 007: context bundle before enrichment
- stage 008: enrichment outputs
- final `context_bundle.json`: merged result after enrichment

That means the stage snapshot semantics were unclear.

### Why this is a problem

- stage outputs should be immutable once the stage is done
- a stage should not point to a file that continues to change later
- it is hard to see what each major pipeline family contributed to the canonical context bundle
- this weakens debugging and benchmark traceability

### Proposed solution

Implemented model for now:

- keep the root `context_bundle.json` as the mutable canonical final runtime bundle
- redefine stage `007` as a markdown handoff stage:
  - `007_markdown_context_handoff`
  - `stage_files/007_markdown_context_handoff/context_bundle_after_markdown.json`
  - `stage_files/007_markdown_context_handoff/markdown_context_payload.json`
- keep stage `008_enrichment` for enrichment-only artifacts and logic
- add an explicit post-enrichment handoff stage:
  - `009_enrichment_context_handoff`
  - `stage_files/009_enrichment_context_handoff/context_bundle_after_enrichment.json`
  - `stage_files/009_enrichment_context_handoff/enrichment_context_payload.json`

This establishes the pattern that each major pipeline family should provide:

- a frozen full context bundle snapshot after the stage family
- the stage-specific payload that was injected into that context bundle
- supporting artifacts that explain how that payload was produced

Planned later extension:

- assumptions should become its own top-level stage family following the same pattern, but that is deferred for now

---

## Finding 6: Enrichment contributes structured evidence, but its contribution to the runtime bundle is not easy to inspect

Status: deferred

### Observation

Enrichment does not add chunk-style evidence in the same format as markdown retrieval. Instead it adds:

- `enriched_fields`
- `external_evidence`
- `external_resolutions`
- `external_no_evidence`
- `non_estimable`

This is valid, but harder to inspect quickly than the markdown evidence flow.

### Why this is a problem

- retrieval and markdown evidence are easy to trace
- enrichment evidence is richer but less immediately navigable
- benchmarks will later benefit from a more normalized evidence structure

### Proposed solution

Defer this until the planned stage-family split is implemented.

Reason:

- right now assumptions still live inside the enrichment family
- a normalized evidence index designed today would likely mix two stage families we already plan to separate
- once assumptions become their own top-level stage family, we can design the evidence index against the stable handoff structure instead of a temporary combined layout

Planned later model:

- add a normalized evidence index per major family, for example:
  - `stage_files/008_enrichment/evidence_index.json`
  - `stage_files/010_assumptions/evidence_index.json`
- or, if a cross-family view is still useful then, add a higher-level combined index built on top of the family-specific indexes

Suggested fields:

- `evidence_id`
- `city`
- `field`
- `evidence_type` such as `external_markdown`, `web_result`, `assumption`, `non_estimable`
- `status` such as `resolved`, `unresolved`, `non_estimable`
- `source_artifact`
- `source_pointer`
- `value_summary`

This should improve discoverability without removing the richer underlying files, but it is better implemented once the stage boundaries are final.

---

## Finding 7: Writer citation coverage is inconsistent across artifacts

Status: implemented

### Observation

In the older runs, `stages/010_writer_citation_coverage.json` or `stages/011_writer_citation_coverage.json` contains:

- `confirmed_city_count = null`
- `required_city_count = null`

But later writer artifacts show:

- `confirmed_city_count = 1`
- `required_city_count = 1`

In `smoke_aachen_02`, this is fixed:

- `stages/011_writer_citation_coverage.json` has `confirmed_city_count = 1`
- `stages/011_writer_citation_coverage.json` has `required_city_count = 1`

### Why this is a problem

- the same concept appears with different values across artifacts
- downstream metrics or dashboards may read the wrong one

### Proposed solution

Implemented model:

- writer citation coverage stage metrics now use the persisted coverage payload fields:
  - `coverage_confirmed`
  - `coverage_required`
- writer citation coverage and writer stage artifacts now report the same city counts

---

## Finding 8: City normalization is inconsistent in one stage summary

Status: implemented

### Observation

In `summary.jsonl` the markdown inputs stage reports:

- `selected_cities_found = ["aachen"]`
- `missing_selected_cities = ["Aachen"]`

This indicates the same city is treated as both found and missing because of normalization mismatch.

### Why this is a problem

- stage metrics become misleading
- city filtering quality becomes harder to trust
- benchmarking and regression analysis may count false mismatches

### Proposed solution

Implemented model:

- normalize city identity once for all selected-city comparisons using `normalize_city_key`
- store normalized city keys as the canonical values in:
  - `inputs.selected_cities_planned`
  - `inputs.selected_cities_found`
- make stage-level missing-city detection compare normalized keys only

Root cause:

- this was not an Aachen-specific bug
- the conceptual issue was that planned cities were logged as raw labels while found cities were logged as normalized keys
- the same mismatch could occur for any city with casing, spaces, hyphens, underscores, or accent variations

Result:

- city matching is now consistent for all cities
- false missing-city mismatches from label formatting differences are removed

---

## Finding 9: The enrichment stage is observable, but executed-substep visibility could be better

Status: implemented

### Observation

Current enrichment metrics capture counts well:

- field count
- gap count
- assumptions
- non-estimable
- web findings
- external evidence
- no evidence

But they do not clearly show which substeps were skipped versus ran with zero results.

### Why this is a problem

- `0` results and `not executed` are different states
- this matters for debugging and later MLflow traces

### Proposed solution

Implemented model:

- `web_research_executed`
- `external_source_search_executed`
- `assumptions_executed`
- `freshness_check_executed`

This makes stage behavior easier to interpret than count metrics alone. In `smoke_aachen_02`, this correctly shows that external-source search did not execute because enrichment found zero city gaps.

---

## Finding 10: Manifest is already the strongest discovery surface and should become the clear source of truth

Status: implemented

### Observation

`manifest.json` is already the best structured artifact index in the run.
It contains aliases and generated files in a way that is easier to consume than `api_state.json`.

### Why this matters

- it is the right base for future MLflow mirroring
- it is the best place to centralize artifact discovery
- it can support humans, APIs, and benchmarks with the same contract

### Proposed solution

- formally define `manifest.json` as the artifact source of truth
- keep alias naming stable and documented
- update consumers to prefer aliases over path assumptions everywhere
- optionally add a small `artifact_categories` or `stage_map` section for even easier discovery

Implemented model:

- API and benchmark consumers use manifest aliases where available
- `api_state.json` no longer contains the full artifact registry
- `manifest.json` remains the artifact locator

---

## Finding 11: `stage_files/` folders are not aligned with numbered stages

Status: implemented

### Observation

In `smoke_aachen_02`, stage detail files are numbered:

- `stages/003_retrieval.json`
- `stages/005_markdown_batching.json`
- `stages/006_markdown_extraction.json`
- `stages/008_enrichment.json`

But the corresponding stage files were still written under unnumbered folders:

- `stage_files/retrieval/`
- `stage_files/markdown_extraction/`
- `stage_files/enrichment/`

### Why this is a problem

- reviewers need to mentally map stage detail files to differently named stage-file folders
- manifest paths are correct, but direct filesystem inspection is less clear
- future MLflow mirroring benefits from a one-to-one stage folder convention

### Proposed solution

Implemented model for future runs:

- stage files now use the same canonical stage number and slug as `stages/`
- expected examples:
  - `stage_files/002_query_preparation/research_question.json`
  - `stage_files/003_retrieval/retrieval.json`
  - `stage_files/005_markdown_batching/batches.json`
  - `stage_files/006_markdown_extraction/excerpts.json`
  - `stage_files/008_enrichment/enrichment_bundle.json`
- markdown batching artifacts are now grouped under `005_markdown_batching`, while markdown extraction decisions and excerpts remain under `006_markdown_extraction`
- README, benchmark docs, and script docstrings now use the numbered paths

---

## Recommended Implementation Order

1. Remove remaining duplicate artifact trees
   - especially `external_sources/`
   - Status: implemented for the top-level `external_sources/` copy

2. Clarify stage naming and execution-state logging
   - rename `008`
   - add executed/skipped flags
   - Status: implemented

3. Add explicit post-enrichment context provenance
   - after-enrichment snapshot or delta artifact
   - Status: open

4. Keep `api_state.json` focused
   - preserve run metadata there, but keep artifact alias lookup in `manifest.json`
   - Status: implemented

5. Normalize repeated metric generation
   - writer coverage
   - city missing/found logic
   - Status: implemented

6. Add normalized enrichment evidence index
   - improves discoverability for review and benchmarks
   - Status: open

7. Align `stage_files/` folders with numbered stages
   - Status: implemented

8. Promote assumptions to a native top-level stage family
   - implemented target:
     - `010_assumptions`
     - `011_assumptions_context_handoff`
   - keep the same handoff pattern used for markdown and enrichment:
     - immutable full context snapshot after the stage family
     - stage-specific payload injected into the canonical context bundle
     - supporting audit artifacts
   - current status:
     - implemented

## Expected Outcome

After these changes, the logging system should be easier to use because:

- each artifact has a clearer purpose
- canonical data is stored once
- runtime state and diagnostic state are easier to distinguish
- reviewers can understand what each stage executed and contributed
- future MLflow integration can mirror a cleaner and more stable artifact model
