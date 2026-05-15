# Data Pipeline Modularization Report

Generated: 2026-05-13

Branch/worktree snapshot: `ON-5708-Align_dev_standard`, with existing uncommitted changes in the checkout. This report is based on static inspection of the current working tree plus current repository memory. It does not include a fresh benchmark run.

## Scope

The target modular pipeline is the full source-to-evidence pipeline, not just
file loading:

1. Read, process, and extract CCC data.
2. Read, process, and extract 3rd party data.
3. Search, process, and extract web data.
4. Generate, review, and apply assumptions.

The goal is for each stage to have a distinct code home, clear input/output contracts, logging, written artifacts, and benchmarks.

In scope for each source stage:

- Source acquisition or handoff from an upstream acquisition process.
- PDF-to-Markdown conversion handoff. The implementation should be based on the other repository that already handles PDF-to-Markdown, rather than redesigned in this repo.
- Retrieval or source selection.
- Data extraction into typed facts/evidence.
- Validation, conflict/no-evidence handling, and provenance.
- Written artifacts and benchmark coverage.

## Executive Summary

The project already has many of the right building blocks, but the boundaries are uneven.

- CCC read/process/extract is the most mature. It has a distinct `backend/modules/markdown_researcher/` package, optional `backend/modules/vector_store/` retrieval, strong run artifacts, and several benchmark paths.
- 3rd party read/process/extract is also fairly mature at the contract level. `documents/source_library/sources.yaml` currently has 46 source entries matching 46 Markdown files, and the governed external-source search has Pydantic models, controlled tools, artifacts, tests, and a Krakow benchmark fixture. The weak point is physical placement: most of this lives under `backend/modules/web_researcher/` instead of a separate external-source module.
- Web search/process/extract is functionally separated across files, but not physically separated from the enrichment package. It has good logging and typed outputs, but weak artifacting. Search plans, raw search results, scrape decisions, and extraction attempts are not persisted as first-class run artifacts.
- Assumptions are split into two concepts: automatic enrichment estimates in `backend/modules/web_researcher/assumptions_estimator.py` and post-run editable assumptions in `backend/api/services/assumptions_review.py`. Both have contracts and tests, but there is no dedicated assumptions module and no dedicated benchmark.

The main architectural issue is that `backend/modules/web_researcher/` has become a mixed responsibility package. It currently owns gap analysis, external-source search, external resolver logic, open web search, freshness checks, context merging, and automatic assumptions. That makes reuse harder because other tools cannot depend on one piece without importing the whole enrichment area.

## Current Pipeline Shape

Current runtime flow in `backend/modules/orchestrator/module.py` and `backend/modules/web_researcher/module.py`:

```text
question
  -> retrieval query preparation
  -> CCC markdown loading or vector retrieval
  -> markdown researcher extraction
  -> context_bundle["markdown"]
  -> optional enrichment
       -> gap analysis
       -> governed external Markdown search
       -> external evidence resolver
       -> optional web research
       -> freshness check
       -> field status merge
       -> assumptions estimator
       -> context_bundle["enrichment"]
  -> writer
  -> final.md
```

Important detail: the current flow already treats CCC as the baseline source. Governed external Markdown, websearch, and assumptions are enrichment layers that run after CCC evidence is available and gaps are identified. What is missing from this view is the upstream PDF-to-Markdown handoff; that should be included in the target scope as a handoff from the existing PDF-to-Markdown implementation in the other repo.

## Stage Readiness Matrix

| Stage | Current code home | Contract quality | Logging | Artifacts | Benchmarks | Modularization verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Read/process/extract CCC data | `backend/modules/markdown_researcher/`, `backend/modules/vector_store/`, orchestrator glue | Partial to strong. `MarkdownResearchResult`, `MarkdownExcerpt`, chunk dicts, and retrieval artifact shape are defined, but chunk inputs are still plain dicts. | Strong enough. Uses module loggers, run log, run summary, progress tracker. | Strong. `markdown/*.json`, `context_bundle.json`, `run.json`, `run.log`, `run_summary.txt`. | Strongest of all stages. Retrieval benchmark, recall benchmark, chunking benchmark, unit tests. | Mostly modular after Markdown exists, but CCC processing is not named as a CCC module, PDF-to-Markdown is out-of-repo, and some artifact assembly lives in orchestrator. |
| Read/process/extract 3rd party data | `documents/source_library/`, `backend/modules/web_researcher/external_sources.py`, `external_agent.py`, `external_resolver.py` | Strong. `SourceMetadata`, `SearchHit`, `EvidenceCandidate`, `ExternalEvidenceClaim`, `ExternalEvidenceResolution`, `NoEvidenceRecord`. | Good. Tool calls and agent failures are logged. | Medium to strong. `external_sources/external_evidence.json` and `enrichment/external_*.json`, but normal run registration of the tool audit is weak. | Medium. Dedicated Krakow external-source benchmark with 4 cases plus unit tests. | Contractually modular after source Markdown exists, physically not modular enough, and missing explicit PDF-to-Markdown handoff. |
| Search/process/extract web data | `backend/modules/web_researcher/search_planner.py`, `search_worker.py`, `search.py`, `scraper.py`, `extractor.py`, `freshness.py`, `tier1_web.py` | Medium. `SearchBatch`, `WebFinding`, `FreshnessResult`, and tier-1 models exist. | Good. Search, scrape, relevance, tier-1, worker, and freshness paths log. | Weak to medium. Final `web_findings.json` and `freshness_results.json` exist only when non-empty; search plans and raw web decisions are not persisted. | Weak. Unit tests exist, but no dedicated live websearch benchmark. | Functionally split by file, but physically buried inside `web_researcher` and missing benchmark/artifact discipline. |
| Generate/review/apply assumptions | `backend/modules/web_researcher/assumptions_estimator.py`, `backend/api/services/assumptions_review.py`, `backend/api/routes/assumptions.py` | Medium. Automatic assumptions use `AssumptionRecord` and `NonEstimableRecord`; review flow uses API models `MissingDataItem`, `AssumptionsPayload`, `RegenerationResult`. | Medium. Service and estimator log key LLM calls and skip/failure paths. | Medium. Automatic assumptions are persisted through enrichment artifacts; review artifacts persist only with `persist_artifacts=true`. | Weak. Unit/API tests exist, but no assumptions benchmark. | Not modular enough. Two assumptions flows live in different areas and should be unified under a dedicated module with API adapters. |

## 1. Read, Process, And Extract CCC Data

### What already looks modular

Current main homes:

- `backend/modules/markdown_researcher/`
- `backend/modules/vector_store/`
- `backend/modules/orchestrator/module.py`
- `documents/*.md`

The CCC source layer is mostly represented as top-level Markdown city files under `documents/`. The normal non-vector path loads those files through `load_markdown_documents()`, chunks them, groups them by normalized city key, and sends them through `extract_markdown_excerpts()`.

In the target scope, this stage starts with a handoff from the other repository's PDF-to-Markdown flow. This repo should consume the converted Markdown as given, then handle chunking, extraction, validation of extracted facts, artifact writing, and benchmarks.

The vector path is also separated in `backend/modules/vector_store/`. When `VECTOR_STORE_ENABLED=true`, the orchestrator retrieves chunks through `retrieve_chunks_for_queries()`, serializes a retrieval artifact with `build_retrieval_artifact()`, and converts retrieved chunks back into markdown researcher input shape with `as_markdown_documents()`.

Current input contract:

- User question and up to 3 retrieval queries.
- Optional selected city filter.
- `MARKDOWN_DIR`, default `documents`.
- Converted CCC Markdown handed off from the PDF-to-Markdown process.
- `MarkdownResearcherConfig` and optional `VectorStoreConfig`.
- Markdown chunk dicts containing at least `path`, `city_name`, `city_key`, `content`, `chunk_id`, and `chunk_index`.

Current output contract:

- `MarkdownResearchResult`
- `MarkdownExcerpt`
- `MarkdownBatchFailure`
- Run-level `context_bundle["markdown"]`
- Citation references in `markdown/references.json`

Artifacts already written:

- `research_question.json`
- `markdown/batches.json`
- `markdown/excerpts.json`
- `markdown/accepted_excerpts.json`
- `markdown/rejected_excerpts.json`
- `markdown/decision_audit.json`
- `markdown/references.json`
- `markdown/retrieval.json` when vector retrieval is enabled
- `context_bundle.json`
- `run.json`
- `run.log`
- `run_summary.txt`

Benchmarks already present:

- Retrieval strategy benchmark: `backend/scripts/run_retrieval_benchmark.py`
- Retrieval benchmark fixtures/config: `backend/benchmarks/prompts/`, `backend/benchmarks/config/`
- Gold recall benchmark: `backend/scripts/benchmark_recall.py`, `backend/benchmarks/gold_recall/`
- Chunking benchmark: `backend/scripts/benchmark_chunking_strategy.py`, `backend/modules/vector_store/benchmarking.py`
- Tests: `tests/test_markdown_researcher.py`, `tests/test_markdown_services.py`, `tests/test_retrieval_benchmark.py`, `tests/test_benchmark_recall.py`, `tests/test_chunking_benchmark.py`, vector-store tests.

### What is not modular enough

- The code is named `markdown_researcher`, not `ccc_reader`. That is accurate technically, but less reusable as a domain stage. Other tools looking for "CCC read/process/extract" would not find a clean CCC boundary.
- PDF-to-Markdown is not represented as an explicit handoff/input contract in this repo yet.
- The orchestrator owns important CCC artifact assembly: batches, accepted/rejected/audit artifacts, references, inspected city metadata, and vector retrieval artifact registration.
- The chunk input contract is still `dict[str, object]`. It works, but it is weaker than the Pydantic model contracts used elsewhere.
- There is no source manifest for top-level CCC markdown files comparable to `documents/source_library/sources.yaml`. The CCC runtime still depends heavily on filename stem as city identity.

### Recommended boundary

Keep `markdown_researcher` as the extractor if desired, but introduce a clearer CCC-facing boundary:

```text
backend/modules/ccc_reader/
  models.py          # CCCSource, CCCChunk, CCCReadResult, CCCExtractionBundle
  source_handoff.py  # consume converted Markdown from PDF-to-Markdown repo
  services.py        # validate, load, select cities, chunk docs
  retrieval.py       # vector and standard retrieval adapters
  extraction.py      # markdown extraction adapter and validation
  artifacts.py       # writes markdown/* and registers run artifacts
  benchmarks/        # CCC/retrieval benchmark fixtures or adapters
```

This does not require rewriting the extraction agent. It means the rest of the system consumes `ccc_reader` contracts rather than knowing how `markdown_researcher` and `vector_store` are wired together.

## 2. Read, Process, And Extract 3rd Party Data

### What already looks modular

Current main homes:

- `documents/source_library/sources.yaml`
- `documents/source_library/**/*.md`
- `backend/modules/web_researcher/external_sources.py`
- `backend/modules/web_researcher/external_agent.py`
- `backend/modules/web_researcher/external_resolver.py`
- `backend/prompts/external_source_researcher_system.md`
- `backend/prompts/external_source_finalizer_system.md`

Current source inventory:

- `documents/source_library/sources.yaml`: 46 entries.
- `documents/source_library/**/*.md`: 46 Markdown files.
- Covered cities currently include Aachen, Dresden, Heidelberg, Klagenfurt, Krakow, Leipzig, Mannheim, Munich, and Munster.
- Covered countries currently include Austria, Germany, and Poland.

This is the strongest contract design in the enrichment area. `SourceRegistry.load()` reads `sources.yaml`, resolves every `source_id` to exactly one Markdown filename stem, and exposes controlled metadata search plus bounded regex tools.

In the target scope, this stage starts with a handoff from the other repository's PDF-to-Markdown flow when 3rd party sources begin as PDFs or documents. This repo should consume the converted source library as given, run controlled source selection/search, extract facts, resolve them against CCC evidence, and persist the audit trail.

Current input contract:

- `GapManifest` from gap analysis.
- `context_bundle` from CCC extraction.
- `documents/source_library/sources.yaml`.
- Markdown files whose stems match `source_id`.
- Converted 3rd party Markdown handed off from the PDF-to-Markdown process when source files begin as PDFs.
- `EnrichmentConfig.external_source_*` limits.

Current controlled tool contract:

- `get_tag_options`
- `list_candidate_sources`
- `regex_search`
- `expand_hits`
- `add_evidence_candidates`
- `list_evidence_candidates`
- `mark_no_evidence_found`

Current output contract:

- `ExternalEvidenceClaim`
- `ExternalEvidenceResolution`
- `NoEvidenceRecord`
- Session tool-call audit records

Artifacts already written:

- `external_sources/external_evidence.json`: run-local tool state with candidates, no-evidence records, and tool calls.
- `enrichment/external_evidence.json`
- `enrichment/external_resolutions.json`
- `enrichment/external_no_evidence.json`
- `enrichment/enrichment_bundle.json`
- `context_bundle.json` with `enrichment.external_*` fields.

Benchmarks already present:

- Fixture: `backend/benchmarks/external_sources/krakow_external_source_benchmark.json`
- Runner: `backend/scripts/benchmark_external_source_pipeline.py`
- Output root: `output/external_source_benchmarks/krakow/<run_id>/`
- Expected benchmark artifacts: `benchmark_summary.json`, `context_bundle.json`, optional `writer_answer.md`, and `external_sources/external_evidence.json`
- Tests: `tests/test_external_sources.py`, `tests/test_writer_citations.py`, external-source portions of `tests/test_enrichment_integration.py`.

### What is not modular enough

- The code lives under `backend/modules/web_researcher/`, even though it is not websearch. It is a governed local source-library reader and should be reusable without importing websearch concerns.
- `run_external_source_enrichment()` returns a tuple. That tuple is clear in code, but a Pydantic result model would be safer and easier for other tools to consume.
- The session tool audit exists on disk, but normal pipeline artifact registration is incomplete. The enrichment serializer registers final external evidence/resolution/no-evidence JSON files, but the low-level tool audit file under `external_sources/external_evidence.json` is not clearly registered in `run.json` by the normal pipeline.
- The current implementation assumes pre-ingested Markdown plus `sources.yaml`. There is no explicit handoff record for consuming converted 3rd party PDFs or newly collected source files from the PDF-to-Markdown repo.
- The benchmark is good, but narrow. It is Krakow-only and has 4 cases.

### Recommended boundary

Move or wrap this functionality as:

```text
backend/modules/external_sources/
  models.py          # SourceMetadata, SearchHit, EvidenceCandidate, ExternalSourceRunResult
  source_handoff.py  # consume converted Markdown library from PDF-to-Markdown repo
  registry.py        # SourceRegistry and source lookup
  tools.py           # ExternalSearchSession controlled tools
  agent.py           # researcher/finalizer agent builders
  resolver.py        # external evidence vs CCC evidence
  artifacts.py       # external_sources/* and enrichment external_* registration
  benchmarks/
    fixtures/
```

Then `web_researcher` or a future `enrichment` orchestrator can call it as a dependency.

## 3. Search, Process, And Extract Web Data

### What already looks modular

Current main homes:

- `backend/modules/web_researcher/search_planner.py`
- `backend/modules/web_researcher/search_worker.py`
- `backend/modules/web_researcher/search.py`
- `backend/modules/web_researcher/scraper.py`
- `backend/modules/web_researcher/extractor.py`
- `backend/modules/web_researcher/relevance.py`
- `backend/modules/web_researcher/deep_diver.py`
- `backend/modules/web_researcher/freshness.py`
- `backend/modules/web_researcher/tier1_web.py`
- `backend/data/tier1_web_sources.yaml`

Current tier-1 web allowlist:

- 14 sources.
- 11 `site_search`, 2 `api`, 1 `auth_required`.

The websearch logic has a reasonable functional split:

- `search_planner.py` turns a `GapManifest` into `SearchBatch` objects.
- `search.py` wraps Serper.
- `scraper.py` wraps Firecrawl and simple scraping.
- `relevance.py` filters candidate web results.
- `extractor.py` extracts field values from scraped content.
- `post_extraction_validator.py` rejects bad findings.
- `search_worker.py` coordinates tier-1 pre-pass, open web pass, deep dive, extraction, and deduplication.
- `freshness.py` compares web findings against CCC evidence.

Current input contract:

- `GapManifest`
- `SearchBatch`
- `AppConfig.enrichment`
- `SERPER_API_KEY`
- `FIRECRAWL_API_KEY`
- Optional tier-1 allowlist at `backend/data/tier1_web_sources.yaml`

Current output contract:

- `WebFinding`
- `FreshnessResult`
- Final `EnrichedField` statuses after context merger.

Artifacts currently written:

- `enrichment/web_findings.json` only when findings exist.
- `enrichment/freshness_results.json` only when freshness results exist.
- `enrichment/enrichment_bundle.json`.
- Progress items in the frontend/run progress flow.

Tests already present:

- `tests/test_search_planner.py`
- `tests/test_search_worker_tier1.py`
- `tests/test_freshness.py`
- Web/freshness portions of context merger and enrichment tests.

### What is not modular enough

- Web search/process/extract is not a separate module. It is one slice of `web_researcher`, which also contains external-source and assumptions logic.
- Search plans are not persisted as first-class artifacts. If websearch finds the wrong thing, there is no standard `web_search/search_plan.json`.
- Raw Serper results are not persisted.
- Scrape attempts, skipped URLs, relevance decisions, extraction attempts, validation rejections, and deep-dive page selection are not persisted as structured artifacts.
- There is no dedicated websearch benchmark equivalent to the external-source benchmark.
- `tier1_web.py` has `api` access entries, but the current worker uses them as site-search coverage hints. There is no separate API reader path for tier-1 sources marked `api`.

### Recommended boundary

Create a distinct websearch module:

```text
backend/modules/web_search/
  models.py          # SearchBatch, SearchResult, WebFinding, FreshnessResult
  planner.py         # query planning
  clients.py         # Serper, Firecrawl, API readers
  relevance.py
  extractor.py
  validator.py
  freshness.py
  artifacts.py       # plans, search results, scrapes, extraction, freshness
  benchmarks/
```

This would let other tools invoke websearch without pulling in external-source or assumptions code.

## 4. Assumptions

### What already looks modular

Current main homes:

- Automatic estimator: `backend/modules/web_researcher/assumptions_estimator.py`
- Post-run review service: `backend/api/services/assumptions_review.py`
- API route: `backend/api/routes/assumptions.py`
- API models: `MissingDataItem`, `AssumptionsPayload`, `RegenerationResult` in `backend/api/models.py`
- Enrichment models: `AssumptionRecord`, `NonEstimableRecord`, `EstimateRange` in `backend/modules/web_researcher/models.py`

There are two related but separate assumptions workflows:

1. Automatic enrichment estimator:
   - Runs inside `run_enrichment_pipeline()`.
   - Inputs `GapManifest`, `EnrichedField[]`, CCC context, national benchmark web findings, and comparative web findings.
   - Outputs `AssumptionRecord[]`, `NonEstimableRecord[]`, and `saturation_warning`.
   - Persists through `enrichment/assumptions.json`, `enrichment/non_estimable.json`, and `enrichment/enrichment_bundle.json`.

2. Post-run review and regenerate:
   - Runs through `/api/v1/runs/{run_id}/assumptions/discover` and `/api/v1/runs/{run_id}/assumptions/apply`.
   - Inputs a completed run, final document, context bundle, and user-edited assumptions.
   - Outputs revised content and optional persisted artifacts.
   - Persists only when `persist_artifacts=true`.

Post-run assumptions artifacts when persistence is enabled:

- `assumptions/discovered.json`
- `assumptions/edited.json`
- `assumptions/revised_context_bundle.json`
- `assumptions/final_with_assumptions.md`

Tests already present:

- `tests/test_assumptions_estimator.py`
- `tests/test_api_assumptions.py`
- Assumptions portions of `tests/test_enrichment_integration.py`, `tests/test_enrichment_models.py`, and `tests/test_enrichment_services.py`.

### What is not modular enough

- Assumptions do not have a dedicated module. The estimator lives in `web_researcher`; the review flow lives in `api/services`.
- There are two output shapes: `enrichment.assumptions[]` for automatic estimates and top-level `context_bundle["assumptions"]` for edited review assumptions. That makes writer/tool reuse harder.
- There is no dedicated assumptions benchmark.
- Known issue documentation already flags estimator quality risks, including anchoring on the wrong CCC fragment and overuse of expert heuristic scaling.
- Review persistence is opt-in. That is useful for UX, but it means written artifact coverage is not uniform unless the caller explicitly requests persistence.

### Recommended boundary

Create:

```text
backend/modules/assumptions/
  models.py          # AssumptionRecord, MissingDataItem, AssumptionRunResult
  estimator.py       # automatic estimator
  reviewer.py        # discover/apply/rewrite business logic
  artifacts.py       # assumptions/* and enrichment assumptions registration
  api_adapter.py     # thin adapter used by backend/api/routes
  benchmarks/
```

The API route should become a thin shell over this module. The writer should consume one canonical assumptions contract, regardless of whether assumptions came from automatic estimation or user review.

## Shared Components Across Modules

| Shared component | Current owner | Shared by | What it carries or creates | Reuse concern |
| --- | --- | --- | --- | --- |
| `AppConfig` and `llm_config.yaml` | `backend/utils/config.py` | Orchestrator, CCC reader, vector store, external sources, websearch, assumptions, writer, benchmarks | Models, token budgets, feature flags, source dirs, websearch toggles, vector settings | Good centralization, but `EnrichmentConfig` now owns unrelated external-source, websearch, and assumptions settings. |
| `RunPaths` | `backend/utils/paths.py` | Orchestrator, run logger, API services, tests | Canonical per-run paths for `run.json`, `context_bundle.json`, `markdown/*`, `final.md` | Good for core run artifacts; enrichment/external/assumptions sub-artifacts are less explicit. |
| `RunLogger` | `backend/services/run_logger.py` | Orchestrator, enrichment serializer, API diagnostics | Structured run log, context bundle, artifact registry, run summary, LLM usage summary | Strong shared service; individual modules should register all sub-artifacts through it. |
| `context_bundle` | Runtime dict built by `RunLogger` and modules | CCC, external sources, websearch, assumptions, writer, API, benchmarks | Cross-stage payload with `markdown`, `enrichment`, final output path, queries, selected cities | Biggest implicit contract. It should have a typed model or versioned schema for reuse. |
| Enrichment Pydantic models | `backend/modules/web_researcher/models.py` | Gap analysis, external sources, websearch, freshness, assumptions, writer tests | `GapManifest`, `WebFinding`, `ExternalEvidenceClaim`, `EnrichedField`, `AssumptionRecord`, etc. | Models are useful but too many unrelated contracts live in one file/package. |
| JSON artifact helpers | `backend/utils/json_io.py`, `backend/modules/orchestrator/utils/io.py` | Scripts, orchestrator, API assumptions, enrichment | JSON read/write helpers and artifact serialization | There are two JSON writer locations. Standardize on one shared helper plus module artifact writers. |
| Agent runtime wrappers | `backend/services/agents.py` | Markdown researcher, external-source agents, writer, other LLM flows | OpenRouter model construction, model settings, sync agent calls | Good shared abstraction. Keep module-specific prompts/contracts outside it. |
| Prompts | `backend/prompts/` | Markdown researcher, orchestrator, external sources, writer, benchmark judges, context chat | LLM behavior contracts | Prompt files are centralized, but runtime schemas must stay aligned with Pydantic models and tests. |
| Retry policy | `backend/utils/retry.py`, `AppConfig.retry` | Markdown researcher, vector retrieval, benchmarks, agent clients | Retry events, backoff, retry exhausted logs | Useful shared behavior; make sure non-CCC stages emit the same retry diagnostics. |
| Progress tracker | `backend/services/progress_tracker.py` | Orchestrator and enrichment | Frontend-visible stage/item progress | Good UX surface, but not a replacement for durable artifacts. |
| City key formatting | `backend/utils/city_normalization.py` | CCC loading, retrieval, context merger, API, tests | Stable city keys and display formatting | Essential shared utility. Some code still uses `.lower()` for city-field keys, which is a known drift risk. |
| Tokenization | `backend/utils/tokenization.py` | Markdown batching, writer, retrieval benchmarking | Token counts, chunking, budgets | Good shared utility. |
| Data folders | `documents/`, `documents/source_library/`, `backend/data/tier1_web_sources.yaml` | CCC reader, external sources, websearch | CCC Markdown, governed source library, tier-1 web allowlist | The source stores are distinct, but only external sources have a strong manifest contract. |
| Benchmark infrastructure | `backend/benchmarks/`, `backend/scripts/*benchmark*.py` | Retrieval, recall, external sources, TEF, chunking | Fixtures, reports, run matrices, benchmark artifacts | Strong for CCC/retrieval; missing equivalent harnesses for open websearch and assumptions. |
| Writer context builder | `backend/modules/writer/utils/multi_pass.py`, `backend/api/services/run_context.py` | Writer, API exports, tests | Writer-safe context subset and rendered context export | Useful, but it hides which fields each upstream module must provide. |

## Artifact Coverage By Stage

| Stage | Strong artifacts today | Missing or weak artifacts |
| --- | --- | --- |
| CCC read/process/extract | `markdown/excerpts.json`, `accepted_excerpts.json`, `rejected_excerpts.json`, `decision_audit.json`, `references.json`, `batches.json`, optional `retrieval.json` | Typed CCC source manifest; standalone CCC reader summary artifact; explicit PDF-to-Markdown handoff artifact. |
| 3rd party read/process/extract | `external_sources/external_evidence.json`, `enrichment/external_evidence.json`, `external_resolutions.json`, `external_no_evidence.json` | Tool audit not clearly registered in normal `run.json`; benchmark is city-specific; explicit PDF-to-Markdown handoff artifact. |
| Web search/process/extract | `enrichment/web_findings.json`, `enrichment/freshness_results.json`, progress items | Search plan, raw results, relevance decisions, scrape attempts, extraction attempts, rejected findings, deep-dive trace. |
| Assumptions | `enrichment/assumptions.json`, `enrichment/non_estimable.json`, optional `assumptions/*.json`, optional `final_with_assumptions.md` | No always-on assumptions audit; no automatic assumptions benchmark report; edited assumptions use a different context location. |

## Benchmark Coverage By Stage

| Stage | Existing benchmark coverage | Gap |
| --- | --- | --- |
| CCC read/process/extract | Retrieval strategy benchmark, gold recall benchmark, chunking benchmark, TEF source-truth benchmark for initiative extraction | Good baseline; CCC reader itself could be named/formalized better and should include PDF-to-Markdown handoff cases. |
| 3rd party read/process/extract | Governed external-source Krakow benchmark with 4 cases | Broaden beyond Krakow, add no-evidence/conflict cases, add PDF-to-Markdown handoff cases, and make benchmark result model reusable. |
| Web search/process/extract | Unit tests for planner, tier-1 worker, freshness | Needs a live or mocked benchmark harness with fixed search/scrape fixtures and artifact scoring. |
| Assumptions | Unit/API tests | Needs benchmark cases for estimate quality, anchor selection, confidence calibration, and non-estimable decisions. |

## What Already Matches The Goal

- The main run already writes inspectable artifacts under `output/<run_id>/`.
- The CCC path has the best artifact discipline and benchmark discipline.
- The external-source path has the best controlled tool contract.
- The enrichment output model already separates `field_manifest`, `gap_manifest`, `external_evidence`, `web_findings`, `freshness_results`, `assumptions`, and `non_estimable`.
- The writer context export deliberately filters run bookkeeping and keeps writer-visible enrichment fields.
- There is a central config module and shared run logger.
- Most functions have type hints and the main script benchmark entrypoints follow the repository's standalone script style.

## What Does Not Yet Match The Goal

- `backend/modules/web_researcher/` is overloaded. It is currently the home for at least four separate reusable capabilities.
- The source-stage boundaries are not visible enough in folder structure.
- Some contracts are typed Pydantic models, while others are loose dict payloads.
- `context_bundle` is the central cross-module contract, but it is not itself a typed, versioned model.
- Web search/process/extract artifacting is too thin for debugging or reproducible benchmarking.
- Assumptions are split between enrichment and API review with different shapes.
- External-source tool-call audit artifacts are not treated as first-class run artifacts in the normal pipeline.
- Web search/process/extract and assumptions do not yet have dedicated benchmark harnesses.

## Proposed Target Structure

One pragmatic target is:

```text
backend/modules/
  ccc_reader/
    models.py
    services.py
    retrieval.py
    artifacts.py
    benchmarks/

  external_sources/
    models.py
    registry.py
    tools.py
    agent.py
    resolver.py
    artifacts.py
    benchmarks/

  web_search/
    models.py
    planner.py
    clients.py
    relevance.py
    extractor.py
    freshness.py
    artifacts.py
    benchmarks/

  assumptions/
    models.py
    estimator.py
    reviewer.py
    artifacts.py
    api_adapter.py
    benchmarks/

  enrichment/
    models.py
    gap_analysis.py
    context_merger.py
    pipeline.py
```

This target preserves current behavior while making each capability reusable by other tools:

- Other tools can call `ccc_reader` to get CCC evidence.
- Other tools can call `external_sources` to search curated 3rd party Markdown.
- Other tools can call `web_search` for open/tier-1 web search, processing, extraction, and freshness evidence.
- Other tools can call `assumptions` for missing-data estimates or review/regenerate flows.
- `enrichment` becomes the orchestrator that composes the four source/estimate modules.

## Priority Refactor Order

1. Define explicit result models for each stage.
   - `CCCReadResult`
   - `ExternalSourceRunResult`
   - `WebSearchRunResult`
   - `AssumptionsRunResult`

2. Add module-level artifact writers.
   - Each stage should own `artifacts.py`.
   - Each artifact writer should register artifacts through `RunLogger.record_artifact()`.

3. Move or wrap external-source code into `backend/modules/external_sources/`.
   - This is the cleanest first extraction because the contracts already exist.

4. Split websearch into `backend/modules/web_search/`.
   - Add persisted search plan and raw decision artifacts while moving.

5. Unify assumptions under `backend/modules/assumptions/`.
   - Keep API routes thin.
   - Normalize automatic and edited assumptions into one writer-facing contract.

6. Formalize `context_bundle`.
   - Add a versioned schema or Pydantic model for the cross-stage context.
   - Keep backwards-compatible reading only if explicitly needed; otherwise avoid dual paths.

7. Add missing benchmarks.
   - Web search/process/extract benchmark with fixed fixtures and scored artifacts.
   - Assumptions benchmark with expected method/confidence/anchor behavior.
   - Broader external-source benchmark beyond Krakow.

## Bottom Line

The project is already close to the desired pipeline at the behavior level. The main gap is packaging and reusable contracts.

CCC data and governed 3rd party data already have meaningful contracts, artifacts, and benchmarks after Markdown exists. The target scope should add the PDF-to-Markdown handoff before those stages, without adding document-normalization responsibility to this repo. Web search/process/extract and assumptions need stronger artifact and benchmark discipline. All four stages should be made physically visible as separate modules or submodules, with `enrichment` acting as the composition layer instead of hosting everything inside `web_researcher`.
