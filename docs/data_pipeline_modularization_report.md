# Data Pipeline Modularization Assessment

## Purpose

This document evaluates whether the repository should prioritize a full
physical modularization of the source-to-evidence pipeline.

It keeps the underlying findings about the current pipeline stages, artifacts,
contracts, and benchmarks, but it changes the main recommendation:

- Do not prioritize a large structural split right now.
- Prefer better documentation, clearer contracts, and targeted hardening within
  the current shared runtime.
- Revisit physical extraction only if reuse, separate ownership, or debugging
  pressure becomes strong enough to justify the added maintenance cost.

The key reason is that the code is not just a set of isolated stage folders.
It is also a shared runtime with common logging, artifact writing, config,
prompt/runtime alignment, retry behavior, path conventions, and a central
`context_bundle` contract. Splitting files without addressing that shared layer
would create more synchronization work while delivering limited practical
benefit in the current repository.

## Scope

The target pipeline under discussion is the full source-to-evidence pipeline,
not just file loading:

1. Read, process, and extract CCC data.
2. Read, process, and extract 3rd party data.
3. Search, process, and extract web data.
4. Generate, review, and apply assumptions.

The intended logical split is into these four parts. The question is whether
that split should become a full physical module reorganization now, or whether
the repository should first document and harden the existing staged design
without spreading shared runtime responsibilities across more packages.

In scope for each source stage:

- Source acquisition or handoff from an upstream acquisition process.
- PDF-to-Markdown conversion handoff. The implementation should be based on the
  other repository that already handles PDF-to-Markdown, rather than
  redesigned in this repo.
- Retrieval or source selection.
- Data extraction into typed facts and evidence.
- Validation, conflict/no-evidence handling, and provenance.
- Written artifacts and benchmark coverage.

## Recommendation At A Glance

The current repository is already reasonably modular at the behavior level.
Its bigger weakness is visibility, not absence of structure.

The best near-term path is:

1. Document the current stage boundaries clearly.
2. Document the shared runtime surface explicitly.
3. Tighten a few contracts and artifact gaps where they matter.
4. Avoid a broad package reorganization unless there is a concrete reuse or
   ownership need.

The best argument against a large refactor is not that the findings are wrong.
The findings are mostly correct. The best argument is that many of the current
benefits of a reorg can be achieved more cheaply through documentation and
small contract hardening, while a full split would create ongoing alignment
work across:

- `RunLogger` and artifact registration
- `AppConfig` and `EnrichmentConfig`
- `RunPaths`
- `context_bundle`
- prompt/runtime schema alignment
- retry policy
- progress reporting
- writer-facing context shaping

That is real maintenance work, not just a one-time move.

## Executive Summary

The project already has many of the right building blocks. The main issue is
that the boundaries are uneven and not visible enough.

Findings that still hold:

- CCC read/process/extract is the most mature area. It has a distinct
  `backend/modules/markdown_researcher/` package, optional
  `backend/modules/vector_store/` retrieval, strong run artifacts, and several
  benchmark paths.
- 3rd party read/process/extract is also fairly mature at the contract level.
  `documents/source_library/sources.yaml` currently has 46 source entries
  matching 46 Markdown files, and the governed external-source search has
  Pydantic models, controlled tools, artifacts, tests, and a Krakow benchmark
  fixture.
- Web search/process/extract is functionally separated across files, but not
  physically separated from the enrichment package. It has good logging and
  typed outputs, but weak artifacting. Search plans, raw search results,
  scrape decisions, and extraction attempts are not persisted as first-class
  run artifacts.
- Assumptions are split into two concepts: automatic enrichment estimates in
  `backend/modules/web_researcher/assumptions_estimator.py` and post-run
  editable assumptions in `backend/api/services/assumptions_review.py`. Both
  have contracts and tests, but there is no dedicated assumptions benchmark.

The previous modularization framing pushed toward creating four physically
separate modules immediately. That conclusion is too strong for the current
need.

The stronger conclusion is:

- The repository already behaves like a staged pipeline.
- The main missing piece is that this shape is not documented clearly enough
  for readers and future reuse decisions.
- The most expensive part of turning these areas into transportable modules is
  not file movement. It is preserving consistent runtime behavior across
  logging, artifacts, config, shared models, and orchestration contracts.
- That cost is high enough that documentation and targeted hardening are the
  better near-term investment.

## Current Pipeline Shape

Current runtime flow in `backend/modules/orchestrator/module.py` and
`backend/modules/web_researcher/module.py`:

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

Important detail: the current flow already treats CCC as the baseline source.
Governed external Markdown, websearch, and assumptions are enrichment layers
that run after CCC evidence is available and gaps are identified.

What is missing from this view is the upstream PDF-to-Markdown handoff. That
handoff should be included in the target scope as an explicit upstream
dependency, but it does not require a structural rewrite inside this repo.

## Why Documentation First Is The Better Recommendation

The case against immediate physical modularization is mostly about shared
runtime coupling and operational consistency.

What a broad split would have to preserve:

- the same logging shape
- the same artifact registration model
- the same per-run path conventions
- the same prompt and Pydantic contract alignment
- the same progress reporting behavior
- the same writer-facing context expectations
- the same retry and error semantics

Those are not incidental details. They are part of how the pipeline is
operated, debugged, benchmarked, and explained.

For the current repository, clearer documentation can provide most of the
reader benefit at much lower cost:

- a written map of stage boundaries
- a written map of shared runtime ownership
- a written artifact catalog
- a written contract summary for each stage
- a written explanation of which central pieces are intentionally shared

That would make the system easier to understand without immediately forcing the
project to duplicate or abstract the shared layer in awkward ways.

## Stage Readiness Matrix

| Stage | Current code home | Contract quality | Logging | Artifacts | Benchmarks | Assessment for now |
| --- | --- | --- | --- | --- | --- | --- |
| Read/process/extract CCC data | `backend/modules/markdown_researcher/`, `backend/modules/vector_store/`, orchestrator glue | Partial to strong. `MarkdownResearchResult`, `MarkdownExcerpt`, chunk dicts, and retrieval artifact shape are defined, but chunk inputs are still plain dicts. | Strong enough. Uses module loggers, run log, run summary, progress tracker. | Strong. `markdown/*.json`, `context_bundle.json`, `run.json`, `run.log`, `run_summary.txt`. | Strongest of all stages. Retrieval benchmark, recall benchmark, chunking benchmark, unit tests. | Mature enough behaviorally. Better naming and contracts would help, but a package move is not the best first step. |
| Read/process/extract 3rd party data | `documents/source_library/`, `backend/modules/web_researcher/external_sources.py`, `external_agent.py`, `external_resolver.py` | Strong. `SourceMetadata`, `SearchHit`, `EvidenceCandidate`, `ExternalEvidenceClaim`, `ExternalEvidenceResolution`, `NoEvidenceRecord`. | Good. Tool calls and agent failures are logged. | Medium to strong. `stage_files/008_enrichment/external_source_search_audit.json` keeps the search trace, while accepted external outputs live in the canonical enrichment bundle. | Medium. Dedicated Krakow external-source benchmark with 4 cases plus unit tests. | Good candidate for future extraction, but the current need is better artifact registration and documentation of boundaries. |
| Search/process/extract web data | `backend/modules/web_researcher/search_planner.py`, `search_worker.py`, `search.py`, `scraper.py`, `extractor.py`, `freshness.py`, `tier1_web.py` | Medium. `SearchBatch`, `WebFinding`, `FreshnessResult`, and tier-1 models exist. | Good. Search, scrape, relevance, tier-1, worker, and freshness paths log. | Medium. Canonical web/freshness outputs live in `stage_files/008_enrichment/enrichment_bundle.json`; `web_research_audit.json` includes non-bundle trace data, structured scrape warnings, and search execution summaries for tier-1/open query volume. | Weak. Unit tests exist, but no dedicated live websearch benchmark. | The main problem is not folder placement. It is missing benchmark discipline and deeper extraction observability. Fix those before considering a module split. |
| Generate/review/apply assumptions | `backend/modules/web_researcher/assumptions_estimator.py`, `backend/api/services/assumptions_review.py`, `backend/api/routes/assumptions.py` | Medium. Automatic assumptions use `AssumptionRecord` and `NonEstimableRecord`; review flow uses API models `MissingDataItem`, `AssumptionsPayload`, `RegenerationResult`. | Medium. Service and estimator log key LLM calls and skip/failure paths. | Medium. Automatic assumptions are persisted through enrichment artifacts; review artifacts persist only with `persist_artifacts=true`. | Weak. Unit/API tests exist, but no assumptions benchmark. | The bigger issue is contract fragmentation, not folder count. Unify the writer-facing contract before restructuring code homes. |

## 1. Read, Process, And Extract CCC Data

### What already looks modular

Current main homes:

- `backend/modules/markdown_researcher/`
- `backend/modules/vector_store/`
- `backend/modules/orchestrator/module.py`
- `documents/*.md`

The CCC source layer is mostly represented as top-level Markdown city files
under `documents/`. The normal non-vector path loads those files through
`load_markdown_documents()`, chunks them, groups them by normalized city key,
and sends them through `extract_markdown_excerpts()`.

In the target scope, this stage starts with a handoff from the other
repository's PDF-to-Markdown flow. This repo should consume the converted
Markdown as given, then handle chunking, extraction, validation of extracted
facts, artifact writing, and benchmarks.

The vector path is also separated in `backend/modules/vector_store/`. When
`VECTOR_STORE_ENABLED=true`, the orchestrator retrieves chunks through
`retrieve_chunks_for_queries()`, serializes a retrieval artifact with
`build_retrieval_artifact()`, and converts retrieved chunks back into markdown
researcher input shape with `as_markdown_documents()`.

Current input contract:

- User question and up to 3 retrieval queries.
- Optional selected city filter.
- `MARKDOWN_DIR`, default `documents`.
- Converted CCC Markdown handed off from the PDF-to-Markdown process.
- `MarkdownResearcherConfig` and optional `VectorStoreConfig`.
- Markdown chunk dicts containing at least `path`, `city_name`, `city_key`,
  `content`, `chunk_id`, and `chunk_index`.

Current output contract:

- `MarkdownResearchResult`
- `MarkdownExcerpt`
- `MarkdownBatchFailure`
- Run-level `context_bundle["markdown"]`
- Citation references derived from `stage_files/006_markdown_extraction/accepted_excerpts.json`

Artifacts already written:

- `stage_files/002_query_preparation/research_question.json`
- `stage_files/005_markdown_batching/batches.json`
- `stage_files/006_markdown_extraction/accepted_excerpts.json`
- `stage_files/006_markdown_extraction/rejected_chunks.json`
- `stage_files/006_markdown_extraction/decision_audit.json`
- `stage_files/003_retrieval/retrieval.json` when vector retrieval is enabled
- `context_bundle.json`
- `run.json`
- `run.log`
- `run_summary.txt`

Benchmarks already present:

- Retrieval strategy benchmark: `backend/scripts/run_retrieval_benchmark.py`
- Retrieval benchmark fixtures/config: `backend/benchmarks/prompts/`,
  `backend/benchmarks/config/`
- Gold recall benchmark: `backend/scripts/benchmark_recall.py`,
  `backend/benchmarks/gold_recall/`
- Chunking benchmark: `backend/scripts/benchmark_chunking_strategy.py`,
  `backend/modules/vector_store/benchmarking.py`
- Tests: `tests/test_markdown_researcher.py`,
  `tests/test_markdown_services.py`, `tests/test_retrieval_benchmark.py`,
  `tests/test_benchmark_recall.py`, `tests/test_chunking_benchmark.py`,
  vector-store tests

### Findings and limitations

- The code is named `markdown_researcher`, not `ccc_reader`. That is accurate
  technically, but less reusable as a domain stage.
- PDF-to-Markdown is not represented as an explicit handoff/input contract in
  this repo yet.
- The orchestrator owns important CCC artifact assembly: batches,
  accepted/rejected/audit artifacts, references, inspected city metadata, and
  vector retrieval artifact registration.
- The chunk input contract is still `dict[str, object]`. It works, but it is
  weaker than the Pydantic model contracts used elsewhere.
- There is no source manifest for top-level CCC markdown files comparable to
  `documents/source_library/sources.yaml`. The CCC runtime still depends
  heavily on filename stem as city identity.

### Why this does not justify an immediate package split

This is already the strongest stage in the system. The missing value is not a
new folder tree. The missing value is a clearer written contract that explains:

- where CCC processing starts after PDF-to-Markdown handoff
- what the chunk contract is
- what artifacts are guaranteed
- which parts are intentionally shared with the orchestrator

In other words, this stage would benefit more from better documentation and a
small amount of contract hardening than from a large move.

### If physical extraction becomes necessary later

If another repository or another team needs a distinct CCC module later, a
reasonable target would be:

```text
backend/modules/ccc_reader/
  models.py
  source_handoff.py
  services.py
  retrieval.py
  extraction.py
  artifacts.py
  benchmarks/
```

That should be treated as a deferred option, not the current recommendation.

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

- `documents/source_library/sources.yaml`: 46 entries
- `documents/source_library/**/*.md`: 46 Markdown files
- Covered cities currently include Aachen, Dresden, Heidelberg, Klagenfurt,
  Krakow, Leipzig, Mannheim, Munich, and Munster
- Covered countries currently include Austria, Germany, and Poland

This is the strongest contract design in the enrichment area.
`SourceRegistry.load()` reads `sources.yaml`, resolves every `source_id` to
exactly one Markdown filename stem, and exposes controlled metadata search plus
bounded regex tools.

In the target scope, this stage starts with a handoff from the other
repository's PDF-to-Markdown flow when 3rd party sources begin as PDFs or
documents. This repo should consume the converted source library as given, run
controlled source selection and search, extract facts, resolve them against CCC
evidence, and persist the audit trail.

Current input contract:

- `GapManifest` from gap analysis
- `context_bundle` from CCC extraction
- `documents/source_library/sources.yaml`
- Markdown files whose stems match `source_id`
- Converted 3rd party Markdown handed off from the PDF-to-Markdown process
  when source files begin as PDFs
- `EnrichmentConfig.external_source_*` limits

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

- `stage_files/008_enrichment/external_source_search_audit.json`: run-local
  tool state with searched city-fields, candidates, validated claims, rejected
  claims, no-evidence records, resolutions, and tool calls
- `stage_files/008_enrichment/enrichment_bundle.json` with
  `external_evidence`, `external_resolutions`, and `external_no_evidence`
- `context_bundle.json` with `enrichment.external_*` fields

Benchmarks already present:

- Fixture:
  `backend/benchmarks/external_sources/krakow_external_source_benchmark.json`
- Runner: `backend/scripts/benchmark_external_source_pipeline.py`
- Output root: `output/external_source_benchmarks/krakow/<run_id>/`
- Expected benchmark artifacts: `benchmark_summary.json`,
  `context_bundle.json`, optional `writer_answer.md`, and
  `stage_files/008_enrichment/external_source_search_audit.json`
- Tests: `tests/test_external_sources.py`, `tests/test_writer_citations.py`,
  external-source portions of `tests/test_enrichment_integration.py`

### Findings and limitations

- The code lives under `backend/modules/web_researcher/`, even though it is
  not websearch.
- `run_external_source_enrichment()` returns a tuple. That tuple is clear in
  code, but a result model would be safer and easier for other tools to
  consume.
- The session tool audit exists on disk, but the runtime-state versus
  diagnostic-artifact split still needs clearer documentation. The low-level
  tool audit is now written into `stage_files/008_enrichment/`, alongside the
  final accepted claims and resolutions.
- The current implementation assumes pre-ingested Markdown plus `sources.yaml`.
  There is no explicit handoff record for consuming converted 3rd party PDFs
  or newly collected source files from the PDF-to-Markdown repo.
- The benchmark is good, but narrow. It is Krakow-only and has 4 cases.

### Why this does not justify an immediate package split

This area is the best candidate for future extraction, but that does not make
it the best current priority.

The current pain points are mostly:

- missing visibility of the boundary
- incomplete artifact registration
- limited benchmark breadth

Those can be addressed while the code stays where it is. If the repository only
needs to keep this capability understandable and reliable, better
documentation and small hardening steps are enough for now.

### If physical extraction becomes necessary later

If a separate module is eventually justified, a reasonable target would be:

```text
backend/modules/external_sources/
  models.py
  source_handoff.py
  registry.py
  tools.py
  agent.py
  resolver.py
  artifacts.py
  benchmarks/
    fixtures/
```

That remains a future extraction path, not the primary recommendation.

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

- 14 sources
- 11 `site_search`, 2 `api`, 1 `auth_required`

The websearch logic has a reasonable functional split:

- `search_planner.py` turns a `GapManifest` into `SearchBatch` objects
- `search.py` wraps Serper
- `scraper.py` wraps Firecrawl and simple scraping
- `relevance.py` filters candidate web results
- `extractor.py` extracts field values from scraped content
- `post_extraction_validator.py` rejects bad findings
- `search_worker.py` coordinates tier-1 pre-pass, open web pass, deep dive,
  extraction, and deduplication
- `freshness.py` compares web findings against CCC evidence

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
- Final `EnrichedField` statuses after context merger

Artifacts currently written:

- `stage_files/008_enrichment/enrichment_bundle.json` with web findings and
  freshness results
- `stage_files/008_enrichment/web_research_audit.json` only when web research
  has trace data beyond the canonical bundle
- Progress items in the frontend/run progress flow
- `stage_files/001_input_snapshot/planned_stages.json`, written once at run
  start, gives the frontend and run-status API the artifact-backed planned
  stage list for new runs
- The run artifacts API exposes a derived `stage_details.enrichment` display
  payload so web research, freshness, and external-source validation can be
  inspected under the same backend enrichment stage

Tests already present:

- `tests/test_search_planner.py`
- `tests/test_search_worker_tier1.py`
- `tests/test_freshness.py`
- Web and freshness portions of context merger and enrichment tests

### Findings and limitations

- Web search/process/extract is not a separate module. It is one slice of
  `web_researcher`, which also contains external-source and assumptions logic.
- Search plans are not persisted as first-class artifacts. If websearch finds
  the wrong thing, there is no standard `web_search/search_plan.json`.
- Raw Serper results are not persisted.
- Scrape attempts, skipped URLs, relevance decisions, extraction attempts,
  validation rejections, and deep-dive page selection are not persisted as
  structured artifacts.
- There is no dedicated websearch benchmark equivalent to the external-source
  benchmark.
- `tier1_web.py` has `api` access entries, but the current worker uses them as
  site-search coverage hints. There is no separate API reader path for tier-1
  sources marked `api`.

### Why this does not justify an immediate package split

This stage is the clearest example of a place where structural separation is
not the main problem.

The real missing pieces are:

- artifact coverage
- benchmark discipline
- clearer written description of how the worker behaves

Even if the code were moved tomorrow, those gaps would still exist. The more
valuable immediate work is to document the current flow and tighten the missing
artifacts. Folder movement should come later, if at all.

### If physical extraction becomes necessary later

If a separate websearch module is eventually needed, a reasonable target would
be:

```text
backend/modules/web_search/
  models.py
  planner.py
  clients.py
  relevance.py
  extractor.py
  validator.py
  freshness.py
  artifacts.py
  benchmarks/
```

That should be treated as a fallback future design, not a current refactor
mandate.

## 4. Assumptions

### What already looks modular

Current main homes:

- Automatic estimator: `backend/modules/web_researcher/assumptions_estimator.py`
- Post-run review service: `backend/api/services/assumptions_review.py`
- API route: `backend/api/routes/assumptions.py`
- API models: `MissingDataItem`, `AssumptionsPayload`,
  `RegenerationResult` in `backend/api/models.py`
- Enrichment models: `AssumptionRecord`, `NonEstimableRecord`,
  `EstimateRange` in `backend/modules/web_researcher/models.py`

There are two related but separate assumptions workflows:

1. Automatic enrichment estimator:
   - Runs inside `run_enrichment_pipeline()`.
   - Inputs `GapManifest`, `EnrichedField[]`, CCC context, national benchmark
     web findings, and comparative web findings.
   - Outputs `AssumptionRecord[]`, `NonEstimableRecord[]`, and
     `saturation_warning`.
   - Persists through `stage_files/010_assumptions/assumptions.json`,
     `stage_files/010_assumptions/non_estimable.json`, and
     `stage_files/010_assumptions/assumptions_bundle.json`.

2. Post-run review and regenerate:
   - Runs through `/api/v1/runs/{run_id}/assumptions/discover` and
     `/api/v1/runs/{run_id}/assumptions/apply`.
   - Inputs a completed run, final document, context bundle, and user-edited
     assumptions.
   - Outputs revised content and optional persisted artifacts.
   - Persists only when `persist_artifacts=true`.

Post-run assumptions artifacts when persistence is enabled:

- `stage_files/assumptions/discovered.json`
- `stage_files/assumptions/edited.json`
- `stage_files/assumptions/revised_context_bundle.json`
- `stage_files/assumptions/final_with_assumptions.md`

Tests already present:

- `tests/test_assumptions_estimator.py`
- `tests/test_api_assumptions.py`
- Assumptions portions of `tests/test_enrichment_integration.py`,
  `tests/test_enrichment_models.py`, and
  `tests/test_enrichment_services.py`

### Findings and limitations

- Automatic assumptions now have dedicated context/artifact helpers.
- Automatic estimates and non-estimable records use top-level
  `context_bundle["assumptions"]`; edited review assumptions also write a
  top-level assumptions block when applied.
- There is no dedicated assumptions benchmark.
- Known issue documentation already flags estimator quality risks, including
  anchoring on the wrong CCC fragment and overuse of expert heuristic scaling.
- Review persistence is opt-in. That is useful for UX, but it means written
  artifact coverage is not uniform unless the caller explicitly requests
  persistence.

### Why this does not justify an immediate package split

This is another case where the main issue is contract shape, not physical
location.

The most valuable immediate work would be:

- document the two assumptions flows clearly
- define one writer-facing assumptions contract
- explain when persistence is guaranteed and when it is optional

Only after that would a folder split carry clear operational value.

### If physical extraction becomes necessary later

If this area is eventually extracted, a reasonable target would be:

```text
backend/modules/assumptions/
  models.py
  estimator.py
  reviewer.py
  artifacts.py
  api_adapter.py
  benchmarks/
```

For now, the recommendation is to unify the contract and documentation first.

## Shared Components Across Modules

This is the strongest argument against immediate physical modularization. The
pipeline is not just four stage folders. It also has a large shared runtime
surface that would have to remain coherent.

| Shared component | Current owner | Shared by | What it carries or creates | Why it matters to the recommendation |
| --- | --- | --- | --- | --- |
| `AppConfig` and `llm_config.yaml` | `backend/utils/config.py` | Orchestrator, CCC reader, vector store, external sources, websearch, assumptions, writer, benchmarks | Models, token budgets, feature flags, source dirs, websearch toggles, vector settings | Centralization is useful today. Splitting modules does not remove the need to keep this aligned. |
| `RunPaths` | `backend/utils/paths.py` | Orchestrator, run logger, API services, tests | Canonical per-run paths for `run.json`, `context_bundle.json`, `markdown/*`, `final.md` | A reorg still needs one shared artifact path contract. |
| `RunLogger` | `backend/services/run_logger.py` | Orchestrator, enrichment serializer, API diagnostics | Structured run log, context bundle, artifact registry, run summary, LLM usage summary | Logging and artifact registration are part of the operational surface that would need coordinated changes. |
| `context_bundle` | Runtime dict built by `RunLogger` and modules | CCC, external sources, websearch, assumptions, writer, API, benchmarks | Cross-stage payload with `markdown`, `enrichment`, final output path, queries, selected cities | This is the biggest implicit contract in the system. Documenting it clearly is more urgent than moving files. |
| Enrichment Pydantic models | `backend/modules/web_researcher/models.py` | Gap analysis, external sources, websearch, freshness, assumptions, writer tests | `GapManifest`, `WebFinding`, `ExternalEvidenceClaim`, `EnrichedField`, `AssumptionRecord`, and related models | The modeling is useful, but it is still a shared seam. A split without a careful contract plan creates drift risk. |
| JSON artifact helpers | `backend/utils/json_io.py`, `backend/modules/orchestrator/utils/io.py` | Scripts, orchestrator, API assumptions, enrichment | JSON read and write helpers and artifact serialization | There are already two writer locations. Multiplying module-specific writers without a documentation and ownership plan would add complexity. |
| Agent runtime wrappers | `backend/services/agents.py` | Markdown researcher, external-source agents, writer, other LLM flows | OpenRouter model construction, model settings, sync agent calls | Runtime execution is already centralized. That is usually a feature, not a defect. |
| Prompts | `backend/prompts/` | Markdown researcher, orchestrator, external sources, writer, benchmark judges, context chat | LLM behavior contracts | Prompt files are centrally managed, which is helpful as long as the contract ownership is documented. |
| Retry policy | `backend/utils/retry.py`, `AppConfig.retry` | Markdown researcher, vector retrieval, benchmarks, agent clients | Retry events, backoff, retry exhausted logs | Shared retry behavior is another cross-cutting concern that must stay consistent. |
| Progress tracker | `backend/services/progress_tracker.py` | Orchestrator and enrichment | Frontend-visible stage and item progress | This is operational UX surface that would need re-coordination after a split. |
| City key formatting | `backend/utils/city_normalization.py` | CCC loading, retrieval, context merger, API, tests | Stable city keys and display formatting | This is a quiet but important shared invariant. |
| Tokenization | `backend/utils/tokenization.py` | Markdown batching, writer, retrieval benchmarking | Token counts, chunking, budgets | Another shared utility whose behavior must stay consistent across stages. |
| Data folders | `documents/`, `documents/source_library/`, `backend/data/tier1_web_sources.yaml` | CCC reader, external sources, websearch | CCC Markdown, governed source library, tier-1 web allowlist | Source placement is already distinct even without a package split. |
| Benchmark infrastructure | `backend/benchmarks/`, `backend/scripts/*benchmark*.py` | Retrieval, recall, external sources, TEF, chunking | Fixtures, reports, run matrices, benchmark artifacts | Benchmark harnesses are shared operational assets, not isolated stage internals. |
| Writer context builder | `backend/modules/writer/utils/multi_pass.py`, `backend/api/services/run_context.py` | Writer, API exports, tests | Writer-safe context subset and rendered context export | This is where upstream stages meet final output expectations. It is a shared contract seam. |

## Artifact Coverage By Stage

| Stage | Strong artifacts today | Missing or weak artifacts | Practical implication |
| --- | --- | --- | --- |
| CCC read/process/extract | `stage_files/006_markdown_extraction/accepted_excerpts.json`, `rejected_chunks.json`, `decision_audit.json`, `stage_files/005_markdown_batching/batches.json`, optional `stage_files/003_retrieval/retrieval.json` | Typed CCC source manifest, standalone CCC reader summary artifact, explicit PDF-to-Markdown handoff artifact | Good enough for operations. Documentation should state this contract clearly. |
| 3rd party read/process/extract | `stage_files/008_enrichment/external_source_search_audit.json`, `stage_files/008_enrichment/enrichment_bundle.json` external fields | Benchmark is city-specific, explicit PDF-to-Markdown handoff artifact | Improve artifact documentation and audit trail before moving code. |
| Web search/process/extract | `stage_files/008_enrichment/enrichment_bundle.json`, optional `stage_files/008_enrichment/web_research_audit.json`, progress items | Relevance decisions, extraction attempts, rejected findings, deeper live-web trace | Scrape warning records and search execution volume summaries now live in the audit artifact; the remaining gaps are still artifact and benchmark discipline problems before they are packaging problems. |
| Assumptions | `stage_files/010_assumptions/assumptions.json`, `stage_files/010_assumptions/non_estimable.json`, `stage_files/010_assumptions/assumptions_stage.json`, optional review/apply artifacts | No automatic assumptions benchmark report | Contract is now normalized at top-level `context_bundle["assumptions"]`; benchmark coverage is the remaining gap. |

## Benchmark Coverage By Stage

| Stage | Existing benchmark coverage | Gap | Practical implication |
| --- | --- | --- | --- |
| CCC read/process/extract | Retrieval strategy benchmark, gold recall benchmark, chunking benchmark, TEF source-truth benchmark for initiative extraction | CCC reader itself could be named and formalized better and should include PDF-to-Markdown handoff cases | Strong enough that documentation can carry a lot of the clarity burden. |
| 3rd party read/process/extract | Governed external-source Krakow benchmark with 4 cases | Broaden beyond Krakow, add no-evidence and conflict cases, add PDF-to-Markdown handoff cases, and make benchmark result model reusable | Worth expanding, but not a reason by itself to restructure folders. |
| Web search/process/extract | Unit tests for planner, tier-1 worker, freshness | Needs a live or mocked benchmark harness with fixed search and scrape fixtures and artifact scoring | This is the highest operational gap in the pipeline. |
| Assumptions | Unit and API tests | Needs benchmark cases for estimate quality, anchor selection, confidence calibration, and non-estimable decisions | Contract and quality work matter more here than physical extraction. |

## What Already Matches The Goal

- The main run already writes inspectable artifacts under `output/<run_id>/`.
- The CCC path has the best artifact discipline and benchmark discipline.
- The external-source path has the best controlled tool contract.
- The enrichment output model already separates `field_manifest`,
  `gap_manifest`, `external_evidence`, `web_findings`,
  `freshness_results`, `assumptions`, and `non_estimable`.
- The writer context export deliberately filters run bookkeeping and keeps
  writer-visible enrichment fields.
- There is a central config module and shared run logger.
- Most functions have type hints and the main script benchmark entrypoints
  follow the repository's standalone script style.

## What Does Not Yet Match The Goal

- `backend/modules/web_researcher/` is overloaded. It is currently the home
  for at least four separate reusable capabilities.
- The source-stage boundaries are not visible enough in folder structure or in
  current documentation.
- Some contracts are typed Pydantic models, while others are loose dict
  payloads.
- `context_bundle` is the central cross-module contract, but it is not itself
  a typed, versioned model.
- Web search/process/extract artifacting is too thin for debugging or
  reproducible benchmarking.
- Assumptions are split between enrichment and API review with different
  shapes.
- External-source tool-call audit artifacts are not treated as first-class run
  artifacts in the normal pipeline.
- Web search/process/extract and assumptions do not yet have dedicated
  benchmark harnesses.

These are real findings. They should remain visible. The recommendation is
simply that they do not all point to the same next move.

## Recommended Near-Term Plan

The recommended next step is not broad modularization. It is documentation and
small hardening.

### Documentation work that would pay off now

1. Write one clear architecture section or document that names the four stages
   and the current code homes for each.
2. Document the shared runtime explicitly:
   `AppConfig`, `RunLogger`, `RunPaths`, `context_bundle`, prompts, retry, and
   writer context shaping.
3. Document guaranteed artifacts by stage and note which ones are optional or
   conditional.
4. Document the PDF-to-Markdown handoff assumption for CCC and 3rd party
   sources.
5. Document the assumptions split clearly, including the difference between
   automatic enrichment estimates and post-run user review.

### Small hardening work that is still worth doing

1. Add explicit result models for stage outputs where tuple or loose-dict
   contracts are doing too much work.
2. Register missing artifacts more consistently through the shared run logger.
3. Add missing benchmarks where operational quality gaps are largest, starting
   with web search and assumptions.
4. Clarify the writer-facing assumptions contract.

These changes improve clarity and maintainability without forcing the shared
runtime to fragment.

## Deferred Target Structure If Modularization Is Ever Required

If modularization later becomes a real requirement, the existing findings still
support a staged target like:

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

This is still a reasonable end state if the repository later needs:

- independent ownership of stages
- transportable capability packages
- stronger standalone benchmark harnesses
- clearer per-stage release and change boundaries

It is just not the best current priority.

## If A Refactor Is Forced Later, Suggested Order

If reuse pressure or team ownership eventually makes a structural split
necessary, the current findings still support this order:

1. Define explicit result models for each stage.
   - `CCCReadResult`
   - `ExternalSourceRunResult`
   - `WebSearchRunResult`
   - `AssumptionsRunResult`
2. Add module-level artifact writers and register them consistently through
   `RunLogger.record_artifact()`.
3. Move or wrap external-source code first, because it is the cleanest current
   transplant candidate.
4. Split websearch next, but only together with persisted search-plan and raw
   decision artifacts.
5. Unify assumptions into one writer-facing contract before or during any move.
6. Formalize `context_bundle` with a versioned schema or typed model.
7. Add missing benchmarks before treating the new structure as stable.

This should be read as a contingency plan, not as the current recommendation.

## Reuse In Another Project: Rewrite Cost And Alignment Surface

If the real goal is to reuse these capabilities in another repository as four
clean modules, this repository should still be treated as donor code and
behavioral reference, not as a package that must be fully modularized first.

The main reason is that the implementation is split between stage-local logic
and shared runtime contracts. Copying the visible module files is not enough.
Any target project also needs to align on:

- config loading
- model execution
- artifact writing
- path conventions
- shared models
- the shape of `context_bundle`
- logging and storage behavior

Approximate implementation surface measured from the current repository tree:

| Capability to reuse | Approximate line count in current repo | What those lines currently include | Rewrite cost in a new project |
| --- | ---: | --- | --- |
| CCC read/process/extract | ~4,100 | `markdown_researcher`, vector retrieval/indexing, benchmark runners, key tests | High. Strong behavior exists, but it is still tied to orchestrator assembly, current retrieval flow, and current document assumptions. |
| 3rd party external sources | ~3,100 | governed source registry, tool loop, resolver, prompts, benchmark runner, key tests | Medium. This is the cleanest transplant candidate, but it still depends on current enrichment config, gap contracts, agent runtime, and artifact conventions. |
| Web search/process/extract | ~3,200 | planner, worker, scraper and search clients, extraction, relevance, validation, freshness, key tests | Medium to high. Functional split is decent, but host-project alignment is heavy because search flow depends on current configs, models, and logging and artifact expectations. |
| Assumptions | ~2,200 | automatic estimator, API review flow, route layer, key tests | Medium to high. The estimator and review path are split across runtime and API code, so this is not a copy-paste module today. |
| Shared alignment surface | ~3,100 | orchestrator, enrichment pipeline shell, shared models, context merger, run logger, config, paths | Mandatory partial rewrite or adapter layer in every target project. This is the main hidden cost. |

Two important observations follow from these counts:

1. The four stage areas alone represent about 12,600 lines of implementation
   surface before counting data files, manifests, YAML source inventories,
   benchmark fixtures, and prompt and schema alignment work.
2. Reusing them as a coherent system also pulls in about 3,100 additional
   lines of shared runtime surface. That shared layer is where most hidden
   coupling currently lives.

This means another project should not assume a simple copy of four folders.
Even if the target project wants the same behavior, it still needs decisions
about:

- whether to preserve `context_bundle` or replace it with explicit typed result
  models between stages
- whether to preserve current run-artifact conventions or map them to the new
  project's logging and storage model
- whether to preserve OpenRouter agent wrappers, prompt loading, and retry
  conventions or swap them for the new project's execution layer
- whether CCC Markdown, governed source-library Markdown, and web or tier-1
  source definitions live in the same places and use the same manifests

### Practical cost framing

For a new project, the main cost is not typing the code again. The main cost is
alignment work.

- Low-cost path: reuse one capability at a time behind a new clean interface.
  The best first candidate is external sources.
- Medium-cost path: reuse two or three capabilities, but replace the shared
  runtime with the new project's own config, logging, and artifact layer.
- Highest-cost path: attempt a near-lift-and-shift of all four modules plus the
  shared orchestration assumptions. This carries the most donor-repo baggage
  and is the least attractive option.

### Recommended reuse strategy

If another repository wants these four capabilities, the better plan is:

1. Define clean target contracts in the new repository first.
2. Map this repository's behavior, prompts, artifacts, and tests onto those
   contracts.
3. Copy only cohesive internals that already match the new contract.
4. Rewrite the orchestration and shared runtime layer in the new repository
   instead of importing this repository's orchestration structure wholesale.

In practice, that means:

- `external_sources` is the best first transplant candidate
- `web_search` is the next best candidate, but it needs stronger target-side
  artifact and client abstractions
- `assumptions` should likely be reassembled in the new project from the
  estimator and review flows rather than copied as-is
- `ccc_reader` should be treated as a selective extraction effort, not a simple
  folder move, because the current behavior spans markdown loading, retrieval,
  and orchestrator-owned artifact assembly

## Bottom Line

The project is already close to the desired pipeline at the behavior level.
The most important gap is clarity, not absence of logic.

The current findings still show real weaknesses:

- `web_researcher` is overloaded
- web artifacting is thin
- assumptions are split
- some contracts are implicit
- some benchmarks are missing

But those findings do not require an immediate repo-wide structural split.

The better near-term recommendation is:

- keep the current shared runtime centralized
- document the stage boundaries and shared contracts clearly
- improve the weakest artifact and benchmark gaps
- defer full physical modularization unless reuse, ownership, or operational
  pressure makes it worth the added synchronization cost

That keeps the real findings in view while avoiding a refactor whose main
effect would be to spread logging, artifact, config, and context-ordering
responsibility across more places than the repository currently needs.

Separately, PDF-to-Markdown conversion should be treated as a different system
concern. It should become a standalone API, but that work is outside the scope
of this repository. This repo should consume the converted Markdown through a
clear handoff contract rather than owning the conversion implementation.
