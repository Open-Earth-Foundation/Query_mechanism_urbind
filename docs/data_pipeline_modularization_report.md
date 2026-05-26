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

### Findings and limitations

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
- Proposed solution: add a dedicated CCC discovery script that scans the
  top-level markdown set and writes a simple manifest or index artifact, so CCC
  source discovery is explicit rather than only inferred from filename stems.

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

This is the strongest contract design in the enrichment area.
`SourceRegistry.load()` reads `sources.yaml`, resolves every `source_id` to
exactly one Markdown filename stem, and exposes controlled metadata search plus
bounded regex tools.

In the target scope, this stage starts with a handoff from the other
repository's PDF-to-Markdown flow when 3rd party sources begin as PDFs or
documents. This repo should consume the converted source library as given, run
controlled source selection and search, extract facts, resolve them against CCC
evidence, and persist the audit trail.

### Findings and limitations

- The code lives under `backend/modules/web_researcher/`, even though it is
  not websearch.
- `run_external_source_enrichment()` returns a tuple. That tuple is clear in
  code, but a result model would be safer and easier for other tools to
  consume.
- The session tool audit exists on disk, but normal pipeline artifact
  registration is incomplete. The enrichment serializer registers final
  external evidence, resolution, and no-evidence JSON files, but the low-level
  tool audit file under `external_sources/external_evidence.json` is not
  clearly registered in `run.json` by the normal pipeline.
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

There are two related but separate assumptions workflows, and they correspond
to two different ways assumptions enter the system:

- automatic assumptions generated inside the enrichment pipeline
- manual assumptions entered by a user through the assumptions review UI and
  submitted through the assumptions API

Those are not just two implementation details. They are two distinct input
paths with different payload shapes and different persistence behavior.

1. Automatic enrichment estimator:
   - Runs inside `run_enrichment_pipeline()`.
   - Inputs `GapManifest`, `EnrichedField[]`, CCC context, national benchmark
     web findings, and comparative web findings.
   - Outputs `AssumptionRecord[]`, `NonEstimableRecord[]`, and
     `saturation_warning`.
   - Persists through `enrichment/assumptions.json`,
     `enrichment/non_estimable.json`, and `enrichment/enrichment_bundle.json`.

2. Post-run review and regenerate:
   - User input starts in the frontend assumptions review workspace, then flows
     through `/api/v1/runs/{run_id}/assumptions/discover` and
     `/api/v1/runs/{run_id}/assumptions/apply`.
   - Inputs a completed run, final document, context bundle, and user-edited
     assumptions.
   - Outputs revised content and optional persisted artifacts.
   - Persists only when `persist_artifacts=true`.

### Findings and limitations

- Assumptions do not have a dedicated module.
- There are two output shapes: `enrichment.assumptions[]` for automatic
  estimates and top-level `context_bundle["assumptions"]` for edited review
  assumptions. That makes writer and tool reuse harder.
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
| `context_bundle` | Runtime dict built by `RunLogger` and modules | CCC, external sources, websearch, assumptions, writer, API, benchmarks | Cross-stage payload with `markdown`, `enrichment`, final output path, queries, selected cities | This is the biggest shared contract in the system. The main concern is preserving one coherent contract as the code evolves, not adding more structural separation around it. |
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

## Deferred Target Structure If Modularization Is Ever Required

This is not the recommended path right now. The current stages still have too
much overlap in shared runtime, contracts, logging, artifact handling, and
writer-facing context shape for a clean split to pay off yet.

If modularization later becomes a real requirement despite that overlap, the
existing findings still support a staged target like:

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

The main reuse surfaces in the current repository are:

| Capability to reuse | What it currently includes | Rewrite cost in a new project |
| --- | --- | --- |
| CCC read/process/extract | `markdown_researcher`, vector retrieval and indexing, benchmark runners, key tests | High. Strong behavior exists, but it is still tied to orchestrator assembly, current retrieval flow, and current document assumptions. |
| 3rd party external sources | governed source registry, tool loop, resolver, prompts, benchmark runner, key tests | Medium. This is the cleanest transplant candidate, but it still depends on current enrichment config, gap contracts, agent runtime, and artifact conventions. |
| Web search/process/extract | planner, worker, scraper and search clients, extraction, relevance, validation, freshness, key tests | Medium to high. Functional split is decent, but host-project alignment is heavy because search flow depends on current configs, models, and logging and artifact expectations. |
| Assumptions | automatic estimator, API review flow, route layer, key tests | Medium to high. The estimator and review path are split across runtime and API code, so this is not a copy-paste module today. |
| Shared alignment surface | orchestrator, enrichment pipeline shell, shared models, context merger, run logger, config, paths | Mandatory partial rewrite or adapter layer in every target project. This is the main hidden cost. |

Two important observations follow:

1. The four stage areas are not self-contained in practice. They come with
   prompts, tests, fixtures, manifests, and contract assumptions.
2. Reusing them as a coherent system still pulls in a shared runtime layer.
   That shared layer is where most hidden coupling currently lives.

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
2. Add missing benchmarks where operational quality gaps are largest, starting
   with web search and assumptions.
3. Clarify the writer-facing assumptions contract.

These changes improve clarity and maintainability without forcing the shared
runtime to fragment.
