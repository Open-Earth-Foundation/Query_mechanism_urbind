# PR 51 Overlap With External Source Search Design

This note maps what [PR #51](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/pull/51)
is trying to implement and how it touches, overlaps with, or conflicts with the
external-source search design described in [plan.md](../plan.md),
[example.md](../example.md), and
[tool_implementation.md](../tool_implementation.md).

## Scope

This document compares PR `#51` against the current external-source design in:

- [plan.md](../plan.md)
- [example.md](../example.md)
- [tool_implementation.md](../tool_implementation.md)

It focuses only on the parts that affect:

- external source ingestion
- local Markdown availability
- metadata and source registries
- runtime search order before assumptions
- evidence and provenance handling
- possible duplication or architecture drift

## Short Answer

PR `#51` is not the same implementation as the planned external-source tool
layer, but it is implementing the lower-level ingestion and provenance substrate
that our design needs.

The biggest overlap is intent:

- curated external sources are ingested ahead of runtime
- local artifacts are used before assumptions
- stable source identifiers and richer provenance are introduced

The biggest conflicts are shape:

- PR `#51` uses a manifest-driven ingestion model, not a standalone
  `external_docs/sources.yaml` runtime registry
- PR `#51` writes city-plan Markdown into `documents/additional/...`, not a
  separate `external_docs/...` snapshot folder
- PR `#51` does not implement the agent-facing tool contract proposed in
  [tool_implementation.md](../tool_implementation.md)
- PR `#51` uses a different metadata vocabulary than the one currently proposed
  in [plan.md](../plan.md)

## What PR 51 Implements

Links in this section point to files on PR `#51`'s head branch
`isaac/feat/urbind-additional-docs`.

PR `#51` introduces a manifest-driven source layer centered on:

- [backend/data/sources_manifest.yaml](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/data/sources_manifest.yaml)
- [backend/modules/sources/manifest.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/modules/sources/manifest.py)
- [backend/modules/sources/runner.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/modules/sources/runner.py)
- [backend/scripts/sources_ingest.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/scripts/sources_ingest.py)
- [backend/scripts/sources_status.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/scripts/sources_status.py)

The main source kinds in the PR are:

- `markdown_corpus`
- `structured_lookup`
- `vector_collection`
- `web_allowlist`

For the additional-documentation repo specifically, PR `#51` does this:

1. Declares the upstream GitHub repo and pins it to a commit SHA.
2. Runs ingestion handlers against declared upstream paths.
3. Converts city-plan PDFs into local Markdown files.
4. Generates structured lookup outputs such as parquet files.
5. Generates a curated tier-1 web allowlist.
6. Makes those artifacts available to the existing enrichment pipeline.

The PDF-to-Markdown path is implemented in
[backend/modules/sources/handlers/pdf_to_markdown.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/modules/sources/handlers/pdf_to_markdown.py).
That handler writes files into:

```text
documents/additional/<city>_<slug>/<city>.md
```

PR `#51` then changes Markdown discovery to recurse through subdirectories in:

- [backend/utils/markdown_files.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/utils/markdown_files.py)
- [backend/modules/markdown_researcher/services.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/modules/markdown_researcher/services.py)

That is what makes the ingested local Markdown files visible at runtime.

## Whole-PR Double Check

After a full pass over the PR branch, the implementation breaks into five
separable buckets:

### 1. Source ingestion and state management

This is the core external-source substrate:

- upstream resolution and pinned GitHub cloning
- per-ingestion state files under `.state/sources/`
- manifest-driven handlers for Markdown, vector, structured lookup, and
  allowlist outputs

This bucket overlaps directly with our external-source design.

### 2. Enrichment pipeline changes that consume those sources

This is the second main overlap area:

- split gap flow flag
- Phase 1 local fanout
- structured lookup grounding
- benchmark retrieval
- tier-1-first search before open Serper
- assumptions short-circuit when deterministic local data is available

This bucket also overlaps directly with our design because it changes the
runtime order before assumptions and enriches provenance.

### 3. Writer, UI, config, and test support for that new pipeline

These files are not a separate source system, but they are part of the same
feature set:

- `llm_config.yaml` adds `use_split_gap_flow`, `tier1_first_search`, and
  `tier1_confidence_threshold`
- writer prompts are updated for source tier, scope safety, financing, and
  derived metrics
- frontend progress UI adds local lookup and benchmark rows
- tests cover manifest loading, tier-1 search, structured lookup grounding,
  and provenance flow

These are adjacent to our design rather than direct conflicts, but they are
important because they show how PR `#51` expects the new source layer to behave
end-to-end.

### 4. Scope / provenance / derived-metric integrity rules

This is part of the same PR, but it is only partially related to our current
external-source search proposal.

It includes:

- `source_id` and `source_tier` propagation
- `bundled_only` field status
- scope-safe aggregation
- financing blocks
- pre-computed derived metrics

The provenance parts overlap with our design, but PR `#51` is also using a
different saved/extracted record structure from the one currently proposed in
our design.

Our current design assumes a search-tool evidence chain like:

- `SearchHit`
- `EvidenceCandidate`
- structured extracted claim
- `NoEvidenceRecord`

That chain is quote-first and snippet-first. It is designed around:

- exact `matched_text`
- exact `quote`
- line references
- explicit evidence-basket persistence before claim extraction
- explicit `mark_no_evidence_found`

PR `#51` instead uses a different runtime chain:

- `StructuredLookupResult`
- `BenchmarkExcerptRecord`
- `WebFinding`
- `EnrichedField`
- `AssumptionRecord`
- `DerivedMetric`

That chain is enrichment-first. It is designed around:

- `(city, field)` resolution
- source provenance and source tier
- field status transitions such as `bundled_only` or `still_missing`
- writer-ready enriched fields and derived metrics

So this is more than a downstream presentation difference. It is a structural
difference in what gets extracted, what gets persisted, and what becomes the
main runtime record.

The practical implication is:

- if we keep our current design unchanged, we will build a quote/evidence-basket
  model beside PR `#51`'s `WebFinding -> EnrichedField` model
- if we build on PR `#51`, we need to decide whether quote-backed
  `SearchHit/EvidenceCandidate` records become a new upstream layer that feeds
  into `EnrichedField`, or whether `EnrichedField` itself becomes the canonical
  saved record and we extend it with quote/line-backed evidence

## How PR 51 Uses External Sources At Runtime

PR `#51` does not expose a new tool loop to the LLM. Instead, it wires local
sources into the existing enrichment flow.

It also does not use agentic search in the way proposed in
[tool_implementation.md](../tool_implementation.md). Our design assumes an
iterative LLM search loop that can narrow candidates, run search, inspect hits,
expand promising snippets, save evidence, and search again. PR `#51` uses a
much more basic retrieval flow:

- deterministic structured lookups
- benchmark similarity retrieval
- curated tier-1 `site:` search
- open web search fallback

That means the two designs align on "use governed sources before assumptions,"
but not on the actual search mechanics.

The main runtime pieces are:

- [backend/modules/web_researcher/module.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/modules/web_researcher/module.py)
- [backend/modules/web_researcher/phase1_fanout.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/modules/web_researcher/phase1_fanout.py)
- [backend/modules/web_researcher/data_lookups/__init__.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/modules/web_researcher/data_lookups/__init__.py)
- [backend/modules/web_researcher/search_worker.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/modules/web_researcher/search_worker.py)
- [backend/modules/web_researcher/tier1_web.py](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/modules/web_researcher/tier1_web.py)

The effective order in PR `#51` is:

1. gap decomposition
2. Phase 1 local fanout
3. structured lookups
4. benchmark retrieval
5. later web-search worker passes, including tier-1-first web probing
6. assumptions estimator only after those earlier sources still leave gaps

So PR `#51` is aligned with the direction that governed local sources should be
used before assumptions. It is only partially aligned with the narrower idea of
"run external Markdown search as a dedicated stage before open web search,"
because PR `#51` splits that work across multiple existing subsystems.

It is also important that PR `#51` is not only "search over added Markdown."
The full implementation combines four different governed-source paths:

- ingested city-plan Markdown
- structured lookups
- benchmark vector retrieval
- curated tier-1 web domains

That is broader than the current design in [tool_implementation.md](../tool_implementation.md),
which is centered on a dedicated search-tool loop over tagged local Markdown.

## Direct Overlaps With Our Current Design

### 1. Curated external sources are pre-ingested, not user-uploaded

This is aligned with the current design direction in [plan.md](../plan.md) and
[tool_implementation.md](../tool_implementation.md).

Our current design assumes converted local artifacts and a runtime metadata
registry. PR `#51` also assumes converted local artifacts and no user upload
step.

### 2. External/local evidence runs before assumptions

This overlaps directly with the current flow described in:

- [plan.md](../plan.md)
- [example.md](../example.md)

PR `#51` already follows the same high-level rule:

- resolve with governed local sources first where possible
- estimate only afterward

### 2b. Alignment on actually getting the gaps

This is one of the strongest alignment points between PR `#51` and the current
design, even though the mechanics differ.

Our current design assumes this broad sequence:

1. CCC retrieval/extraction runs first.
2. unresolved fields are identified.
3. external governed sources are searched for those unresolved fields.
4. only fields still unresolved after that proceed to assumptions.

That is captured most clearly in:

- [plan.md](../plan.md)
- [example.md](../example.md)

PR `#51` implements the same intent in a more explicit split flow:

1. Phase 0 decomposes the question into granular fields.
2. Phase 1 runs local source fanout before per-city gap detection.
3. Phase 2 detects which `(city, field)` cells are still gaps.
4. later retrieval/search passes try to close those gaps.
5. only remaining gaps go to the assumptions estimator.

So the alignment is real:

- both approaches make gap detection an explicit control point before
  assumptions
- both approaches try to close only unresolved fields rather than searching
  everything blindly
- both approaches treat governed external/local sources as part of gap closure,
  not as a post-hoc optional supplement

The main difference is where the gap is formalized:

- our current design describes unresolved fields more from the outside-in:
  CCC first, then external search for still-missing fields
- PR `#51` formalizes the gap process inside `web_researcher` with explicit
  field decomposition and per-city gap manifests

This is a design difference, not a contradiction. In practice, PR `#51`'s gap
manifest could become the driver for the external search-tool layer instead of
our design inventing a second unresolved-field tracker.

### 3. Benchmark-first thinking

The benchmark concept is not identical, but there is still overlap.

Our design requires a benchmark-first implementation discipline in
[plan.md](../plan.md). PR `#51` already introduces benchmark-oriented local
retrieval and benchmark artifacts, which could reduce duplicated benchmark
infrastructure if aligned carefully.

### 4. Gap status vocabulary still differs

Even though the gap-closing intent is aligned, the status vocabulary is not yet
the same.

PR `#51` introduces a richer internal gap/field-status model with concepts
such as:

- `blank_fields`
- `stale_flags`
- `bundled_fields`
- `bundled_only`
- `still_missing`
- `partially_resolved`

Our current design is simpler and more search-tool oriented:

- unresolved after CCC
- resolved by external evidence
- conflict that must be surfaced
- no evidence found
- pass to assumptions

This is not a hard conflict, but it is an integration point. If PR `#51` is
merged first, the external-source search-tool layer should probably consume the
existing gap manifest and field statuses rather than introducing a parallel gap
state model with overlapping meanings.

## Conflicts And Architecture Drift

### 1. Source registry shape is different

Our current design assumes a search-oriented runtime registry centered on
`external_docs/sources.yaml`, as described in:

- [plan.md](../plan.md)
- [tool_implementation.md](../tool_implementation.md)

PR `#51` instead uses a broader ingestion manifest:

- [backend/data/sources_manifest.yaml](https://github.com/Open-Earth-Foundation/Query_mechanism_urbind/blob/isaac/feat/urbind-additional-docs/backend/data/sources_manifest.yaml)

The two documents do not serve the same purpose:

- `sources_manifest.yaml` describes upstream sources, ingestion kinds, handlers,
  outputs, and coverage
- our planned `sources.yaml` describes search-time metadata for already-converted
  documents

This is the largest design conflict. If both are kept independently, the repo
will end up with two source-of-truth layers for the same documents.

### 2. File layout is different

Our current design assumes a dedicated external-doc search area, for example:

```text
external_docs/
  sources.yaml
  krakow/
    krakow_electromobility_strategy_2030.md
```

PR `#51` writes to:

```text
documents/additional/<city>_<slug>/<city>.md
```

That layout is optimized around the existing markdown researcher rule
`Path.stem == city`, not around a separate external-source registry and search
snapshot model.

This is not a hard blocker, but it means our current `external_docs/current`
snapshot design in [tool_implementation.md](../tool_implementation.md) would
need to be adapted if PR `#51` becomes the canonical ingestion path.

### 3. Metadata vocabulary is different

Our current design uses search-oriented metadata such as:

- `city`
- `country`
- `verticals`
- `source_type`
- `publication_year`
- `tef_sectors`

PR `#51` manifest coverage is narrower and runtime-oriented:

- `cities`
- `country` or `countries`
- `fields`
- `scope`

That means PR `#51` does not yet expose enough metadata to satisfy the public
filtering contract currently proposed in [plan.md](../plan.md) and
[tool_implementation.md](../tool_implementation.md), especially for:

- `verticals`
- `source_type`
- `publication_year`
- `tef_sectors`

It is also important that PR `#51` does not use TEF in the way our current
external-source design does. Our design expects TEF sector tags to help narrow
the local external-document search scope. PR `#51` does not use TEF sectors as
part of its runtime retrieval contract here; its manifest coverage is based on
city/country/field/scope instead.

### 4. The LLM-facing search-tool contract is still missing

Our current design explicitly proposes:

- `get_tag_options`
- `list_candidate_sources`
- `regex_search`
- `expand_hits`
- `add_evidence_candidates`
- `list_evidence_candidates`
- `mark_no_evidence_found`

See:

- [plan.md](../plan.md)
- [tool_implementation.md](../tool_implementation.md)
- [example.md](../example.md)

PR `#51` does not implement this tool surface. Instead, it enhances the existing
pipeline internals.

This is an overlap in goal, but not in implementation. If both paths move
forward independently, the repo could end up with:

- an internal manifest-driven retrieval path
- a second external-document tool subsystem layered beside it

That would be unnecessary duplication unless one is clearly treated as the
backend substrate for the other.

### 5. Provenance and evidence model are not the same

This should not be treated as an alignment point.

At the intention level, both designs want stable source-backed outputs. At the
implementation level, the structures are different enough that they are not
aligned.

PR `#51` uses provenance through field-oriented runtime records such as:

- `StructuredLookupResult`
- `BenchmarkExcerptRecord`
- `WebFinding`
- `EnrichedField`
- `AssumptionRecord`
- `DerivedMetric`

Our current design uses provenance through a snippet/evidence chain such as:

- `SearchHit`
- `EvidenceCandidate`
- structured extracted claim
- `NoEvidenceRecord`

So even where both designs use `source_id`, they do not use the same provenance
structure, they do not persist the same kinds of records, and they do not move
through the same extraction steps.

Our current design requires:

- exact `matched_text`
- exact `quote`
- line-backed snippets
- explicit evidence-basket persistence
- explicit `mark_no_evidence_found` records

See:

- [example.md](../example.md)
- [plan.md](../plan.md)
- [tool_implementation.md](../tool_implementation.md)

PR `#51` improves provenance, but it does not yet implement the same explicit
quote-first evidence-selection model. That means the design intent is aligned,
but the operational evidence interface is still different.

More concretely, the mismatch is not only about missing fields. It is about the
shape of the saved and extracted records:

- our design saves snippets first, then extracts claims from those saved
  snippets
- PR `#51` stores retrieved/extracted values into field-oriented runtime
  records such as `WebFinding` and `EnrichedField`

So a merge between the two approaches will need a deliberate contract choice,
not just extra provenance keys.

### 6. Benchmark means different things

In our current design, the benchmark is primarily an implementation acceptance
set for search and extraction behavior.

In PR `#51`, benchmark retrieval refers to a vector-backed local benchmark
collection used as part of runtime enrichment.

These are related, but not the same. They should not be merged conceptually
without a clear distinction between:

- implementation benchmark fixtures
- runtime benchmark retrieval corpus
