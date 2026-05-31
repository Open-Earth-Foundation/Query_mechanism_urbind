# Agentic Research Writer Plan

## Purpose

Readjust the writer so it can actively search, filter, and save relevant context before drafting the final answer.

The goal is not to replace the existing section writer. The current section-first writer flow is already the right writing shape for aggregate answers:

```text
writer section planner
-> section writer agents
-> section composer
-> final markdown answer
```

The missing capability is an agentic research step immediately before section planning. That step should search the available context and third-party source evidence, save the best excerpts into a writer evidence basket, and then let the existing section writer agents write from those saved, citeable excerpts.

## Current App Flow

The current backend flow is:

```text
frontend run request
-> run executor
-> orchestrator
-> markdown retrieval or vector retrieval
-> markdown researcher extracts accepted CCC excerpts
-> context_bundle.json is assembled
-> optional enrichment runs gap analysis, external source search, web search, and assumptions
-> writer builds a writer-safe context bundle
-> aggregate mode uses section-first writing when enabled
-> final.md is written
```

Important current implementation points:

- `backend/modules/orchestrator/module.py` owns the run sequence and calls enrichment before writer.
- `backend/modules/web_researcher/external_sources.py` already has governed third-party Markdown search tools:
  - `list_candidate_sources`
  - `regex_search`
  - `expand_hits`
  - `add_evidence_candidates`
  - `list_evidence_candidates`
  - `mark_no_evidence_found`
- `backend/modules/writer/agent.py` already has section-first aggregate writing:
  - section planner
  - section writer agent
  - section composer
- `backend/modules/writer/utils/section_first.py` builds the planner catalog and narrows context per section.
- `backend/modules/writer/utils/multi_pass.py` builds the writer-safe context bundle and filters enrichment by city.

## Design Decision

Add a new writer research and evidence-curation stage before the existing section planner.

The new stage should:

1. Search through writer-visible context using regex and metadata filters.
2. Filter by city, source kind, field, and section hint.
3. Save useful snippets, facts, or excerpts into a run-local evidence basket.
4. Convert saved snippets into citation-compatible writer evidence.
5. Pass the saved evidence catalog to the existing section planner.
6. Let the existing section writer agents write one section each from assigned saved evidence.

Recommended target flow:

```text
CCC markdown research
-> enrichment and third-party source evidence
-> build writer-safe context
-> writer research curator searches context and saves evidence
-> section planner divides saved evidence into sections
-> existing section writer agents write sections
-> existing composer assembles final answer
-> citation validation and final.md
```

This keeps research and writing separate:

- The research curator decides what evidence is worth saving.
- The section planner decides how to divide the answer.
- The section writers write prose only from assigned evidence.
- The composer removes duplication and produces the final answer.

## How To Use Both Writer Layers

### 1. Agentic Research Writer

This is a new pre-writing agent. It does not draft final prose.

Responsibilities:

- inspect the user question and selected cities;
- search CCC excerpts, external evidence, web findings, assumptions, and optionally source chunks;
- use regex for numbers, years, units, target terms, policy terms, and source-language wording;
- filter hits by city;
- save relevant excerpts or facts into a persistent writer evidence basket;
- mark unresolved areas when searches produce no useful evidence;
- return a structured evidence-selection summary.

This agent answers: "What should the writer use?"

### 2. Existing Section Planner

The section planner should stay responsible for answer structure.

Adjustment:

- It should receive a compact `saved_evidence_catalog` instead of, or before, the broader raw evidence catalog.
- It should assign saved evidence IDs to each planned section.
- It should still backfill important unassigned evidence when necessary.

This agent answers: "How should the saved evidence be divided into sections?"

### 3. Existing Section Writer Agents

The section writer agents should remain pure writing agents.

Adjustment:

- Each section writer receives only the saved evidence assigned to that section.
- It should not search, expand, or save evidence.
- It should cite every factual claim using assigned evidence IDs.

This agent answers: "How do I write this one section from assigned evidence?"

### 4. Existing Section Composer

The composer should remain the final assembly step.

Adjustment:

- It composes only from section drafts.
- It does not introduce new facts.
- It preserves citations and removes duplication.

This agent answers: "How do I assemble the cited section drafts into one final answer?"

## Search Scope

The agentic research writer should search over a normalized writer context index. The index should include:

1. CCC accepted excerpts from `context_bundle.markdown.excerpts`.
2. Source chunks behind CCC `source_chunk_ids`, where available.
3. External Markdown evidence from `context_bundle.enrichment.external_evidence`.
4. External resolver decisions from `context_bundle.enrichment.external_resolutions`.
5. Web findings from `context_bundle.enrichment.web_findings`.
6. Assumptions and non-estimable records.
7. Enriched field status records.

Each searchable item should have a common shape:

```json
{
  "context_item_id": "ctx_1",
  "source_kind": "ccc_excerpt",
  "city": "Krakow",
  "field": "public_ev_chargers_2030_target",
  "source_ref_id": "ref_12",
  "source_id": "krakow-secap-2025",
  "title": "Krakow SECAP 2025",
  "text": "Searchable snippet or excerpt text.",
  "quote": "Exact quote available for final citation.",
  "line_start": 10,
  "line_end": 18,
  "source_url": "https://example.org/source.pdf"
}
```

`source_kind` should be one of:

- `ccc_excerpt`
- `ccc_source_chunk`
- `external_markdown`
- `external_resolution`
- `web_finding`
- `assumption`
- `non_estimable`
- `enriched_field`

## Writer Research Tools

The writer research curator can reuse the external-source tool pattern, but the tools should operate on the assembled writer context.

Recommended MVP tools:

```python
list_context_sources(
    cities: list[str] | None = None,
    source_kinds: list[str] | None = None,
    fields: list[str] | None = None,
    max_items: int = 100,
) -> list[ContextSourceSummary]
```

```python
regex_search_context(
    pattern: str,
    cities: list[str] | None = None,
    source_kinds: list[str] | None = None,
    fields: list[str] | None = None,
    case_sensitive: bool = False,
    context_words: int | None = None,
    context_lines: int | None = None,
    max_matches: int | None = None,
) -> list[ContextSearchHit]
```

```python
expand_context_hits(
    hit_ids: list[str],
    context_words: int | None = None,
    context_lines: int | None = None,
) -> list[ContextSearchHit]
```

```python
save_context_evidence(
    candidates: list[SavedEvidenceInput],
) -> list[SavedWriterEvidence]
```

```python
list_saved_context_evidence() -> list[SavedWriterEvidence]
```

```python
mark_context_evidence_missing(
    city: str,
    topic: str,
    searched_context_item_ids: list[str],
    search_summary: str,
) -> WriterMissingEvidenceRecord
```

The tool behavior should follow the current external-source guardrails:

- require at least one meaningful filter or prior candidate set before broad regex search;
- cap pattern length;
- reject unsafe regex patterns;
- cap matches and snippet size;
- keep hit IDs run-local;
- allow evidence saving only from current-run hits;
- persist tool calls and saved evidence as artifacts.

## Saved Evidence Store

Saved evidence should be run-local and writer-owned.

Recommended artifacts:

```text
output/<run_id>/writer/evidence_workspace.json
output/<run_id>/writer/saved_evidence.json
```

`evidence_workspace.json` should contain:

- tool calls;
- regex patterns;
- filters;
- hit counts;
- selected hit IDs;
- unresolved/no-evidence records.

`saved_evidence.json` should contain:

```json
{
  "run_id": "20260530_1200",
  "schema_version": 1,
  "saved_evidence": [
    {
      "saved_id": "wref_1",
      "citation_ref_id": "ref_103",
      "source_context_item_id": "ctx_17",
      "source_kind": "external_markdown",
      "city": "Krakow",
      "topic": "public EV chargers",
      "section_hint": "Charging Infrastructure Targets",
      "quote": "The city plans at least 150 new public charging stations...",
      "summary": "Krakow has a concrete public charging-station target.",
      "line_start": 44,
      "line_end": 48,
      "source_id": "krakow-secap-2025",
      "source_url": "https://example.org/krakow-secap.pdf",
      "confidence": 0.86,
      "reason": "Contains a concrete city-level infrastructure target."
    }
  ],
  "no_evidence": []
}
```

`citation_ref_id` is important. The final writer already validates `[ref_n]` citations, so saved writer evidence should become citation-compatible rather than using a separate citation style.

## Citation Strategy

The simplest compatible strategy is:

1. Preserve existing `ref_n` IDs when saved evidence points to an existing CCC excerpt.
2. Assign new `ref_n` IDs to saved third-party, web, and assumption records after the existing markdown references.
3. Extend `markdown/references.json` or add a writer reference artifact that the frontend can resolve.
4. Make the writer-safe context expose saved evidence as regular cited evidence.

This avoids forcing the section writer and frontend to understand two unrelated citation systems.

## Section Planning From Saved Evidence

The section planner payload should be expanded from:

```json
{
  "evidence_catalog": [...]
}
```

to:

```json
{
  "saved_evidence_catalog": [...],
  "fallback_evidence_catalog": [...],
  "enrichment_summary": {...}
}
```

Planner rules:

- Prefer `saved_evidence_catalog`.
- Use `fallback_evidence_catalog` only when saved evidence is empty or misses selected cities.
- Assign every saved citation ref to at least one section.
- Group sections by the actual analytical task: metric, sector, city comparison, source type, time horizon, or data gap.
- Do not create generic sections like "Analysis" unless the question itself is generic.

The existing `WriterSectionSpec.required_ref_ids` can remain the main assignment field. It should point to saved citation ref IDs.

## Section Writer Context

`build_section_context_bundle()` should be adjusted so a section receives:

- only the saved evidence assigned to `section.required_ref_ids`;
- city-filtered enrichment records relevant to the assigned cities;
- no broad raw context unless configured as fallback.

The section writer prompt should stay mostly unchanged, but its input contract should mention saved writer evidence as a first-class source.

Required behavior:

- cite every claim with assigned `ref_n` IDs;
- do not search;
- do not use unassigned refs;
- separate observed CCC evidence, third-party evidence, web evidence, and assumptions where relevant.

## Configuration

Recommended config additions:

```yaml
writer:
  evidence_curator_enabled: true
  evidence_curator_max_turns: 6
  evidence_curator_max_saved_items: 80
  evidence_curator_max_regex_searches: 12
  evidence_curator_default_context_words: 80
  evidence_curator_max_context_words: 250
  evidence_curator_max_matches_per_search: 100
  evidence_curator_use_source_chunks: true
```

Keep existing section-first settings:

```yaml
writer:
  section_first_aggregate_enabled: true
  section_planner_max_input_tokens: 80000
  section_max_workers: 3
```

The default should be:

- curator enabled before both aggregate and city-by-city writer paths;
- section-first writer still enabled and canonical for aggregate mode;
- excerpt-only fallback available through config and per-run opt-out.

## Implementation Phases

### Phase 1: Writer Context Index

Add a module such as:

```text
backend/modules/writer/utils/research_context.py
```

Responsibilities:

- flatten writer-visible context into `ContextItem` records;
- normalize city keys;
- preserve original `ref_id` when present;
- include enrichment source metadata and provenance;
- optionally resolve source chunks behind `source_chunk_ids`.

Tests:

- CCC excerpts become searchable context items.
- External evidence becomes searchable context items.
- City filtering normalizes aliases.
- Empty context produces an empty index without failing.

### Phase 2: Writer Evidence Session

Add a module such as:

```text
backend/modules/writer/utils/research_session.py
```

Responsibilities:

- implement `list_context_sources`;
- implement bounded `regex_search_context`;
- implement `expand_context_hits`;
- implement `save_context_evidence`;
- persist workspace artifacts.

Reuse the safe regex validation pattern from `ExternalSearchSession` where practical.

Tests:

- regex search returns bounded hits.
- unsafe regex is rejected.
- save only accepts current-run hit IDs.
- saved evidence writes atomically.
- city and source-kind filters work.

### Phase 3: Writer Research Curator Agent

Add a prompt:

```text
backend/prompts/writer_research_curator_system.md
```

Add models:

```text
WriterEvidenceSelection
SavedWriterEvidence
WriterMissingEvidenceRecord
```

The curator output should summarize:

- saved evidence count;
- covered cities;
- uncovered cities or topics;
- high-risk conflicts;
- suggested section hints.

Tests:

- agent tools are built with gated visibility.
- fake agent can save evidence and return structured output.
- no saved evidence falls back to current writer behavior.

### Phase 4: Section-First Integration

Update writer flow:

```text
write_markdown()
-> build_writer_context_bundle()
-> maybe run writer evidence curator
-> build section plan from saved evidence
-> write section drafts
-> compose final output
```

Changes:

- `write_markdown()` invokes curator before `_write_markdown_section_first()`.
- `_build_writer_section_plan()` includes saved evidence catalog.
- `sanitize_writer_section_plan()` validates saved refs.
- `build_section_context_bundle()` narrows by saved refs.
- saved-evidence diagnostics are persisted alongside section-first diagnostics, not as a replacement for the chapter writer.

Tests:

- planner receives saved evidence catalog.
- section writer receives only assigned saved refs.
- composer still preserves citations.
- existing section-first tests continue to pass.

### Phase 5: Artifacts, API, and Frontend

Backend:

- record `writer_saved_evidence` artifact in `run.json`;
- include saved evidence in writer context export;
- expose saved evidence through diagnostics or a dedicated run artifact endpoint.

Frontend:

- initially show saved evidence only in dev diagnostics or writer-context export;
- later add a compact "Writer Evidence" inspection panel next to the generated document.

Avoid adding broad UI work in the MVP unless needed for debugging.

### Phase 6: Benchmarks and Acceptance Tests

Add benchmark cases where the correct final answer requires third-party evidence that is present in enrichment but easy for the writer to miss without pre-selection.

Minimum cases:

1. City-specific target value in external Markdown.
2. CCC excerpt plus external confirmation.
3. External conflict that should be surfaced separately.
4. Multi-city aggregate answer where each section needs different evidence.

Acceptance criteria:

- writer saved evidence is persisted;
- final answer cites saved third-party evidence;
- section plan assigns saved evidence to sections;
- section writers do not receive unrelated raw evidence;
- no regression in existing writer citation coverage;
- no regression in existing external-source search tests.

## Guardrails

- Do not let section writer agents search.
- Do not let the curator write final prose.
- Do not pass the full raw context to every section when saved evidence exists.
- Do not invent citation IDs outside the writer reference registry.
- Do not silently resolve CCC versus third-party conflicts.
- Keep source documents read-only.
- Keep all saved evidence traceable to a search hit or existing context item.

## Open Questions

- Should the curator run for city-by-city mode in the first implementation, or only aggregate mode?
- Should saved web findings receive the same `ref_n` citation treatment as saved Markdown evidence?
- How much saved-evidence detail should be visible in the frontend MVP?
- Should source chunks be loaded eagerly for all candidate refs, or only after regex hits need expansion?
- Should the writer curator search all enrichment records by default, or only the records matching selected cities?

## Recommendation

Implement this as a writer evidence-curation layer that prepares a saved, citeable evidence basket. Then reuse the existing section-first writer exactly where it is strongest: dividing the answer into sections, writing each section with a dedicated agent, and composing the final cited report.

This gives us the requested agentic search and save behavior without creating a second competing writer pipeline.
