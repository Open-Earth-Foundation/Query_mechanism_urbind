# External Tagged Markdown Search Plan

## Context

We want to improve coverage for Net Zero Cities (NZC) questions by using external data sources that complement the existing CCC-derived Markdown corpus.

The current CCC search and extraction path is already working and should remain the primary evidence path. The new work should not replace, weaken, or destabilize that path. Instead, it should add a governed enrichment layer that can consult approved external Markdown documents when CCC evidence is missing, stale, or needs confirmation.

External documents may include city Climate Action Plans (CAPs), updated CCC/NZC documents, mobility plans, energy plans, built environment plans, national datasets rendered as Markdown, EU programme docs, and other curated sources.

## Decision

Keep the existing CCC search unchanged and add a separate agentic search layer over tagged external Markdown files.

The agentic layer should behave like a shell-style research harness, but it should not run arbitrary shell commands in production. Instead, the model should use controlled search tools that support literal search, proximity search, and validated regex search over a scoped set of tagged Markdown files.

The new stage should produce structured evidence records, not directly rewrite final answers or mutate the primary context. A resolver should decide whether external evidence fills a CCC gap, confirms CCC evidence, supersedes stale CCC evidence, creates a conflict requiring review, or still leaves the field unresolved.

Recommended pipeline position:

```text
question refinement
-> existing CCC markdown retrieval and extraction
-> gap/staleness detection
-> agentic search over tagged external Markdown
-> evidence resolver
-> assumptions estimator for unresolved gaps only
-> writer
```

## Why This Direction

We discussed several options:

- Expanding the existing CCC search directly.
- Adding web search against the open internet.
- Using RAG over additional documents.
- Letting an agent search local Markdown with shell-like commands.
- Letting the model use regex-like searches.

The chosen approach is a hybrid of RAG-style governance and agentic search flexibility:

- CCC remains the primary source of truth.
- Tagged external Markdown gives us scalable source onboarding by adding files plus metadata.
- Agentic search gives the model the ability to iterate like a human researcher: search synonyms, inspect snippets, search for numbers near units, then extract claims.
- Controlled tools give us production safety, reproducibility, and auditability.
- Regex support gives the model useful expressiveness without giving it raw shell access.

This keeps the system extensible while avoiding unbounded web noise and avoiding production dependence on arbitrary model-selected shell commands.

## Source Tagging Model

Every external Markdown source should have metadata. For MVP, source metadata
should live in a folder-level YAML file at the top of the external Markdown
collection, not in per-file frontmatter and not in scattered sidecar files.

Recommended convention:

```text
external_docs/
  sources.yaml
  krakow/
    krakow_electromobility_strategy_2030.md
  munich/
    munich_mobility_strategy_2035.md
```

The metadata entry's `source_id` should match the Markdown filename stem. This
lets the loader resolve files without a manual `path` tag.

Example:

```yaml
sources:
  - source_id: vienna_cap_2024
    title: Vienna Climate Action Plan 2024
    city: Vienna
    country: Austria
    publication_year: 2024
    description: City climate action plan covering mobility, energy, and buildings targets.
    source_type: city_cap
    publisher: City of Vienna
    verticals: [mobility, energy, built_environment]
    tef_sectors: [transport, energy, buildings]
    tef_transitions: [charging_infrastructure, public_transport, building_renovation]
    source_url: https://...
```

MVP required tags:

- `source_id`: stable source identifier.
- `title`: human-readable source title.
- `upstream_group`: source group from the copied additional-docs catalogue.
- `city`: city the document is about.
- `country`: source jurisdiction.
- `publication_year`: publication year, not necessarily the data year or target year.
- `description`: short summary of what the document covers.
- `source_type`: city_cap, mobility_plan, energy_plan, national_dataset, eu_dataset, operator_report, think_tank_report, news, etc.
- `verticals`: mobility, energy, built_environment, waste, adaptation, finance, governance, etc.
- `tef_sectors`: broad TEF sector-level hints for filtering.
- `tef_transitions`: broad document-level transition hints for filtering.

Optional but useful tags:

- `publisher`: useful for citations and credibility review.
- `source_url`: original URL when available.
- `data_years`: years for observed data contained in the document.
- `target_years`: future years referenced by targets or plans.

No per-initiative tags are required for MVP. They are too narrow and hard to
maintain when a document contains many initiatives. TEF tags should be broad,
document-level hints only; initiative-level TEF classification can happen later
through the existing TEF mapping flow.

No `path` field is needed in MVP because the loader can map `source_id` to the
Markdown filename stem under the external docs folder. If we later need multiple
files per source or non-standard file names, a path field can be added as a
future expansion.

Possible future expansion:

- `source_tier`: official_city, official_national, official_eu, operator, academic, think_tank, media.
- `last_checked_at`: date when the source URL or file freshness was last checked.

## Data Catalogue Copy

We should copy a lightweight catalogue snapshot from the additional-docs source
repo into this repository so the search layer has a local source inventory to
validate against. The upstream repo currently groups sources into city plans,
national datasets, European databases, think tanks, and EU programmes. We should
preserve that grouping as `upstream_group` only; it is not the same thing as a
future source credibility tier.

Recommended local planning file:

```text
assets/external_sources/catalogue.yaml
```

The catalogue should track the source inventory that is in scope for Markdown
conversion and scanning. If a source is in this catalogue, it is expected to be
converted to Markdown. Sources that cannot or should not be converted should
stay outside this MVP scan catalogue.

At runtime, the same metadata shape can live as `external_docs/sources.yaml` at
the top of the converted Markdown folder. The planning catalogue and runtime
folder-level metadata should use the same fields.

Catalogue responsibilities:

- list every upstream source we plan to convert or reference;
- record which target cities, countries, and verticals each source can apply to;
- store enough metadata for `get_tag_options()` and `list_candidate_sources()`;
- avoid using source ranking or authority logic in the MVP.

For broad sources, `city` may be empty or omitted. Instead, the catalogue should
use coverage fields such as `countries`, `target_cities`, and `geographic_scope`
so national and European sources can still be considered for selected cities.

Recommended catalogue fields:

```yaml
source_id: bundesnetzagentur_ladesaeulenregister
title: Bundesnetzagentur Ladesaeulenregister
upstream_group: tier_2_national_datasets
geographic_scope: national
target_cities: [Aachen, Dresden, Heidelberg, Leipzig, Mannheim, Munich, Munster]
countries: [Germany]
publication_year: null
description: Public charging point registry for Germany.
source_type: national_dataset
publisher: Bundesnetzagentur
verticals: [mobility]
tef_sectors: [transport]
tef_transitions: [charging_infrastructure]
```

## Controlled Search Tools

The model should not receive raw shell access in production. It should receive a bounded search interface.

Agreed MVP tools:

1. `get_tag_options`

   ```python
   get_tag_options() -> TagOptions
   ```

   Returns the available metadata values the agent can choose from, such as cities,
   countries, publication years, source types, verticals, TEF sectors, and TEF
   transitions. This keeps the agent from inventing filters.

2. `list_candidate_sources`

   ```python
   list_candidate_sources(
       cities: list[str] | None = None,
       countries: list[str] | None = None,
       verticals: list[str] | None = None,
       tef_sectors: list[str] | None = None,
       tef_transitions: list[str] | None = None,
       source_types: list[str] | None = None,
       publication_year_min: int | None = None,
       publication_year_max: int | None = None,
       max_files: int = 50,
   ) -> list[SourceSummary]
   ```

   Scans source metadata and returns candidate files before text search starts. The
   agent should use this to narrow the search scope by city, country, vertical,
   source type, year, and optional TEF tags.

3. `regex_search`

   ```python
   regex_search(
       pattern: str,
       source_ids: list[str] | None = None,
       cities: list[str] | None = None,
       countries: list[str] | None = None,
       verticals: list[str] | None = None,
       tef_sectors: list[str] | None = None,
       tef_transitions: list[str] | None = None,
       source_types: list[str] | None = None,
       case_sensitive: bool = False,
       context_words: int = 80,
       context_lines: int = 2,
       max_matches: int = 100,
   ) -> list[SearchHit]
   ```

   Runs a validated regex over either explicit `source_ids` or a scoped metadata
   filter. Each `SearchHit` must include the snippet directly, using
   `context_words` and `context_lines`, plus line references and heading metadata.

4. `expand_hit`

   ```python
   expand_hit(
       hit_id: str,
       context_words: int = 250,
       context_lines: int = 10,
   ) -> SearchHit
   ```

   Expands an existing search hit when the original snippet is promising but too
   small. We do not need a generic `read_snippet` tool for MVP because search
   results already return snippets; expansion should be anchored to a prior hit.

5. `add_evidence_candidate`

   ```python
   add_evidence_candidate(
       hit_id: str,
       city: str,
       field: str,
       reason: str,
       confidence: float,
   ) -> EvidenceCandidate
   ```

   Saves a useful hit into the run's evidence basket. This does not mutate the
   source document; it marks a snippet as evidence for later claim extraction and
   resolution.

6. `list_evidence_candidates`

   ```python
   list_evidence_candidates() -> list[EvidenceCandidate]
   ```

   Lets the agent review selected evidence and avoid duplicate or contradictory
   candidate snippets.

7. `mark_no_evidence_found`

   ```python
   mark_no_evidence_found(
       city: str,
       field: str,
       searched_source_ids: list[str],
       search_summary: str,
   ) -> NoEvidenceRecord
   ```

   Records that the agent searched relevant sources and did not find usable
   evidence. This is important because it separates "not searched" from "searched
   and not found" before the assumptions stage.

LLM-recommended tool, but not yet agreed for MVP:

8. `proximity_search`

   ```python
   proximity_search(
       terms: list[str],
       near_terms: list[str],
       source_ids: list[str] | None = None,
       cities: list[str] | None = None,
       countries: list[str] | None = None,
       verticals: list[str] | None = None,
       tef_sectors: list[str] | None = None,
       tef_transitions: list[str] | None = None,
       source_types: list[str] | None = None,
       max_distance_words: int = 40,
       context_words: int = 80,
       context_lines: int = 2,
       max_matches: int = 100,
   ) -> list[SearchHit]
   ```

   Finds snippets where one set of terms appears near another set of terms. This
   may be easier and safer than regex for searches like "charging" near
   "2030 target" or "renovation" near "m2", but we still need to decide whether
   it belongs in MVP or should wait until regex search proves insufficient.

Every search result should include enough context to be directly useful:

```json
{
  "hit_id": "hit_123",
  "source_id": "munich_mobility_strategy_2035",
  "title": "Munich Mobility Strategy 2035",
  "city": "Munich",
  "line_start": 210,
  "line_end": 218,
  "matched_text": "18,000 charging points",
  "snippet": "...matched text with requested context words and context lines...",
  "heading_path": ["Mobility", "Charging infrastructure"]
}
```

The implementation can use `ripgrep`, Python file scanning, a Markdown parser,
or vector metadata under the hood. The model only sees safe, scoped tools.

## Regex Support

Regex should be supported because many climate-plan facts are easiest to find by searching for years, numeric values, units, and nearby concepts.

Useful examples:

```regex
(?i)\b(public|normal|fast)\s+charging\s+(points|stations|infrastructure)\b
```

```regex
(?i)\b(2030|2040|2050)\b.{0,120}\b(target|goal|aim|planned|expected)\b
```

```regex
(?i)\b(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s*(MW|GWh|tCO2e|charging points|vehicles)\b
```

Guardrails:

- Cap pattern length.
- Validate regex before execution.
- Reject expensive patterns such as nested quantifiers.
- Avoid backreferences unless there is a concrete need.
- Use timeouts for regex execution.
- Search only files selected by tags.
- Cap scanned files, scanned bytes, matches, and context lines.
- Log pattern, filters, source IDs, hit counts, and elapsed time.

In most cases, the model should start with targeted regex searches for numeric
or unit-heavy follow-ups. Proximity search is a possible helper, but it is not
yet agreed for MVP.

## Evidence Record

The agentic search stage should return structured claims.

Example:

```json
{
  "city": "Vienna",
  "field": "public_ev_chargers_2030_target",
  "value": 18000,
  "unit": "charging points",
  "source_id": "vienna_cap_2024",
  "source_type": "city_cap",
  "publication_year": 2024,
  "line_start": 212,
  "line_end": 219,
  "quote": "The city targets 18,000 public charging points by 2030.",
  "confidence": 0.86,
  "claim_role": "fills_missing"
}
```

Required fields:

- `city`
- `field`
- `value`
- `unit`
- `source_id`
- `line_start`
- `line_end`
- `quote`
- `confidence`
- `claim_role`

The quote and line references are important because they make the result inspectable and debuggable. The source file path can be resolved internally from `source_id`; it does not need to be manually tagged.

## Resolver Rules

The resolver should determine how CCC and external evidence interact.

Initial policy:

- If CCC has a current value and external evidence agrees, keep CCC primary and record external confirmation.
- If CCC is missing a field and external evidence comes from a tagged candidate source, use external evidence to fill the gap when confidence is high enough.
- If sources materially conflict, preserve both and flag `conflict_review_required`.
- Source-tier based automatic conflict resolution is out of MVP scope and should be treated as a future expansion.
- If nothing reliable is found, pass the field to the assumptions estimator.

This keeps the distinction clear:

```text
CCC = primary evidence
tagged external Markdown = governed enrichment evidence
assumptions = last resort
```

## MVP Scope

1. Define the external source metadata schema.
2. Add a small curated set of tagged Markdown files.
3. Build controlled literal/proximity/regex search over those files.
4. Add a simple agent loop that searches for missing fields and returns evidence claims.
5. Add resolver rules for fill, confirm, supersede, conflict, and unresolved.
6. Wire the resolved evidence into the enrichment stage before assumptions.
7. Persist artifacts for audit and debugging.
8. Add tests for filtering, regex validation, claim extraction, and resolver behavior.

## Open Questions

- What is the first vertical for the MVP: mobility, energy, or built environment?
- Should external tagged Markdown search run only when CCC is missing/stale, or also when CCC already has data so external sources can confirm or challenge it?
- When CCC and external sources materially conflict, should MVP always preserve both for review, or can a newer source be preferred in some cases?
- If we later introduce source tiers, which tiers can supersede CCC automatically?
- Should conflicts block writer output or appear as flagged caveats?
- How much search trace should be exposed in the frontend?
- Do we want multilingual synonym expansion in the first version?
- Should `proximity_search` be included in MVP, or should we start with regex search plus hit expansion and add proximity search only if needed?

## Success Criteria

- Existing CCC search behavior remains unchanged.
- External sources are only considered when selected city and source tags match.
- The model can iteratively search tagged Markdown using bounded regex tools, with proximity search treated as an optional extension.
- Every accepted external fact has source provenance and line references.
- Conflicts are explicit, not silently resolved by the model.
- Assumptions are generated only after CCC and tagged external evidence cannot resolve a field.
- Adding a new external source usually requires adding Markdown plus metadata, not changing code.
