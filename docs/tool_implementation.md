# Tool Implementation Deep Dive

## Goal

Build a controlled tool layer that lets an LLM search tagged external Markdown
sources without raw shell access. The tool layer should feel flexible enough for
agentic research, but stay bounded, auditable, and safe in production.

The MVP tool contract from `plan.md` is:

1. `get_tag_options`
2. `list_candidate_sources`
3. `regex_search`
4. `expand_hits`
5. `add_evidence_candidates`
6. `list_evidence_candidates`
7. `mark_no_evidence_found`

`proximity_search` is an LLM-recommended possible extension, but not agreed for
MVP yet.

## Runtime Inputs

The external Markdown folder should look like:

```text
external_docs/
  sources.yaml
  krakow/
    krakow_electromobility_strategy_2030.md
  munich/
    munich_mobility_strategy_2035.md
```

`sources.yaml` is the metadata entrypoint. Each `source_id` maps to a Markdown
file whose filename stem is the same as `source_id`. Example:

```yaml
sources:
  - source_id: krakow_electromobility_strategy_2030
    title: Krakow Electromobility Strategy 2030
    upstream_group: tier_1_city_plans
    city: [Krakow]
    country: [Poland]
    publication_year: 2024
    description: City electromobility strategy covering public charging infrastructure.
    source_type: mobility_plan
    verticals: [mobility]
    tef_sectors: [transport]
```

For MVP, these TEF sector tags are populated by the current TEF mapping
pipeline at document level. We do not require hand-authored per-initiative TEF
tagging in this search layer.

No manual `path` field is required for MVP. The loader resolves a file by
searching under `external_docs/` for `<source_id>.md`. If duplicate stems exist,
metadata validation fails.

## Agent Loop and Tool Visibility

This agent should be modeled as a bounded research loop, not as a single
search followed by a single answer. After each `regex_search`, it may receive
many candidate hits. It then has to decide which hits to ignore, which hits to
expand, which hits are already strong enough to save as evidence, and whether
the current search pattern was wrong and needs to be replaced with another one.

The high-level loop is:

```text
get_tag_options
-> list_candidate_sources
-> regex_search
-> triage returned hits
-> expand_hits for up to 3 promising hits at once
-> add_evidence_candidates for any confirmed useful hits
-> run another regex_search if coverage is still weak
-> repeat until the field is resolved or search budget is exhausted
-> mark_no_evidence_found if nothing usable was found
```

The important detail is that the agent may pass through the middle of this loop
several times for the same field. That is expected behavior. The tool layer
should support it directly instead of assuming one search is enough.

### Stage 1: Source discovery

Goal:
- find the smallest relevant source set before any text search starts

What the agent knows:
- the user question
- the selected city
- the field or gap that CCC did not answer

Tools available:
- `get_tag_options`
- `list_candidate_sources`
- `regex_search`
- `list_evidence_candidates`
- `mark_no_evidence_found`

What the agent does:
- call `get_tag_options` to see the valid cities, verticals, source types, and
  TEF sectors
- call `list_candidate_sources` to narrow the file set
- avoid searching the whole corpus unless the backend explicitly allows it

Why `expand_hits` is not shown yet:
- before a search result exists, expansion has no anchor and becomes document
  browsing

### Stage 2: First search pass

Goal:
- run a small, scoped search to test whether the candidate files contain useful
  evidence

What the agent does:
- draft source-language synonyms or equivalent search terms with the LLM before
  building the first bounded search pattern
- call `regex_search` with a narrow pattern
- request modest snippet context first
- inspect the returned hit list rather than immediately expanding everything

Expected outcome:
- the agent gets a batch of hits, not a final answer

At this point the tool surface grows because the agent now has anchored hit IDs
from the current run.

Additional tools now available:
- `expand_hits`
- `add_evidence_candidates`

### Stage 3: Hit triage

Goal:
- decide what to do with each returned hit without flooding the context window

After one `regex_search`, the agent should classify hits into four groups:

1. Ignore:
   the snippet is clearly irrelevant, too generic, or a duplicate of a better
   hit
2. Save immediately:
   the snippet already contains a concrete value, city scope, and enough
   surrounding context to be trusted
3. Expand:
   the snippet looks promising but is missing surrounding explanation, table
   context, or a sentence that confirms the exact metric
4. Search again:
   the current pattern was too broad, too narrow, or aimed at the wrong wording

This triage step is where most of the agentic behavior happens. The agent is
not just “reading results”; it is actively deciding which branch to take next.

### Stage 4: Controlled expansion

Goal:
- inspect only the few hits that deserve more context

What the agent does:
- call `expand_hits` for at most 3 hits in one call
- use expansion only for hits that are plausible evidence candidates
- avoid expanding weak or duplicate hits

Why the limit matters:
- expansion is the most expensive read step because it increases snippet size
- a limit of 3 forces the agent to rank hits instead of expanding everything

### Stage 5: Evidence capture

Goal:
- persist only the snippets that are actually useful for later claim extraction

What the agent does:
- call `add_evidence_candidates` with one or more selected hits
- use `list_evidence_candidates` to see what is already saved
- avoid saving multiple weak variants when one strong snippet is enough

What this changes:
- once a snippet is in the evidence basket, the run no longer depends on the
  agent remembering it in context
- later steps can work from saved evidence rather than the entire search trace

### Stage 6: Repeat or stop

Goal:
- decide whether the current field is resolved

The agent should continue the loop when:
- saved evidence is still ambiguous
- no hit contains the actual target value
- the first search pattern only found supporting wording, not the metric itself

The agent should stop and record failure when:
- relevant sources were searched
- several patterns were tried
- no usable evidence was found within budget

At that point it should call `mark_no_evidence_found`.

### Why tool visibility changes over time

The agent should only see the tools that are useful for the current stage.

1. Before any search:
   show `get_tag_options`, `list_candidate_sources`, `regex_search`,
   `list_evidence_candidates`, and `mark_no_evidence_found`
2. After at least one `regex_search` returned hits for the current city-field task:
   add `expand_hits` and `add_evidence_candidates`
3. After evidence has been saved:
   keep `list_evidence_candidates` visible so the agent can avoid duplicates
   and inspect current run state

This staged tool visibility keeps the early context smaller and nudges the
agent toward the intended workflow: narrow first, search second, expand only
when justified, and save evidence instead of carrying everything in memory.

### Concrete example: Krakow missing EV charging target

Assume CCC does not provide a usable answer for
`public_ev_chargers_2030_target` in Krakow.

1. The agent starts with `get_tag_options`, `list_candidate_sources`,
   `regex_search`, `list_evidence_candidates`, and
   `mark_no_evidence_found`.
2. It calls `get_tag_options` and then `list_candidate_sources` with filters
   such as `cities=["Krakow"]`, `verticals=["mobility"]`, and optional TEF
   sector filters.
3. It runs `regex_search` over the narrowed metadata-scoped candidate set with a
   narrow pattern around charging targets and 2030.
4. Once the first search returns hit IDs, the agent also gets access to
   `expand_hits` and `add_evidence_candidates`.
5. If the search returns many hits, the agent ignores weak ones, saves obvious
   ones immediately, and selects only the strongest ambiguous hits for
   expansion.
6. If 3 hits need more context, it calls
   `expand_hits(hit_ids=["h4", "h7", "h9"])`.
7. If expanded hits `h7` and `h9` contain concrete Krakow targets, it saves
   them with `add_evidence_candidates(...)`.
8. If the field is still unresolved, it runs another `regex_search` with a
   refined pattern. If relevant sources were searched and no usable evidence
   was found, it calls `mark_no_evidence_found`.

### Large review sets

If the saved snippets or expansion results for one field exceed roughly 100k
tokens in total, do not switch to a smaller model. Split the review set into
batches of at most 50k tokens, review those batches in parallel agents, and
merge the selected evidence back into the main run state. The parallel agents
should review bounded snippet batches, not reopen the whole source corpus.

## Tool Schemas

### 1. `get_tag_options`

```python
def get_tag_options() -> TagOptions: ...
```

Returns distinct values derived from `sources.yaml`:

```json
{
  "cities": ["Krakow", "Munich"],
  "countries": ["Germany", "Poland"],
  "publication_years": [2021, 2024],
  "source_types": ["city_cap", "mobility_plan"],
  "verticals": ["mobility", "energy"],
  "tef_sectors": ["transport"]
}
```

### 2. `list_candidate_sources`

```python
def list_candidate_sources(
    cities: list[str] | None = None,
    countries: list[str] | None = None,
    verticals: list[str] | None = None,
    tef_sectors: list[str] | None = None,
    source_types: list[str] | None = None,
    publication_year_min: int | None = None,
    publication_year_max: int | None = None,
    max_files: int = 50,
) -> list[SourceSummary]: ...
```

Important behavior:

- Validate all requested filter values against known tag options.
- Normalize simple strings case-insensitively for matching.
- Use OR within the same filter, AND across filter groups.
- Apply a hard server-side `max_files`, even if the LLM requests more.
- Return source summaries, not full documents.

Example source summary:

```json
{
  "source_id": "krakow_electromobility_strategy_2030",
  "title": "Krakow Electromobility Strategy 2030",
  "city": "Krakow",
  "country": "Poland",
  "publication_year": 2024,
  "source_type": "mobility_plan",
  "verticals": ["mobility"],
  "tef_sectors": ["transport"],
  "description": "City electromobility strategy covering public charging infrastructure."
}
```

### 3. `regex_search`

```python
def regex_search(
    pattern: str,
    cities: list[str] | None = None,
    countries: list[str] | None = None,
    verticals: list[str] | None = None,
    tef_sectors: list[str] | None = None,
    source_types: list[str] | None = None,
    case_sensitive: bool = False,
    context_words: int = 80,
    context_lines: int = 2,
    max_matches: int = 100,
) -> list[SearchHit]: ...
```

Search scope rules:

- Require at least one metadata filter, unless the current run already has a
  candidate set from `list_candidate_sources`.
- When the current run already has a candidate set, intersect it with any
  additional metadata filters instead of exposing explicit `source_ids` in the
  public tool contract.
- Refuse a search that would scan every source without filters.
- Cap `max_matches`, `context_words`, and `context_lines`.
- Return snippets directly; do not require a separate read tool.

Example hit:

```json
{
  "search_id": "s1",
  "hit_id": "h1",
  "source_id": "krakow_electromobility_strategy_2030",
  "title": "Krakow Electromobility Strategy 2030",
  "city": "Krakow",
  "line_start": 136,
  "line_end": 158,
  "matched_text": "1,200 public charging points",
  "snippet": "By 2030, Krakow plans to expand public charging infrastructure to 1,200 public charging points...",
  "heading_path": ["Charging Infrastructure", "Public network targets"],
  "truncated": false
}
```

### 4. `expand_hits`

```python
def expand_hits(
    hit_ids: list[str],
    context_words: int = 250,
    context_lines: int = 10,
) -> list[SearchHit]: ...
```

`expand_hits` should only work for hit IDs created in the current run/session.
This avoids arbitrary document browsing. The tool finds each original hit,
recomputes a wider snippet around the original match, and returns the same hit
shape with larger context.

Validation:

- `hit_ids` must contain between 1 and 3 IDs.
- all hit IDs must exist in the current run/session.
- duplicate hit IDs in the same request should be rejected.
- return hits in the same order they were requested.

The agent should only see this tool after tool 3 (`regex_search`) has returned
at least one hit for the current city-field task. This keeps the initial tool
surface smaller, limits context use, and prevents the LLM from trying to browse
documents before a bounded search has established a specific hit anchor.

### 5. `add_evidence_candidates`

```python
def add_evidence_candidates(
    candidates: list[EvidenceCandidateInput],
) -> list[EvidenceCandidate]: ...
```

This stores one or more selected hits into a per-run evidence basket. It should
not mutate the source Markdown file. Every stored evidence candidate should
carry the exact matched text and the exact quote that justified selection.

Validation:

- every `hit_id` must exist in the current run/session.
- every `confidence` must be clamped to `0.0 <= confidence <= 1.0`.
- `city`, `field`, and `reason` are required for every candidate.
- duplicate `hit_id + field` entries should be ignored or replaced.
- there is no separate business cap on the number of evidence candidates saved
  in a run for MVP.

### 6. `list_evidence_candidates`

```python
def list_evidence_candidates() -> list[EvidenceCandidate]: ...
```

Returns evidence already selected in the current run/session. Keep this concise:
candidate ID, source ID, field, line range, matched text, short quote preview,
confidence, and reason.

### 7. `mark_no_evidence_found`

```python
def mark_no_evidence_found(
    city: str,
    field: str,
    searched_source_ids: list[str],
    search_summary: str,
) -> NoEvidenceRecord: ...
```

This records that relevant sources were searched but no usable snippet was
found. It is useful downstream because "searched and not found" is different
from "not searched".

## Document Parsing and Search Index

For MVP, we do not need vector search. A simple per-file text index is
enough.

Parse each Markdown file into a `DocumentIndex`:

```python
@dataclass(frozen=True)
class DocumentIndex:
    source: SourceMetadata
    text: str
    lines: tuple[str, ...]
    line_start_offsets: tuple[int, ...]
    heading_ranges: tuple[HeadingRange, ...]
    word_spans: tuple[WordSpan, ...]
```

`heading_ranges` can reuse the logic already present in
`backend/modules/vector_store/markdown_blocks.py`, which parses Markdown blocks
with heading paths and line ranges. For the search tools, we need line-to-heading
lookup:

```python
@dataclass(frozen=True)
class HeadingRange:
    start_line: int
    end_line: int
    heading_path: tuple[str, ...]
```

If a hit occurs on line 142, find the deepest heading range that contains line
142. If no heading exists, return an empty heading path or the document title.

## Regex Search Implementation

Use Python regex search over full document text, not line-by-line search. Full
text search makes it possible to match multi-line phrases and then map the match
back to lines.

Basic flow:

```text
validate pattern
-> resolve candidate source IDs
-> load DocumentIndex snapshots
-> compile regex
-> scan each document text
-> for each match, calculate line range
-> calculate context range by words and lines
-> derive heading_path
-> create SearchHit
-> stop at max_matches / snippet limits
```

### Regex Safety

Python's standard `re` module has no built-in per-match timeout. For MVP, keep
the pattern language constrained enough to reduce regex denial-of-service risk.

Recommended validation:

- `pattern` length max, e.g. 300 characters.
- Reject empty patterns.
- Reject too many alternatives, e.g. more than 30 `|` tokens.
- Reject nested unbounded quantifiers such as `(.*)+`, `(.+)*`, `(.{0,100})+`.
- Reject backreferences like `\1` unless we explicitly decide to support them.
- Reject lookbehind initially.
- Compile before scanning and return a validation error if compilation fails.
- Enforce document count, byte count, match count, and elapsed-time caps.

If regex safety becomes a concern, use a safer engine later:

- shell out internally to `rg` with fixed caps and no user shell access;
- or add a regex library with timeout support;
- or implement a simpler query DSL for common numeric/unit cases.

The LLM never receives shell access either way.

## Response Size and Context Budget

The LLM context window is finite, so every tool response needs hard limits.

Recommended server defaults:

```yaml
external_search:
  max_files_per_search: 50
  max_matches_per_search: 100
  max_regex_searches_per_field: 5
  default_context_words: 80
  max_context_words: 250
  default_context_lines: 2
  max_context_lines: 10
  max_expand_hits_per_call: 3
  max_snippet_chars: 4000
  max_pattern_chars: 300
  review_batch_trigger_tokens: 100000
  review_batch_max_tokens: 50000
```

Response trimming policy:

- Cap per-hit snippet chars.
- Prefer returning fewer complete hits over many truncated hits.
- Include `truncated: true` when a snippet was shortened.
- Include `more_matches_available: true` at the response level if the search hit
  a cap.
- If snippet review for one field grows beyond `review_batch_trigger_tokens`,
  split it into review batches capped at `review_batch_max_tokens`.
- Review those batches in parallel agents rather than switching to a smaller
  model.

For `list_candidate_sources`, return summaries only. If there are too many
candidate files, return the top `max_files` sorted by direct city match first,
then country match, then publication year descending.

## Hit IDs and Session State

Tool-generated IDs should be stable within one run and simple enough that the
LLM does not waste tokens on them.

```text
searches: s1, s2, s3, ...
hits: h1, h2, h3, ...
evidence candidates: e1, e2, e3, ...
```

All three sequences reset per run.

The backend keeps a per-run hit cache:

```python
@dataclass(frozen=True)
class HitRecord:
    hit_id: str
    source_id: str
    match_start: int
    match_end: int
    pattern: str
    line_start: int
    line_end: int
```

`expand_hits` and `add_evidence_candidates` must refer to this cache. This
avoids letting the LLM read arbitrary line ranges and makes every saved evidence
item traceable to a prior search.

## Evidence Basket Persistence

Evidence candidates are per run, not global. The source docs remain read-only.

Recommended artifact shape:

```json
{
  "run_id": "run_123",
  "candidates": [
    {
      "candidate_id": "e1",
      "hit_id": "h7",
      "source_id": "krakow_electromobility_strategy_2030",
      "field": "public_ev_chargers_2030_target",
      "line_start": 136,
      "line_end": 158,
      "confidence": 0.9,
      "reason": "Contains a concrete 2030 city-level target."
    }
  ],
  "no_evidence": []
}
```

Use atomic writes:

```text
write output/<run_id>/stage_files/008_enrichment/external_source_search_audit.tmp
fsync if needed
rename to output/<run_id>/stage_files/008_enrichment/external_source_search_audit.json
```

In Python, use `Path.write_text` to a temporary file and `Path.replace` for the
final move. For multi-process deployments, guard per-run evidence writes with a
lock file.

## Production Concurrency

The main concurrency risk is not multiple users reading the same Markdown files.
Concurrent reads are safe. The real risks are:

- reading a file while a conversion job is writing it;
- reading metadata while it is being replaced;
- two tool calls writing the same per-run evidence artifact at the same time;
- stale in-memory registry state after source updates.

### Source files should be immutable at runtime

Production should treat converted Markdown as read-only:

```text
external_docs/releases/2026-04-28T120000Z/
  sources.yaml
  ...
external_docs/current -> releases/2026-04-28T120000Z
```

Search workers read from `external_docs/current`. A source update writes a new
release directory and atomically switches `current` after validation. Existing
requests keep using the snapshot they started with. New requests use the new
snapshot.

This avoids partial reads and avoids needing locks for normal searches.

### Conversion jobs must write elsewhere first

Conversion should never write directly into the live folder. It should:

1. write converted Markdown and `sources.yaml` to a staging folder;
2. validate metadata and file existence;
3. build or warm the search index if desired;
4. publish by atomic directory/symlink switch.

### In-process cache

The app can cache `SourceRegistry` and `DocumentIndex` objects by snapshot ID:

```text
cache key = resolved current folder path + sources.yaml mtime/hash
```

Use an in-process read/write lock for cache refresh. Search calls can share the
same immutable objects without locking. Only cache rebuild needs a lock.

### Per-run writes need locking

Evidence basket writes are mutable. Guard them with:

- an in-process `threading.Lock` keyed by `run_id`;
- plus a filesystem lock if multiple API worker processes can handle the same
  run.

Without adding dependencies, a simple standard-library lock file can be:

```text
output/<run_id>/external_evidence.lock
```

Acquire by `os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)`. Release
by closing and deleting the lock file. Add a timeout and stale-lock cleanup.

If we already use a database or queue for run state later, move evidence basket
writes there and avoid filesystem lock complexity.

## File Reading Strategy

Search should read files as UTF-8 text. Do not keep long-lived open file handles.

Recommended:

- load each Markdown file into an immutable `DocumentIndex`;
- close the file immediately;
- search in memory;
- invalidate cache only when the external docs snapshot changes.

This prevents "multiple people opening the same files" from being a production
issue. The files are not modified by requests, and readers do not keep handles
open across source updates.

## Error Handling

Tool errors should be structured and useful to the LLM:

```json
{
  "error": {
    "code": "INVALID_FILTER",
    "message": "Unknown city filter: Krakov",
    "allowed_values": ["Krakow"]
  }
}
```

Recommended error codes:

- `INVALID_FILTER`
- `SOURCE_NOT_FOUND`
- `SOURCE_SCOPE_REQUIRED`
- `REGEX_TOO_LONG`
- `REGEX_UNSAFE`
- `REGEX_COMPILE_ERROR`
- `SEARCH_LIMIT_EXCEEDED`
- `HIT_NOT_FOUND`
- `EVIDENCE_WRITE_CONFLICT`

Do not expose raw stack traces to the LLM.

## Audit Logging

Every tool call should be logged with enough metadata for debugging:

```json
{
  "tool": "regex_search",
  "run_id": "run_123",
  "resolved_source_ids": ["krakow_electromobility_strategy_2030"],
  "filters": {
    "cities": ["Krakow"],
    "verticals": ["mobility"]
  },
  "pattern": "...",
  "match_count": 3,
  "elapsed_ms": 42,
  "truncated": false
}
```

Do not log full document text. Persist bounded per-run logs that include search
queries, filters, selected source IDs, hit counts, elapsed time, candidate IDs,
matched text, quote previews, and resolver outcomes. Full snippets should stay
inside normal evidence artifacts and follow the same artifact/privacy policy as
existing run artifacts.

## Practical MVP Cut

The smallest useful implementation is:

1. Load `external_docs/sources.yaml`.
2. Validate and index converted Markdown files into immutable in-memory
   `DocumentIndex` objects.
3. Expose `get_tag_options`, `list_candidate_sources`, `regex_search`,
   `expand_hits`, `add_evidence_candidates`, `list_evidence_candidates`, and
   `mark_no_evidence_found`.
4. Return snippets directly from `regex_search`.
5. Allow batched expansion of up to 3 hits per call.
6. Persist selected evidence per run with atomic writes.
7. Keep source docs read-only in production.

This gives the LLM agent enough power to perform iterative local research while
keeping the production system scoped, deterministic, and auditable.
