# Retrieval Evidence Diversity Analysis

## Purpose

This note captures a retrieval-only diagnosis of the benchmark misses and a
priority-ordered set of fixes. The core claim is that the current issue is not
retrieval volume. The system already pulls many chunks. The issue is that it
does not retrieve enough different evidence shapes.

In the seven-city case, the pipeline already retrieved 442 chunks with 20
fallback chunks per city/query and `max_distance=1.0`. That means the main
problem is not "more chunks." It is query diversity, score fusion, and chunk
representation in retrieval.

## Relevant Code Paths

- [`backend/prompts/orchestrator_research_question_system.md`](../prompts/orchestrator_research_question_system.md)
- [`backend/modules/orchestrator/module.py`](../modules/orchestrator/module.py)
- [`backend/modules/vector_store/retriever.py`](../modules/vector_store/retriever.py)
- [`backend/modules/vector_store/chunk_packer.py`](../modules/vector_store/chunk_packer.py)
- [`backend/modules/vector_store/table_utils.py`](../modules/vector_store/table_utils.py)
- [`backend/utils/config.py`](../utils/config.py)

## What Is Broken

### Query generation is too narrow

The query refiner is forced to produce only 2 extra retrieval variants, and the
pipeline caps total retrieval queries at 3. In practice these queries are
usually semantic near-duplicates of the original question, not orthogonal
evidence probes.

### Multi-query fusion is too primitive

[`retriever.py`](../modules/vector_store/retriever.py) currently merges results
with a "best distance wins" rule. That favors one generic good-sounding chunk
over chunks that are supported across multiple query families.

### Table retrieval is underpowered

In [`chunk_packer.py`](../modules/vector_store/chunk_packer.py) and
[`table_utils.py`](../modules/vector_store/table_utils.py), table embeddings are
built from a summary plus only the first few preview rows. A target row later
in the row group can be effectively invisible to vector search.

### Paragraph embeddings are missing structural context

For non-table chunks, the embedding text is just raw paragraph text, without
heading-path context. That weakens generic retrieval on scenario, program,
funding, and implementation sections.

## Retrieval-Only Changes

### 1. Expand retrieval from 3 queries to 5 or 6 fixed evidence families

The prompt in
[`orchestrator_research_question_system.md`](../prompts/orchestrator_research_question_system.md)
should move away from "one keyword-heavy + one evidence-oriented" variants and
instead emit fixed retrieval families such as:

- canonical question
- measure/program family:
  `initiative action measure program project package funding scheme policy`
- quantitative family:
  `target count budget cost investment CAPEX funding amount year timeline milestone`
- scenario/projection family:
  `scenario assumption projection forecast pathway study roadmap strategy`
- implementation family:
  `pilot demonstration tender procurement rollout installed planned under construction`
- tabular/indicator family:
  `table metric indicator row emissions energy demand public private split`

This remains city-agnostic. The selected-city filter already scopes retrieval.

### 2. Add a second retrieval pass for missing evidence dimensions

Do not stop after the initial semantic query set. After pass 1, inspect each
city's seed set for missing evidence shapes. Trigger a second generic
evidence-family query when the question clearly needs a dimension that is not
yet represented.

Examples:

- no currency/year chunks when the question asks for budgets or timelines
- no table chunks when the question asks for metrics or targets
- no scenario/program chunks when the question asks for projections,
  assumptions, or broad plans

This remains generalized and does not require city names or guessed project
names.

### 3. Replace "best distance wins" with fused ranking

[`retriever.py`](../modules/vector_store/retriever.py) should not treat
`_merge_rows_best_distance` as the main merge rule. Use reciprocal-rank fusion
or a similar simple fused score that boosts chunks appearing across multiple
query families.

Why this matters:

- generic charging chunks stop dominating because one query liked them
- scenario/program/table chunks that appear across specialized families rise

### 4. Change table indexing so rows are actually retrievable

This is the biggest indexing-side retrieval fix.

Current behavior uses a summary plus early-row previews. Instead, one of these
should happen:

- embed each row group with full row text when it fits
- embed both the summary and the raw row-group text as separate retrievable
  variants
- sample rows from the start, middle, and end of the row group rather than only
  the first rows

This likely matters for misses like the Dresden `-260,900 tCO2eq/a` row and
similar indicator-table misses.

### 5. Add heading-path text to paragraph embeddings

For non-table chunks, prepend heading context into the embedded document rather
than storing it only in metadata.

Example shape:

```text
Heading: Mobility > Climate-friendly drives and fuels
Content: ...
```

This improves retrieval for scenario, funding, and program sections without any
city-specific logic.

### 6. Add diversity constraints to seed retrieval

Per city, reserve some seed slots by chunk type or evidence shape, for example:

- at least a few table chunks
- at least a few narrative paragraphs
- optional heading diversity

Without this, one city can be saturated with many near-duplicate charging
infrastructure paragraphs.

### 7. Increase context expansion modestly as a secondary fix

Current defaults in [`config.py`](../utils/config.py) are:

- `context_window_chunks=0`
- `table_context_window_chunks=1`

A reasonable follow-up adjustment is:

- paragraph neighbors: `1`
- table neighbors: `2`

This is not the main fix, but it helps when a nearby chunk contains the exact
number while the seed chunk carries only the framing.

## Which Changes Map To The Actual Misses

- Aachen `EUR 715.1M`, Dresden NeutralPath, and Munster `9,000 / 79,000`:
  query-family expansion plus second-pass evidence coverage
- Dresden `-260,900 tCO2eq/a` and Mannheim `749 EVs`: table indexing fix
- Aachen `1,600 vs 1,800` conflict: query fusion and diversity constraints
  reduce dominance of one generic surrogate
- Jessener / Vonovia style project paragraphs: program and implementation query
  families plus heading-aware paragraph embeddings

## What Not To Do First

- do not start by increasing `max_distance` or just pulling more chunks; the
  pipeline already pulls a lot
- do not inject city names or guessed project names
- do not treat this as a markdown or writer problem when Stage A is the actual
  miss

## Priority Order

1. Query family expansion
2. Table/index representation fix
3. Multi-query fusion instead of best-distance merge
4. Coverage-seeking second pass
5. Diversity slots and modest neighbor-window increase
