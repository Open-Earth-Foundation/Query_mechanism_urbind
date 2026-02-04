# Implementation Plan: Structured Extractions + Recursive Summarization

## Goal
Prevent large multi-city extractions from overwhelming the orchestrator and writer by combining:
1) Structured metadata and indexing (topic clusters + source mapping), and
2) Parallel recursive summarization (tree summaries with controlled token budgets).

## Combined approach (high level)
- Step 1: Keep per-chunk extractions but require structured fields (topic, subtopic, relevance score, source id).
- Step 2: Build a topic index that clusters excerpts by subtopic and calculates coverage + relevance.
- Step 3: If excerpts exceed a size threshold, run recursive tree summarization per cluster (parallel), then roll up to a master summary.
- Step 4: Feed the orchestrator the index + summaries (not raw excerpts) so it can validate topic-by-topic.
- Step 5: Feed the writer the master summary + high-relevance cluster summaries with source mapping.

## Data contract changes
### New/updated markdown extraction fields
Add structured metadata to each excerpt. Keep values deterministic where possible.

Fields to add to `MarkdownExcerpt`:
- `source_id`: string, deterministic id. Example: `path|chunk_index`.
- `source_path`: string, document path.
- `chunk_index`: int.
- `chunk_count`: int.
- `topic`: string (canonical label, e.g., "transport", "buildings").
- `subtopic`: string (more specific, e.g., "public transit electrification").
- `relevance_score`: float in [0, 1].

Notes:
- Populate `source_id`, `source_path`, `chunk_index`, `chunk_count`, and `city_name` in Python from the document metadata.
- Let the LLM fill `topic`, `subtopic`, `snippet`, `answer`, `relevant`, and `relevance_score`.

### New index/summaries payload
Add a new `markdown_index` + `markdown_summaries` payload to the context bundle.

Example shape:
```
{
  "markdown_index": {
    "clusters": [
      {
        "cluster_id": "transport::public-transit",
        "topic": "transport",
        "subtopic": "public transit electrification",
        "relevance_score": 0.92,
        "city_names": ["Munich", "Berlin"],
        "source_ids": ["..."],
        "excerpt_count": 18
      }
    ],
    "coverage": {
      "cities": 5,
      "clusters": 12,
      "excerpts": 84
    }
  },
  "markdown_summaries": {
    "cluster_summaries": [
      {
        "cluster_id": "transport::public-transit",
        "summary": "...",
        "source_ids": ["..."],
        "city_names": ["..."],
        "relevance_score": 0.92
      }
    ],
    "master_summary": {
      "summary": "...",
      "source_ids": ["..."],
      "city_names": ["..."],
      "cluster_ids": ["..."]
    }
  }
}
```

## Pipeline changes
1. Extraction (markdown researcher)
   - Update prompt + models to output structured metadata fields.
   - Inject deterministic metadata in Python after the LLM returns excerpts.

2. Indexing / clustering
   - Add a small indexer module that groups excerpts by (topic, subtopic).
   - Compute cluster-level relevance (max or weighted by excerpt scores).
   - Persist index JSON and update context bundle.

3. Recursive summarization (parallel)
   - Add a summarizer agent that accepts a list of excerpts and outputs a short summary + source_ids.
   - Implement tree summarization:
     - Batch excerpts within a token budget.
     - Summarize each batch in parallel.
     - Merge batch summaries into cluster summaries.
     - Merge cluster summaries into a master summary.
   - Persist summaries JSON and update context bundle.

4. Orchestrator + writer integration
   - Orchestrator sees `markdown_index` + `markdown_summaries` instead of raw excerpts.
   - Writer receives the same, focusing on master + top clusters.

## Orchestrator behavior changes
- Use `markdown_index.clusters` to prioritize high relevance clusters.
- Validate topic-by-topic: if a required topic has low coverage, request more markdown.
- If token budget is constrained, prefer index + summaries and avoid raw excerpts.
- Decision rules update: for multi-city questions, require cluster coverage across cities before "write".

## Writer behavior changes
- Use `master_summary` as the backbone.
- Expand using cluster summaries, especially for high relevance clusters.
- When detailing per-city differences, reference cluster summaries filtered by city.
- Never mention source ids or internal metadata in the final answer.

## Config additions (suggested)
Add a summarizer config block (or extend markdown_researcher config):
- `summarizer.max_input_tokens`
- `summarizer.max_output_tokens`
- `summarizer.batch_token_budget`
- `summarizer.max_workers`
- `summarizer.min_excerpts_for_summary` (skip if small)
- `summarizer.cluster_min_relevance` (optional filter)

## Files to update (planned)
- `app/modules/markdown_researcher/models.py`
- `app/prompts/markdown_researcher_system.md`
- `app/modules/markdown_researcher/agent.py`
- `app/modules/markdown_researcher/services.py` (helper for deterministic metadata)
- `app/modules/markdown_researcher/indexer.py` (new)
- `app/modules/summarizer/` (new agent + models + prompt)
- `app/services/run_logger.py` (store new bundle fields)
- `app/modules/orchestrator/module.py` (wire index + summaries)
- `app/prompts/orchestrator_system.md`
- `app/prompts/writer_system.md`
- `app/utils/config.py` + `llm_config.yaml`
- `tests/` (new tests for indexer + summarizer)

## Validation plan
- Dry-run on a multi-city dataset (5-10 cities).
- Log counts: excerpts, clusters, summary sizes, and token budgets per stage.
- Compare orchestrator decisions before/after (should rely on index + summaries).
- Verify writer output stays grounded while remaining complete.

## Testing plan
- Unit tests for:
  - `source_id` generation.
  - clustering logic (topic/subtopic grouping).
  - summarization tree planner (batch sizes, deterministic ordering).
- Integration test that builds context bundle with index + summaries and ensures orchestrator can decide "write" without raw excerpts.
