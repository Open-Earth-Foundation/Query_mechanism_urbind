# Implementation Plan: Structured Extractions + Recursive Summarization

## Goal

Prevent large multi-city extractions from overwhelming the orchestrator and writer by combining:

1. Structured metadata and indexing (topic clusters + source mapping), and
2. Parallel recursive summarization (tree summaries with controlled token budgets).

## Combined approach (high level):

- Step 1: Keep per-chunk extractions but require structured fields (topic, source id).
- Step 2: Build a topic list that collects all topics from excerpts.
- Step 3: If excerpts exceed a size threshold, run recursive tree summarization per topic (parallel), then roll up to a master summary.
- Step 4: Feed the orchestrator the topic list + summaries (not raw excerpts) so it can validate topic-by-topic.
- Step 5: Feed the writer the master summary + topic summaries with source mapping.

## Data contract changes

### New/updated markdown extraction fields

Add structured metadata to each excerpt. Keep values deterministic where possible.

Fields to add to `MarkdownExcerpt`:

- `source_id`: string, deterministic id. Example: `path|chunk_index`.
- `source_path`: string, document path.
- `chunk_index`: int.
- `chunk_count`: int.
- `topic`: string (canonical label, e.g., "transport", "buildings", "energy").

Notes:

- Populate `source_id`, `source_path`, `chunk_index`, `chunk_count`, and `city_name` in Python from the document metadata.
- Let the LLM fill `topic`, `snippet`, `answer`, and `relevant`.

### New index/summaries payload

Add a new `markdown_index` + `markdown_summaries` payload to the context bundle.

Example shape:

```
{
  "markdown_index": {
    "topics": [
      "transport",
      "buildings",
      "energy",
      "waste",
      "water",
      "air_quality",
      "green_space"
    ],
    "coverage": {
      "cities": 5,
      "topics": 7,
      "excerpts": 84
    }
  },
  "markdown_summaries": {
    "topic_summaries": [
      {
        "topic": "transport",
        "summary": "...",
        "source_ids": ["..."],
        "city_names": ["..."]
      }
    ],
    "master_summary": {
      "summary": "...",
      "source_ids": ["..."],
      "city_names": ["..."],
      "topics": ["..."]
    }
  }
}
```

## Pipeline changes

1. Extraction (markdown researcher)

   - Update prompt + models to output structured metadata fields.
   - Inject deterministic metadata in Python after the LLM returns excerpts.

2. Indexing / topic collection

   - Add a small indexer module that collects all unique topics from excerpts.
   - Build a topic list and track which cities contribute to each topic.
   - Persist index JSON and update context bundle.

3. Recursive summarization (parallel)

   - Add a summarizer agent that accepts a list of excerpts and outputs a short summary + source_ids.
   - Implement tree summarization per topic:
     - Batch excerpts within a token budget.
     - Summarize each batch in parallel.
     - Merge batch summaries into topic summaries.
     - Merge topic summaries into a master summary.
   - Persist summaries JSON and update context bundle.

4. Orchestrator + writer integration
   - Orchestrator sees `markdown_index` + `markdown_summaries` instead of raw excerpts.
   - Writer receives the same, focusing on master + top clusters.

## Orchestrator behavior changes

- Use `markdown_index.topics` to check available topics.
- Validate topic-by-topic: if a required topic is missing, request more markdown.
- If token budget is constrained, prefer index + summaries and avoid raw excerpts.
- Decision rules update: for multi-city questions, require topic coverage across cities before "write".

## Writer behavior changes

- Use `master_summary` as the backbone.
- Expand using topic summaries as needed.
- When detailing per-city differences, reference topic summaries filtered by city.
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
  - topic collection logic.
  - summarization tree planner (batch sizes, deterministic ordering).
- Integration test that builds context bundle with index + summaries and ensures orchestrator can decide "write" without raw excerpts.
