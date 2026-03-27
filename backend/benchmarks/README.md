# Benchmark Configuration

This folder contains benchmark-only setup, separate from normal runtime settings.

## Files

- `prompts/retrieval_questions.txt`: benchmark questions.
- `prompts/retrieval_query_overrides.json`: fixed canonical + retrieval queries per question (optional, recommended for stable chunk counts).
- `config/base.env`: shared settings applied to both benchmark modes.
- `config/mode_standard.env`: overrides for standard chunking runs.
- `config/mode_vector.env`: overrides for vector-store runs.

## Override order

The benchmark runner loads env files in this order for each mode:

1. `config/base.env`
2. mode-specific env (`config/mode_standard.env` or `config/mode_vector.env`)

If a key appears in both, the mode-specific value wins.

## Notes

- Vector benchmark mode uses the existing default Chroma store/collection unless
  overridden in the main environment.
- Vector-store retrieval/embedding tuning is read from `llm_config.yaml`
  (`vector_store.*`), not benchmark env files.
- Markdown researcher benchmark sizing/concurrency is configured from benchmark CLI
  options (`--markdown-option <batch_max_chunks>:<max_workers>`).
- Default benchmark markdown options are:
  - `16:8`
  - `32:4`
  - `32:8`
- Benchmark runs do not build/update vector index; they measure runtime behavior
  with the currently available index.
- To reduce run-to-run variance in retrieval behavior, the benchmark script can use
  fixed canonical + retrieval queries from `prompts/retrieval_query_overrides.json`.
- Benchmark includes LLM-as-judge scoring (OpenRouter `openai/gpt-5.4-mini`) for each
  standard-vs-vector pair on the same question/repetition/markdown option.
- Benchmark report includes speed metrics (runtime + tokens/sec) and LLM issue
  counters (rate limits, retry exhausted, max-turns, and non-working calls).
- Failed runs are kept in the report with error details and issue counters, and
  benchmark execution continues with remaining runs.
- For ad-hoc comparison of two files, use:
  `python -m backend.scripts.judge_final_outputs --left-final <path_a> --right-final <path_b> --question "..."`

## Gold recall benchmark

Use `python -m backend.scripts.benchmark_recall --gold-file tests/fixtures/benchmark_gold.json`
to measure information loss across retrieval, markdown extraction, and final writing.

- Stage A strict metrics (`retrieval_recall`, `retrieval_precision`, `mrr`) use
  direct hits from `retrieval.json.seed_chunks[]`.
- Stage A supplemental delivery metrics (`delivery_recall`,
  `delivery_precision`) use the final delivered context in
  `retrieval.json.chunks[]`.
- Stage B uses `excerpts[].source_chunk_ids` for extraction recall and an LLM
  fact judge for fact extraction rate.
- Stage C uses an LLM fact judge on `final.md` plus citation coverage derived
  from cited `ref_id` values mapped through `references.json`.

Gold fixtures live in `tests/fixtures/benchmark_gold.json` and use the versioned
schema `{"version": 1, "cases": [...]}` with `case_id`, `question`,
`gold_chunk_ids`, `gold_facts`, `gold_city`, and optional `selected_cities` plus
`cached_run_dir`.

Cached runs must contain `markdown/retrieval.json`, `markdown/excerpts.json`,
`markdown/references.json`, and `final.md`, and the cached question must match
the gold question exactly. Legacy runs without `seed_chunks[]` are rejected
because they cannot support strict Stage A metrics.

The fact judge is separate from the pairwise benchmark judge and defaults to
OpenRouter `openai/gpt-5.4-mini`.

Outputs are written under `output/benchmarks/recall/<benchmark_id>/`:

- `benchmark_report.json`
- `benchmark_report.md`
- `runs/<case_id>/...` for any case executed live
