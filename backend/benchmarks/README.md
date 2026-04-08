# Benchmark Configuration

This folder contains benchmark-only prompts and documentation for the markdown-only
runtime.

## Files

- `prompts/retrieval_questions.txt`: benchmark questions.
- `prompts/retrieval_query_overrides.json`: optional fixed canonical and retrieval
  queries per benchmark question for stable reruns.

## Markdown chunking benchmark

Use `python -m backend.scripts.run_retrieval_benchmark` to compare
batching/concurrency settings for the standard markdown chunking pipeline.

- Benchmark sizing is controlled with `--markdown-option <batch_max_chunks>:<max_workers>`.
- Default markdown options are `16:8`, `32:4`, and `32:8`.
- Query overrides can pin the canonical and retrieval queries so repeated runs use
  the same prompt inputs.
- Reports include runtime, token throughput, markdown chunk counts, excerpt counts,
  and LLM issue counters.
- Failed runs stay in the report with their error details so the remaining matrix can
  continue.

Outputs are written under `output/benchmarks/<benchmark_id>/`:

- `benchmark_report.json`
- `benchmark_report.md`
- `runs/standard_chunking/<run_id>/...`

For ad-hoc comparison of any two final documents, use
`python -m backend.scripts.judge_final_outputs --left-final <path_a> --right-final <path_b> --question "..."`

## Gold recall benchmark

Use `python -m backend.scripts.benchmark_recall --gold-file tests/fixtures/benchmark_gold.json`
to measure information loss across delivered markdown chunks, markdown extraction,
and final writing.

- Stage A uses delivered chunks reconstructed from `markdown/batches.json` plus the
  live markdown files. Metrics are `delivery_recall`, `delivery_precision`, and `mrr`.
- Stage B uses `excerpts[].source_chunk_ids` for extraction recall and an LLM fact
  judge for fact extraction rate.
- Stage C uses an LLM fact judge on `final.md` plus citation coverage derived from
  cited `ref_id` values mapped through `references.json`.

Gold fixtures live in `tests/fixtures/benchmark_gold.json` and use the versioned
schema `{"version": 1, "cases": [...]}` with `case_id`, `question`,
`gold_chunk_ids`, `gold_facts`, `gold_city`, and optional `selected_cities`,
`gold_chunk_texts`, and `gold_chunk_alternatives`.

- `gold_chunk_texts` should hold the canonical chunk text for each gold slot.
- `gold_chunk_alternatives` lets one gold chunk slot accept specific alternative
  runtime chunks without changing the benchmark denominator while still storing both
  `chunk_id` and `chunk_text`.

Every benchmark case executes the live pipeline and is then scored from the freshly
produced `markdown/batches.json`, `markdown/excerpts.json`,
`markdown/references.json`, and `final.md` artifacts.

Outputs are written under `output/benchmarks/recall/<benchmark_id>/`:

- `benchmark_report.json`
- `benchmark_report.md`
- `runs/<case_id>/...`
