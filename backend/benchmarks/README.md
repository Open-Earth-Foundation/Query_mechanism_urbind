# Benchmark Configuration

This folder contains benchmark-only setup, separate from normal runtime settings.

## Files

- `prompts/retrieval_questions.txt`: benchmark questions.
- `prompts/retrieval_query_overrides.json`: fixed optional retrieval queries per question (optional, recommended for stable chunk counts). The legacy `canonical_research_query` field must mirror the benchmark question; the benchmark uses the question itself as query 1.
- `config/base.env`: shared settings applied to both benchmark modes.
- `config/mode_standard.env`: overrides for standard chunking runs.
- `config/mode_vector.env`: overrides for vector-store runs.
- `retrieval_evidence_diversity_analysis.md`: retrieval-only diagnosis of current benchmark misses and prioritized fixes for query diversity, fusion, and chunk representation.

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
  fixed optional retrieval queries from `prompts/retrieval_query_overrides.json`.
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
`gold_chunk_ids`, `gold_facts`, `gold_city`, and optional `selected_cities`,
`gold_chunk_texts`, and `gold_chunk_alternatives`.

- `gold_chunk_texts` should hold the canonical chunk text for each gold slot.
  The scorer still supports containment fallback, but the fixture should keep
  the actual chunk text in JSON.
- `gold_chunk_alternatives` lets one gold chunk slot accept specific
  alternative runtime chunks without changing the benchmark denominator, while
  storing both `chunk_id` and `chunk_text` in the fixture JSON.

Every benchmark case executes the live pipeline and is then scored from the
freshly produced `stage_files/003_retrieval/retrieval.json`,
`stage_files/006_markdown_extraction/excerpts.json`,
`stage_files/006_markdown_extraction/references.json`, and `final.md` artifacts.

Per-case `benchmark_report.json` chunk diagnostics keep the canonical gold
`chunk_id` and, when different, the `matched_chunk_id` that actually satisfied
that benchmark slot.

The fact judge is separate from the pairwise benchmark judge and defaults to
OpenRouter `openai/gpt-5.4-mini`.

Outputs are written under `output/benchmarks/recall/<benchmark_id>/`:

- `benchmark_report.json`
- `benchmark_report.md`
- `runs/<case_id>/...` for each benchmark case

## Writer numeric benchmark

Use `python -m backend.scripts.benchmark_writer_numbers` to compare final writer
numbers against a frozen manual baseline for Krakow, the Poland group, and the
optional all-cities corpus snapshot.

- The fixture lives at
  `backend/benchmarks/writer_numeric/writer_numeric_benchmark.json` and uses the
  versioned schema `{"version": 1, "default_mode": "...", "cases": [...]}`.
- Every case freezes `selected_cities` explicitly, including the all-cities
  case. Dynamic placeholders such as `all_cities` or group tokens are rejected
  by the loader.
- Cases can be marked with `requires_explicit_include=true`. Those cases are
  skipped unless `--include-optional-cases` is passed.
- The frozen all-cities case is marked optional because it can take a long time
  and consume a large number of LLM tokens.
- Every `baseline_metrics[]` entry stores the benchmarked `metric_id`, label,
  unit, `expected_value`, and manual `components[]` used to justify the
  baseline.
- Default execution mode is `ccc_only`. Use `--mode full_pipeline` to force the
  enrichment stack on, or `--mode both` to run each case in both modes.
- Use `--include-optional-cases` when you intentionally want to run the
  expensive all-cities case.
- The numeric extractor is configured separately from the fact judge under
  `benchmark_number_extractor` in `llm_config.yaml`. It receives metric ids,
  labels, and units, while baseline expected values are used only by the
  deterministic comparison step after extraction.

Outputs are written under `output/benchmarks/writer_numeric/<benchmark_id>/`:

- `benchmark_summary.json`: full persisted report payload with case-level
  comparisons.
- `benchmark_report.md`: human-readable diff report that shows baseline value,
  extracted value, status, and writer snippet per metric.
- Some cases now opt into row-level audits for the writer's city/count table.
  Those reports include per-city match, mismatch, missing, and extra rows for
  the configured combined-total metric.
- The optional all-cities bus case now also emits a heuristic retrieval audit
  that compares selected source documents with numeric bus-count language
  against the set of cities that actually surfaced in accepted excerpts.
- `runs/<case_id>__<mode>/final.md`: live writer output for each run.
- `runs/<case_id>__<mode>/context_bundle.json`: live writer context bundle.
- `runs/<case_id>__<mode>/extracted_numbers.json`: structured extractor output.
