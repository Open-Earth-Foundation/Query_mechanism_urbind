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
- Benchmark includes LLM-as-judge scoring (OpenRouter `openai/gpt-5.2`) for each
  standard-vs-vector pair on the same question/repetition/markdown option.
- Benchmark report includes speed metrics (runtime + tokens/sec) and LLM issue
  counters (rate limits, retry exhausted, max-turns, and non-working calls).
- For ad-hoc comparison of two files, use:
  `python -m backend.scripts.judge_final_outputs --left-final <path_a> --right-final <path_b> --question "..."`
