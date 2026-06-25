# MLflow Artifact Overview

This document describes the run artifacts produced under `output/<run_id>/` and the proposed subset that should be uploaded to MLflow for the `URBIND` experiment.

## Proposed MLflow Artifact Policy

Upload these artifacts by default:

| Artifact | Upload | Purpose |
| --- | --- | --- |
| `stage_files/**` | Yes | Detailed stage payloads for snapshots, retrieval, batching, evidence extraction, enrichment, assumptions, and context handoffs. |
| `context_bundle.json` | Yes | Final assembled writer/runtime context for the run. |
| `final.md` | Yes | Final user-facing markdown output. |
| `error_log.txt` | Yes, when present | Filtered error and traceback-focused log artifact. |

Do not upload these artifacts by default:

| Artifact | Upload | Why Not By Default |
| --- | --- | --- |
| `manifest.json` | No | Mostly repeats artifact paths and aliases that are useful locally but less useful once MLflow stores files directly. |
| `api_state.json` | No | Repeats status, timestamps, metrics, and error fields that should be logged as MLflow tags or metrics. |
| `summary.jsonl` | No | Repeats compact stage timeline information that is already visible through `stage_files/**` plus MLflow run metadata. |
| `run_summary.txt` | No | Human-readable wrapper around data that should be captured by uploaded artifacts and MLflow tags/metrics. |
| `stages/*.json` | No | Stage summary files duplicate the detailed payloads in `stage_files/**`. |
| `run.log` | No | Full logs can contain large payloads and verbose request details. Use `error_log.txt` for the default error view. |
| Raw LLM request/response payloads | No | These can be large and may include sensitive or user-provided content. |

Log these values as MLflow tags or metrics instead of uploading the wrapper files:

- `run_id`
- `status`
- `finish_reason`
- `environment`
- `query_mode`
- `analysis_mode`
- selected city count
- markdown chunk count
- markdown excerpt count
- final output character count
- LLM call count
- input, output, and total token counts
- retry count
- exhausted retry count
- run duration
- error code and error type when available

## Uploaded Artifact Details

### `stage_files/**`

`stage_files/` contains larger, stage-owned payloads. These are the main artifacts to inspect when debugging what the pipeline actually saw or produced.

| Path | Contents |
| --- | --- |
| `stage_files/001_input_snapshot/execution_snapshot.json` | CLI/runtime invocation details: argv, working directory, config path, requested run id, resolved run id, and rerunnable invocation command when available. |
| `stage_files/001_input_snapshot/code_snapshot.json` | Git snapshot: repo root, commit, branch, dirty flag, and changed files. |
| `stage_files/001_input_snapshot/config_snapshot.json` | Resolved application config and config file hash. |
| `stage_files/001_input_snapshot/vector_store_snapshot.json` | Vector-store settings, Chroma paths, collection name, manifest hash, manifest summary, and auto-update diagnostics when an update runs. |
| `stage_files/001_input_snapshot/documents_snapshot.json` | Markdown corpus snapshot: document directory, file count, selected-city file summary, source-library count, file manifest, and snapshot hash. |
| `stage_files/002_query_preparation/research_question.json` | Original question, query mode, canonical research query, retrieval query list, and explicit retrieval query slots. |
| `stage_files/003_retrieval/retrieval.json` | Written when vector retrieval is enabled. Contains retrieval queries, optional city filter, retrieval tuning metadata, seed chunks, final delivered chunks, distance/provenance details, and retrieval totals. |
| `stage_files/005_markdown_batching/batches.json` | Markdown researcher batching plan: city batches, estimated token counts, chunk ordering, source paths, chunk indexes, and chunk ids. |
| `stage_files/005_markdown_batching/source_chunk_index.json` | Chunk id to source metadata lookup used by source/reference views. |
| `stage_files/006_markdown_extraction/accepted_excerpts.json` | Accepted evidence excerpts. Each excerpt includes citation id, quote, city fields, partial answer, and source chunk ids. |
| `stage_files/006_markdown_extraction/rejected_chunks.json` | Rejected chunk ids, rejected-per-city grouping, extraction status, and counts. |
| `stage_files/006_markdown_extraction/decision_audit.json` | Decision reconciliation: retrieved, accepted, rejected, unresolved, invariant status, unknown ids, missing ids, overlap ids, and batch failures. |
| `stage_files/006_markdown_extraction/city_summary.json` | City-level extraction summary: per-city batch counts, decision counts, excerpt counts, status/error rollups, and city lists with excerpts, without excerpts, or with failures. |
| `stage_files/007_markdown_context_handoff/context_bundle_after_markdown.json` | Immutable full context snapshot after markdown extraction and before enrichment/writer work. |
| `stage_files/008_enrichment/enrichment_bundle.json` | Canonical enrichment payload when enrichment runs: field manifest, gap manifest, enriched fields, evidence, and enrichment metadata. |
| `stage_files/008_enrichment/external_source_search_audit.json` | External-source search trace when external source search runs: searched fields, candidates, validated/rejected claims, no-evidence records, resolutions, tool calls, and metrics. |
| `stage_files/008_enrichment/web_research_audit.json` | Web research trace when non-bundle trace outputs exist: search batches, findings, benchmark traces, and other web-search diagnostics. |
| `stage_files/009_enrichment_context_handoff/context_bundle_after_enrichment.json` | Immutable full context snapshot after enrichment completes. |
| `stage_files/010_assumptions/assumptions_bundle.json` | Assumptions payload: model estimates, non-estimable records, saturation warnings, and assumptions metadata. |
| `stage_files/010_assumptions/assumptions_stage.json` | Assumptions stage support artifact with flags, outputs, and metrics. |
| `stage_files/011_assumptions_context_handoff/context_bundle_after_assumptions.json` | Immutable full context snapshot after assumptions are merged. |
| `stage_files/012_writer_multi_pass/...` | Writer multi-pass planning and intermediate payloads when the writer splits oversized context into batches. |
| `stage_files/013_writer_citation_coverage/...` | Writer citation coverage diagnostics when coverage checks are recorded. |
| `stage_files/101_assumptions_discovery/...` | Assumption-discovery artifacts created by API assumption review flows when artifact persistence is enabled. |
| `stage_files/102_assumptions_apply/...` | Assumption-apply artifacts such as edited assumptions, revised context, and regenerated output when artifact persistence is enabled. |

### `context_bundle.json`

Final assembled context shared across the pipeline and later API features. It includes:

- original question
- research question
- query mode
- retrieval queries
- selected and inspected city fields
- analysis mode
- markdown evidence bundle
- enrichment payload when enrichment runs
- assumptions payload when assumptions run
- final output path

This is uploaded because it is the easiest single file for understanding what the writer and follow-up flows had available.

### `final.md`

Final delivered markdown answer. It usually contains:

- `# Question`
- the original user question
- generated answer body from the writer
- inline citation references such as `ref_1`, depending on the answer

This is uploaded because it is the actual user-facing result.

### `error_log.txt`

Filtered error view extracted from `run.log`. It contains:

- `ERROR` log entries
- `CRITICAL` log entries
- `RETRY_EXHAUSTED` entries
- continuation lines after selected log records, including tracebacks
- a no-error placeholder when no matching error entries exist

This is uploaded by default because it preserves useful failure context without uploading the full verbose `run.log`.

## Local-Only Artifact Details

These files stay in the local run directory by default. Their most useful fields should be mirrored into MLflow tags or metrics.

| Path | Contents |
| --- | --- |
| `manifest.json` | Canonical local artifact registry. Contains generated file list, alias map, run id, summary event path, and final metadata. |
| `api_state.json` | Machine-readable API state. Contains run status, timestamps, finish reason, structured error payload, inputs, decisions, LLM usage summary, retry summary, writer coverage, and writer multi-pass state. |
| `summary.jsonl` | Append-only stage timeline. Each line has event index, event type, run id, timestamp, stage number, and compact stage payload. |
| `stages/NNN_<stage>.json` | Stage detail summaries with structured `inputs`, `outputs`, `metrics`, and stage-local decisions. |
| `run_summary.txt` | Human-readable run overview: question, query mode, selected cities, markdown counts, status, runtime, LLM usage, retry summary, artifact list, decisions, and markdown failure summary. |
| `run.log` | Full runtime log, including `LLM_USAGE`, retry events, provider diagnostics, warnings, and full exception traces. Not uploaded by default because it is verbose and can include sensitive or large payload context. |
| `progress.json` | Live progress state for API polling and frontend display. Includes step labels plus canonical stage names and stage numbers. |
| `chat/<conversation_id>.json` | Context-chat session history when follow-up chat is used. |
| `chat_jobs/<conversation_id>/<job_id>.json` | Split-mode chat job state when long follow-up chat is processed asynchronously. |
| `chat_cache/evidence_chunks.json` | Lazy compact evidence cache for overflowed chat prompts. |

## Where Run Errors Live

Run errors are stored in several places locally:

| Location | Error Shape |
| --- | --- |
| `error_log.txt` | Filtered error, critical, exhausted retry, and traceback lines. This is the default MLflow error artifact. |
| `api_state.json` | Structured terminal error fields: `status`, `finish_reason`, and `error`. These are not uploaded as a file by default, but should become MLflow tags. |
| `run.log` | Full raw log context. Not uploaded by default. |
| `manifest.json` | Registers `error_log` when an error log artifact is produced. Not uploaded by default. |
| Relevant `stage_files/**` entries | Stage-specific structured failure payloads when the failing stage wrote them. |

## What This Avoids Duplicating

The proposed MLflow artifact subset intentionally avoids uploading wrapper files that repeat the same content in different forms:

- `stage_files/**` keeps the detailed payloads, so `stages/*.json` summaries are not uploaded.
- `final.md` keeps the user-facing answer, so `run_summary.txt` does not need to repeat it.
- MLflow tags/metrics keep status, timing, token, retry, and error summaries, so `api_state.json` is not uploaded by default.
- MLflow's file list plus the known policy makes `manifest.json` less necessary in MLflow.
- `error_log.txt` keeps the useful failure slice, so full `run.log` stays local by default.
