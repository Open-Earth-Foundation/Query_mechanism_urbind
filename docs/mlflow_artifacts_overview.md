# MLflow Artifact Overview

This document describes the run artifacts produced under `output/<run_id>/` and the implemented MLflow upload and tracing policy for the `URBIND` experiment.

## MLflow Artifact Policy

MLflow is optional and disabled by default. When `MLFLOW_ENABLED=true`, the backend uploads every finalized run artifact by default. MLflow mirrors the full `output/<run_id>/` directory so local inspection and MLflow inspection show the same evidence, wrappers, logs, document snapshots, generated outputs, and raw LLM call records.

| Artifact | Upload | Purpose |
| --- | --- | --- |
| `**/*` | Yes | Complete run mirror, including all payloads, wrappers, logs, snapshots, and generated outputs. |
| `stage_files/**` | Yes | Detailed stage payloads for snapshots, retrieval, batching, evidence extraction, enrichment, assumptions, and context handoffs. |
| `stages/*.json` | Yes | Compact stage summaries with inputs, outputs, metrics, and stage-local decisions. |
| `manifest.json` | Yes | Canonical artifact registry and alias map for local and MLflow artifact discovery. |
| `api_state.json` | Yes | Machine-readable status, timestamps, inputs, decisions, metrics, LLM usage summary, retry summary, writer coverage, and API state. |
| `summary.jsonl` | Yes | Append-only event timeline for stage checkpoints and run diagnostics. |
| `run_summary.txt` | Yes | Human-readable run index for quick review. |
| `run.log` | Yes | Full runtime log, including LLM usage, retries, warnings, diagnostics, and tracebacks. |
| `context_bundle.json` | Yes | Final assembled writer/runtime context for the run. |
| `final.md` | Yes | Final user-facing markdown output. |
| `error_log.txt` | Yes, when present | Filtered error and traceback-focused log artifact. |
| `llm_calls/index.jsonl` | Yes, when present | Run-level ordered index of recorded LLM calls. |
| `stage_files/**/llm_calls/*.json` | Yes, when present | Full prompt/request, response/output, provider metadata, token usage, timestamps, status, error, and stage tags for each recorded LLM call. |

Do not exclude artifacts only because they duplicate another artifact or were previously treated as local-only/untracked. Duplication is acceptable because this policy optimizes for complete auditability and reproducibility. If redaction is required, perform it before local artifact creation or through an explicit redacted export mode; the default MLflow mirror does not silently drop files.

Log these values as MLflow tags or metrics in addition to uploading the wrapper files:

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

Runtime controls:

- `MLFLOW_ENABLED=false`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME=URBIND`
- `MLFLOW_ARTIFACT_PATH=run_artifacts`
- `MLFLOW_TRACE_MODE=consolidated`
- `MLFLOW_FAIL_ON_ERROR=false`

MLflow failures are best-effort warnings by default. Set `MLFLOW_FAIL_ON_ERROR=true` only when pipeline success should depend on MLflow upload/tracing success.

Sync hardening:

- Before MLflow calls run, the backend switches process console streams to UTF-8 when Python supports stream reconfiguration. This prevents Windows console encoding errors from MLflow status output from failing an otherwise successful sync.
- MLflow sync is idempotent for a completed local run directory. If `api_state.json["mlflow"]` already records a completed sync with an MLflow run id, a later sync returns that metadata instead of opening a new MLflow run.
- If a previous sync recorded an MLflow run id or trace ids but failed before artifact upload completed, the retry reuses the stored run id and trace metadata, then uploads artifacts into the same MLflow run instead of creating duplicate runs or traces.
- Assumption review/apply API artifacts created after pipeline finalization force a re-sync into the existing MLflow run for the same local `run_id`, so post-run `stage_files/101_assumptions_discovery/**`, `stage_files/102_assumptions_apply/**`, and raw LLM call records are mirrored as well.
  When the original run already has trace metadata, post-run assumption calls are recorded in a supplemental `post_run_assumptions` trace instead of recreating the original pipeline trace.

## MLflow Run And Trace Policy

Create one MLflow run for each pipeline `run_id`. All uploaded artifacts, metrics, tags, and traces for that pipeline execution belong under that single MLflow run.

Preferred trace layout:

- Record one consolidated MLflow trace for the whole pipeline execution.
- Include markdown researcher calls and assumptions calls in that same trace.
- Use child spans, tags, or attributes so each LLM/tool call is easy to distinguish.
- Tag each call with at least `run_id`, `stage_number`, `stage_name`, `stage_family`, `agent`, `call_kind`, `provider`, `model`, and token counts when available.
- Use `stage_family=markdown` and `agent=markdown_researcher` for markdown extraction calls.
- Use `stage_family=assumptions` and agent values such as `assumptions_estimator`, `assumptions_reviewer`, or `assumptions_apply` for assumptions calls.
- Preserve call order so the MLflow trace reads in the same sequence as the pipeline run.

Fallback trace layout when one consolidated trace cannot represent both flows cleanly:

- Keep one MLflow run for the pipeline `run_id`.
- Record a markdown-agent trace first with `trace_family=markdown` and `agent=markdown_researcher`.
- Record one assumptions trace for all assumptions-related calls with `trace_family=assumptions`.
- Tag both traces with the same `run_id` and `trace_group=<run_id>` so they can be found together from the MLflow run.
- Upload the full raw LLM call artifacts for both flows even when traces are split.

MLflow sync metadata is persisted back into local artifacts:

- `api_state.json["mlflow"]`
- `manifest.json["metadata"]["mlflow"]`

The metadata includes MLflow enabled state, sync status, MLflow run id when available, experiment name, artifact path, trace ids when available, fallback status,
supplemental post-run trace ids when applicable, the highest post-run LLM call index already traced, and any best-effort sync error payload.

## Raw LLM Call Artifacts

The run-local LLM recorder writes one JSON file per recorded provider call plus one run-level index. Recording is enabled automatically for MLflow-enabled runs and does not depend on verbose `run.log` payload logging.

Per-call files live under the owning stage:

- markdown extraction: `stage_files/006_markdown_extraction/llm_calls/`
- enrichment calls: `stage_files/008_enrichment/llm_calls/`
- assumptions estimation: `stage_files/010_assumptions/llm_calls/`
- writer calls: `stage_files/014_writer/llm_calls/`
- assumptions review/apply API flows: `stage_files/101_assumptions_discovery/llm_calls/` and `stage_files/102_assumptions_apply/llm_calls/`

Each call record includes:

- `run_id`
- `call_index`
- `stage_number`
- `stage_name`
- `stage_family`
- `agent`
- `call_kind`
- `provider`
- `model`
- `started_at` and `ended_at`
- `status`
- `metadata`
- `request`
- `response`
- `error`

The run-level `llm_calls/index.jsonl` stores the same ordering and points to each per-call JSON file.

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

This is uploaded by default because it preserves a quick failure-focused view alongside the full `run.log`.

## Additional Artifact Details

These files are uploaded by default even when they duplicate data in more detailed artifacts. Their most useful fields should also be mirrored into MLflow tags or metrics.

| Path | Contents |
| --- | --- |
| `manifest.json` | Canonical local artifact registry. Contains generated file list, alias map, run id, summary event path, and final metadata. |
| `api_state.json` | Machine-readable API state. Contains run status, timestamps, finish reason, structured error payload, inputs, decisions, LLM usage summary, retry summary, writer coverage, and writer multi-pass state. |
| `summary.jsonl` | Append-only stage timeline. Each line has event index, event type, run id, timestamp, stage number, and compact stage payload. |
| `stages/NNN_<stage>.json` | Stage detail summaries with structured `inputs`, `outputs`, `metrics`, and stage-local decisions. |
| `run_summary.txt` | Human-readable run overview: question, query mode, selected cities, markdown counts, status, runtime, LLM usage, retry summary, artifact list, decisions, and markdown failure summary. |
| `run.log` | Full runtime log, including `LLM_USAGE`, retry events, provider diagnostics, warnings, and full exception traces. |
| `progress.json` | Live progress state for API polling and frontend display. Includes step labels plus canonical stage names and stage numbers. |
| `chat/<conversation_id>.json` | Context-chat session history when follow-up chat is used. |
| `chat_jobs/<conversation_id>/<job_id>.json` | Split-mode chat job state when long follow-up chat is processed asynchronously. |
| `chat_cache/evidence_chunks.json` | Lazy compact evidence cache for overflowed chat prompts. |

## Duplication And Trace Organization

The MLflow policy intentionally accepts duplication so each run is complete and inspectable from MLflow alone:

- `stage_files/**` keeps the detailed payloads, and `stages/*.json` keeps compact stage summaries.
- `final.md` keeps the user-facing answer, and `run_summary.txt` keeps the quick human-readable index.
- MLflow tags/metrics keep status, timing, token, retry, and error summaries, and `api_state.json` remains uploaded as the machine-readable API view.
- `manifest.json` remains uploaded as the artifact alias source of truth.
- `error_log.txt` keeps the useful failure slice, and `run.log` remains uploaded for full diagnostics.
- Raw LLM call artifacts remain uploaded for exact prompt/response replay and audit.
- MLflow traces provide the navigable execution view, with markdown and assumptions calls distinguished by tags even when they share one consolidated trace.
