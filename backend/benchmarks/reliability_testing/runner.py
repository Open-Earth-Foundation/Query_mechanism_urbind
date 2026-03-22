from __future__ import annotations

import inspect
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.benchmarks.runner import _collect_llm_issue_counts
from backend.models import ErrorInfo
from backend.modules.markdown_researcher.agent import extract_markdown_excerpts
from backend.modules.markdown_researcher.models import (
    MarkdownBatchFailure,
    MarkdownResearchResult,
)
from backend.modules.markdown_researcher.services import (
    build_city_batches,
    resolve_batch_input_token_limit,
    split_documents_by_city,
)
from backend.modules.orchestrator.utils import (
    attach_run_file_logger,
    build_markdown_artifacts,
)
from backend.modules.orchestrator.utils.error_handlers import detach_run_file_logger
from backend.modules.vector_store.models import RetrievedChunk
from backend.modules.vector_store.retriever import (
    as_markdown_documents,
    retrieve_chunks_for_queries,
)
from backend.services.error_log_artifact import write_error_log_artifact
from backend.utils.config import AppConfig, get_openrouter_api_key, load_config
from backend.utils.json_io import write_json
from backend.utils.tokenization import count_tokens

from backend.benchmarks.reliability_testing.models import (
    BenchmarkRunStatus,
    PayloadCaptureMode,
    ReliabilityBenchmarkMatrix,
    ReliabilityBenchmarkProgress,
    ReliabilityBenchmarkReport,
    ReliabilityMarkdownDefaults,
    ReliabilityModelConfig,
    ReliabilityModelResult,
)

logger = logging.getLogger(__name__)

DEFAULT_MATRIX_CONFIG_PATH = Path(
    "backend/benchmarks/reliability_testing/config/markdown_model_matrix.yml"
)
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _timestamp_slug() -> str:
    """Create a UTC timestamp slug for benchmark ids."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return unique non-empty strings while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = str(value).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _slugify_model_id(model_id: str) -> str:
    """Create a stable filesystem-safe slug from an OpenRouter model id."""
    slug = _SLUG_RE.sub("_", model_id.strip().lower()).strip("._-")
    return slug or "model"


def load_reliability_matrix(path: Path) -> ReliabilityBenchmarkMatrix:
    """Load and validate a reliability benchmark matrix YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Reliability matrix config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ReliabilityBenchmarkMatrix.model_validate(raw)


def select_benchmark_models(
    matrix: ReliabilityBenchmarkMatrix,
    selected_models: list[str] | None = None,
) -> list[ReliabilityModelConfig]:
    """Return the models that should run for this benchmark invocation."""
    if selected_models:
        model_by_key = {model.id.casefold(): model for model in matrix.models}
        resolved: list[ReliabilityModelConfig] = []
        seen: set[str] = set()
        missing: list[str] = []
        for raw_model in selected_models:
            key = str(raw_model).strip().casefold()
            if not key:
                continue
            model = model_by_key.get(key)
            if model is None:
                missing.append(str(raw_model))
                continue
            if key in seen:
                continue
            seen.add(key)
            resolved.append(model)
        if missing:
            raise ValueError(
                "Unknown model ids requested via --model: " + ", ".join(sorted(missing))
            )
        if not resolved:
            raise ValueError("No models matched the provided --model filters.")
        return resolved

    enabled_models = [model for model in matrix.models if model.enabled]
    if not enabled_models:
        raise ValueError("Reliability matrix has no enabled models to run.")
    return enabled_models


def _apply_markdown_defaults(
    config: AppConfig,
    defaults: ReliabilityMarkdownDefaults,
) -> AppConfig:
    """Return a deep-copied config with benchmark-wide markdown defaults applied."""
    resolved = config.model_copy(deep=True)
    resolved.enable_sql = False
    resolved.markdown_researcher.max_turns = defaults.max_turns
    resolved.markdown_researcher.batch_max_chunks = defaults.batch_max_chunks
    resolved.markdown_researcher.max_workers = defaults.max_workers
    resolved.markdown_researcher.reasoning_effort = defaults.reasoning_effort
    return resolved


def _apply_model_overrides(
    config: AppConfig,
    model_config: ReliabilityModelConfig,
) -> AppConfig:
    """Return one config copy with the model id and explicit overrides applied."""
    resolved = config.model_copy(deep=True)
    resolved.markdown_researcher.model = model_config.id
    if "reasoning_effort" in model_config.model_fields_set:
        resolved.markdown_researcher.reasoning_effort = model_config.reasoning_effort
    return resolved


def _serialize_retrieval_payload(
    *,
    question: str,
    retrieval_queries: list[str],
    selected_cities: list[str],
    chunks: list[RetrievedChunk],
    meta: dict[str, Any],
) -> dict[str, object]:
    """Build the persisted retrieval corpus artifact."""
    return {
        "question": question,
        "queries": list(retrieval_queries),
        "selected_cities": list(selected_cities),
        "retrieved_count": len(chunks),
        "meta": meta,
        "chunks": [asdict(chunk) for chunk in chunks],
    }


def _serialize_batch_plan(
    *,
    question: str,
    retrieval_queries: list[str],
    batch_token_limit: int,
    batch_max_chunks: int,
    tasks: list[tuple[str, int, list[dict[str, object]]]],
) -> dict[str, object]:
    """Build the persisted benchmark batch-plan artifact."""
    serialized_batches: list[dict[str, object]] = []
    for city_name, batch_index, batch in tasks:
        chunk_ids = _dedupe_preserve_order(
            [str(document.get("chunk_id", "")).strip() for document in batch]
        )
        input_tokens = sum(count_tokens(str(document.get("content", ""))) for document in batch)
        serialized_batches.append(
            {
                "city_name": city_name,
                "batch_index": batch_index,
                "chunk_count": len(batch),
                "input_tokens": input_tokens,
                "chunk_ids": chunk_ids,
            }
        )
    return {
        "question": question,
        "queries": list(retrieval_queries),
        "batch_count": len(tasks),
        "batch_input_token_limit": batch_token_limit,
        "batch_max_chunks": batch_max_chunks,
        "batches": serialized_batches,
    }


def _build_failed_result(
    tasks: list[tuple[str, int, list[dict[str, object]]]],
    exc: Exception,
) -> MarkdownResearchResult:
    """Create a synthetic result when the markdown stage aborts entirely."""
    unresolved_chunk_ids: list[str] = []
    batch_failures: list[MarkdownBatchFailure] = []
    for city_name, batch_index, batch in tasks:
        batch_chunk_ids = _dedupe_preserve_order(
            [str(document.get("chunk_id", "")).strip() for document in batch]
        )
        unresolved_chunk_ids.extend(batch_chunk_ids)
        batch_failures.append(
            MarkdownBatchFailure(
                city_name=city_name,
                batch_index=batch_index,
                reason=type(exc).__name__,
                unresolved_chunk_ids=batch_chunk_ids,
            )
        )
    return MarkdownResearchResult(
        status="error",
        excerpts=[],
        unresolved_chunk_ids=_dedupe_preserve_order(unresolved_chunk_ids),
        batch_failures=batch_failures,
        error=ErrorInfo(
            code="MARKDOWN_BENCHMARK_EXCEPTION",
            message=f"{type(exc).__name__}: {exc}",
        ),
    )


def _invoke_markdown_func(
    markdown_func: Callable[..., MarkdownResearchResult],
    *,
    question: str,
    documents: list[dict[str, object]],
    config: AppConfig,
    api_key: str,
    run_id: str,
    payload_capture_mode: PayloadCaptureMode,
    payload_recorder: Callable[[dict[str, object]], None] | None,
) -> MarkdownResearchResult:
    """Invoke the markdown function with only the supported keyword arguments."""
    signature = inspect.signature(markdown_func)
    kwargs: dict[str, object] = {
        "question": question,
        "documents": documents,
        "config": config,
        "api_key": api_key,
    }
    if "log_llm_payload" in signature.parameters:
        kwargs["log_llm_payload"] = False
    if "run_id" in signature.parameters:
        kwargs["run_id"] = run_id
    if "batch_payload_mode" in signature.parameters:
        kwargs["batch_payload_mode"] = payload_capture_mode
    if "batch_payload_recorder" in signature.parameters:
        kwargs["batch_payload_recorder"] = payload_recorder
    return markdown_func(**kwargs)


def _count_log_marker(run_log_path: Path, marker: str) -> int:
    """Count lines containing the given marker in a run.log file."""
    if not run_log_path.exists():
        return 0
    count = 0
    with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker in line:
                count += 1
    return count


def _extract_token_value(usage: Mapping[str, object], keys: list[str]) -> int:
    """Return the first numeric token value found for the provided key aliases."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _parse_llm_usage_lines(run_log_path: Path) -> tuple[int, int, int]:
    """Parse run.log and return (calls, input_tokens, output_tokens)."""
    if not run_log_path.exists():
        return 0, 0, 0
    call_count = 0
    total_input = 0
    total_output = 0
    with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "LLM_USAGE " not in line:
                continue
            payload = line.split("LLM_USAGE ", 1)[1].strip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = data.get("usage")
            if not isinstance(usage, Mapping):
                continue
            call_count += 1
            total_input += _extract_token_value(usage, ["input_tokens", "prompt_tokens"])
            total_output += _extract_token_value(
                usage, ["output_tokens", "completion_tokens"]
            )
    return call_count, total_input, total_output


def _summarize_failure_reasons(
    batch_failures: list[MarkdownBatchFailure],
) -> dict[str, int]:
    """Count failed batches by structured failure reason."""
    return dict(Counter(failure.reason for failure in batch_failures))


def _summarize_benchmark_results(
    results: list[ReliabilityModelResult],
    retrieved_count: int,
) -> dict[str, object]:
    """Build the benchmark-level rollup section."""
    if not results:
        return {
            "model_count": 0,
            "models_with_failed_batches": 0,
            "models_failed_entirely": 0,
        }

    def _failed_batch_rate(result: ReliabilityModelResult) -> float:
        if result.total_batches <= 0:
            return 0.0
        return float(result.failed_batches) / float(result.total_batches)

    def _unresolved_rate(result: ReliabilityModelResult) -> float:
        if retrieved_count <= 0:
            return 0.0
        return float(result.unresolved_total) / float(retrieved_count)

    def _pick_best(key: Callable[[ReliabilityModelResult], float]) -> dict[str, object]:
        chosen = min(results, key=lambda item: (key(item), item.model_id.casefold()))
        return {"model_id": chosen.model_id, "rate": key(chosen)}

    def _pick_worst(key: Callable[[ReliabilityModelResult], float]) -> dict[str, object]:
        chosen = max(results, key=lambda item: (key(item), item.model_id.casefold()))
        return {"model_id": chosen.model_id, "rate": key(chosen)}

    return {
        "model_count": len(results),
        "models_with_failed_batches": sum(1 for result in results if result.failed_batches > 0),
        "models_failed_entirely": sum(1 for result in results if result.failed_entirely),
        "best_failed_batch_rate_model": _pick_best(_failed_batch_rate),
        "worst_failed_batch_rate_model": _pick_worst(_failed_batch_rate),
        "best_unresolved_rate_model": _pick_best(_unresolved_rate),
        "worst_unresolved_rate_model": _pick_worst(_unresolved_rate),
    }


def _render_benchmark_report(report: ReliabilityBenchmarkReport) -> str:
    """Render a concise markdown report for the reliability benchmark."""
    lines = [
        "# Markdown Reliability Benchmark",
        "",
        f"- Benchmark ID: {report.benchmark_id}",
        f"- Status: {report.status}",
        f"- Started at: {report.started_at}",
        f"- Generated at: {report.generated_at}",
        f"- Output dir: `{report.output_dir}`",
        f"- Retrieved chunks: {report.retrieved_count}",
        f"- Batch count: {report.batch_count}",
        f"- Payload capture mode: `{report.payload_capture_mode}`",
        "",
        "## Queries",
        "",
        f"- Question: {report.question}",
    ]
    for index, query in enumerate(report.retrieval_queries, start=1):
        lines.append(f"- Retrieval query {index}: {query}")

    lines.extend(
        [
            "",
            "## Model Summary",
            "",
            "| Model | Failed/Total Batches | Unresolved | Retries | Retry Exhausted | Max Turns | Bad Output | Failed Entirely | Runtime (s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in report.results:
        lines.append(
            "| {model} | {failed}/{total} | {unresolved} | {retries} | {retry_exhausted} | {max_turns} | {bad_output} | {failed_entirely} | {runtime:.2f} |".format(
                model=result.model_id,
                failed=result.failed_batches,
                total=result.total_batches,
                unresolved=result.unresolved_total,
                retries=result.retry_event_count,
                retry_exhausted=result.retry_exhausted_count,
                max_turns=result.max_turns_count,
                bad_output=result.bad_output_count,
                failed_entirely="yes" if result.failed_entirely else "no",
                runtime=result.runtime_seconds,
            )
        )

    lines.extend(["", "## Failure Reasons", ""])
    for result in report.results:
        if result.failure_reasons:
            rendered = ", ".join(
                f"{reason}={count}" for reason, count in sorted(result.failure_reasons.items())
            )
        else:
            rendered = "none"
        lines.append(f"- {result.model_id}: {rendered}")

    return "\n".join(lines) + "\n"


def _build_benchmark_report(
    *,
    benchmark_id: str,
    status: BenchmarkRunStatus,
    started_at: str,
    output_dir: Path,
    config_path: Path,
    matrix_config_path: Path,
    matrix: ReliabilityBenchmarkMatrix,
    selected_cities: list[str],
    retrieved_count: int,
    batch_count: int,
    results: list[ReliabilityModelResult],
) -> ReliabilityBenchmarkReport:
    """Build one benchmark report snapshot for the current execution state."""
    return ReliabilityBenchmarkReport(
        benchmark_id=benchmark_id,
        status=status,
        started_at=started_at,
        generated_at=datetime.now(timezone.utc).isoformat(),
        output_dir=str(output_dir),
        config_path=str(config_path),
        matrix_config_path=str(matrix_config_path),
        question=matrix.question,
        retrieval_queries=list(matrix.retrieval_queries),
        selected_cities=selected_cities,
        payload_capture_mode=matrix.payload_capture_mode,
        retrieved_count=retrieved_count,
        batch_count=batch_count,
        results=results,
        summary=_summarize_benchmark_results(results, retrieved_count),
    )


def _write_report_snapshot(
    benchmark_root: Path,
    report: ReliabilityBenchmarkReport,
) -> None:
    """Persist the current benchmark report JSON and markdown snapshot."""
    write_json(benchmark_root / "benchmark_report.json", report.model_dump())
    (benchmark_root / "benchmark_report.md").write_text(
        _render_benchmark_report(report),
        encoding="utf-8",
    )


def _write_progress_snapshot(
    benchmark_root: Path,
    progress: ReliabilityBenchmarkProgress,
) -> None:
    """Persist the live benchmark progress artifact."""
    write_json(benchmark_root / "progress.json", progress.model_dump())


def run_markdown_reliability_benchmark(
    *,
    benchmark_id: str | None = None,
    output_dir: Path = Path("output/reliability_testing"),
    config_path: Path = Path("llm_config.yaml"),
    matrix_config_path: Path = DEFAULT_MATRIX_CONFIG_PATH,
    selected_models: list[str] | None = None,
    selected_cities: list[str] | None = None,
    api_key_override: str | None = None,
    retrieve_func: Callable[..., tuple[list[RetrievedChunk], dict[str, Any]]] = retrieve_chunks_for_queries,
    markdown_func: Callable[..., MarkdownResearchResult] = extract_markdown_excerpts,
) -> ReliabilityBenchmarkReport:
    """Run a markdown-only reliability benchmark with one frozen retrieval corpus."""
    matrix = load_reliability_matrix(matrix_config_path)
    models_to_run = select_benchmark_models(matrix, selected_models)

    config = _apply_markdown_defaults(load_config(config_path), matrix.markdown_defaults)
    if not config.vector_store.enabled:
        raise ValueError(
            "Markdown reliability benchmark requires vector_store.enabled=true in the config."
        )

    resolved_benchmark_id = benchmark_id or _timestamp_slug()
    benchmark_root = output_dir / resolved_benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    api_key = (
        api_key_override.strip()
        if isinstance(api_key_override, str) and api_key_override.strip()
        else get_openrouter_api_key()
    )
    resolved_selected_cities = (
        list(selected_cities)
        if selected_cities is not None
        else list(matrix.selected_cities)
    )
    results: list[ReliabilityModelResult] = []
    retrieval_chunks: list[RetrievedChunk] = []
    retrieval_meta: dict[str, Any] = {}
    batch_plan: list[tuple[str, int, list[dict[str, object]]]] = []
    batch_token_limit = 0
    progress = ReliabilityBenchmarkProgress(
        benchmark_id=resolved_benchmark_id,
        status="running",
        output_dir=str(benchmark_root),
        started_at=started_at,
        updated_at=started_at,
        total_models=len(models_to_run),
    )
    _write_progress_snapshot(benchmark_root, progress)
    _write_report_snapshot(
        benchmark_root,
        _build_benchmark_report(
            benchmark_id=resolved_benchmark_id,
            status="running",
            started_at=started_at,
            output_dir=benchmark_root,
            config_path=config_path,
            matrix_config_path=matrix_config_path,
            matrix=matrix,
            selected_cities=resolved_selected_cities,
            retrieved_count=0,
            batch_count=0,
            results=results,
        ),
    )

    try:
        retrieval_chunks, retrieval_meta = retrieve_func(
            queries=list(matrix.retrieval_queries),
            config=config,
            docs_dir=config.markdown_dir,
            selected_cities=resolved_selected_cities,
            run_id=f"{resolved_benchmark_id}:retrieval",
        )
        if not retrieval_chunks:
            raise ValueError("Reliability benchmark retrieval returned no chunks.")

        markdown_documents = as_markdown_documents(retrieval_chunks)
        documents_by_city = split_documents_by_city(markdown_documents)
        batch_token_limit = resolve_batch_input_token_limit(config)
        batch_plan = build_city_batches(
            documents_by_city=documents_by_city,
            max_batch_input_tokens=batch_token_limit,
            max_batch_chunks=max(config.markdown_researcher.batch_max_chunks, 1),
        )
        if not batch_plan:
            raise ValueError("Reliability benchmark produced no markdown batches.")

        write_json(
            benchmark_root / "retrieval.json",
            _serialize_retrieval_payload(
                question=matrix.question,
                retrieval_queries=matrix.retrieval_queries,
                selected_cities=resolved_selected_cities,
                chunks=retrieval_chunks,
                meta=retrieval_meta,
            ),
        )
        write_json(
            benchmark_root / "batches.json",
            _serialize_batch_plan(
                question=matrix.question,
                retrieval_queries=matrix.retrieval_queries,
                batch_token_limit=batch_token_limit,
                batch_max_chunks=max(config.markdown_researcher.batch_max_chunks, 1),
                tasks=batch_plan,
            ),
        )
        progress.retrieval_written = True
        progress.retrieved_count = len(retrieval_chunks)
        progress.batch_count = len(batch_plan)
        progress.updated_at = datetime.now(timezone.utc).isoformat()
        _write_progress_snapshot(benchmark_root, progress)
        _write_report_snapshot(
            benchmark_root,
            _build_benchmark_report(
                benchmark_id=resolved_benchmark_id,
                status="running",
                started_at=started_at,
                output_dir=benchmark_root,
                config_path=config_path,
                matrix_config_path=matrix_config_path,
                matrix=matrix,
                selected_cities=resolved_selected_cities,
                retrieved_count=len(retrieval_chunks),
                batch_count=len(batch_plan),
                results=results,
            ),
        )

        for model_config in models_to_run:
            model_slug = _slugify_model_id(model_config.id)
            model_dir = benchmark_root / model_slug
            markdown_dir = model_dir / "markdown"
            markdown_dir.mkdir(parents=True, exist_ok=True)

            progress.current_model_id = model_config.id
            progress.updated_at = datetime.now(timezone.utc).isoformat()
            _write_progress_snapshot(benchmark_root, progress)

            run_log_handler = attach_run_file_logger(model_dir)
            payload_records: list[dict[str, object]] = []
            model_started_at = time.perf_counter()
            markdown_result: MarkdownResearchResult
            try:
                logger.info(
                    "Starting reliability benchmark model=%s batches=%d",
                    model_config.id,
                    len(batch_plan),
                )
                model_run_config = _apply_model_overrides(config, model_config)
                markdown_result = _invoke_markdown_func(
                    markdown_func,
                    question=matrix.question,
                    documents=[dict(document) for document in markdown_documents],
                    config=model_run_config,
                    api_key=api_key,
                    run_id=f"{resolved_benchmark_id}:{model_slug}",
                    payload_capture_mode=matrix.payload_capture_mode,
                    payload_recorder=(
                        payload_records.append
                        if matrix.payload_capture_mode != "off"
                        else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Markdown reliability benchmark failed for model=%s", model_config.id
                )
                markdown_result = _build_failed_result(batch_plan, exc)
            finally:
                runtime_seconds = max(time.perf_counter() - model_started_at, 0.0)
                detach_run_file_logger(run_log_handler)

            artifacts = build_markdown_artifacts(
                markdown_chunks=markdown_documents,
                markdown_result=markdown_result,
                run_id=f"{resolved_benchmark_id}:{model_slug}",
                markdown_source_mode="vector_store_retrieval",
                analysis_mode="aggregate",
                retrieval_queries=list(matrix.retrieval_queries),
                selected_cities=resolved_selected_cities,
            )
            write_json(markdown_dir / "accepted_excerpts.json", artifacts.accepted_payload)
            write_json(markdown_dir / "rejected_excerpts.json", artifacts.rejected_payload)
            write_json(markdown_dir / "decision_audit.json", artifacts.decision_audit_payload)
            write_json(markdown_dir / "references.json", artifacts.references_payload)
            write_json(markdown_dir / "excerpts.json", artifacts.excerpts_payload)

            failed_payloads_path: Path | None = None
            if matrix.payload_capture_mode != "off":
                failed_payloads_path = markdown_dir / "failed_batch_payloads.json"
                write_json(failed_payloads_path, payload_records)

            run_log_path = model_dir / "run.log"
            error_log_path = model_dir / "error_log.txt"
            write_error_log_artifact(run_log_path, error_log_path)

            issue_counts = _collect_llm_issue_counts(run_log_path)
            retry_event_count = _count_log_marker(run_log_path, "RETRY_EVENT ")
            llm_calls, input_tokens, output_tokens = _parse_llm_usage_lines(run_log_path)
            total_tokens = input_tokens + output_tokens
            decision_audit = artifacts.decision_audit_payload
            batch_failures = list(markdown_result.batch_failures)
            failed_batches = len(batch_failures)
            total_batches = len(batch_plan)
            successful_batches = max(total_batches - failed_batches, 0)

            model_result = ReliabilityModelResult(
                model_id=model_config.id,
                model_slug=model_slug,
                total_batches=total_batches,
                successful_batches=successful_batches,
                failed_batches=failed_batches,
                cities_with_failed_batches=sorted(
                    {failure.city_name for failure in batch_failures}
                ),
                retry_event_count=retry_event_count,
                retry_exhausted_count=issue_counts["retry_exhausted_count"],
                max_turns_count=issue_counts["max_turns_count"],
                bad_output_count=issue_counts["bad_output_count"],
                failure_reasons=_summarize_failure_reasons(batch_failures),
                accepted_total=int(decision_audit.get("accepted_total", 0)),
                rejected_total=int(decision_audit.get("rejected_total", 0)),
                unresolved_total=int(decision_audit.get("unresolved_total", 0)),
                excerpt_count=int(artifacts.excerpts_payload.get("excerpt_count", 0)),
                runtime_seconds=runtime_seconds,
                llm_calls=llm_calls,
                total_tokens=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                failed_entirely=successful_batches == 0,
                error_code=markdown_result.error.code if markdown_result.error else "",
                error_message=markdown_result.error.message if markdown_result.error else "",
                run_dir=str(model_dir),
                run_log_path=str(run_log_path),
                error_log_path=str(error_log_path),
                failed_batch_payloads_path=(
                    str(failed_payloads_path) if failed_payloads_path is not None else None
                ),
            )
            write_json(model_dir / "model_result.json", model_result.model_dump())
            results.append(model_result)

            progress.current_model_id = None
            progress.last_completed_model_id = model_config.id
            progress.completed_model_count = len(results)
            progress.completed_model_ids = [result.model_id for result in results]
            progress.results_written = len(results)
            progress.updated_at = datetime.now(timezone.utc).isoformat()
            _write_progress_snapshot(benchmark_root, progress)
            _write_report_snapshot(
                benchmark_root,
                _build_benchmark_report(
                    benchmark_id=resolved_benchmark_id,
                    status="running",
                    started_at=started_at,
                    output_dir=benchmark_root,
                    config_path=config_path,
                    matrix_config_path=matrix_config_path,
                    matrix=matrix,
                    selected_cities=resolved_selected_cities,
                    retrieved_count=len(retrieval_chunks),
                    batch_count=len(batch_plan),
                    results=results,
                ),
            )
    except Exception as exc:
        progress.status = "failed"
        progress.current_model_id = None
        progress.error_type = type(exc).__name__
        progress.error_message = str(exc)
        progress.updated_at = datetime.now(timezone.utc).isoformat()
        _write_progress_snapshot(benchmark_root, progress)
        _write_report_snapshot(
            benchmark_root,
            _build_benchmark_report(
                benchmark_id=resolved_benchmark_id,
                status="failed",
                started_at=started_at,
                output_dir=benchmark_root,
                config_path=config_path,
                matrix_config_path=matrix_config_path,
                matrix=matrix,
                selected_cities=resolved_selected_cities,
                retrieved_count=len(retrieval_chunks),
                batch_count=len(batch_plan),
                results=results,
            ),
        )
        raise

    progress.status = "completed"
    progress.current_model_id = None
    progress.updated_at = datetime.now(timezone.utc).isoformat()
    _write_progress_snapshot(benchmark_root, progress)
    report = _build_benchmark_report(
        benchmark_id=resolved_benchmark_id,
        status="completed",
        started_at=started_at,
        output_dir=benchmark_root,
        config_path=config_path,
        matrix_config_path=matrix_config_path,
        matrix=matrix,
        selected_cities=resolved_selected_cities,
        retrieved_count=len(retrieval_chunks),
        batch_count=len(batch_plan),
        results=results,
    )
    _write_report_snapshot(benchmark_root, report)
    return report


__all__ = [
    "DEFAULT_MATRIX_CONFIG_PATH",
    "load_reliability_matrix",
    "run_markdown_reliability_benchmark",
    "select_benchmark_models",
]
