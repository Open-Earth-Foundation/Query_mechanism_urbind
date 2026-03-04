from __future__ import annotations

import json
import logging
import os
import re
import shlex
import statistics
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from dotenv import dotenv_values

from backend.benchmarks.judge import judge_final_outputs
from backend.benchmarks.models import BenchmarkJudgeEvaluation
from backend.modules.orchestrator.module import run_pipeline
from backend.modules.orchestrator.models import ResearchQuestionRefinement
from backend.utils.config import get_openrouter_api_key, load_config

logger = logging.getLogger(__name__)

_HTTP_STATUS_RE = re.compile(r"->\s*(\d{3})\b")
_HTTP_429_RE = re.compile(r"HTTP/\d(?:\.\d)?\s+429\b", re.IGNORECASE)


@dataclass(frozen=True)
class BenchmarkModeConfig:
    """Benchmark execution mode and its env override files."""

    name: str
    env_files: list[Path]


@dataclass(frozen=True)
class BenchmarkMarkdownConfig:
    """Benchmark markdown batching and concurrency settings."""

    name: str
    batch_max_chunks: int
    max_workers: int


@dataclass(frozen=True)
class BenchmarkQuestionResult:
    """Per-run benchmark measurement for one question and mode."""

    mode: str
    markdown_config: str
    batch_max_chunks: int
    max_workers: int
    repetition: int
    question: str
    run_id: str
    success: bool
    error_type: str
    error_message: str
    runtime_seconds: float
    tokens_per_second: float
    llm_calls: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    llm_issue_total: int
    llm_not_working_count: int
    llm_rate_limit_count: int
    llm_http_error_count: int
    llm_retry_exhausted_count: int
    llm_max_turns_count: int
    llm_bad_output_count: int
    markdown_chunk_count: int
    markdown_excerpt_count: int
    markdown_source_mode: str
    final_output_path: str
    run_summary_path: str
    run_log_path: str


@dataclass(frozen=True)
class BenchmarkReport:
    """Benchmark report payload persisted as JSON and markdown."""

    benchmark_id: str
    generated_at: str
    output_dir: str
    questions_file: str
    questions: list[str]
    selected_cities: list[str]
    docs_dir: str
    repetitions: int
    mode_configs: list[dict[str, object]]
    markdown_configs: list[dict[str, object]]
    results: list[BenchmarkQuestionResult]
    mode_summary: dict[str, dict[str, float]]
    mode_config_summary: dict[str, dict[str, float]]
    judge_results: list[dict[str, object]]
    judge_summary: dict[str, dict[str, float]]
    judge_mode_config_summary: dict[str, dict[str, float]]


def _mode_config_key(mode_name: str, markdown_config_name: str) -> str:
    """Build a stable summary key for (mode, markdown config)."""
    return f"{mode_name} | {markdown_config_name}"


def _load_env_overrides(files: list[Path]) -> dict[str, str]:
    """Load key/value overrides from ordered env files."""
    merged: dict[str, str] = {}
    for env_file in files:
        if not env_file.exists():
            raise FileNotFoundError(f"Benchmark env file not found: {env_file}")
        values = dotenv_values(env_file)
        for key, value in values.items():
            if value is None:
                continue
            merged[key] = value
    return merged


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily apply environment variable overrides."""
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _load_questions(questions_file: Path) -> list[str]:
    """Read benchmark questions from a newline-delimited file."""
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    questions: list[str] = []
    for line in questions_file.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        questions.append(cleaned)
    if not questions:
        raise ValueError(f"No benchmark questions found in {questions_file}")
    return questions


def _load_query_overrides(
    path: Path,
) -> dict[str, ResearchQuestionRefinement]:
    """Load benchmark-stable refinement outputs from a JSON mapping file."""
    if not path.exists():
        raise FileNotFoundError(f"Query overrides file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Query overrides must be a JSON object at {path}")
    overrides: dict[str, ResearchQuestionRefinement] = {}
    for question, value in payload.items():
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(value, dict):
            continue
        canonical = str(value.get("canonical_research_query", "")).strip()
        raw_queries = value.get("retrieval_queries", [])
        retrieval_queries: list[str] = []
        if isinstance(raw_queries, list):
            retrieval_queries = [
                str(item).strip() for item in raw_queries if str(item).strip()
            ]
        if not canonical:
            raise ValueError(f"Missing canonical_research_query for question={question!r}")
        if not retrieval_queries:
            raise ValueError(f"Missing retrieval_queries for question={question!r}")
        overrides[question] = ResearchQuestionRefinement(
            research_question=canonical,
            retrieval_queries=retrieval_queries,
        )
    if not overrides:
        raise ValueError(f"No query overrides loaded from {path}")
    return overrides


def _build_fixed_refiner(
    overrides: dict[str, ResearchQuestionRefinement],
) -> Callable[..., ResearchQuestionRefinement]:
    """Create a refinement function that returns stable, cached queries."""

    def _refine(
        question: str,
        config,  # noqa: ANN001
        api_key: str,  # noqa: ARG001
        selected_cities=None,  # noqa: ANN001, ARG001
        log_llm_payload: bool = True,  # noqa: ARG001, FBT001
    ) -> ResearchQuestionRefinement:
        del config
        key = str(question).strip()
        if key not in overrides:
            raise KeyError(
                "No fixed query overrides found for benchmark question: "
                f"{question!r}"
            )
        refinement = overrides[key]
        return ResearchQuestionRefinement(
            research_question=refinement.research_question,
            retrieval_queries=list(refinement.retrieval_queries),
        )

    return _refine


def _timestamp_slug() -> str:
    """Create a UTC timestamp slug for benchmark runs."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _compute_runtime_seconds(started_at: str, completed_at: str) -> float:
    """Compute wall-clock runtime from ISO timestamps."""
    if not started_at or not completed_at:
        return 0.0
    started_text = str(started_at).strip()
    completed_text = str(completed_at).strip()
    if not started_text or not completed_text:
        return 0.0
    if started_text.casefold() == "none" or completed_text.casefold() == "none":
        return 0.0
    try:
        started = datetime.fromisoformat(started_text)
        completed = datetime.fromisoformat(completed_text)
    except ValueError:
        return 0.0
    return max((completed - started).total_seconds(), 0.0)


def _parse_structured_log_payload(line: str, marker: str) -> dict[str, object] | None:
    """Parse retry payload in JSON or key=value format after a marker."""
    if marker not in line:
        return None
    payload_raw = line.split(marker, 1)[1].strip()
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload

    try:
        tokens = shlex.split(payload_raw)
    except ValueError:
        return None
    parsed: dict[str, object] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key_clean = key.strip()
        if not key_clean:
            continue
        parsed[key_clean] = value.strip()
    return parsed or None


def _has_rate_limit_signal(value: str) -> bool:
    """Return True when text indicates a rate-limit style failure."""
    lowered = value.casefold()
    return "rate limit" in lowered or "ratelimit" in lowered or "429" in lowered


def _collect_llm_issue_counts(run_text_log: Path) -> dict[str, int]:
    """Count benchmark-relevant LLM issue signals from run.log."""
    counts: dict[str, int] = {
        "rate_limit_count": 0,
        "not_working_count": 0,
        "http_error_count": 0,
        "retry_exhausted_count": 0,
        "max_turns_count": 0,
        "bad_output_count": 0,
    }
    if not run_text_log.exists():
        return counts

    fallback_max_turns_hits = 0
    with run_text_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lowered = line.casefold()
            if "openai http error response:" in lowered:
                counts["http_error_count"] += 1
                status_match = _HTTP_STATUS_RE.search(line)
                if status_match and status_match.group(1) == "429":
                    counts["rate_limit_count"] += 1
                else:
                    counts["not_working_count"] += 1
            elif _HTTP_429_RE.search(line):
                counts["rate_limit_count"] += 1

            if "RETRY_EXHAUSTED " in line:
                counts["retry_exhausted_count"] += 1
                payload = _parse_structured_log_payload(line, "RETRY_EXHAUSTED ")
                if payload is None:
                    counts["not_working_count"] += 1
                    continue

                error_type = str(payload.get("error_type", "")).strip()
                error_message = str(payload.get("error_message", "")).strip()
                is_rate_limited = _has_rate_limit_signal(
                    error_type
                ) or _has_rate_limit_signal(error_message)
                if is_rate_limited:
                    counts["rate_limit_count"] += 1
                else:
                    counts["not_working_count"] += 1
                if error_type == "RetryableBadOutput":
                    counts["bad_output_count"] += 1

            if "AGENT_MAX_TURNS_DIAGNOSTICS " in line:
                counts["max_turns_count"] += 1
                counts["not_working_count"] += 1
            elif "hit max turns limit" in lowered:
                fallback_max_turns_hits += 1

    if counts["max_turns_count"] == 0 and fallback_max_turns_hits > 0:
        counts["max_turns_count"] = fallback_max_turns_hits
        counts["not_working_count"] += fallback_max_turns_hits

    return counts


def _load_run_log_json(path: Path) -> dict[str, object]:
    """Load run.json payload when present, otherwise return an empty mapping."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _build_failed_result(
    *,
    mode_name: str,
    markdown_config: BenchmarkMarkdownConfig,
    question: str,
    repetition: int,
    run_id: str,
    run_dir: Path,
    error: Exception,
) -> BenchmarkQuestionResult:
    """Build a benchmark result row for a failed run and retain issue counters."""
    run_log_json_path = run_dir / "run.json"
    run_text_log_path = run_dir / "run.log"
    run_summary_path = run_dir / "run_summary.txt"
    final_output_path = run_dir / "final.md"

    run_log = _load_run_log_json(run_log_json_path)
    usage = run_log.get("llm_usage", {})
    totals = usage.get("totals", {})
    inputs = run_log.get("inputs", {})
    runtime_seconds = _compute_runtime_seconds(
        str(run_log.get("started_at", "")),
        str(run_log.get("completed_at", "")),
    )
    total_tokens = int(totals.get("total_tokens", 0))
    tokens_per_second = (float(total_tokens) / runtime_seconds) if runtime_seconds > 0 else 0.0

    issue_counts = _collect_llm_issue_counts(run_text_log_path)
    exception_text = f"{type(error).__name__}: {error}"
    if sum(issue_counts.values()) == 0:
        if _has_rate_limit_signal(exception_text):
            issue_counts["rate_limit_count"] += 1
        else:
            issue_counts["not_working_count"] += 1
        if re.search(r"\b[45]\d{2}\b", exception_text):
            issue_counts["http_error_count"] += 1

    rate_limit_count = int(issue_counts["rate_limit_count"])
    not_working_count = int(issue_counts["not_working_count"])

    return BenchmarkQuestionResult(
        mode=mode_name,
        markdown_config=markdown_config.name,
        batch_max_chunks=int(markdown_config.batch_max_chunks),
        max_workers=int(markdown_config.max_workers),
        repetition=repetition,
        question=question,
        run_id=str(run_log.get("run_id", run_id)),
        success=False,
        error_type=type(error).__name__,
        error_message=str(error),
        runtime_seconds=runtime_seconds,
        tokens_per_second=tokens_per_second,
        llm_calls=int(usage.get("calls", 0)),
        total_tokens=total_tokens,
        input_tokens=int(totals.get("input_tokens", 0)),
        output_tokens=int(totals.get("output_tokens", 0)),
        llm_issue_total=rate_limit_count + not_working_count,
        llm_not_working_count=not_working_count,
        llm_rate_limit_count=rate_limit_count,
        llm_http_error_count=int(issue_counts["http_error_count"]),
        llm_retry_exhausted_count=int(issue_counts["retry_exhausted_count"]),
        llm_max_turns_count=int(issue_counts["max_turns_count"]),
        llm_bad_output_count=int(issue_counts["bad_output_count"]),
        markdown_chunk_count=int(inputs.get("markdown_chunk_count", 0)),
        markdown_excerpt_count=int(inputs.get("markdown_excerpt_count", 0)),
        markdown_source_mode=str(inputs.get("markdown_source_mode", "unknown")),
        final_output_path=str(final_output_path),
        run_summary_path=str(run_summary_path),
        run_log_path=str(run_log_json_path),
    )


def _run_mode_question(
    mode_name: str,
    markdown_config: BenchmarkMarkdownConfig,
    question: str,
    repetition: int,
    run_index: int,
    run_id: str,
    config_path: Path,
    docs_dir: Path,
    runs_dir: Path,
    selected_cities: list[str],
    log_llm_payload: bool,
    env_overrides: dict[str, str],
    refine_question_func: Callable[..., ResearchQuestionRefinement] | None,
) -> BenchmarkQuestionResult:
    """Run one benchmark question for one mode and collect metrics."""
    with _temporary_env(env_overrides):
        config = load_config(config_path)
        config.enable_sql = False
        config.markdown_dir = docs_dir
        config.runs_dir = runs_dir
        config.markdown_researcher.batch_max_chunks = max(
            int(markdown_config.batch_max_chunks),
            1,
        )
        config.markdown_researcher.max_workers = max(int(markdown_config.max_workers), 1)
        pipeline_kwargs: dict[str, object] = {
            "question": question,
            "config": config,
            "run_id": run_id,
            "log_llm_payload": log_llm_payload,
            "selected_cities": selected_cities,
        }
        if refine_question_func is not None:
            pipeline_kwargs["refine_question_func"] = refine_question_func
        run_paths = run_pipeline(**pipeline_kwargs)  # type: ignore[arg-type]
    run_log = json.loads(run_paths.run_log.read_text(encoding="utf-8"))
    usage = run_log.get("llm_usage", {})
    totals = usage.get("totals", {})
    inputs = run_log.get("inputs", {})
    runtime_seconds = _compute_runtime_seconds(
        str(run_log.get("started_at", "")),
        str(run_log.get("completed_at", "")),
    )
    total_tokens = int(totals.get("total_tokens", 0))
    tokens_per_second = (float(total_tokens) / runtime_seconds) if runtime_seconds > 0 else 0.0

    run_text_log = run_paths.run_log.parent / "run.log"
    issue_counts = _collect_llm_issue_counts(run_text_log)
    rate_limit_count = int(issue_counts["rate_limit_count"])
    not_working_count = int(issue_counts["not_working_count"])

    return BenchmarkQuestionResult(
        mode=mode_name,
        markdown_config=markdown_config.name,
        batch_max_chunks=int(markdown_config.batch_max_chunks),
        max_workers=int(markdown_config.max_workers),
        repetition=repetition,
        question=question,
        run_id=run_log.get("run_id", run_id),
        success=True,
        error_type="",
        error_message="",
        runtime_seconds=runtime_seconds,
        tokens_per_second=tokens_per_second,
        llm_calls=int(usage.get("calls", 0)),
        total_tokens=total_tokens,
        input_tokens=int(totals.get("input_tokens", 0)),
        output_tokens=int(totals.get("output_tokens", 0)),
        llm_issue_total=rate_limit_count + not_working_count,
        llm_not_working_count=not_working_count,
        llm_rate_limit_count=rate_limit_count,
        llm_http_error_count=int(issue_counts["http_error_count"]),
        llm_retry_exhausted_count=int(issue_counts["retry_exhausted_count"]),
        llm_max_turns_count=int(issue_counts["max_turns_count"]),
        llm_bad_output_count=int(issue_counts["bad_output_count"]),
        markdown_chunk_count=int(inputs.get("markdown_chunk_count", 0)),
        markdown_excerpt_count=int(inputs.get("markdown_excerpt_count", 0)),
        markdown_source_mode=str(inputs.get("markdown_source_mode", "unknown")),
        final_output_path=str(run_paths.final_output),
        run_summary_path=str(run_paths.run_summary),
        run_log_path=str(run_paths.run_log),
    )


def _summarize_result_rows(
    mode_results: list[BenchmarkQuestionResult],
) -> dict[str, float]:
    """Aggregate means/medians for a homogeneous group of benchmark runs."""
    successful_runs = [row for row in mode_results if row.success]
    sample_rows = successful_runs if successful_runs else mode_results
    runtimes = [row.runtime_seconds for row in sample_rows]
    tokens = [float(row.total_tokens) for row in sample_rows]
    tokens_per_second = [row.tokens_per_second for row in sample_rows]
    excerpts = [float(row.markdown_excerpt_count) for row in sample_rows]
    chunks = [float(row.markdown_chunk_count) for row in sample_rows]
    llm_issues = [float(row.llm_issue_total) for row in mode_results]
    llm_rate_limits = [float(row.llm_rate_limit_count) for row in mode_results]
    llm_not_working = [float(row.llm_not_working_count) for row in mode_results]
    llm_retry_exhausted = [float(row.llm_retry_exhausted_count) for row in mode_results]
    llm_max_turns = [float(row.llm_max_turns_count) for row in mode_results]
    runs_with_issues = float(sum(1 for row in mode_results if row.llm_issue_total > 0))
    runs_total = len(mode_results)
    runs_succeeded = len(successful_runs)
    runs_failed = runs_total - runs_succeeded

    return {
        "runs": float(runs_total),
        "runs_succeeded": float(runs_succeeded),
        "runs_failed": float(runs_failed),
        "success_rate": (
            float(runs_succeeded) / float(runs_total) if runs_total > 0 else 0.0
        ),
        "runtime_seconds_mean": statistics.fmean(runtimes),
        "runtime_seconds_median": statistics.median(runtimes),
        "total_tokens_mean": statistics.fmean(tokens),
        "total_tokens_median": statistics.median(tokens),
        "tokens_per_second_mean": statistics.fmean(tokens_per_second),
        "tokens_per_second_median": statistics.median(tokens_per_second),
        "markdown_excerpt_count_mean": statistics.fmean(excerpts),
        "markdown_chunk_count_mean": statistics.fmean(chunks),
        "llm_issue_total": float(sum(llm_issues)),
        "llm_issue_mean_per_run": statistics.fmean(llm_issues),
        "llm_issue_run_rate": runs_with_issues / float(len(mode_results)),
        "llm_rate_limit_total": float(sum(llm_rate_limits)),
        "llm_not_working_total": float(sum(llm_not_working)),
        "llm_retry_exhausted_total": float(sum(llm_retry_exhausted)),
        "llm_max_turns_total": float(sum(llm_max_turns)),
    }


def _summarize_mode(results: list[BenchmarkQuestionResult]) -> dict[str, dict[str, float]]:
    """Aggregate benchmark metrics per mode (across all markdown configs)."""
    grouped: dict[str, list[BenchmarkQuestionResult]] = {}
    for result in results:
        grouped.setdefault(result.mode, []).append(result)

    summary: dict[str, dict[str, float]] = {}
    for mode_name, mode_results in grouped.items():
        summary[mode_name] = _summarize_result_rows(mode_results)
    return summary


def _summarize_mode_config(
    results: list[BenchmarkQuestionResult],
) -> dict[str, dict[str, float]]:
    """Aggregate benchmark metrics per (mode, markdown config)."""
    grouped: dict[str, list[BenchmarkQuestionResult]] = {}
    for result in results:
        key = _mode_config_key(result.mode, result.markdown_config)
        grouped.setdefault(key, []).append(result)

    summary: dict[str, dict[str, float]] = {}
    for key, mode_results in grouped.items():
        row = _summarize_result_rows(mode_results)
        row["batch_max_chunks"] = float(mode_results[0].batch_max_chunks)
        row["max_workers"] = float(mode_results[0].max_workers)
        summary[key] = row
    return summary


def _write_markdown_report(path: Path, report: BenchmarkReport) -> None:
    """Write a human-readable benchmark summary."""
    lines: list[str] = [
        "# Retrieval Strategy Benchmark",
        "",
        f"- Benchmark ID: {report.benchmark_id}",
        f"- Generated at: {report.generated_at}",
        f"- Questions file: {report.questions_file}",
        f"- Docs dir: {report.docs_dir}",
        f"- Selected cities: {', '.join(report.selected_cities) if report.selected_cities else '(all)'}",
        f"- Repetitions: {report.repetitions}",
    ]
    lines.extend(["", "## Markdown benchmark configs", ""])
    for config in report.markdown_configs:
        lines.append(
            "- "
            f"{config['name']}: batch_max_chunks={config['batch_max_chunks']} "
            f"max_workers={config['max_workers']}"
        )
    lines.extend(["", "## Questions", ""])
    lines.extend([f"- {question}" for question in report.questions])
    lines.extend(["", "## Mode + config summary", ""])
    for key, metrics in sorted(report.mode_config_summary.items()):
        lines.extend(
            [
                f"### {key}",
                "",
                f"- Batch max chunks: {int(metrics['batch_max_chunks'])}",
                f"- Max workers: {int(metrics['max_workers'])}",
                f"- Runs: {int(metrics['runs'])}",
                f"- Successful runs: {int(metrics['runs_succeeded'])}",
                f"- Failed runs: {int(metrics['runs_failed'])}",
                f"- Success rate: {metrics['success_rate']:.2%}",
                f"- Runtime mean (s): {metrics['runtime_seconds_mean']:.2f}",
                f"- Runtime median (s): {metrics['runtime_seconds_median']:.2f}",
                f"- Tokens/sec mean: {metrics['tokens_per_second_mean']:.2f}",
                f"- Tokens/sec median: {metrics['tokens_per_second_median']:.2f}",
                f"- Total tokens mean: {metrics['total_tokens_mean']:.0f}",
                f"- Chunk count mean: {metrics['markdown_chunk_count_mean']:.1f}",
                f"- Excerpt count mean: {metrics['markdown_excerpt_count_mean']:.1f}",
                f"- LLM issues total: {int(metrics['llm_issue_total'])}",
                f"- LLM issue mean/run: {metrics['llm_issue_mean_per_run']:.2f}",
                f"- LLM issue run-rate: {metrics['llm_issue_run_rate']:.2%}",
                f"- Rate limit issues total: {int(metrics['llm_rate_limit_total'])}",
                f"- Non-working LLM issues total: {int(metrics['llm_not_working_total'])}",
                f"- Retry exhausted total: {int(metrics['llm_retry_exhausted_total'])}",
                f"- Max turns total: {int(metrics['llm_max_turns_total'])}",
                "",
            ]
        )
    lines.extend(["## Mode summary", ""])
    for mode_name, metrics in sorted(report.mode_summary.items()):
        lines.extend(
            [
                f"### {mode_name}",
                "",
                f"- Runs: {int(metrics['runs'])}",
                f"- Successful runs: {int(metrics['runs_succeeded'])}",
                f"- Failed runs: {int(metrics['runs_failed'])}",
                f"- Success rate: {metrics['success_rate']:.2%}",
                f"- Runtime mean (s): {metrics['runtime_seconds_mean']:.2f}",
                f"- Runtime median (s): {metrics['runtime_seconds_median']:.2f}",
                f"- Tokens/sec mean: {metrics['tokens_per_second_mean']:.2f}",
                f"- LLM issues total: {int(metrics['llm_issue_total'])}",
                "",
            ]
        )

    if report.judge_mode_config_summary:
        lines.extend(["## Judge mode + config summary", ""])
        for key, metrics in sorted(report.judge_mode_config_summary.items()):
            lines.extend(
                [
                    f"### {key}",
                    "",
                    f"- Mean judge score: {metrics['mean_judge_score']:.2f}",
                    f"- Median judge score: {metrics['median_judge_score']:.2f}",
                    f"- Wins: {int(metrics['wins'])}",
                    f"- Ties: {int(metrics['ties'])}",
                    "",
                ]
            )

    if report.judge_summary:
        lines.extend(["## Judge mode summary", ""])
        for mode_name, metrics in sorted(report.judge_summary.items()):
            lines.extend(
                [
                    f"### {mode_name}",
                    "",
                    f"- Mean judge score: {metrics['mean_judge_score']:.2f}",
                    f"- Median judge score: {metrics['median_judge_score']:.2f}",
                    f"- Wins: {int(metrics['wins'])}",
                    f"- Ties: {int(metrics['ties'])}",
                    "",
                ]
            )

    lines.extend(["## Run details", ""])
    for result in report.results:
        lines.extend(
            [
                (
                    f"- Mode={result.mode} config={result.markdown_config} "
                    f"batch={result.batch_max_chunks} workers={result.max_workers} "
                    f"rep={result.repetition} run_id={result.run_id} "
                    f"status={'ok' if result.success else 'failed'} "
                    f"runtime={result.runtime_seconds:.2f}s "
                    f"tokens={result.total_tokens} tokens_per_s={result.tokens_per_second:.2f} "
                    f"issues={result.llm_issue_total} rate_limit={result.llm_rate_limit_count} "
                    f"not_working={result.llm_not_working_count}"
                ),
                f"  - Question: {result.question}",
                f"  - Final output: {result.final_output_path}",
                f"  - Run summary: {result.run_summary_path}",
            ]
        )
        if not result.success:
            lines.append(
                f"  - Error: {result.error_type}: {result.error_message}"
            )
    if report.judge_results:
        lines.extend(["", "## Judge run details", ""])
        for row in report.judge_results:
            lines.extend(
                [
                    (
                        f"- Question={row['question']} rep={row['repetition']} "
                        f"config={row['markdown_config']} "
                        f"batch={row['batch_max_chunks']} workers={row['max_workers']} "
                        f"{row['left_label']}={row['left_total_score']} "
                        f"{row['right_label']}={row['right_total_score']} "
                        f"winner={row['winner']} confidence={row['confidence']:.2f}"
                    ),
                    f"  - Comparative rationale: {row['comparative_rationale']}",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_judge_pairs(
    results: list[BenchmarkQuestionResult],
) -> list[tuple[BenchmarkQuestionResult, BenchmarkQuestionResult]]:
    """Pair standard and vector outputs by question+repetition+markdown config."""
    by_key: dict[tuple[str, int, str], dict[str, BenchmarkQuestionResult]] = {}
    for result in results:
        if not result.success:
            continue
        if not Path(result.final_output_path).exists():
            logger.warning(
                "Skipping judge candidate with missing final output file: %s",
                result.final_output_path,
            )
            continue
        key = (result.question, result.repetition, result.markdown_config)
        slot = by_key.setdefault(key, {})
        slot[result.mode] = result
    pairs: list[tuple[BenchmarkQuestionResult, BenchmarkQuestionResult]] = []
    for key in sorted(by_key.keys()):
        slot = by_key[key]
        left = slot.get("standard_chunking")
        right = slot.get("vector_store")
        if left and right:
            pairs.append((left, right))
    return pairs


def _summarize_judge_scores(
    scores_by_group: dict[str, list[float]],
    wins_by_group: dict[str, int],
    ties_by_group: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Aggregate judge scores into mean/median/win/tie stats."""
    summary: dict[str, dict[str, float]] = {}
    for group_name, scores in scores_by_group.items():
        if not scores:
            continue
        summary[group_name] = {
            "mean_judge_score": statistics.fmean(scores),
            "median_judge_score": statistics.median(scores),
            "wins": float(wins_by_group.get(group_name, 0)),
            "ties": float(ties_by_group.get(group_name, 0)),
        }
    return summary


def _run_benchmark_judging(
    *,
    results: list[BenchmarkQuestionResult],
    config_path: Path,
    log_llm_payload: bool,
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """Run pairwise LLM-as-judge comparisons and aggregate score summaries."""
    pairs = _build_judge_pairs(results)
    if not pairs:
        return [], {}, {}

    config = load_config(config_path)
    api_key = get_openrouter_api_key()
    rows: list[dict[str, object]] = []
    scores_by_mode: dict[str, list[float]] = {"standard_chunking": [], "vector_store": []}
    wins_by_mode: dict[str, int] = {"standard_chunking": 0, "vector_store": 0}
    ties_by_mode: dict[str, int] = {"standard_chunking": 0, "vector_store": 0}
    scores_by_mode_config: dict[str, list[float]] = {}
    wins_by_mode_config: dict[str, int] = {}
    ties_by_mode_config: dict[str, int] = {}

    for left, right in pairs:
        try:
            left_text = Path(left.final_output_path).read_text(encoding="utf-8")
            right_text = Path(right.final_output_path).read_text(encoding="utf-8")
            evaluation: BenchmarkJudgeEvaluation = judge_final_outputs(
                question=left.question,
                left_label=left.mode,
                left_text=left_text,
                right_label=right.mode,
                right_text=right_text,
                config=config,
                api_key=api_key,
                log_llm_payload=log_llm_payload,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Judge evaluation failed question=%s repetition=%d config=%s",
                left.question,
                left.repetition,
                left.markdown_config,
            )
            continue
        left_mode_config = _mode_config_key(left.mode, left.markdown_config)
        right_mode_config = _mode_config_key(right.mode, right.markdown_config)
        scores_by_mode[left.mode].append(float(evaluation.left.total_score))
        scores_by_mode[right.mode].append(float(evaluation.right.total_score))
        scores_by_mode_config.setdefault(left_mode_config, []).append(
            float(evaluation.left.total_score)
        )
        scores_by_mode_config.setdefault(right_mode_config, []).append(
            float(evaluation.right.total_score)
        )
        if evaluation.winner == "left":
            wins_by_mode[left.mode] += 1
            wins_by_mode_config[left_mode_config] = (
                wins_by_mode_config.get(left_mode_config, 0) + 1
            )
        elif evaluation.winner == "right":
            wins_by_mode[right.mode] += 1
            wins_by_mode_config[right_mode_config] = (
                wins_by_mode_config.get(right_mode_config, 0) + 1
            )
        else:
            ties_by_mode[left.mode] += 1
            ties_by_mode[right.mode] += 1
            ties_by_mode_config[left_mode_config] = (
                ties_by_mode_config.get(left_mode_config, 0) + 1
            )
            ties_by_mode_config[right_mode_config] = (
                ties_by_mode_config.get(right_mode_config, 0) + 1
            )

        rows.append(
            {
                "question": left.question,
                "repetition": left.repetition,
                "markdown_config": left.markdown_config,
                "batch_max_chunks": left.batch_max_chunks,
                "max_workers": left.max_workers,
                "left_label": evaluation.left_label,
                "right_label": evaluation.right_label,
                "left_total_score": evaluation.left.total_score,
                "right_total_score": evaluation.right.total_score,
                "winner": evaluation.winner,
                "confidence": evaluation.confidence,
                "comparative_rationale": evaluation.comparative_rationale,
                "left_breakdown": evaluation.left.model_dump(),
                "right_breakdown": evaluation.right.model_dump(),
            }
        )

    judge_summary = _summarize_judge_scores(
        scores_by_mode,
        wins_by_mode,
        ties_by_mode,
    )
    judge_mode_config_summary = _summarize_judge_scores(
        scores_by_mode_config,
        wins_by_mode_config,
        ties_by_mode_config,
    )
    return rows, judge_summary, judge_mode_config_summary


def run_retrieval_strategy_benchmark(
    benchmark_id: str,
    output_dir: Path,
    config_path: Path,
    docs_dir: Path,
    questions_file: Path,
    selected_cities: list[str],
    repetitions: int,
    mode_configs: list[BenchmarkModeConfig],
    markdown_configs: list[BenchmarkMarkdownConfig],
    use_query_overrides: bool,
    query_overrides_path: Path | None,
    log_llm_payload: bool,
) -> BenchmarkReport:
    """Execute retrieval strategy benchmark for configured modes and markdown options."""
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if not mode_configs:
        raise ValueError("At least one benchmark mode is required.")
    if not markdown_configs:
        raise ValueError("At least one markdown benchmark config is required.")

    questions = _load_questions(questions_file)
    query_overrides: dict[str, ResearchQuestionRefinement] | None = None
    fixed_refiner: Callable[..., ResearchQuestionRefinement] | None = None
    if use_query_overrides:
        if query_overrides_path is None:
            raise ValueError("query_overrides_path must be set when use_query_overrides=true")
        query_overrides = _load_query_overrides(query_overrides_path)
        missing = [question for question in questions if question not in query_overrides]
        if missing:
            raise ValueError(
                "Query overrides are missing entries for benchmark questions: "
                + "; ".join(missing)
            )
        fixed_refiner = _build_fixed_refiner(query_overrides)

    benchmark_root = output_dir / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkQuestionResult] = []
    for repetition in range(1, repetitions + 1):
        for question_index, question in enumerate(questions, start=1):
            for markdown_config in markdown_configs:
                for mode in mode_configs:
                    env_overrides = _load_env_overrides(mode.env_files)
                    mode_runs_dir = benchmark_root / "runs" / mode.name
                    mode_runs_dir.mkdir(parents=True, exist_ok=True)
                    run_id = (
                        f"{mode.name}_{markdown_config.name}_"
                        f"r{repetition:02d}_q{question_index:02d}_{_timestamp_slug()}"
                    )
                    logger.info(
                        (
                            "Benchmark run mode=%s config=%s batch=%d workers=%d run_id=%s "
                            "repetition=%d question_index=%d"
                        ),
                        mode.name,
                        markdown_config.name,
                        markdown_config.batch_max_chunks,
                        markdown_config.max_workers,
                        run_id,
                        repetition,
                        question_index,
                    )
                    try:
                        result = _run_mode_question(
                            mode_name=mode.name,
                            markdown_config=markdown_config,
                            question=question,
                            repetition=repetition,
                            run_index=question_index,
                            run_id=run_id,
                            config_path=config_path,
                            docs_dir=docs_dir,
                            runs_dir=mode_runs_dir,
                            selected_cities=selected_cities,
                            log_llm_payload=log_llm_payload,
                            env_overrides=env_overrides,
                            refine_question_func=fixed_refiner,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "Benchmark run failed mode=%s config=%s run_id=%s",
                            mode.name,
                            markdown_config.name,
                            run_id,
                        )
                        result = _build_failed_result(
                            mode_name=mode.name,
                            markdown_config=markdown_config,
                            question=question,
                            repetition=repetition,
                            run_id=run_id,
                            run_dir=mode_runs_dir / run_id,
                            error=exc,
                        )
                    results.append(result)

    mode_summary = _summarize_mode(results)
    mode_config_summary = _summarize_mode_config(results)
    judge_results, judge_summary, judge_mode_config_summary = _run_benchmark_judging(
        results=results,
        config_path=config_path,
        log_llm_payload=log_llm_payload,
    )
    report = BenchmarkReport(
        benchmark_id=benchmark_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        output_dir=str(benchmark_root),
        questions_file=str(questions_file),
        questions=questions,
        selected_cities=selected_cities,
        docs_dir=str(docs_dir),
        repetitions=repetitions,
        mode_configs=[
            {"name": mode.name, "env_files": [str(path) for path in mode.env_files]}
            for mode in mode_configs
        ],
        markdown_configs=[
            {
                "name": cfg.name,
                "batch_max_chunks": cfg.batch_max_chunks,
                "max_workers": cfg.max_workers,
            }
            for cfg in markdown_configs
        ],
        results=results,
        mode_summary=mode_summary,
        mode_config_summary=mode_config_summary,
        judge_results=judge_results,
        judge_summary=judge_summary,
        judge_mode_config_summary=judge_mode_config_summary,
    )

    report_json_path = benchmark_root / "benchmark_report.json"
    report_markdown_path = benchmark_root / "benchmark_report.md"
    serializable_report = asdict(report)
    report_json_path.write_text(
        json.dumps(serializable_report, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(report_markdown_path, report)
    logger.info("Benchmark report JSON: %s", report_json_path.as_posix())
    logger.info("Benchmark report markdown: %s", report_markdown_path.as_posix())
    return report


__all__ = [
    "BenchmarkModeConfig",
    "BenchmarkMarkdownConfig",
    "BenchmarkQuestionResult",
    "BenchmarkReport",
    "run_retrieval_strategy_benchmark",
]
