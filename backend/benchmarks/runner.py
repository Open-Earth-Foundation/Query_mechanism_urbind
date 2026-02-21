from __future__ import annotations

import json
import logging
import os
import statistics
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import dotenv_values

from backend.benchmarks.judge import judge_final_outputs
from backend.benchmarks.models import BenchmarkJudgeEvaluation
from backend.modules.orchestrator.module import run_pipeline
from backend.utils.config import get_openrouter_api_key, load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkModeConfig:
    """Benchmark execution mode and its env override files."""

    name: str
    env_files: list[Path]


@dataclass(frozen=True)
class BenchmarkQuestionResult:
    """Per-run benchmark measurement for one question and mode."""

    mode: str
    repetition: int
    question: str
    run_id: str
    runtime_seconds: float
    llm_calls: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
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
    results: list[BenchmarkQuestionResult]
    mode_summary: dict[str, dict[str, float]]
    judge_results: list[dict[str, object]]
    judge_summary: dict[str, dict[str, float]]


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


def _timestamp_slug() -> str:
    """Create a UTC timestamp slug for benchmark runs."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _compute_runtime_seconds(started_at: str, completed_at: str) -> float:
    """Compute wall-clock runtime from ISO timestamps."""
    if not started_at or not completed_at:
        return 0.0
    started = datetime.fromisoformat(started_at)
    completed = datetime.fromisoformat(completed_at)
    return max((completed - started).total_seconds(), 0.0)


def _run_mode_question(
    mode_name: str,
    question: str,
    repetition: int,
    run_index: int,
    config_path: Path,
    docs_dir: Path,
    runs_dir: Path,
    selected_cities: list[str],
    log_llm_payload: bool,
    env_overrides: dict[str, str],
) -> BenchmarkQuestionResult:
    """Run one benchmark question for one mode and collect metrics."""
    with _temporary_env(env_overrides):
        config = load_config(config_path)
        config.enable_sql = False
        config.markdown_dir = docs_dir
        config.runs_dir = runs_dir
        run_id = f"{mode_name}_r{repetition:02d}_q{run_index:02d}_{_timestamp_slug()}"
        run_paths = run_pipeline(
            question=question,
            config=config,
            run_id=run_id,
            log_llm_payload=log_llm_payload,
            selected_cities=selected_cities,
        )
    run_log = json.loads(run_paths.run_log.read_text(encoding="utf-8"))
    usage = run_log.get("llm_usage", {})
    totals = usage.get("totals", {})
    inputs = run_log.get("inputs", {})
    return BenchmarkQuestionResult(
        mode=mode_name,
        repetition=repetition,
        question=question,
        run_id=run_log.get("run_id", run_id),
        runtime_seconds=_compute_runtime_seconds(
            str(run_log.get("started_at", "")),
            str(run_log.get("completed_at", "")),
        ),
        llm_calls=int(usage.get("calls", 0)),
        total_tokens=int(totals.get("total_tokens", 0)),
        input_tokens=int(totals.get("input_tokens", 0)),
        output_tokens=int(totals.get("output_tokens", 0)),
        markdown_chunk_count=int(inputs.get("markdown_chunk_count", 0)),
        markdown_excerpt_count=int(inputs.get("markdown_excerpt_count", 0)),
        markdown_source_mode=str(inputs.get("markdown_source_mode", "unknown")),
        final_output_path=str(run_paths.final_output),
        run_summary_path=str(run_paths.run_summary),
        run_log_path=str(run_paths.run_log),
    )


def _summarize_mode(results: list[BenchmarkQuestionResult]) -> dict[str, dict[str, float]]:
    """Aggregate per-mode medians and means for key metrics."""
    grouped: dict[str, list[BenchmarkQuestionResult]] = {}
    for result in results:
        grouped.setdefault(result.mode, []).append(result)
    summary: dict[str, dict[str, float]] = {}
    for mode_name, mode_results in grouped.items():
        runtimes = [row.runtime_seconds for row in mode_results]
        tokens = [float(row.total_tokens) for row in mode_results]
        excerpts = [float(row.markdown_excerpt_count) for row in mode_results]
        chunks = [float(row.markdown_chunk_count) for row in mode_results]
        summary[mode_name] = {
            "runs": float(len(mode_results)),
            "runtime_seconds_mean": statistics.fmean(runtimes),
            "runtime_seconds_median": statistics.median(runtimes),
            "total_tokens_mean": statistics.fmean(tokens),
            "total_tokens_median": statistics.median(tokens),
            "markdown_excerpt_count_mean": statistics.fmean(excerpts),
            "markdown_chunk_count_mean": statistics.fmean(chunks),
        }
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
    lines.extend(["", "## Questions", ""])
    lines.extend([f"- {question}" for question in report.questions])
    lines.extend(["", "## Mode summary", ""])
    for mode_name, metrics in report.mode_summary.items():
        lines.extend(
            [
                f"### {mode_name}",
                "",
                f"- Runs: {int(metrics['runs'])}",
                f"- Runtime mean (s): {metrics['runtime_seconds_mean']:.2f}",
                f"- Runtime median (s): {metrics['runtime_seconds_median']:.2f}",
                f"- Total tokens mean: {metrics['total_tokens_mean']:.0f}",
                f"- Total tokens median: {metrics['total_tokens_median']:.0f}",
                f"- Chunk count mean: {metrics['markdown_chunk_count_mean']:.1f}",
                f"- Excerpt count mean: {metrics['markdown_excerpt_count_mean']:.1f}",
                "",
            ]
        )
    if report.judge_summary:
        lines.extend(["## Judge summary", ""])
        for mode_name, metrics in report.judge_summary.items():
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
                    f"- Mode={result.mode} rep={result.repetition} run_id={result.run_id} "
                    f"runtime={result.runtime_seconds:.2f}s tokens={result.total_tokens} "
                    f"chunks={result.markdown_chunk_count} excerpts={result.markdown_excerpt_count}"
                ),
                f"  - Question: {result.question}",
                f"  - Final output: {result.final_output_path}",
                f"  - Run summary: {result.run_summary_path}",
            ]
        )
    if report.judge_results:
        lines.extend(["", "## Judge run details", ""])
        for row in report.judge_results:
            lines.extend(
                [
                    (
                        f"- Question={row['question']} rep={row['repetition']} "
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
    """Pair standard and vector outputs by question+repetition."""
    by_key: dict[tuple[str, int], dict[str, BenchmarkQuestionResult]] = {}
    for result in results:
        key = (result.question, result.repetition)
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


def _run_benchmark_judging(
    *,
    results: list[BenchmarkQuestionResult],
    config_path: Path,
    log_llm_payload: bool,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    """Run pairwise LLM-as-judge comparisons and aggregate by mode."""
    pairs = _build_judge_pairs(results)
    if not pairs:
        return [], {}

    config = load_config(config_path)
    api_key = get_openrouter_api_key()
    rows: list[dict[str, object]] = []
    scores_by_mode: dict[str, list[float]] = {"standard_chunking": [], "vector_store": []}
    wins_by_mode: dict[str, int] = {"standard_chunking": 0, "vector_store": 0}
    ties_by_mode: dict[str, int] = {"standard_chunking": 0, "vector_store": 0}

    for left, right in pairs:
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
        scores_by_mode[left.mode].append(float(evaluation.left.total_score))
        scores_by_mode[right.mode].append(float(evaluation.right.total_score))
        if evaluation.winner == "left":
            wins_by_mode[left.mode] += 1
        elif evaluation.winner == "right":
            wins_by_mode[right.mode] += 1
        else:
            ties_by_mode[left.mode] += 1
            ties_by_mode[right.mode] += 1

        rows.append(
            {
                "question": left.question,
                "repetition": left.repetition,
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

    summary: dict[str, dict[str, float]] = {}
    for mode_name, mode_scores in scores_by_mode.items():
        if not mode_scores:
            continue
        summary[mode_name] = {
            "mean_judge_score": statistics.fmean(mode_scores),
            "median_judge_score": statistics.median(mode_scores),
            "wins": float(wins_by_mode[mode_name]),
            "ties": float(ties_by_mode[mode_name]),
        }
    return rows, summary


def run_retrieval_strategy_benchmark(
    benchmark_id: str,
    output_dir: Path,
    config_path: Path,
    docs_dir: Path,
    questions_file: Path,
    selected_cities: list[str],
    repetitions: int,
    mode_configs: list[BenchmarkModeConfig],
    log_llm_payload: bool,
) -> BenchmarkReport:
    """Execute retrieval strategy benchmark for standard and vector modes."""
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if not mode_configs:
        raise ValueError("At least one benchmark mode is required.")

    questions = _load_questions(questions_file)
    benchmark_root = output_dir / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkQuestionResult] = []
    for repetition in range(1, repetitions + 1):
        for question_index, question in enumerate(questions, start=1):
            for mode in mode_configs:
                env_overrides = _load_env_overrides(mode.env_files)
                mode_runs_dir = benchmark_root / "runs" / mode.name
                mode_runs_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Benchmark run mode=%s repetition=%d question_index=%d",
                    mode.name,
                    repetition,
                    question_index,
                )
                result = _run_mode_question(
                    mode_name=mode.name,
                    question=question,
                    repetition=repetition,
                    run_index=question_index,
                    config_path=config_path,
                    docs_dir=docs_dir,
                    runs_dir=mode_runs_dir,
                    selected_cities=selected_cities,
                    log_llm_payload=log_llm_payload,
                    env_overrides=env_overrides,
                )
                results.append(result)

    mode_summary = _summarize_mode(results)
    judge_results, judge_summary = _run_benchmark_judging(
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
        results=results,
        mode_summary=mode_summary,
        judge_results=judge_results,
        judge_summary=judge_summary,
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
    "BenchmarkQuestionResult",
    "BenchmarkReport",
    "run_retrieval_strategy_benchmark",
]
