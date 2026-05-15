from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable

from backend.benchmarks.writer_numeric.extractor import extract_writer_numbers
from backend.benchmarks.writer_numeric.models import (
    MetricComparison,
    PipelineMode,
    RequestedMode,
    WriterMetricExtraction,
    WriterNumericBenchmarkCase,
    WriterNumericBenchmarkDataset,
    WriterNumericBenchmarkReport,
    WriterNumericBenchmarkSummary,
    WriterNumericCaseResult,
)
from backend.modules.orchestrator.module import run_pipeline
from backend.utils.config import AppConfig
from backend.utils.json_io import write_json

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_FILE = Path(
    "backend/benchmarks/writer_numeric/writer_numeric_benchmark.json"
)
DEFAULT_OUTPUT_DIR = Path("output/benchmarks/writer_numeric")

_WHITESPACE_RE = re.compile(r"[\s\u00A0\u202F]+")
_NON_NUMERIC_RE = re.compile(r"[^0-9,.\-]")
_THOUSANDS_COMMA_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_THOUSANDS_DOT_RE = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")


PipelineRunner = Callable[..., object]
MetricExtractor = Callable[..., object]


def load_writer_numeric_benchmark_dataset(path: Path) -> WriterNumericBenchmarkDataset:
    """Load and validate the versioned writer numeric benchmark fixture."""
    return WriterNumericBenchmarkDataset.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_requested_modes(mode: RequestedMode) -> list[PipelineMode]:
    """Expand the user-facing mode flag into concrete pipeline modes."""
    if mode == "both":
        return ["ccc_only", "full_pipeline"]
    return [mode]


def apply_pipeline_mode(config: AppConfig, mode: PipelineMode, runs_dir: Path) -> AppConfig:
    """Clone config and force the enrichment flags for one benchmark mode."""
    mode_config = config.model_copy(deep=True)
    mode_config.runs_dir = runs_dir
    if mode == "ccc_only":
        mode_config.enrichment = mode_config.enrichment.model_copy(
            update={
                "enabled": False,
                "external_source_search_enabled": False,
                "web_research_enabled": False,
            }
        )
        return mode_config
    mode_config.enrichment = mode_config.enrichment.model_copy(
        update={
            "enabled": True,
            "external_source_search_enabled": True,
            "web_research_enabled": True,
        }
    )
    return mode_config


def normalize_metric_value(value: object, unit: str | None = None) -> str | None:
    """Normalize one numeric value into a deterministic comparison string."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(Decimal(str(value)).normalize(), "f")

    raw_text = str(value).strip()
    if not raw_text:
        return None

    lowered = _WHITESPACE_RE.sub(" ", raw_text.lower())
    unit_hint = (unit or "").strip().lower()
    for token in (
        unit_hint,
        unit_hint.replace("%", "percent"),
        "pln",
        "eur",
        "usd",
        "gbp",
        "percent",
        "percentage",
        "pct",
        "%",
        "€",
        "$",
        "£",
    ):
        if not token:
            continue
        lowered = lowered.replace(token, " ")

    candidate = _NON_NUMERIC_RE.sub("", lowered)
    candidate = candidate.strip()
    if not candidate:
        return None

    if _THOUSANDS_COMMA_RE.match(candidate):
        candidate = candidate.replace(",", "")
    elif _THOUSANDS_DOT_RE.match(candidate):
        candidate = candidate.replace(".", "").replace(",", ".")
    elif "," in candidate and "." not in candidate:
        digits_after_last_comma = len(candidate.rsplit(",", 1)[-1])
        if digits_after_last_comma == 3:
            candidate = candidate.replace(",", "")
        else:
            candidate = candidate.replace(",", ".")

    candidate = candidate.replace(" ", "")
    try:
        normalized_decimal = Decimal(candidate)
    except InvalidOperation:
        return None

    normalized_decimal = normalized_decimal.normalize()
    normalized_text = format(normalized_decimal, "f")
    if "." in normalized_text:
        normalized_text = normalized_text.rstrip("0").rstrip(".")
    return normalized_text


def run_writer_numeric_benchmark(
    *,
    benchmark_file: Path,
    output_dir: Path,
    benchmark_id: str,
    requested_mode: RequestedMode,
    config: AppConfig,
    api_key: str,
    pipeline_runner: PipelineRunner = run_pipeline,
    extractor: MetricExtractor = extract_writer_numbers,
    log_llm_payload: bool = False,
) -> WriterNumericBenchmarkReport:
    """Run the writer numeric benchmark and persist report artifacts."""
    dataset = load_writer_numeric_benchmark_dataset(benchmark_file)
    executed_modes = resolve_requested_modes(requested_mode)
    benchmark_root = output_dir / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)
    runs_dir = benchmark_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    results: list[WriterNumericCaseResult] = []
    for mode in executed_modes:
        mode_config = apply_pipeline_mode(config, mode, runs_dir)
        for case in dataset.cases:
            result = _run_case(
                case=case,
                mode=mode,
                config=mode_config,
                api_key=api_key,
                pipeline_runner=pipeline_runner,
                extractor=extractor,
                log_llm_payload=log_llm_payload,
            )
            results.append(result)

    report = WriterNumericBenchmarkReport(
        benchmark_id=benchmark_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        benchmark_file=str(benchmark_file),
        output_dir=str(benchmark_root),
        requested_mode=requested_mode,
        executed_modes=executed_modes,
        results=results,
        summary=_build_summary(results),
    )
    write_json(
        benchmark_root / "benchmark_summary.json",
        report.model_dump(),
        ensure_ascii=False,
    )
    (benchmark_root / "benchmark_report.md").write_text(
        _render_report_markdown(report),
        encoding="utf-8",
    )
    return report


def _run_case(
    *,
    case: WriterNumericBenchmarkCase,
    mode: PipelineMode,
    config: AppConfig,
    api_key: str,
    pipeline_runner: PipelineRunner,
    extractor: MetricExtractor,
    log_llm_payload: bool,
) -> WriterNumericCaseResult:
    """Execute one benchmark case for one pipeline mode."""
    run_id = f"{case.case_id}__{mode}"
    logger.info(
        "Running writer numeric benchmark case_id=%s mode=%s selected_cities=%d",
        case.case_id,
        mode,
        len(case.selected_cities),
    )
    run_paths = pipeline_runner(
        question=case.question,
        config=config,
        run_id=run_id,
        log_llm_payload=log_llm_payload,
        selected_cities=case.selected_cities,
        analysis_mode="aggregate",
        api_key_override=api_key,
    )
    final_output = Path(getattr(run_paths, "final_output"))
    context_bundle = Path(getattr(run_paths, "context_bundle"))
    if not final_output.exists():
        raise FileNotFoundError(f"Benchmark run missing final output: {final_output}")
    if not context_bundle.exists():
        raise FileNotFoundError(f"Benchmark run missing context bundle: {context_bundle}")

    candidate_text = final_output.read_text(encoding="utf-8")
    extraction = extractor(
        case=case,
        candidate_text=candidate_text,
        config=config,
        api_key=api_key,
        log_llm_payload=log_llm_payload,
    )
    extraction_path = final_output.parent / "extracted_numbers.json"
    write_json(extraction_path, extraction.model_dump(), ensure_ascii=False)

    comparisons = _compare_metrics(case, extraction.metrics)
    match_count = sum(1 for comparison in comparisons if comparison.status == "match")
    mismatch_count = sum(
        1 for comparison in comparisons if comparison.status == "mismatch"
    )
    missing_count = sum(1 for comparison in comparisons if comparison.status == "missing")
    return WriterNumericCaseResult(
        case_id=case.case_id,
        mode=mode,
        question=case.question,
        selected_cities=case.selected_cities,
        run_id=run_id,
        run_dir=str(final_output.parent),
        final_output_path=str(final_output),
        context_bundle_path=str(context_bundle),
        extracted_numbers_path=str(extraction_path),
        metric_results=comparisons,
        match_count=match_count,
        mismatch_count=mismatch_count,
        missing_count=missing_count,
    )


def _compare_metrics(
    case: WriterNumericBenchmarkCase,
    extracted_metrics: list[WriterMetricExtraction],
) -> list[MetricComparison]:
    """Compare extracted metric values against the manual baseline."""
    extracted_by_id = {metric.metric_id: metric for metric in extracted_metrics}
    comparisons: list[MetricComparison] = []
    for metric in case.baseline_metrics:
        extracted = extracted_by_id.get(metric.metric_id)
        expected_normalized = normalize_metric_value(metric.expected_value, metric.unit)
        extracted_raw_value = extracted.normalized_value if extracted else None
        extracted_normalized = normalize_metric_value(extracted_raw_value, metric.unit)
        found = bool(extracted and extracted.found)
        if not found or extracted_normalized is None:
            status = "missing"
        elif extracted_normalized == expected_normalized:
            status = "match"
        else:
            status = "mismatch"
        comparisons.append(
            MetricComparison(
                metric_id=metric.metric_id,
                label=metric.label,
                unit=metric.unit,
                expected_value=metric.expected_value,
                expected_normalized=expected_normalized,
                extracted_value=extracted_raw_value,
                extracted_normalized=extracted_normalized,
                found=found,
                status=status,
                raw_snippet=extracted.raw_snippet if extracted else None,
                notes=extracted.notes if extracted else "",
            )
        )
    return comparisons


def _build_summary(results: list[WriterNumericCaseResult]) -> WriterNumericBenchmarkSummary:
    """Aggregate benchmark totals across all case outputs."""
    metric_results = [metric for result in results for metric in result.metric_results]
    return WriterNumericBenchmarkSummary(
        case_count=len({result.case_id for result in results}),
        output_count=len(results),
        metric_count=len(metric_results),
        match_count=sum(1 for metric in metric_results if metric.status == "match"),
        mismatch_count=sum(1 for metric in metric_results if metric.status == "mismatch"),
        missing_count=sum(1 for metric in metric_results if metric.status == "missing"),
    )


def _render_report_markdown(report: WriterNumericBenchmarkReport) -> str:
    """Render a human-readable diff report with per-metric snippets."""
    lines = [
        "# Writer Numeric Benchmark Report",
        "",
        f"- Benchmark ID: `{report.benchmark_id}`",
        f"- Requested mode: `{report.requested_mode}`",
        f"- Executed modes: `{', '.join(report.executed_modes)}`",
        f"- Fixture: `{report.benchmark_file}`",
        f"- Output directory: `{report.output_dir}`",
        "",
        "## Summary",
        "",
        f"- Case count: {report.summary.case_count}",
        f"- Output count: {report.summary.output_count}",
        f"- Metric count: {report.summary.metric_count}",
        f"- Matches: {report.summary.match_count}",
        f"- Mismatches: {report.summary.mismatch_count}",
        f"- Missing: {report.summary.missing_count}",
    ]
    for result in report.results:
        lines.extend(
            [
                "",
                f"## {result.mode} / {result.case_id}",
                "",
                f"- Run directory: `{result.run_dir}`",
                f"- Selected cities: {len(result.selected_cities)}",
                f"- Match / mismatch / missing: {result.match_count} / {result.mismatch_count} / {result.missing_count}",
                "",
                "| Metric | Baseline | Extracted | Status | Writer snippet |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for metric in result.metric_results:
            snippet = (metric.raw_snippet or "").replace("\n", " ").replace("|", "\\|")
            expected = f"{metric.expected_value} {metric.unit}".strip()
            extracted = (
                f"{metric.extracted_value} {metric.unit}".strip()
                if metric.extracted_value is not None
                else "missing"
            )
            lines.append(
                f"| `{metric.metric_id}` | {expected} | {extracted} | `{metric.status}` | {snippet or '-'} |"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_BENCHMARK_FILE",
    "DEFAULT_OUTPUT_DIR",
    "apply_pipeline_mode",
    "load_writer_numeric_benchmark_dataset",
    "normalize_metric_value",
    "resolve_requested_modes",
    "run_writer_numeric_benchmark",
]
