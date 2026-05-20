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
    BaselineMetric,
    ComponentRowComparison,
    MetricComparison,
    MetricComponentAudit,
    PipelineMode,
    RequestedMode,
    RetrievalCandidateAudit,
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
DEFAULT_DOCUMENTS_DIR = Path("documents")

_WHITESPACE_RE = re.compile(r"[\s\u00A0\u202F]+")
_NON_NUMERIC_RE = re.compile(r"[^0-9,.\-]")
_THOUSANDS_COMMA_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_THOUSANDS_DOT_RE = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_CITY_NUMERIC_PATTERN_STRINGS = (
    r"\b\d+[\d,.]*\b.{0,80}\belectric buses?\b",
    r"\belectric buses?\b.{0,80}\b\d+[\d,.]*\b",
    r"\b\d+[\d,.]*\b.{0,80}\bzero[- ]emission buses?\b",
    r"\bzero[- ]emission buses?\b.{0,80}\b\d+[\d,.]*\b",
    r"\b\d+[\d,.]*\b.{0,80}\be-?buses?\b",
    r"\be-?buses?\b.{0,80}\b\d+[\d,.]*\b",
    r"\b\d+[\d,.]*\b.{0,80}\bfuel[- ]cell buses?\b",
    r"\bfuel[- ]cell buses?\b.{0,80}\b\d+[\d,.]*\b",
    r"\b\d+[\d,.]*\b.{0,80}\bhydrogen buses?\b",
    r"\bhydrogen buses?\b.{0,80}\b\d+[\d,.]*\b",
    r"\b\d+[\d,.]*\b.{0,80}\bemission[- ]free buses?\b",
    r"\bemission[- ]free buses?\b.{0,80}\b\d+[\d,.]*\b",
)
_CITY_NUMERIC_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in _CITY_NUMERIC_PATTERN_STRINGS
]


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


def select_benchmark_cases(
    dataset: WriterNumericBenchmarkDataset,
    *,
    include_optional_cases: bool,
) -> list[WriterNumericBenchmarkCase]:
    """Return the benchmark cases that should run for one invocation."""
    if include_optional_cases:
        return dataset.cases
    return [case for case in dataset.cases if not case.requires_explicit_include]


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
    include_optional_cases: bool = False,
    pipeline_runner: PipelineRunner = run_pipeline,
    extractor: MetricExtractor = extract_writer_numbers,
    log_llm_payload: bool = False,
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR,
) -> WriterNumericBenchmarkReport:
    """Run the writer numeric benchmark and persist report artifacts."""
    dataset = load_writer_numeric_benchmark_dataset(benchmark_file)
    executed_modes = resolve_requested_modes(requested_mode)
    selected_cases = select_benchmark_cases(
        dataset,
        include_optional_cases=include_optional_cases,
    )
    if not selected_cases:
        raise ValueError("No writer numeric benchmark cases selected for execution.")
    benchmark_root = output_dir / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)
    runs_dir = benchmark_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    results: list[WriterNumericCaseResult] = []
    for mode in executed_modes:
        mode_config = apply_pipeline_mode(config, mode, runs_dir)
        for case in selected_cases:
            result = _run_case(
                case=case,
                mode=mode,
                config=mode_config,
                api_key=api_key,
                pipeline_runner=pipeline_runner,
                extractor=extractor,
                log_llm_payload=log_llm_payload,
                documents_dir=documents_dir,
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
    documents_dir: Path,
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
    component_audits = _build_component_audits(case, candidate_text)
    retrieval_audit = _build_retrieval_audit(
        case=case,
        context_bundle_path=context_bundle,
        documents_dir=documents_dir,
    )
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
        component_audits=component_audits,
        retrieval_audit=retrieval_audit,
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


def _build_component_audits(
    case: WriterNumericBenchmarkCase,
    candidate_text: str,
) -> list[MetricComponentAudit]:
    """Compare expected row components against the writer's city/count table."""
    table_rows = _extract_city_count_rows(candidate_text)
    audits: list[MetricComponentAudit] = []
    for metric in case.baseline_metrics:
        if not metric.display_metadata.get("compare_components_as_rows"):
            continue
        audits.append(_build_metric_component_audit(metric, table_rows))
    return audits


def _build_metric_component_audit(
    metric: BaselineMetric,
    table_rows: list[tuple[str, str, str]],
) -> MetricComponentAudit:
    """Build one row-level audit for one baseline metric."""
    extracted_by_label: dict[str, tuple[str, str, str]] = {}
    for label, raw_value, raw_row in table_rows:
        normalized_label = _normalize_row_label(label)
        if normalized_label and normalized_label not in extracted_by_label:
            extracted_by_label[normalized_label] = (label, raw_value, raw_row)

    row_results: list[ComponentRowComparison] = []
    expected_label_order: list[str] = []
    for component in metric.components:
        normalized_label = _normalize_row_label(component.label)
        expected_label_order.append(normalized_label)
        extracted = extracted_by_label.get(normalized_label)
        expected_normalized = normalize_metric_value(component.value, metric.unit)
        if extracted is None:
            row_results.append(
                ComponentRowComparison(
                    label=component.label,
                    expected_value=component.value,
                    expected_normalized=expected_normalized,
                    extracted_value=None,
                    extracted_normalized=None,
                    status="missing",
                    raw_row=None,
                    notes="Expected row was not present in the writer table.",
                )
            )
            continue

        extracted_label, extracted_value, raw_row = extracted
        extracted_normalized = normalize_metric_value(extracted_value, metric.unit)
        status = (
            "match"
            if extracted_normalized is not None and extracted_normalized == expected_normalized
            else "mismatch"
        )
        label_note = ""
        if extracted_label != component.label:
            label_note = f"Writer row label: {extracted_label}"
        row_results.append(
            ComponentRowComparison(
                label=component.label,
                expected_value=component.value,
                expected_normalized=expected_normalized,
                extracted_value=extracted_value,
                extracted_normalized=extracted_normalized,
                status=status,
                raw_row=raw_row,
                notes=label_note,
            )
        )

    expected_label_set = set(expected_label_order)
    for label, raw_value, raw_row in table_rows:
        normalized_label = _normalize_row_label(label)
        if normalized_label in expected_label_set:
            continue
        row_results.append(
            ComponentRowComparison(
                label=label,
                expected_value=None,
                expected_normalized=None,
                extracted_value=raw_value,
                extracted_normalized=normalize_metric_value(raw_value, metric.unit),
                status="extra",
                raw_row=raw_row,
                notes="Writer included a row that is not present in the frozen baseline.",
            )
        )

    return MetricComponentAudit(
        metric_id=metric.metric_id,
        label=metric.label,
        unit=metric.unit,
        expected_row_count=len(metric.components),
        extracted_row_count=len(table_rows),
        match_count=sum(1 for row in row_results if row.status == "match"),
        mismatch_count=sum(1 for row in row_results if row.status == "mismatch"),
        missing_count=sum(1 for row in row_results if row.status == "missing"),
        extra_count=sum(1 for row in row_results if row.status == "extra"),
        row_results=row_results,
    )


def _extract_city_count_rows(candidate_text: str) -> list[tuple[str, str, str]]:
    """Return the first markdown table that looks like a city/count table."""
    table_lines: list[str] = []
    for line in candidate_text.splitlines():
        if line.lstrip().startswith("|"):
            table_lines.append(line.rstrip())
            continue
        if table_lines:
            rows = _parse_city_count_table(table_lines)
            if rows:
                return rows
            table_lines = []
    if not table_lines:
        return []
    return _parse_city_count_table(table_lines)


def _parse_city_count_table(table_lines: list[str]) -> list[tuple[str, str, str]]:
    """Parse one markdown table block when it has city/count headers."""
    if len(table_lines) < 3:
        return []
    header_cells = _split_markdown_row(table_lines[0])
    divider_cells = _split_markdown_row(table_lines[1])
    if len(header_cells) < 2 or not _is_markdown_divider_row(divider_cells):
        return []
    normalized_headers = [_normalize_row_label(cell) for cell in header_cells[:2]]
    if normalized_headers[0] not in {"city", "cities"}:
        return []
    if normalized_headers[1] not in {
        "count",
        "counts",
        "number",
        "numbers",
        "figure",
        "figures",
    }:
        return []

    rows: list[tuple[str, str, str]] = []
    for raw_row in table_lines[2:]:
        cells = _split_markdown_row(raw_row)
        if len(cells) < 2:
            continue
        label = cells[0]
        raw_value = cells[1]
        if not label.strip():
            continue
        rows.append((label, raw_value, raw_row.strip()))
    return rows


def _split_markdown_row(row: str) -> list[str]:
    """Split one markdown table row into trimmed cell strings."""
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_markdown_divider_row(cells: list[str]) -> bool:
    """Return True when cells look like a markdown table divider row."""
    if not cells:
        return False
    for cell in cells:
        normalized = cell.replace(":", "").replace("-", "").strip()
        if normalized:
            return False
    return True


def _normalize_row_label(value: str) -> str:
    """Normalize a row label into a deterministic city key."""
    lowered = value.strip().lower()
    lowered = lowered.replace("_", " ")
    normalized = _NON_ALNUM_RE.sub(" ", lowered)
    return " ".join(normalized.split())


def _build_retrieval_audit(
    *,
    case: WriterNumericBenchmarkCase,
    context_bundle_path: Path,
    documents_dir: Path,
) -> RetrievalCandidateAudit | None:
    """Build an optional heuristic recall audit for one case."""
    if case.retrieval_audit_preset is None:
        return None
    excerpt_cities = _load_excerpt_city_keys(context_bundle_path)
    candidate_cities = _scan_document_candidates(
        documents_dir=documents_dir,
        preset=case.retrieval_audit_preset,
    )
    excerpt_set = {city.lower(): city for city in excerpt_cities}
    candidate_set = {city.lower(): city for city in candidate_cities}
    candidate_only = sorted(
        [city for key, city in candidate_set.items() if key not in excerpt_set],
        key=str.lower,
    )
    excerpt_only = sorted(
        [city for key, city in excerpt_set.items() if key not in candidate_set],
        key=str.lower,
    )
    return RetrievalCandidateAudit(
        preset=case.retrieval_audit_preset,
        candidate_city_count=len(candidate_cities),
        excerpt_city_count=len(excerpt_cities),
        candidate_only_cities=candidate_only,
        excerpt_only_cities=excerpt_only,
    )


def _load_excerpt_city_keys(context_bundle_path: Path) -> list[str]:
    """Load distinct city keys from the saved markdown excerpt bundle."""
    payload = json.loads(context_bundle_path.read_text(encoding="utf-8"))
    excerpts = payload.get("markdown", {}).get("excerpts", [])
    seen: dict[str, str] = {}
    for excerpt in excerpts:
        city_key = excerpt.get("city_key") or excerpt.get("city_name")
        if not city_key:
            continue
        normalized = str(city_key).strip().lower()
        if normalized and normalized not in seen:
            seen[normalized] = str(city_key)
    return sorted(seen.values(), key=str.lower)


def _scan_document_candidates(*, documents_dir: Path, preset: str) -> list[str]:
    """Heuristically scan city documents for candidate numeric evidence."""
    if preset != "electric_bus_numeric_mentions":
        raise ValueError(f"Unsupported retrieval audit preset: {preset}")
    matches: list[str] = []
    for path in sorted(documents_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in _CITY_NUMERIC_PATTERNS):
            matches.append(path.stem)
    return matches


def _build_summary(results: list[WriterNumericCaseResult]) -> WriterNumericBenchmarkSummary:
    """Aggregate benchmark totals across all case outputs."""
    metric_results = [metric for result in results for metric in result.metric_results]
    component_audits = [audit for result in results for audit in result.component_audits]
    component_rows = [row for audit in component_audits for row in audit.row_results]
    retrieval_audits = [
        result.retrieval_audit
        for result in results
        if result.retrieval_audit is not None
    ]
    return WriterNumericBenchmarkSummary(
        case_count=len({result.case_id for result in results}),
        output_count=len(results),
        metric_count=len(metric_results),
        match_count=sum(1 for metric in metric_results if metric.status == "match"),
        mismatch_count=sum(1 for metric in metric_results if metric.status == "mismatch"),
        missing_count=sum(1 for metric in metric_results if metric.status == "missing"),
        component_audit_count=len(component_audits),
        component_row_count=len(component_rows),
        component_row_match_count=sum(
            1 for row in component_rows if row.status == "match"
        ),
        component_row_mismatch_count=sum(
            1 for row in component_rows if row.status == "mismatch"
        ),
        component_row_missing_count=sum(
            1 for row in component_rows if row.status == "missing"
        ),
        component_row_extra_count=sum(
            1 for row in component_rows if row.status == "extra"
        ),
        retrieval_audit_count=len(retrieval_audits),
        retrieval_candidate_city_count=sum(
            audit.candidate_city_count for audit in retrieval_audits
        ),
        retrieval_excerpt_city_count=sum(
            audit.excerpt_city_count for audit in retrieval_audits
        ),
        retrieval_candidate_only_count=sum(
            len(audit.candidate_only_cities) for audit in retrieval_audits
        ),
        retrieval_excerpt_only_count=sum(
            len(audit.excerpt_only_cities) for audit in retrieval_audits
        ),
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
        f"- Component audits: {report.summary.component_audit_count}",
        (
            "- Component rows"
            f" (match / mismatch / missing / extra): "
            f"{report.summary.component_row_match_count} / "
            f"{report.summary.component_row_mismatch_count} / "
            f"{report.summary.component_row_missing_count} / "
            f"{report.summary.component_row_extra_count}"
        ),
        (
            "- Retrieval audits"
            f" (candidate / excerpt / candidate-only / excerpt-only): "
            f"{report.summary.retrieval_candidate_city_count} / "
            f"{report.summary.retrieval_excerpt_city_count} / "
            f"{report.summary.retrieval_candidate_only_count} / "
            f"{report.summary.retrieval_excerpt_only_count}"
        ),
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
        for audit in result.component_audits:
            lines.extend(
                [
                    "",
                    f"### Row Audit / {audit.metric_id}",
                    "",
                    (
                        f"- Expected rows: {audit.expected_row_count}"
                        f", extracted rows: {audit.extracted_row_count}"
                    ),
                    (
                        "- Match / mismatch / missing / extra: "
                        f"{audit.match_count} / {audit.mismatch_count} / "
                        f"{audit.missing_count} / {audit.extra_count}"
                    ),
                    "",
                    "| Row | Baseline | Extracted | Status | Writer row |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for row in audit.row_results:
                expected = (
                    f"{row.expected_value} {audit.unit}".strip()
                    if row.expected_value is not None
                    else "n/a"
                )
                extracted = (
                    f"{row.extracted_value} {audit.unit}".strip()
                    if row.extracted_value is not None
                    else "missing"
                )
                raw_row = (row.raw_row or row.notes or "-").replace("|", "\\|")
                lines.append(
                    f"| {row.label} | {expected} | {extracted} | `{row.status}` | {raw_row} |"
                )
        if result.retrieval_audit is not None:
            audit = result.retrieval_audit
            lines.extend(
                [
                    "",
                    f"### Retrieval Audit / {audit.preset}",
                    "",
                    f"- Candidate city docs: {audit.candidate_city_count}",
                    f"- Excerpt-bearing cities: {audit.excerpt_city_count}",
                    (
                        "- Candidate docs missing from excerpts: "
                        f"{len(audit.candidate_only_cities)}"
                    ),
                    (
                        "- Excerpt-bearing cities missing from heuristic scan: "
                        f"{len(audit.excerpt_only_cities)}"
                    ),
                ]
            )
            if audit.candidate_only_cities:
                lines.append(
                    "- Candidate-only cities: "
                    + ", ".join(f"`{city}`" for city in audit.candidate_only_cities)
                )
            if audit.excerpt_only_cities:
                lines.append(
                    "- Excerpt-only cities: "
                    + ", ".join(f"`{city}`" for city in audit.excerpt_only_cities)
                )
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_BENCHMARK_FILE",
    "DEFAULT_DOCUMENTS_DIR",
    "DEFAULT_OUTPUT_DIR",
    "apply_pipeline_mode",
    "load_writer_numeric_benchmark_dataset",
    "normalize_metric_value",
    "resolve_requested_modes",
    "run_writer_numeric_benchmark",
    "select_benchmark_cases",
]
