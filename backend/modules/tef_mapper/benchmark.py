"""Krakow TEF benchmark helpers for source-truth comparison."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.modules.initiative_extractor.models import (
    InitiativeExtraction,
    InitiativeExtractionRecord,
    InitiativeNumbers,
    InitiativeSourceRef,
)
from backend.modules.tef_mapper.agent import map_initiatives_to_tef
from backend.utils.config import AppConfig
from backend.utils.json_io import read_json_object, write_json

logger = logging.getLogger(__name__)

KRAKOW_SOURCE_TRUTH_DIR = Path("backend/benchmarks/tef_mapping/krakow_source_truth")
DEFAULT_KRAKOW_TEF_SOURCE_TRUTH = (
    KRAKOW_SOURCE_TRUTH_DIR / "all_correct_initiatives_mapped_to_tef.json"
)
DEFAULT_KRAKOW_BENCHMARK_OUTPUT_ROOT = Path(
    "output/tef_benchmarks/krakow_tef_mapping"
)
MAPPING_RUN_ID = "01_tef_mapping"
COMPARISON_DIR_NAME = "02_comparison"

JsonDict = dict[str, Any]
MappingKey = tuple[str, str, str]


@dataclass(frozen=True)
class IssueClassification:
    """Computed benchmark classification for one initiative comparison."""

    status: str
    priority: str | None
    primary_exact: bool
    mapping_set_exact: bool
    source_primary_present_in_fresh: bool
    fresh_primary_is_source_secondary: bool


@dataclass(frozen=True)
class TefBenchmarkComparison:
    """Source-truth comparison payload produced by the benchmark."""

    source_truth_file: str
    candidate_run_dir: str
    summary: JsonDict
    issues: list[JsonDict]


@dataclass(frozen=True)
class TefBenchmarkRunResult:
    """Paths and counts returned by a full Krakow TEF benchmark run."""

    benchmark_id: str
    benchmark_dir: str
    source_truth_file: str
    input_path: str
    mapping_run_dir: str
    report_path: str
    issues_path: str
    summary_path: str
    mapping_result: JsonDict
    comparison_summary: JsonDict


def load_source_truth_rows(source_truth_path: Path, limit: int | None = None) -> list[JsonDict]:
    """Load source-truth initiative rows from the curated mapped Krakow fixture."""
    payload = read_json_object(source_truth_path)
    if payload is None or not isinstance(payload.get("initiatives"), list):
        raise ValueError(
            f"Source-truth file must contain an `initiatives` list: {source_truth_path}"
        )
    rows = payload["initiatives"]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Source-truth initiatives must be JSON objects: {source_truth_path}")
    if limit is None:
        return rows
    return rows[: max(limit, 0)]


def source_truth_row_to_initiative_record(row: JsonDict) -> InitiativeExtractionRecord:
    """Convert one mapped source-truth row into the mapper input record shape."""
    return InitiativeExtractionRecord(
        record_id=row["record_id"],
        source_document=row["source_document"],
        document_local_code=row.get("local_code"),
        initiative=InitiativeExtraction(
            city=row["city"],
            initiative_name=row["initiative_name"],
            general_description=row.get("general_description"),
            objective_text=row.get("objective_text"),
            implementation_text=row.get("implementation_text"),
            planned_outputs_text=row.get("planned_outputs_text"),
            delivery_text=row.get("delivery_text"),
            funding_text=row.get("funding_text"),
            timeline_text=row.get("timeline_text"),
            numbers=InitiativeNumbers.model_validate(row.get("numbers") or {}),
        ),
        source_refs=_source_refs_from_truth_row(row),
    )


def write_source_truth_initiatives_jsonl(rows: list[JsonDict], output_path: Path) -> None:
    """Write curated source-truth initiatives as mapper-ready JSONL records."""
    records = [source_truth_row_to_initiative_record(row) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )


def compare_mapping_run_to_source_truth(
    *,
    source_truth_path: Path,
    mapping_run_dir: Path,
    limit: int | None = None,
) -> TefBenchmarkComparison:
    """Compare a TEF mapping run against the curated Krakow source-truth mappings."""
    rows = load_source_truth_rows(source_truth_path, limit=limit)
    final_rows = _read_jsonl(mapping_run_dir / "05_final_mappings" / "final_mappings.jsonl")
    review_rows = _read_jsonl(mapping_run_dir / "06_review" / "review_items.jsonl")
    return compare_rows_to_final_mappings(
        source_truth_rows=rows,
        final_rows=final_rows,
        review_rows=review_rows,
        source_truth_path=source_truth_path,
        mapping_run_dir=mapping_run_dir,
    )


def compare_rows_to_final_mappings(
    *,
    source_truth_rows: list[JsonDict],
    final_rows: list[JsonDict],
    review_rows: list[JsonDict],
    source_truth_path: Path,
    mapping_run_dir: Path,
) -> TefBenchmarkComparison:
    """Compare source-truth rows with final mapping rows and classify issues."""
    final_by_record = _group_by_key(final_rows, "initiative_record_id")
    reviews_by_record = _group_by_key(review_rows, "initiative_record_id")
    status_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    issues: list[JsonDict] = []
    clean_count = 0

    for source_row in source_truth_rows:
        record_id = source_row["record_id"]
        source_mappings = source_row.get("tef_mappings") or []
        fresh_mappings = final_by_record.get(record_id, [])
        classification = classify_mapping_set(source_mappings, fresh_mappings)
        status_counts[classification.status] += 1

        if classification.status == "clean_match":
            clean_count += 1
            continue

        if classification.priority is not None:
            priority_counts[classification.priority] += 1

        issues.append(
            _build_issue(
                source_row=source_row,
                fresh_mappings=fresh_mappings,
                review_rows=reviews_by_record.get(record_id, []),
                classification=classification,
            )
        )

    summary = {
        "cases_compared": len(source_truth_rows),
        "clean_primary_and_set_matches": clean_count,
        "issues_marked": len(issues),
        "status_counts": dict(status_counts),
        "priority_counts": dict(priority_counts),
    }
    return TefBenchmarkComparison(
        source_truth_file=str(source_truth_path),
        candidate_run_dir=str(mapping_run_dir),
        summary=summary,
        issues=sorted(
            issues,
            key=lambda issue: (
                _priority_sort_key(issue["priority"]),
                str(issue["local_code"]),
            ),
        ),
    )


def classify_mapping_set(
    source_mappings: list[JsonDict],
    fresh_mappings: list[JsonDict],
) -> IssueClassification:
    """Classify one initiative's mapping comparison as clean, P1, P2, or P3."""
    source_primary = _primary_mapping(source_mappings)
    fresh_primary = _primary_mapping(fresh_mappings)
    source_primary_key = _mapping_key(source_primary)
    fresh_primary_key = _mapping_key(fresh_primary)
    source_keys = _mapping_keys(source_mappings)
    fresh_keys = _mapping_keys(fresh_mappings)
    primary_exact = source_primary_key == fresh_primary_key
    mapping_set_exact = source_keys == fresh_keys
    source_primary_present = bool(source_primary_key and source_primary_key in fresh_keys)
    fresh_primary_is_source_secondary = bool(
        fresh_primary_key
        and fresh_primary_key in source_keys
        and fresh_primary_key != source_primary_key
    )
    same_path_different_target = bool(
        source_primary
        and fresh_primary
        and source_primary.get("target_path") == fresh_primary.get("target_path")
        and source_primary_key != fresh_primary_key
    )

    if primary_exact and mapping_set_exact:
        return IssueClassification(
            status="clean_match",
            priority=None,
            primary_exact=primary_exact,
            mapping_set_exact=mapping_set_exact,
            source_primary_present_in_fresh=source_primary_present,
            fresh_primary_is_source_secondary=fresh_primary_is_source_secondary,
        )
    if primary_exact:
        status = "nonprimary_mapping_set_drift"
        priority = "P3"
    elif not fresh_mappings:
        status = "missing_fresh_mapping"
        priority = "P1"
    elif source_primary_present:
        status = "source_primary_demoted"
        priority = "P2"
    elif fresh_primary_is_source_secondary:
        status = "source_secondary_promoted"
        priority = "P2"
    elif same_path_different_target:
        status = "same_path_different_target"
        priority = "P2"
    else:
        status = "primary_target_mismatch"
        priority = "P1"

    return IssueClassification(
        status=status,
        priority=priority,
        primary_exact=primary_exact,
        mapping_set_exact=mapping_set_exact,
        source_primary_present_in_fresh=source_primary_present,
        fresh_primary_is_source_secondary=fresh_primary_is_source_secondary,
    )


def write_comparison_artifacts(
    comparison: TefBenchmarkComparison,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write machine-readable and Markdown benchmark comparison artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    issues_path = output_dir / "tef_benchmark_issues.json"
    report_path = output_dir / "tef_benchmark_report.md"
    write_json(issues_path, asdict(comparison), ensure_ascii=False)
    report_path.write_text(_render_markdown_report(comparison), encoding="utf-8")
    return issues_path, report_path


def run_krakow_tef_benchmark(
    *,
    config: AppConfig,
    api_key: str,
    source_truth_path: Path = DEFAULT_KRAKOW_TEF_SOURCE_TRUTH,
    tef_catalog_dir: Path = Path("tef_mapping"),
    output_root: Path = DEFAULT_KRAKOW_BENCHMARK_OUTPUT_ROOT,
    benchmark_id: str | None = None,
    max_workers: int | None = None,
    limit: int | None = None,
    log_llm_payload: bool = False,
) -> TefBenchmarkRunResult:
    """Run the full Krakow TEF benchmark and write comparison artifacts."""
    resolved_benchmark_id = benchmark_id or _default_benchmark_id()
    benchmark_dir = output_root / resolved_benchmark_id
    source_rows = load_source_truth_rows(source_truth_path, limit=limit)
    input_path = benchmark_dir / "00_inputs" / "initiatives.jsonl"
    write_source_truth_initiatives_jsonl(source_rows, input_path)

    logger.info(
        "Starting Krakow TEF benchmark benchmark_id=%s initiatives=%d",
        resolved_benchmark_id,
        len(source_rows),
    )
    mapping_result = map_initiatives_to_tef(
        config=config,
        api_key=api_key,
        tef_catalog_dir=tef_catalog_dir,
        output_root=benchmark_dir,
        initiatives_jsonl=input_path,
        run_id=MAPPING_RUN_ID,
        selected_cities=["Krakow"],
        max_workers=max_workers,
        log_llm_payload=log_llm_payload,
    )
    mapping_run_dir = Path(mapping_result.output_dir)
    comparison = compare_mapping_run_to_source_truth(
        source_truth_path=source_truth_path,
        mapping_run_dir=mapping_run_dir,
        limit=limit,
    )
    comparison_dir = benchmark_dir / COMPARISON_DIR_NAME
    issues_path, report_path = write_comparison_artifacts(comparison, comparison_dir)
    summary_path = benchmark_dir / "benchmark_summary.json"
    result = TefBenchmarkRunResult(
        benchmark_id=resolved_benchmark_id,
        benchmark_dir=str(benchmark_dir),
        source_truth_file=str(source_truth_path),
        input_path=str(input_path),
        mapping_run_dir=str(mapping_run_dir),
        report_path=str(report_path),
        issues_path=str(issues_path),
        summary_path=str(summary_path),
        mapping_result=mapping_result.model_dump(mode="json"),
        comparison_summary=comparison.summary,
    )
    write_json(summary_path, asdict(result), ensure_ascii=False)
    logger.info("Krakow TEF benchmark finished: %s", asdict(result))
    return result


def _source_refs_from_truth_row(row: JsonDict) -> list[InitiativeSourceRef]:
    """Build source refs from source-truth row refs, skipping incomplete refs."""
    refs: list[InitiativeSourceRef] = []
    for index, ref in enumerate(row.get("source_refs") or [], start=1):
        start_line = ref.get("start_line")
        end_line = ref.get("end_line")
        if start_line is None or end_line is None:
            continue
        refs.append(
            InitiativeSourceRef(
                source_document=ref.get("source_document", row["source_document"]),
                segment_id=f"{row['record_id']}:source:{index}",
                start_line=int(start_line),
                end_line=int(end_line),
            )
        )
    return refs


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read JSONL rows from a benchmark or mapper artifact."""
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _group_by_key(rows: list[JsonDict], key: str) -> dict[str, list[JsonDict]]:
    """Group JSON rows by a string key."""
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if isinstance(value, str):
            grouped[value].append(row)
    return grouped


def _mapping_key(row: JsonDict | None) -> MappingKey | None:
    """Return the comparison identity for one TEF mapping row."""
    if not row:
        return None
    target_type = row.get("target_type")
    target_id = row.get("target_id")
    target_path = row.get("target_path")
    if not all(isinstance(value, str) for value in [target_type, target_id, target_path]):
        return None
    return target_type, target_id, target_path


def _mapping_keys(rows: list[JsonDict]) -> set[MappingKey]:
    """Return all comparable mapping identities from rows."""
    return {key for row in rows if (key := _mapping_key(row)) is not None}


def _primary_mapping(rows: list[JsonDict]) -> JsonDict | None:
    """Return the primary mapping row, falling back to the first row when needed."""
    for row in rows:
        if row.get("is_primary") is True:
            return row
    return rows[0] if rows else None


def _build_issue(
    *,
    source_row: JsonDict,
    fresh_mappings: list[JsonDict],
    review_rows: list[JsonDict],
    classification: IssueClassification,
) -> JsonDict:
    """Build one issue record with source context and comparison details."""
    source_mappings = source_row.get("tef_mappings") or []
    source_primary = source_row.get("primary_tef_mapping") or _primary_mapping(source_mappings)
    fresh_primary = _primary_mapping(fresh_mappings)
    source_keys = _mapping_keys(source_mappings)
    fresh_keys = _mapping_keys(fresh_mappings)
    issue: JsonDict = {
        "local_code": source_row.get("local_code"),
        "priority": classification.priority,
        "status": classification.status,
        "initiative_name": source_row.get("initiative_name"),
        "initiative_about": _initiative_about(source_row),
        "source_reference": _source_reference(source_row),
        "source_primary": source_primary,
        "fresh_primary": fresh_primary,
        "primary_exact": classification.primary_exact,
        "mapping_set_exact": classification.mapping_set_exact,
        "source_primary_present_in_fresh": (
            classification.source_primary_present_in_fresh
        ),
        "fresh_primary_is_source_secondary": (
            classification.fresh_primary_is_source_secondary
        ),
        "missing_source_mapping_keys": sorted(source_keys - fresh_keys),
        "extra_fresh_mapping_keys": sorted(fresh_keys - source_keys),
        "path_divergence": _path_divergence(
            source_primary=source_primary,
            fresh_primary=fresh_primary,
            primary_exact=classification.primary_exact,
            mapping_set_exact=classification.mapping_set_exact,
        ),
        "fresh_rationale": fresh_primary.get("rationale") if fresh_primary else None,
        "review_flags": sorted(
            {
                row.get("review_type")
                for row in review_rows
                if isinstance(row.get("review_type"), str)
            }
        ),
    }
    issue["possible_reason"] = _possible_reason(issue)
    return issue


def _initiative_about(row: JsonDict) -> str:
    """Summarize the initiative from source-truth extraction fields."""
    pieces = []
    for field in ["general_description", "objective_text", "planned_outputs_text"]:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            label = {
                "general_description": "",
                "objective_text": "Objective: ",
                "planned_outputs_text": "Planned output: ",
            }[field]
            pieces.append(f"{label}{value.strip()}")
    return " ".join(pieces) or "No extracted initiative description available."


def _source_reference(row: JsonDict) -> str:
    """Render source refs as compact document-line references."""
    refs = row.get("source_refs") or []
    rendered = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        document = ref.get("source_document") or ref.get("source_path") or row.get(
            "source_document",
            "source document",
        )
        start_line = ref.get("start_line")
        end_line = ref.get("end_line")
        if start_line and end_line:
            rendered.append(f"{document}:L{start_line}-L{end_line}")
        else:
            rendered.append(str(document))
    return "; ".join(rendered) if rendered else "No source reference recorded."


def _path_divergence(
    *,
    source_primary: JsonDict | None,
    fresh_primary: JsonDict | None,
    primary_exact: bool,
    mapping_set_exact: bool,
) -> str:
    """Explain where the source and fresh primary TEF paths diverged."""
    if not source_primary:
        return "source truth has no primary mapping"
    if not fresh_primary:
        return "new run produced no primary mapping"
    source_path = str(source_primary.get("target_path") or "")
    fresh_path = str(fresh_primary.get("target_path") or "")
    if primary_exact and mapping_set_exact:
        return "primary target and full mapping set match"
    if primary_exact:
        return "primary target identical; non-primary mapping set differs"
    if source_path == fresh_path:
        return "same TEF path, but target type/id differs"

    source_parts = source_path.split("/") if source_path else []
    fresh_parts = fresh_path.split("/") if fresh_path else []
    common = []
    for source_part, fresh_part in zip(source_parts, fresh_parts):
        if source_part != fresh_part:
            break
        common.append(source_part)
    index = len(common)
    expected = source_parts[index] if index < len(source_parts) else "<end>"
    got = fresh_parts[index] if index < len(fresh_parts) else "<end>"
    if common:
        return f"aligned through `{'/'.join(common)}`; expected `{expected}`, got `{got}`"
    return f"diverged at TEF sector; expected `{expected}`, got `{got}`"


def _possible_reason(issue: JsonDict) -> str:
    """Provide a concise generic reason for the issue class."""
    status = issue["status"]
    if status == "nonprimary_mapping_set_drift":
        return (
            "The primary route is stable, but secondary mappings drifted; this usually "
            "means the initiative has multiple plausible co-benefits."
        )
    if status in {"source_primary_demoted", "source_secondary_promoted"}:
        return (
            "The run found a source-truth target but ranked it differently, suggesting "
            "a weighting issue between accepted initiative readings."
        )
    if status == "same_path_different_target":
        return (
            "The run reached the same TEF branch but chose a different target id or "
            "target type, usually because category and transition-element options overlap."
        )
    if status == "missing_fresh_mapping":
        return "The mapper produced no final mapping row for this source-truth initiative."
    return (
        "The run selected a different primary TEF route from source truth, usually due "
        "to ambiguous initiative wording or overlapping sibling category guidance."
    )


def _render_markdown_report(comparison: TefBenchmarkComparison) -> str:
    """Render the benchmark comparison as a Markdown report."""
    summary = comparison.summary
    lines = [
        "# Krakow TEF Benchmark Report",
        "",
        "This report compares one full Krakow TEF mapping run against the curated CCC "
        "source-truth TEF mappings.",
        "",
        "## Inputs",
        "",
        f"- Source truth: `{comparison.source_truth_file}`",
        f"- Candidate run: `{comparison.candidate_run_dir}`",
        "",
        "## Summary",
        "",
        f"- Cases compared: {summary['cases_compared']}",
        f"- Clean primary and full mapping-set matches: {summary['clean_primary_and_set_matches']}",
        f"- Issues marked: {summary['issues_marked']}",
        f"- Priority counts: {_format_counts(summary.get('priority_counts', {}))}",
        f"- Status counts: {_format_counts(summary.get('status_counts', {}))}",
        "",
        "P1 means the new primary target does not match source truth. P2 means the run "
        "stayed close to source truth but promoted, demoted, or changed the target "
        "within the same path. P3 means the primary target matches and only secondary "
        "mapping rows drifted.",
        "",
        "## Issue Index",
        "",
    ]
    for priority in ["P1", "P2", "P3"]:
        codes = [
            str(issue["local_code"])
            for issue in comparison.issues
            if issue.get("priority") == priority
        ]
        lines.append(f"- {priority}: {', '.join(codes) if codes else 'none'}")
    lines.extend(["", "## Detailed Issues", ""])
    for issue in comparison.issues:
        lines.extend(_render_issue(issue))
    return "\n".join(lines)


def _render_issue(issue: JsonDict) -> list[str]:
    """Render one benchmark issue as Markdown lines."""
    missing = [_format_mapping_key(tuple(key)) for key in issue["missing_source_mapping_keys"]]
    extra = [_format_mapping_key(tuple(key)) for key in issue["extra_fresh_mapping_keys"]]
    lines = [
        f"### {issue['priority']} {issue['local_code']} - {issue['status']}",
        "",
        f"- Initiative: {issue['initiative_name']}",
        f"- What the initiative is about: {issue['initiative_about']}",
        f"- Source reference: {issue['source_reference']}",
        f"- Source-truth primary: {_format_mapping(issue.get('source_primary'))}",
        f"- New-run primary: {_format_mapping(issue.get('fresh_primary'))}",
        f"- TEF divergence: {issue['path_divergence']}",
        f"- Possible reason: {issue['possible_reason']}",
        f"- Missing source-truth mappings: {'; '.join(missing) if missing else 'none'}",
        f"- Extra new-run mappings: {'; '.join(extra) if extra else 'none'}",
    ]
    if issue.get("fresh_rationale"):
        lines.append(f"- New-run rationale: {issue['fresh_rationale']}")
    flags = issue.get("review_flags") or []
    lines.extend(
        [
            f"- New-run review flags: {', '.join(flags) if flags else 'none'}",
            "",
        ]
    )
    return lines


def _format_counts(counts: object) -> str:
    """Render a dictionary of counters for Markdown."""
    if not isinstance(counts, dict) or not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _format_mapping(row: JsonDict | None) -> str:
    """Render a TEF mapping row identity."""
    return _format_mapping_key(_mapping_key(row))


def _format_mapping_key(key: tuple[object, ...] | None) -> str:
    """Render a mapping identity tuple."""
    if key is None:
        return "<none>"
    target_type, target_id, target_path = key
    return f"{target_type} | {target_id} | `{target_path}`"


def _priority_sort_key(priority: object) -> int:
    """Sort P1 before P2 before P3."""
    return {"P1": 0, "P2": 1, "P3": 2}.get(priority, 9)


def _default_benchmark_id() -> str:
    """Return a timestamped default benchmark id."""
    return datetime.now(UTC).strftime("krakow_tef_benchmark_%Y%m%d_%H%M%S")


__all__ = [
    "COMPARISON_DIR_NAME",
    "DEFAULT_KRAKOW_BENCHMARK_OUTPUT_ROOT",
    "DEFAULT_KRAKOW_TEF_SOURCE_TRUTH",
    "MAPPING_RUN_ID",
    "IssueClassification",
    "TefBenchmarkComparison",
    "TefBenchmarkRunResult",
    "classify_mapping_set",
    "compare_mapping_run_to_source_truth",
    "compare_rows_to_final_mappings",
    "load_source_truth_rows",
    "run_krakow_tef_benchmark",
    "source_truth_row_to_initiative_record",
    "write_comparison_artifacts",
    "write_source_truth_initiatives_jsonl",
]
