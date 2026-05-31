"""Helpers for loading developer-facing run diagnostics artifacts."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from pathlib import Path

from backend.api.models import (
    RunDiagnosticsArtifactPaths,
    RunDiagnosticsResponse,
    RunWriterCitationCoverage,
    RunWriterMultiPass,
    RunWriterMultiPassBatch,
    RunWriterSectionPlan,
    RunWriterSectionPlanSection,
)
from backend.api.services.run_store import RunRecord
from backend.utils.json_io import read_json_object

_WARNING_MARKERS = (" - WARNING - ", "RETRY_EVENT ", "RETRY_EXHAUSTED ")
_ERROR_MARKERS = (" - ERROR - ", " - CRITICAL - ", "RETRY_EXHAUSTED ")
_MAX_WARNING_ENTRIES = 200
_MAX_LOG_TAIL_LINES = 120


def build_run_diagnostics(
    record: RunRecord,
    *,
    runs_dir: Path,
) -> RunDiagnosticsResponse:
    """Build one diagnostics payload from persisted run artifacts."""
    run_dir = resolve_run_dir(record, runs_dir)
    run_log_payload = read_json_object(run_dir / "run.json") or {}
    artifacts_payload = (
        run_log_payload.get("artifacts")
        if isinstance(run_log_payload.get("artifacts"), dict)
        else {}
    )
    run_log_path = _resolve_artifact_path(None, run_dir, "run.log")
    run_summary_path = _resolve_artifact_path(
        artifacts_payload.get("run_summary"),
        run_dir,
        "run_summary.txt",
    )
    error_log_path = _resolve_artifact_path(
        artifacts_payload.get("error_log"),
        run_dir,
        "error_log.txt",
    )

    llm_usage = run_log_payload.get("llm_usage")
    if not isinstance(llm_usage, dict):
        llm_usage = None

    retry_summary = run_log_payload.get("retry_summary")
    if not isinstance(retry_summary, dict):
        retry_summary = None

    return RunDiagnosticsResponse(
        run_id=record.run_id,
        question=record.question,
        status=record.status,
        started_at=record.started_at,
        completed_at=record.completed_at,
        finish_reason=record.finish_reason,
        error=record.error,
        artifacts=RunDiagnosticsArtifactPaths(
            run_log=_build_artifact_label(run_log_path, run_dir),
            run_summary=_build_artifact_label(run_summary_path, run_dir),
            error_log=_build_artifact_label(error_log_path, run_dir),
        ),
        writer_citation_coverage=_read_writer_citation_coverage(
            run_log_payload,
            run_log_path,
        ),
        writer_multi_pass=_read_writer_multi_pass(run_log_payload, run_log_path),
        writer_section_plan=_read_writer_section_plan(run_log_payload, run_log_path),
        llm_usage=llm_usage,
        retry_summary=retry_summary,
        warning_entries=_read_warning_entries(run_log_path),
        log_tail=_read_log_tail(run_log_path),
        error_log_text=_read_error_log_text(error_log_path, run_log_path),
    )


def resolve_run_dir(record: RunRecord, runs_dir: Path) -> Path:
    """Resolve the artifact directory for one run record."""
    if record.run_log_path is not None:
        return record.run_log_path.parent
    if record.context_bundle_path is not None:
        return record.context_bundle_path.parent
    if record.final_output_path is not None:
        return record.final_output_path.parent
    return runs_dir / record.run_id


def _resolve_artifact_path(
    raw_value: object,
    run_dir: Path,
    fallback_name: str,
) -> Path | None:
    """Resolve one artifact path from run.json while constraining it to ``run_dir``."""
    run_dir_resolved = run_dir.resolve(strict=False)
    candidates: list[Path] = []
    if isinstance(raw_value, str) and raw_value.strip():
        configured_path = Path(raw_value.strip())
        relative_candidate = _coerce_run_relative_path(
            configured_path,
            run_dir_resolved=run_dir_resolved,
        )
        if relative_candidate is not None:
            candidates.append(run_dir / relative_candidate)
        basename_candidate = _coerce_simple_relative_path(Path(configured_path.name))
        if basename_candidate is not None:
            candidates.append(run_dir / basename_candidate)
    fallback_candidate = _coerce_simple_relative_path(Path(fallback_name))
    if fallback_candidate is not None:
        candidates.append(run_dir / fallback_candidate)
    for candidate in candidates:
        if _is_run_local_path(candidate, run_dir_resolved) and candidate.exists():
            return candidate
    return None


def _build_artifact_label(path: Path | None, run_dir: Path) -> str | None:
    """Return a run-local artifact label instead of exposing a host filesystem path."""
    if path is None:
        return None
    return path.relative_to(run_dir).as_posix()


def _coerce_run_relative_path(
    candidate: Path,
    *,
    run_dir_resolved: Path,
) -> Path | None:
    """Convert one configured artifact path into a safe path relative to ``run_dir``."""
    if candidate.is_absolute():
        try:
            return candidate.resolve(strict=False).relative_to(run_dir_resolved)
        except ValueError:
            return None
    return _coerce_simple_relative_path(candidate)


def _coerce_simple_relative_path(candidate: Path) -> Path | None:
    """Return a relative path only when it stays inside the run directory."""
    if candidate.is_absolute() or candidate.drive or candidate.anchor or not candidate.parts:
        return None
    if any(part == ".." for part in candidate.parts):
        return None
    return candidate


def _is_run_local_path(candidate: Path, run_dir_resolved: Path) -> bool:
    """Return True when one candidate resolves inside the current run directory."""
    try:
        candidate.resolve(strict=False).relative_to(run_dir_resolved)
    except ValueError:
        return False
    return True


def _read_warning_entries(run_log_path: Path | None) -> list[str]:
    """Return warning and retry lines from ``run.log``."""
    if run_log_path is None or not run_log_path.exists():
        return []
    entries: list[str] = []
    try:
        with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                if any(marker in stripped for marker in _WARNING_MARKERS):
                    entries.append(stripped)
                    if len(entries) >= _MAX_WARNING_ENTRIES:
                        break
    except OSError:
        return []
    return entries


def _read_writer_citation_coverage(
    run_log_payload: dict[str, object],
    run_log_path: Path | None,
) -> RunWriterCitationCoverage | None:
    """Read persisted writer coverage metadata or infer it from ``run.log``."""
    payload = _normalize_writer_citation_coverage(
        run_log_payload.get("writer_citation_coverage")
    )
    if payload is not None:
        return payload
    if run_log_path is None or not run_log_path.exists():
        return None
    last_payload: RunWriterCitationCoverage | None = None
    try:
        with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "WRITER_CITATION_COVERAGE " not in line:
                    continue
                payload_raw = line.split("WRITER_CITATION_COVERAGE ", 1)[1].strip()
                normalized = _normalize_writer_citation_coverage_from_json(payload_raw)
                if normalized is not None:
                    last_payload = normalized
    except OSError:
        return None
    return last_payload


def _read_writer_multi_pass(
    run_log_payload: dict[str, object],
    run_log_path: Path | None,
) -> RunWriterMultiPass | None:
    """Read persisted writer multi-pass payload or infer it from ``run.log``."""
    payload = _normalize_writer_multi_pass(run_log_payload.get("writer_multi_pass"))
    if payload is not None:
        return payload
    if run_log_path is None or not run_log_path.exists():
        return None
    last_payload: RunWriterMultiPass | None = None
    try:
        with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "WRITER_MULTI_PASS " not in line:
                    continue
                payload_raw = line.split("WRITER_MULTI_PASS ", 1)[1].strip()
                normalized = _normalize_writer_multi_pass_from_json(payload_raw)
                if normalized is not None:
                    last_payload = normalized
    except OSError:
        return None
    return last_payload


def _read_writer_section_plan(
    run_log_payload: dict[str, object],
    run_log_path: Path | None,
) -> RunWriterSectionPlan | None:
    """Read persisted section-first writer diagnostics or infer from ``run.log``."""
    payload = _normalize_writer_section_plan(run_log_payload.get("writer_section_plan"))
    if payload is not None:
        return payload
    if run_log_path is None or not run_log_path.exists():
        return None
    last_payload: RunWriterSectionPlan | None = None
    try:
        with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "WRITER_SECTION_PLAN " not in line:
                    continue
                payload_raw = line.split("WRITER_SECTION_PLAN ", 1)[1].strip()
                normalized = _normalize_writer_section_plan_from_json(payload_raw)
                if normalized is not None:
                    last_payload = normalized
    except OSError:
        return None
    return last_payload


def _normalize_writer_citation_coverage(raw_value: object) -> RunWriterCitationCoverage | None:
    """Validate writer coverage payload loaded from structured artifacts."""
    if not isinstance(raw_value, dict):
        return None
    return _build_writer_citation_coverage(raw_value)


def _normalize_writer_citation_coverage_from_json(
    payload_raw: str,
) -> RunWriterCitationCoverage | None:
    """Parse a JSON-formatted writer coverage log line."""
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _build_writer_citation_coverage(payload)


def _normalize_writer_multi_pass_from_json(
    payload_raw: str,
) -> RunWriterMultiPass | None:
    """Parse one JSON-formatted writer multi-pass log line."""
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _normalize_writer_multi_pass(payload)


def _normalize_writer_section_plan_from_json(
    payload_raw: str,
) -> RunWriterSectionPlan | None:
    """Parse one JSON-formatted section-first writer log line."""
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _normalize_writer_section_plan(payload)


def _build_writer_citation_coverage(
    payload: dict[str, object],
) -> RunWriterCitationCoverage | None:
    """Coerce a coverage payload into the API diagnostics schema."""
    status_raw = payload.get("status")
    coverage_confirmed = payload.get("coverage_confirmed")
    coverage_required = payload.get("coverage_required")
    coverage_ratio = payload.get("coverage_ratio")
    if (
        not isinstance(status_raw, str)
        or status_raw not in {"confirmed", "partial", "retrying", "exhausted"}
        or not isinstance(coverage_confirmed, int)
        or not isinstance(coverage_required, int)
        or not isinstance(coverage_ratio, str)
    ):
        return None

    missing_cities_raw = payload.get("missing_cities")
    missing_cities = (
        [city for city in missing_cities_raw if isinstance(city, str)]
        if isinstance(missing_cities_raw, list)
        else []
    )
    attempt_raw = payload.get("attempt")
    max_attempts_raw = payload.get("max_attempts")
    analysis_mode_raw = payload.get("analysis_mode")
    return RunWriterCitationCoverage(
        status=status_raw,
        attempt=attempt_raw if isinstance(attempt_raw, int) else None,
        max_attempts=max_attempts_raw if isinstance(max_attempts_raw, int) else None,
        coverage_confirmed=coverage_confirmed,
        coverage_required=coverage_required,
        coverage_ratio=coverage_ratio,
        missing_cities=missing_cities,
        analysis_mode=analysis_mode_raw if isinstance(analysis_mode_raw, str) else None,
    )


def _normalize_writer_multi_pass(raw_value: object) -> RunWriterMultiPass | None:
    """Validate writer multi-pass payload loaded from structured artifacts."""
    if not isinstance(raw_value, dict):
        return None

    strategy = raw_value.get("strategy")
    combine_strategy = raw_value.get("combine_strategy")
    analysis_mode = raw_value.get("analysis_mode")
    payload_tokens = raw_value.get("payload_tokens")
    threshold_tokens = raw_value.get("threshold_tokens")
    batch_count = raw_value.get("batch_count")
    batches_raw = raw_value.get("batches")
    if (
        strategy != "split_by_city"
        or combine_strategy != "draft_merge"
        or not isinstance(analysis_mode, str)
        or not isinstance(payload_tokens, int)
        or not isinstance(threshold_tokens, int)
        or not isinstance(batch_count, int)
        or not isinstance(batches_raw, list)
    ):
        return None

    batches: list[RunWriterMultiPassBatch] = []
    for entry in batches_raw:
        if not isinstance(entry, dict):
            return None
        batch_index = entry.get("batch_index")
        city_names_raw = entry.get("city_names")
        excerpt_count = entry.get("excerpt_count")
        batch_payload_tokens = entry.get("payload_tokens")
        if (
            not isinstance(batch_index, int)
            or not isinstance(city_names_raw, list)
            or not isinstance(excerpt_count, int)
            or not isinstance(batch_payload_tokens, int)
        ):
            return None
        city_names = [city for city in city_names_raw if isinstance(city, str)]
        batches.append(
            RunWriterMultiPassBatch(
                batch_index=batch_index,
                city_names=city_names,
                excerpt_count=excerpt_count,
                payload_tokens=batch_payload_tokens,
            )
        )

    return RunWriterMultiPass(
        strategy="split_by_city",
        combine_strategy="draft_merge",
        analysis_mode=analysis_mode,
        payload_tokens=payload_tokens,
        threshold_tokens=threshold_tokens,
        batch_count=batch_count,
        batches=batches,
    )


def _normalize_writer_section_plan(raw_value: object) -> RunWriterSectionPlan | None:
    """Validate section-first writer diagnostics loaded from artifacts."""
    if not isinstance(raw_value, dict):
        return None

    strategy = raw_value.get("strategy")
    analysis_mode = raw_value.get("analysis_mode")
    planner_input_tokens = raw_value.get("planner_input_tokens")
    catalog_truncated = raw_value.get("catalog_truncated")
    section_count = raw_value.get("section_count")
    sections_raw = raw_value.get("sections")
    if (
        strategy != "section_first"
        or not isinstance(analysis_mode, str)
        or not isinstance(planner_input_tokens, int)
        or not isinstance(catalog_truncated, bool)
        or not isinstance(section_count, int)
        or not isinstance(sections_raw, list)
    ):
        return None

    sections: list[RunWriterSectionPlanSection] = []
    for entry in sections_raw:
        section = _normalize_writer_section_plan_section(entry)
        if section is None:
            return None
        sections.append(section)

    return RunWriterSectionPlan(
        strategy="section_first",
        analysis_mode=analysis_mode,
        planner_input_tokens=planner_input_tokens,
        catalog_truncated=catalog_truncated,
        section_count=section_count,
        sections=sections,
    )


def _normalize_writer_section_plan_section(
    raw_value: object,
) -> RunWriterSectionPlanSection | None:
    """Coerce one section-first diagnostic section."""
    if not isinstance(raw_value, dict):
        return None
    section_id = raw_value.get("section_id")
    title = raw_value.get("title")
    section_type = raw_value.get("section_type")
    purpose = raw_value.get("purpose")
    writing_instructions = raw_value.get("writing_instructions")
    required_ref_ids_raw = raw_value.get("required_ref_ids")
    city_names_raw = raw_value.get("city_names")
    if (
        not isinstance(section_id, str)
        or not isinstance(title, str)
        or not isinstance(section_type, str)
        or not isinstance(purpose, str)
        or not isinstance(writing_instructions, str)
        or not isinstance(required_ref_ids_raw, list)
        or not isinstance(city_names_raw, list)
    ):
        return None

    payload_tokens = raw_value.get("payload_tokens")
    draft_length_chars = raw_value.get("draft_length_chars")
    batch_count = raw_value.get("batch_count")
    return RunWriterSectionPlanSection(
        section_id=section_id,
        title=title,
        section_type=section_type,
        purpose=purpose,
        required_ref_ids=[
            ref_id for ref_id in required_ref_ids_raw if isinstance(ref_id, str)
        ],
        city_names=[city for city in city_names_raw if isinstance(city, str)],
        writing_instructions=writing_instructions,
        payload_tokens=payload_tokens if isinstance(payload_tokens, int) else None,
        draft_length_chars=draft_length_chars
        if isinstance(draft_length_chars, int)
        else None,
        batch_count=batch_count if isinstance(batch_count, int) else None,
    )


def _read_log_tail(run_log_path: Path | None) -> list[str]:
    """Return the most recent run log lines."""
    if run_log_path is None or not run_log_path.exists():
        return []
    lines: deque[str] = deque(maxlen=_MAX_LOG_TAIL_LINES)
    try:
        with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lines.append(line.rstrip("\n"))
    except OSError:
        return []
    return list(lines)


def _read_error_log_text(
    error_log_path: Path | None,
    run_log_path: Path | None,
) -> str | None:
    """Return persisted error log text or derive one from ``run.log``."""
    if error_log_path is not None:
        text = _read_text_file(error_log_path)
        if text:
            return text
    return _extract_error_blocks(run_log_path)


def _read_text_file(path: Path) -> str | None:
    """Read one UTF-8 text artifact and strip trailing whitespace."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return text or None


def _extract_error_blocks(run_log_path: Path | None) -> str | None:
    """Extract ERROR, CRITICAL, and exhausted retry blocks from ``run.log``."""
    if run_log_path is None or not run_log_path.exists():
        return None
    selected_lines: list[str] = []
    in_error_block = False
    try:
        with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                if _is_log_entry_start(stripped):
                    in_error_block = any(marker in stripped for marker in _ERROR_MARKERS)
                    if in_error_block:
                        selected_lines.append(stripped)
                elif in_error_block:
                    selected_lines.append(stripped)
    except OSError:
        return None
    return _join_lines(selected_lines)


def _is_log_entry_start(line: str) -> bool:
    """Return True when a line starts a new timestamped log entry."""
    return len(line) > 4 and line[:4].isdigit() and line[4] == "-"


def _join_lines(lines: Iterable[str]) -> str | None:
    """Return joined non-empty lines or ``None`` when nothing was collected."""
    normalized = [line for line in lines if line]
    if not normalized:
        return None
    return "\n".join(normalized)


__all__ = ["build_run_diagnostics", "resolve_run_dir"]
