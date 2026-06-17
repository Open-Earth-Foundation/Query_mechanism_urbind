"""Controlled search tools for governed external Markdown sources."""

from __future__ import annotations

import bisect
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import yaml

from backend.modules.web_researcher.models import (
    EvidenceCandidate,
    EvidenceCandidateInput,
    NoEvidenceRecord,
    SearchHit,
    SourceMetadata,
    SourceSummary,
    TagOptions,
)
from backend.utils.config import AppConfig

logger = logging.getLogger(__name__)
EXTERNAL_SOURCE_SEARCH_AUDIT_FILENAME = "external_source_search_audit.json"


class ExternalSourceToolError(ValueError):
    """Structured error raised by external-source tools."""

    def __init__(
        self,
        code: str,
        message: str,
        allowed_values: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.allowed_values = allowed_values or []

    def to_dict(self) -> dict[str, object]:
        """Return a compact error payload suitable for an LLM tool result."""
        payload: dict[str, object] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.allowed_values:
            payload["error"]["allowed_values"] = self.allowed_values  # type: ignore[index]
        return payload


@dataclass(frozen=True)
class WordSpan:
    """Character span for one non-whitespace token in a Markdown document."""

    start: int
    end: int


@dataclass(frozen=True)
class DocumentIndex:
    """Immutable in-memory index for one external Markdown file."""

    source: SourceMetadata
    path: Path
    text: str
    lines: tuple[str, ...]
    line_start_offsets: tuple[int, ...]
    heading_paths_by_line: tuple[tuple[str, ...], ...]
    word_spans: tuple[WordSpan, ...]


@dataclass(frozen=True)
class HitRecord:
    """Internal hit cache record used by expansion and evidence capture."""

    hit_id: str
    source_id: str
    match_start: int
    match_end: int
    pattern: str
    search_id: str


@dataclass(frozen=True)
class ExternalSearchLimits:
    """Hard caps applied by the external-source tool layer."""

    max_files_per_search: int
    max_matches_per_search: int
    max_regex_searches_per_field: int
    default_context_words: int
    max_context_words: int
    default_context_lines: int
    max_context_lines: int
    max_expand_hits_per_call: int
    max_snippet_chars: int
    max_pattern_chars: int


def build_external_search_limits(config: AppConfig) -> ExternalSearchLimits:
    """Read external-source search caps from application config."""
    enrichment = config.enrichment
    return ExternalSearchLimits(
        max_files_per_search=enrichment.max_external_sources_per_search,
        max_matches_per_search=enrichment.max_external_matches_per_search,
        max_regex_searches_per_field=enrichment.max_external_regex_searches_per_field,
        default_context_words=enrichment.external_default_context_words,
        max_context_words=enrichment.external_max_context_words,
        default_context_lines=enrichment.external_default_context_lines,
        max_context_lines=enrichment.external_max_context_lines,
        max_expand_hits_per_call=enrichment.external_max_expand_hits_per_call,
        max_snippet_chars=enrichment.external_max_snippet_chars,
        max_pattern_chars=enrichment.external_max_pattern_chars,
    )


class SourceRegistry:
    """Validated metadata and document indexes for one external source folder."""

    def __init__(
        self,
        root_dir: Path,
        sources: list[SourceMetadata],
        source_paths: dict[str, Path],
    ) -> None:
        self.root_dir = root_dir
        self.sources = sources
        self.source_paths = source_paths
        self._source_by_id = {source.source_id: source for source in sources}
        self._indexes: dict[str, DocumentIndex] = {}

    @classmethod
    def load(cls, root_dir: Path) -> "SourceRegistry":
        """Load and validate `sources.yaml` plus Markdown files under `root_dir`."""
        root = root_dir.resolve()
        metadata_path = root / "sources.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"External source metadata not found: {metadata_path}")

        raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        raw_sources = raw.get("sources")
        if not isinstance(raw_sources, list):
            raise ExternalSourceToolError(
                "SOURCE_NOT_FOUND",
                "sources.yaml must contain a top-level `sources` list.",
            )

        sources = [SourceMetadata.model_validate(item) for item in raw_sources]
        source_paths = _resolve_source_paths(root, sources)
        return cls(root, sources, source_paths)

    def has_sources(self) -> bool:
        """Return True when at least one tagged source is available."""
        return bool(self.sources)

    def get_tag_options(self) -> TagOptions:
        """Return available metadata filters derived from source tags."""
        return TagOptions(
            cities=_sorted_distinct(_flatten(source.city for source in self.sources)),
            countries=_sorted_distinct(_flatten(source.country for source in self.sources)),
            publication_years=sorted(
                {
                    source.publication_year
                    for source in self.sources
                    if source.publication_year is not None
                }
            ),
            source_types=_sorted_distinct(source.source_type for source in self.sources),
            verticals=_sorted_distinct(_flatten(source.verticals for source in self.sources)),
            tef_sectors=_sorted_distinct(_flatten(source.tef_sectors for source in self.sources)),
        )

    def list_candidate_sources(
        self,
        *,
        cities: list[str] | None = None,
        countries: list[str] | None = None,
        verticals: list[str] | None = None,
        tef_sectors: list[str] | None = None,
        source_types: list[str] | None = None,
        publication_year_min: int | None = None,
        publication_year_max: int | None = None,
        max_files: int = 50,
    ) -> list[SourceSummary]:
        """Return source summaries matching metadata filters."""
        options = self.get_tag_options()
        filters = _ValidatedFilters(
            cities=_validate_filter("city", cities, options.cities),
            countries=_validate_filter("country", countries, options.countries),
            verticals=_validate_filter("vertical", verticals, options.verticals),
            tef_sectors=_validate_filter("tef_sector", tef_sectors, options.tef_sectors),
            source_types=_validate_filter("source_type", source_types, options.source_types),
            publication_year_min=publication_year_min,
            publication_year_max=publication_year_max,
        )
        matched = [source for source in self.sources if _source_matches(source, filters)]
        matched.sort(key=lambda source: _source_sort_key(source, filters))
        limit = max(0, min(max_files, len(matched)))
        return [_source_summary(source) for source in matched[:limit]]

    def get_source(self, source_id: str) -> SourceMetadata:
        """Return metadata for a source id or raise a structured error."""
        source = self._source_by_id.get(source_id)
        if source is None:
            raise ExternalSourceToolError("SOURCE_NOT_FOUND", f"Unknown source_id: {source_id}")
        return source

    def get_index(self, source_id: str) -> DocumentIndex:
        """Return a cached immutable Markdown index for one source."""
        index = self._indexes.get(source_id)
        if index is not None:
            return index

        source = self.get_source(source_id)
        path = self.source_paths[source_id]
        text = path.read_text(encoding="utf-8")
        index = _build_document_index(source, path, text)
        self._indexes[source_id] = index
        return index


class ExternalSearchSession:
    """Per-run controlled tool session over the external source registry."""

    _BASE_VISIBLE_TOOL_NAMES = (
        "get_tag_options",
        "list_candidate_sources",
        "regex_search",
        "list_evidence_candidates",
        "mark_no_evidence_found",
    )
    _HIT_VISIBLE_TOOL_NAMES = ("expand_hits", "add_evidence_candidates")

    def __init__(
        self,
        *,
        run_id: str,
        registry: SourceRegistry,
        limits: ExternalSearchLimits,
        artifact_dir: Path | None = None,
    ) -> None:
        self.run_id = run_id
        self.registry = registry
        self.limits = limits
        self.artifact_dir = artifact_dir
        self._last_candidate_source_ids: list[str] = []
        self._search_counter = 0
        self._hit_counter = 0
        self._evidence_counter = 0
        self._no_evidence_counter = 0
        self._field_regex_search_count = 0
        self._active_city = ""
        self._active_field = ""
        self._hit_records: dict[str, HitRecord] = {}
        self._hit_ids_by_task: dict[tuple[str, str], list[str]] = {}
        self._expanded_hit_ids_by_task: dict[tuple[str, str], list[str]] = {}
        self._candidates: list[EvidenceCandidate] = []
        self._no_evidence: list[NoEvidenceRecord] = []
        self._tool_calls: list[dict[str, object]] = []

    def set_active_task(self, city: str, field: str) -> None:
        """Start a field-scoped research budget while preserving run-level IDs."""
        self._active_city = city
        self._active_field = field
        self._last_candidate_source_ids = []
        self._field_regex_search_count = 0
        self._hit_ids_by_task.setdefault(self._active_task_key(), [])
        self._expanded_hit_ids_by_task.setdefault(self._active_task_key(), [])

    def active_task(self) -> tuple[str, str]:
        """Return the city and field currently being researched."""
        return self._active_city, self._active_field

    def has_hits_for_active_task(self) -> bool:
        """Return whether the active city-field task has current-run regex hits."""
        if not self._active_city or not self._active_field:
            return False
        return bool(self._hit_ids_by_task.get(self._active_task_key(), []))

    def allowed_tool_names_for_active_task(self) -> list[str]:
        """Return tool names currently visible for the active city-field task."""
        tool_names = list(self._BASE_VISIBLE_TOOL_NAMES)
        if self.has_hits_for_active_task():
            tool_names.extend(self._HIT_VISIBLE_TOOL_NAMES)
        return tool_names

    def get_tag_options(self) -> TagOptions:
        """Tool: return available metadata filter values."""
        start = time.monotonic()
        options = self.registry.get_tag_options()
        self._record_tool_call("get_tag_options", {}, [], 0, start)
        return options

    def list_candidate_sources(
        self,
        cities: list[str] | None = None,
        countries: list[str] | None = None,
        verticals: list[str] | None = None,
        tef_sectors: list[str] | None = None,
        source_types: list[str] | None = None,
        publication_year_min: int | None = None,
        publication_year_max: int | None = None,
        max_files: int = 50,
    ) -> list[SourceSummary]:
        """Tool: select source files by metadata before text search."""
        start = time.monotonic()
        capped_max = min(max_files, self.limits.max_files_per_search)
        summaries = self.registry.list_candidate_sources(
            cities=cities,
            countries=countries,
            verticals=verticals,
            tef_sectors=tef_sectors,
            source_types=source_types,
            publication_year_min=publication_year_min,
            publication_year_max=publication_year_max,
            max_files=capped_max,
        )
        if self._active_field:
            summaries = _sort_summaries_for_field(summaries, self._active_field)
        self._last_candidate_source_ids = [summary.source_id for summary in summaries]
        self._record_tool_call(
            "list_candidate_sources",
            {
                "cities": cities,
                "countries": countries,
                "verticals": verticals,
                "tef_sectors": tef_sectors,
                "source_types": source_types,
                "publication_year_min": publication_year_min,
                "publication_year_max": publication_year_max,
                "max_files": capped_max,
            },
            self._last_candidate_source_ids,
            len(summaries),
            start,
        )
        return summaries

    def regex_search(
        self,
        pattern: str,
        cities: list[str] | None = None,
        countries: list[str] | None = None,
        verticals: list[str] | None = None,
        tef_sectors: list[str] | None = None,
        source_types: list[str] | None = None,
        case_sensitive: bool = False,
        context_words: int | None = None,
        context_lines: int | None = None,
        max_matches: int | None = None,
    ) -> list[SearchHit]:
        """Tool: run a validated regex over scoped external Markdown sources."""
        start = time.monotonic()
        if self._field_regex_search_count >= self.limits.max_regex_searches_per_field:
            raise ExternalSourceToolError(
                "SEARCH_LIMIT_EXCEEDED",
                "The regex search budget for this external-source session is exhausted.",
            )
        source_ids = self._resolve_source_ids(
            cities=cities,
            countries=countries,
            verticals=verticals,
            tef_sectors=tef_sectors,
            source_types=source_types,
        )
        regex = _compile_safe_pattern(pattern, case_sensitive, self.limits)
        capped_words = _cap_int(
            context_words,
            self.limits.default_context_words,
            self.limits.max_context_words,
        )
        capped_lines = _cap_int(
            context_lines,
            self.limits.default_context_lines,
            self.limits.max_context_lines,
        )
        capped_matches = _cap_int(
            max_matches,
            self.limits.max_matches_per_search,
            self.limits.max_matches_per_search,
        )

        self._search_counter += 1
        self._field_regex_search_count += 1
        search_id = f"s{self._search_counter}"
        hits: list[SearchHit] = []
        for source_id in source_ids:
            index = self.registry.get_index(source_id)
            for match in regex.finditer(index.text):
                if len(hits) >= capped_matches:
                    break
                self._hit_counter += 1
                hit_id = f"h{self._hit_counter}"
                hit = _build_search_hit(
                    search_id=search_id,
                    hit_id=hit_id,
                    index=index,
                    match=match,
                    context_words=capped_words,
                    context_lines=capped_lines,
                    max_snippet_chars=self.limits.max_snippet_chars,
                )
                self._hit_records[hit_id] = HitRecord(
                    hit_id=hit_id,
                    source_id=source_id,
                    match_start=match.start(),
                    match_end=match.end(),
                    pattern=pattern,
                    search_id=search_id,
                )
                self._remember_hit(hit_id)
                hits.append(hit)
            if len(hits) >= capped_matches:
                break

        self._record_tool_call(
            "regex_search",
            {
                "pattern": pattern,
                "cities": cities,
                "countries": countries,
                "verticals": verticals,
                "tef_sectors": tef_sectors,
                "source_types": source_types,
                "case_sensitive": case_sensitive,
                "context_words": capped_words,
                "context_lines": capped_lines,
                "max_matches": capped_matches,
            },
            source_ids,
            len(hits),
            start,
        )
        self._write_state()
        return hits

    def expand_hits(
        self,
        hit_ids: list[str],
        context_words: int | None = None,
        context_lines: int | None = None,
    ) -> list[SearchHit]:
        """Tool: expand up to three current-run hit snippets."""
        start = time.monotonic()
        if not hit_ids:
            raise ExternalSourceToolError("HIT_NOT_FOUND", "At least one hit_id is required.")
        if len(hit_ids) > self.limits.max_expand_hits_per_call:
            raise ExternalSourceToolError(
                "SEARCH_LIMIT_EXCEEDED",
                f"expand_hits accepts at most {self.limits.max_expand_hits_per_call} hit IDs.",
            )
        if len(set(hit_ids)) != len(hit_ids):
            raise ExternalSourceToolError("HIT_NOT_FOUND", "Duplicate hit_ids are not allowed.")

        capped_words = _cap_int(
            context_words,
            self.limits.max_context_words,
            self.limits.max_context_words,
        )
        capped_lines = _cap_int(
            context_lines,
            self.limits.max_context_lines,
            self.limits.max_context_lines,
        )
        expanded: list[SearchHit] = []
        source_ids: list[str] = []
        for hit_id in hit_ids:
            record = self._get_active_task_hit_record(hit_id)
            index = self.registry.get_index(record.source_id)
            source_ids.append(record.source_id)
            self._remember_expanded_hit(record.hit_id)
            expanded.append(
                _build_search_hit_from_record(
                    record,
                    index=index,
                    context_words=capped_words,
                    context_lines=capped_lines,
                    max_snippet_chars=self.limits.max_snippet_chars,
                )
            )

        self._record_tool_call(
            "expand_hits",
            {"hit_ids": hit_ids, "context_words": capped_words, "context_lines": capped_lines},
            source_ids,
            len(expanded),
            start,
        )
        self._write_state()
        return expanded

    def stage_expanded_hits_for_active_task(self) -> list[EvidenceCandidate]:
        """Save expanded or recent hits for the active task as fallback candidates."""
        city, field = self.active_task()
        hit_ids = self._expanded_hit_ids_by_task.get(self._active_task_key(), [])
        if not hit_ids:
            hit_ids = self._hit_ids_by_task.get(self._active_task_key(), [])[-3:]
        if not city or not field or not hit_ids:
            return []
        inputs = [
            EvidenceCandidateInput(
                hit_id=hit_id,
                city=city,
                field=field,
                reason=(
                    "Fallback candidate staged from a hit the external-source "
                    "researcher expanded before exhausting its turn budget."
                ),
                confidence=0.7,
            )
            for hit_id in hit_ids
        ]
        return self.add_evidence_candidates(inputs)

    def evidence_candidates_for_active_task(self) -> list[EvidenceCandidate]:
        """Return saved evidence candidates for the currently active city-field pair."""
        city, field = self.active_task()
        return [
            candidate
            for candidate in self._candidates
            if candidate.city.casefold() == city.casefold()
            and candidate.field.casefold() == field.casefold()
        ]

    def add_evidence_candidates(
        self,
        candidates: list[EvidenceCandidateInput],
    ) -> list[EvidenceCandidate]:
        """Tool: save selected hit snippets into the run evidence basket."""
        start = time.monotonic()
        saved: list[EvidenceCandidate] = []
        for candidate_input in candidates:
            record = self._get_active_task_hit_record(candidate_input.hit_id)
            hit = _build_search_hit_from_record(
                record,
                index=self.registry.get_index(record.source_id),
                context_words=self.limits.max_context_words,
                context_lines=self.limits.max_context_lines,
                max_snippet_chars=self.limits.max_snippet_chars,
            )
            source = self.registry.get_source(record.source_id)
            existing_index = _find_candidate_index(
                self._candidates,
                candidate_input.hit_id,
                candidate_input.field,
            )
            if existing_index is None:
                self._evidence_counter += 1
                candidate_id = f"e{self._evidence_counter}"
            else:
                candidate_id = self._candidates[existing_index].candidate_id
            candidate = EvidenceCandidate(
                candidate_id=candidate_id,
                hit_id=candidate_input.hit_id,
                source_id=hit.source_id,
                title=hit.title,
                city=candidate_input.city,
                field=candidate_input.field,
                matched_text=hit.matched_text,
                quote=hit.snippet,
                line_start=hit.line_start,
                line_end=hit.line_end,
                heading_path=hit.heading_path,
                confidence=max(0.0, min(float(candidate_input.confidence), 1.0)),
                reason=candidate_input.reason,
                source_type=source.source_type,
                publication_year=source.publication_year,
                source_url=source.source_url,
            )
            if existing_index is None:
                self._candidates.append(candidate)
            else:
                self._candidates[existing_index] = candidate
            saved.append(candidate)

        self._record_tool_call(
            "add_evidence_candidates",
            {"candidate_count": len(candidates)},
            sorted({candidate.source_id for candidate in saved}),
            len(saved),
            start,
        )
        self._write_state()
        return saved

    def list_evidence_candidates(self) -> list[EvidenceCandidate]:
        """Tool: return evidence already saved in the current run."""
        start = time.monotonic()
        self._record_tool_call("list_evidence_candidates", {}, [], len(self._candidates), start)
        return list(self._candidates)

    def mark_no_evidence_found(
        self,
        city: str,
        field: str,
        searched_source_ids: list[str],
        search_summary: str,
    ) -> NoEvidenceRecord:
        """Tool: record a searched field with no usable external evidence."""
        start = time.monotonic()
        for source_id in searched_source_ids:
            self.registry.get_source(source_id)
        self._no_evidence_counter += 1
        record = NoEvidenceRecord(
            record_id=f"n{self._no_evidence_counter}",
            city=city,
            field=field,
            searched_source_ids=searched_source_ids,
            search_summary=search_summary,
        )
        self._no_evidence.append(record)
        self._record_tool_call(
            "mark_no_evidence_found",
            {"city": city, "field": field, "search_summary": search_summary},
            searched_source_ids,
            0,
            start,
        )
        self._write_state()
        return record

    def evidence_candidates(self) -> list[EvidenceCandidate]:
        """Return saved evidence candidates for resolver validation."""
        return list(self._candidates)

    def no_evidence_records(self) -> list[NoEvidenceRecord]:
        """Return current no-evidence records."""
        return list(self._no_evidence)

    def tool_call_log(self) -> list[dict[str, object]]:
        """Return bounded audit records for every tool call in this session."""
        return list(self._tool_calls)

    def _resolve_source_ids(
        self,
        *,
        cities: list[str] | None,
        countries: list[str] | None,
        verticals: list[str] | None,
        tef_sectors: list[str] | None,
        source_types: list[str] | None,
    ) -> list[str]:
        """Resolve metadata filters into source IDs, intersecting prior candidates."""
        has_filters = any([cities, countries, verticals, tef_sectors, source_types])
        if not has_filters and not self._last_candidate_source_ids:
            raise ExternalSourceToolError(
                "SOURCE_SCOPE_REQUIRED",
                "regex_search requires a metadata filter or a prior candidate-source list.",
            )

        if has_filters:
            summaries = self.registry.list_candidate_sources(
                cities=cities,
                countries=countries,
                verticals=verticals,
                tef_sectors=tef_sectors,
                source_types=source_types,
                max_files=self.limits.max_files_per_search,
            )
            source_ids = [summary.source_id for summary in summaries]
            if self._last_candidate_source_ids:
                allowed = set(self._last_candidate_source_ids)
                source_ids = [source_id for source_id in source_ids if source_id in allowed]
                order = {
                    source_id: index
                    for index, source_id in enumerate(self._last_candidate_source_ids)
                }
                source_ids.sort(key=lambda source_id: order.get(source_id, len(order)))
            return source_ids[: self.limits.max_files_per_search]

        return self._last_candidate_source_ids[: self.limits.max_files_per_search]

    def _record_tool_call(
        self,
        tool: str,
        payload: dict[str, object],
        resolved_source_ids: list[str],
        hit_count: int,
        start_time: float,
    ) -> None:
        """Append a compact audit log entry for one tool call."""
        self._tool_calls.append(
            {
                "tool": tool,
                "run_id": self.run_id,
                "payload": payload,
                "resolved_source_ids": resolved_source_ids,
                "hit_count": hit_count,
                "elapsed_ms": round((time.monotonic() - start_time) * 1000, 2),
            }
        )

    def _active_task_key(self) -> tuple[str, str]:
        """Return the normalized active task key."""
        return self._active_city.casefold(), self._active_field.casefold()

    def _remember_expanded_hit(self, hit_id: str) -> None:
        """Track expanded hits so an over-budget agent can still be finalized."""
        task_hit_ids = self._expanded_hit_ids_by_task.setdefault(self._active_task_key(), [])
        if hit_id not in task_hit_ids:
            task_hit_ids.append(hit_id)

    def _remember_hit(self, hit_id: str) -> None:
        """Track regex hits so a premature no-evidence result can be finalized."""
        task_hit_ids = self._hit_ids_by_task.setdefault(self._active_task_key(), [])
        if hit_id not in task_hit_ids:
            task_hit_ids.append(hit_id)

    def _get_active_task_hit_record(self, hit_id: str) -> HitRecord:
        """Return a hit only when it belongs to the current active city-field task."""
        record = self._hit_records.get(hit_id)
        if record is None:
            raise ExternalSourceToolError("HIT_NOT_FOUND", f"Unknown hit_id: {hit_id}")
        if hit_id not in self._hit_ids_by_task.get(self._active_task_key(), []):
            raise ExternalSourceToolError(
                "HIT_NOT_FOUND",
                f"hit_id does not belong to the active city-field task: {hit_id}",
            )
        return record

    def _write_state(self) -> None:
        """Persist current session evidence and tool logs with an atomic replace."""
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "candidates": [candidate.model_dump(mode="json") for candidate in self._candidates],
            "no_evidence": [record.model_dump(mode="json") for record in self._no_evidence],
            "tool_calls": self._tool_calls,
        }
        target = self.artifact_dir / EXTERNAL_SOURCE_SEARCH_AUDIT_FILENAME
        tmp = self.artifact_dir / "external_source_search_audit.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)


def try_load_external_source_registry(root_dir: Path) -> SourceRegistry | None:
    """Load the source registry, returning None when it is missing or invalid."""
    try:
        registry = SourceRegistry.load(root_dir)
    except FileNotFoundError:
        logger.info("External source library skipped; sources.yaml not found under %s", root_dir)
        return None
    except (ExternalSourceToolError, ValidationError, yaml.YAMLError) as exc:
        logger.warning(
            "External source library skipped; invalid sources.yaml under %s: %s",
            root_dir,
            exc,
            exc_info=True,
        )
        return None
    if not registry.has_sources():
        logger.info("External source library skipped; no sources found under %s", root_dir)
        return None
    return registry


@dataclass(frozen=True)
class _ValidatedFilters:
    """Normalized metadata filters used internally for source matching."""

    cities: list[str]
    countries: list[str]
    verticals: list[str]
    tef_sectors: list[str]
    source_types: list[str]
    publication_year_min: int | None
    publication_year_max: int | None


def _resolve_source_paths(root: Path, sources: list[SourceMetadata]) -> dict[str, Path]:
    """Resolve every source_id to exactly one Markdown file by filename stem."""
    paths_by_stem: dict[str, list[Path]] = {}
    for path in root.rglob("*.md"):
        paths_by_stem.setdefault(path.stem, []).append(path)

    resolved: dict[str, Path] = {}
    seen_ids: set[str] = set()
    for source in sources:
        if source.source_id in seen_ids:
            raise ExternalSourceToolError(
                "SOURCE_NOT_FOUND",
                f"Duplicate source_id in sources.yaml: {source.source_id}",
            )
        seen_ids.add(source.source_id)
        matches = paths_by_stem.get(source.source_id, [])
        if not matches:
            raise ExternalSourceToolError(
                "SOURCE_NOT_FOUND",
                f"No Markdown file found for source_id `{source.source_id}`.",
            )
        if len(matches) > 1:
            raise ExternalSourceToolError(
                "SOURCE_NOT_FOUND",
                f"Multiple Markdown files found for source_id `{source.source_id}`.",
            )
        resolved[source.source_id] = matches[0]
    return resolved


def _build_document_index(source: SourceMetadata, path: Path, text: str) -> DocumentIndex:
    """Build line, heading, and word indexes for one Markdown file."""
    raw_lines = text.splitlines(keepends=True)
    display_lines = tuple(line.rstrip("\r\n") for line in raw_lines)
    offsets: list[int] = []
    offset = 0
    for raw_line in raw_lines:
        offsets.append(offset)
        offset += len(raw_line)
    if not offsets:
        offsets = [0]

    heading_paths: list[tuple[str, ...]] = []
    current_headings: list[str] = []
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    for line in display_lines:
        match = heading_pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            current_headings = current_headings[: level - 1]
            current_headings.append(title)
        heading_paths.append(tuple(current_headings))

    word_spans = tuple(
        WordSpan(match.start(), match.end()) for match in re.finditer(r"\S+", text)
    )
    return DocumentIndex(
        source=source,
        path=path,
        text=text,
        lines=display_lines,
        line_start_offsets=tuple(offsets),
        heading_paths_by_line=tuple(heading_paths),
        word_spans=word_spans,
    )


def _compile_safe_pattern(
    pattern: str,
    case_sensitive: bool,
    limits: ExternalSearchLimits,
) -> re.Pattern[str]:
    """Validate and compile the bounded MVP regex language."""
    cleaned = pattern.strip()
    if not cleaned:
        raise ExternalSourceToolError("REGEX_COMPILE_ERROR", "Regex pattern cannot be empty.")
    if len(cleaned) > limits.max_pattern_chars:
        raise ExternalSourceToolError(
            "REGEX_TOO_LONG",
            f"Regex pattern exceeds {limits.max_pattern_chars} characters.",
        )
    if cleaned.count("|") > 30:
        raise ExternalSourceToolError("REGEX_UNSAFE", "Regex pattern has too many alternatives.")
    if re.search(r"\\[1-9]", cleaned):
        raise ExternalSourceToolError("REGEX_UNSAFE", "Backreferences are not supported.")
    if "(?<=" in cleaned or "(?<!" in cleaned:
        raise ExternalSourceToolError("REGEX_UNSAFE", "Lookbehind is not supported.")
    if re.search(r"\((?:\.\*|\.\+|\.\{0,\d+\})\)[+*]", cleaned):
        raise ExternalSourceToolError("REGEX_UNSAFE", "Nested unbounded quantifiers are unsafe.")

    flags = re.MULTILINE | re.DOTALL
    if not case_sensitive:
        flags |= re.IGNORECASE
    try:
        return re.compile(cleaned, flags)
    except re.error as exc:
        raise ExternalSourceToolError("REGEX_COMPILE_ERROR", str(exc)) from exc


def _build_search_hit(
    *,
    search_id: str,
    hit_id: str,
    index: DocumentIndex,
    match: re.Match[str],
    context_words: int,
    context_lines: int,
    max_snippet_chars: int,
) -> SearchHit:
    """Build a public search-hit payload for one regex match."""
    return _build_search_hit_from_span(
        search_id=search_id,
        hit_id=hit_id,
        index=index,
        match_start=match.start(),
        match_end=match.end(),
        matched_text=match.group(0),
        context_words=context_words,
        context_lines=context_lines,
        max_snippet_chars=max_snippet_chars,
    )


def _build_search_hit_from_record(
    record: HitRecord,
    *,
    index: DocumentIndex,
    context_words: int,
    context_lines: int,
    max_snippet_chars: int,
) -> SearchHit:
    """Rebuild a search-hit payload from an internal hit cache record."""
    return _build_search_hit_from_span(
        search_id=record.search_id,
        hit_id=record.hit_id,
        index=index,
        match_start=record.match_start,
        match_end=record.match_end,
        matched_text=index.text[record.match_start : record.match_end],
        context_words=context_words,
        context_lines=context_lines,
        max_snippet_chars=max_snippet_chars,
    )


def _build_search_hit_from_span(
    *,
    search_id: str,
    hit_id: str,
    index: DocumentIndex,
    match_start: int,
    match_end: int,
    matched_text: str,
    context_words: int,
    context_lines: int,
    max_snippet_chars: int,
) -> SearchHit:
    """Create a bounded search-hit snippet around a character span."""
    line_start = _line_number_for_offset(index, match_start)
    line_end = _line_number_for_offset(index, max(match_end - 1, match_start))
    snippet_start, snippet_end = _context_char_range(
        index,
        match_start,
        match_end,
        line_start,
        line_end,
        context_words,
        context_lines,
    )
    raw_snippet = index.text[snippet_start:snippet_end]
    snippet = raw_snippet.strip()
    truncated = False
    if len(snippet) > max_snippet_chars:
        snippet = _center_truncated_snippet(
            raw_snippet=raw_snippet,
            match_start=match_start,
            match_end=match_end,
            snippet_start=snippet_start,
            max_snippet_chars=max_snippet_chars,
        )
        truncated = True
    heading_path = _heading_path_for_line(index, line_start)
    return SearchHit(
        search_id=search_id,
        hit_id=hit_id,
        source_id=index.source.source_id,
        title=index.source.title,
        city=index.source.city,
        line_start=max(1, line_start - context_lines),
        line_end=min(len(index.lines), line_end + context_lines),
        matched_text=matched_text.strip()[:500],
        snippet=snippet,
        heading_path=list(heading_path),
        truncated=truncated,
    )


def _center_truncated_snippet(
    *,
    raw_snippet: str,
    match_start: int,
    match_end: int,
    snippet_start: int,
    max_snippet_chars: int,
) -> str:
    """Truncate long snippets around the regex match instead of dropping it."""
    if max_snippet_chars <= 0:
        return ""
    relative_start = max(0, match_start - snippet_start)
    relative_end = max(relative_start, match_end - snippet_start)
    context_before = max_snippet_chars // 3
    window_start = max(0, relative_start - context_before)
    window_end = min(len(raw_snippet), window_start + max_snippet_chars)
    window_start = max(0, min(window_start, window_end - max_snippet_chars))

    body = raw_snippet[window_start:window_end].strip()
    prefix = "[...snippet truncated...]\n" if window_start > 0 else ""
    suffix = "\n[...snippet truncated...]" if window_end < len(raw_snippet) else ""
    if relative_start >= window_end or relative_end <= window_start:
        return raw_snippet[:max_snippet_chars].rstrip() + "\n[...snippet truncated...]"
    return f"{prefix}{body}{suffix}"


def _line_number_for_offset(index: DocumentIndex, offset: int) -> int:
    """Return one-based line number for a character offset."""
    position = bisect.bisect_right(index.line_start_offsets, max(0, offset))
    return max(1, min(position, len(index.lines)))


def _heading_path_for_line(index: DocumentIndex, line_number: int) -> tuple[str, ...]:
    """Return the heading path active at a one-based line number."""
    if not index.heading_paths_by_line:
        return ()
    line_index = max(0, min(line_number - 1, len(index.heading_paths_by_line) - 1))
    return index.heading_paths_by_line[line_index]


def _context_char_range(
    index: DocumentIndex,
    match_start: int,
    match_end: int,
    line_start: int,
    line_end: int,
    context_words: int,
    context_lines: int,
) -> tuple[int, int]:
    """Return a snippet range expanded by both word and line context."""
    line_context_start = max(1, line_start - context_lines)
    line_context_end = min(len(index.lines), line_end + context_lines)
    line_start_char = index.line_start_offsets[line_context_start - 1]
    if line_context_end < len(index.line_start_offsets):
        line_end_char = index.line_start_offsets[line_context_end]
    else:
        line_end_char = len(index.text)

    if not index.word_spans:
        return line_start_char, line_end_char

    starts = [span.start for span in index.word_spans]
    word_start_index = max(0, bisect.bisect_left(starts, match_start) - context_words)
    word_end_index = min(
        len(index.word_spans) - 1,
        bisect.bisect_right(starts, match_end) + context_words,
    )
    word_start_char = index.word_spans[word_start_index].start
    word_end_char = index.word_spans[word_end_index].end
    return min(line_start_char, word_start_char), max(line_end_char, word_end_char)


def _source_summary(source: SourceMetadata) -> SourceSummary:
    """Convert full source metadata into an LLM-safe summary."""
    return SourceSummary(
        source_id=source.source_id,
        title=source.title,
        city=source.city,
        country=source.country,
        publication_year=source.publication_year,
        source_type=source.source_type,
        verticals=source.verticals,
        tef_sectors=source.tef_sectors,
        description=source.description,
    )


def _validate_filter(
    name: str,
    values: list[str] | None,
    allowed_values: list[str],
) -> list[str]:
    """Validate filter values against known tag options and preserve display casing."""
    if not values:
        return []
    allowed_by_key = {value.casefold(): value for value in allowed_values}
    resolved: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in allowed_by_key:
            raise ExternalSourceToolError(
                "INVALID_FILTER",
                f"Unknown {name} filter: {value}",
                allowed_values,
            )
        resolved.append(allowed_by_key[key])
    return resolved


def _source_matches(source: SourceMetadata, filters: _ValidatedFilters) -> bool:
    """Return True when a source matches OR-within and AND-across filters."""
    if filters.cities and not _tag_matches(
        source.city,
        filters.cities,
        allow_broad=True,
        scope=source.geographic_scope,
    ):
        return False
    if filters.countries and not _tag_matches(
        source.country,
        filters.countries,
        allow_broad=True,
        scope=source.geographic_scope,
    ):
        return False
    if filters.verticals and not _tag_matches(source.verticals, filters.verticals):
        return False
    if filters.tef_sectors and not _tag_matches(source.tef_sectors, filters.tef_sectors):
        return False
    if filters.source_types and not _tag_matches([source.source_type], filters.source_types):
        return False
    if filters.publication_year_min is not None:
        if source.publication_year is None or source.publication_year < filters.publication_year_min:
            return False
    if filters.publication_year_max is not None:
        if source.publication_year is None or source.publication_year > filters.publication_year_max:
            return False
    return True


def _tag_matches(
    source_values: list[str],
    requested_values: list[str],
    *,
    allow_broad: bool = False,
    scope: str = "",
) -> bool:
    """Match metadata tags case-insensitively."""
    if allow_broad and not source_values and scope.casefold() in {"european", "global"}:
        return True
    source_keys = {value.casefold() for value in source_values}
    requested_keys = {value.casefold() for value in requested_values}
    return bool(source_keys & requested_keys)


def _source_sort_key(source: SourceMetadata, filters: _ValidatedFilters) -> tuple[int, int, int, str]:
    """Sort direct city sources first, then direct country, then newest year."""
    direct_city = 0
    if filters.cities:
        direct_city = 0 if _tag_matches(source.city, filters.cities) else 1
    direct_country = 0
    if filters.countries:
        direct_country = 0 if _tag_matches(source.country, filters.countries) else 1
    newest_year = -(source.publication_year or 0)
    return direct_city, direct_country, newest_year, source.title.casefold()


def _sort_summaries_for_field(
    summaries: list[SourceSummary],
    field: str,
) -> list[SourceSummary]:
    """Prefer sources whose metadata overlaps with the active field name."""
    field_tokens = _token_set(field)
    if not field_tokens:
        return summaries
    return sorted(
        summaries,
        key=lambda summary: (
            -len(field_tokens & _token_set(
                " ".join([summary.source_id, summary.title, summary.description])
            )),
            -(summary.publication_year or 0),
            summary.title.casefold(),
        ),
    )


def _token_set(value: str) -> set[str]:
    """Tokenize field/source text for generic source-priority matching."""
    return {token for token in re.split(r"[^a-z0-9]+", value.casefold()) if len(token) > 2}


def _find_candidate_index(
    candidates: list[EvidenceCandidate],
    hit_id: str,
    field: str,
) -> int | None:
    """Find an existing candidate by hit and field."""
    for index, candidate in enumerate(candidates):
        if candidate.hit_id == hit_id and candidate.field.casefold() == field.casefold():
            return index
    return None


def _cap_int(value: int | None, default: int, maximum: int) -> int:
    """Clamp optional integer tool parameters to a configured positive cap."""
    if value is None:
        return max(0, default)
    return max(0, min(int(value), maximum))


def _flatten(values: Any) -> list[str]:
    """Flatten an iterable of string iterables into a list of strings."""
    flattened: list[str] = []
    for item in values:
        if isinstance(item, str):
            flattened.append(item)
            continue
        flattened.extend(str(value) for value in item if isinstance(value, str))
    return flattened


def _sorted_distinct(values: Any) -> list[str]:
    """Return case-insensitive distinct string values in display order."""
    by_key: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned:
            by_key.setdefault(cleaned.casefold(), cleaned)
    return [by_key[key] for key in sorted(by_key)]


__all__ = [
    "EXTERNAL_SOURCE_SEARCH_AUDIT_FILENAME",
    "ExternalSearchLimits",
    "ExternalSearchSession",
    "ExternalSourceToolError",
    "SourceRegistry",
    "build_external_search_limits",
    "try_load_external_source_registry",
]
