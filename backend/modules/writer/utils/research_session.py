"""Controlled writer context search tools for the research curator."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast, get_args

from backend.modules.orchestrator.utils.references import REF_ID_PATTERN
from backend.modules.writer.models import (
    WriterContextItem,
    WriterContextSearchHit,
    WriterContextSourceKind,
    WriterContextSourceSummary,
    WriterMissingEvidenceRecord,
    WriterSavedEvidence,
)
from backend.utils.config import AppConfig
from backend.utils.json_io import write_json

_SOURCE_KIND_VALUES = set(get_args(WriterContextSourceKind))


class WriterResearchToolError(ValueError):
    """Structured error returned by writer research tools."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, object]:
        """Return a compact tool error payload."""
        return {"error": {"code": self.code, "message": self.message}}


@dataclass(frozen=True)
class WriterResearchLimits:
    """Caps applied to one writer research-curator session."""

    max_saved_items: int
    max_regex_searches: int
    max_matches_per_search: int
    default_context_words: int
    max_context_words: int
    max_snippet_chars: int
    max_pattern_chars: int


@dataclass(frozen=True)
class _HitRecord:
    """Internal hit cache for expansion and evidence saving."""

    hit_id: str
    item_id: str
    match_start: int
    match_end: int
    pattern: str
    search_id: str


def build_writer_research_limits(config: AppConfig) -> WriterResearchLimits:
    """Read writer research-curator caps from config."""
    writer = config.writer
    return WriterResearchLimits(
        max_saved_items=writer.evidence_curator_max_saved_items,
        max_regex_searches=writer.evidence_curator_max_regex_searches,
        max_matches_per_search=writer.evidence_curator_max_matches_per_search,
        default_context_words=writer.evidence_curator_default_context_words,
        max_context_words=writer.evidence_curator_max_context_words,
        max_snippet_chars=writer.evidence_curator_max_snippet_chars,
        max_pattern_chars=writer.evidence_curator_max_pattern_chars,
    )


class WriterResearchSession:
    """Per-run controlled search session over writer-visible context."""

    def __init__(
        self,
        *,
        run_id: str,
        items: list[WriterContextItem],
        limits: WriterResearchLimits,
        artifact_dir: Path | None = None,
        initial_missing_records: list[WriterMissingEvidenceRecord] | None = None,
    ) -> None:
        self.run_id = run_id
        self.items = items
        self.limits = limits
        self.artifact_dir = artifact_dir
        self._items_by_id = {item.item_id: item for item in items}
        self._search_counter = 0
        self._hit_counter = 0
        self._saved_counter = 0
        self._missing_counter = 0
        self._next_ref_index = _next_ref_index(items)
        self._hit_records: dict[str, _HitRecord] = {}
        self._saved_by_item_id: dict[str, WriterSavedEvidence] = {}
        self._saved_evidence: list[WriterSavedEvidence] = []
        self._missing_records = list(initial_missing_records or [])
        self._tool_calls: list[dict[str, object]] = []

    def source_summaries(self) -> list[WriterContextSourceSummary]:
        """Return source summaries for all indexed context."""
        return _summarize_sources(self.items)

    def saved_evidence(self) -> list[WriterSavedEvidence]:
        """Return saved evidence records in capture order."""
        return list(self._saved_evidence)

    def missing_records(self) -> list[WriterMissingEvidenceRecord]:
        """Return missing-evidence records in capture order."""
        return list(self._missing_records)

    def list_context_sources(
        self,
        cities: list[str] | None = None,
        source_kinds: list[str] | None = None,
        fields: list[str] | None = None,
    ) -> list[WriterContextSourceSummary]:
        """Tool: summarize context sources matching optional filters."""
        start = time.monotonic()
        filtered = self._filter_items(
            cities=cities,
            source_kinds=source_kinds,
            fields=fields,
        )
        summaries = _summarize_sources(filtered)
        self._record_tool_call(
            "list_context_sources",
            {"cities": cities, "source_kinds": source_kinds, "fields": fields},
            len(summaries),
            start,
        )
        self._write_state()
        return summaries

    def regex_search_context(
        self,
        pattern: str,
        cities: list[str] | None = None,
        source_kinds: list[str] | None = None,
        fields: list[str] | None = None,
        case_sensitive: bool = False,
        context_words: int | None = None,
        max_matches: int | None = None,
    ) -> list[WriterContextSearchHit]:
        """Tool: run a safe regex over writer-visible context."""
        start = time.monotonic()
        if self._search_counter >= self.limits.max_regex_searches:
            raise WriterResearchToolError(
                "SEARCH_LIMIT_EXCEEDED",
                "The writer research regex search budget is exhausted.",
            )
        regex = _compile_safe_pattern(pattern, case_sensitive, self.limits)
        capped_words = _cap_int(
            context_words,
            self.limits.default_context_words,
            self.limits.max_context_words,
        )
        capped_matches = _cap_int(
            max_matches,
            self.limits.max_matches_per_search,
            self.limits.max_matches_per_search,
        )
        filtered = self._filter_items(
            cities=cities,
            source_kinds=source_kinds,
            fields=fields,
        )

        self._search_counter += 1
        search_id = f"s{self._search_counter}"
        hits: list[WriterContextSearchHit] = []
        for item in filtered:
            for match in regex.finditer(item.text):
                if len(hits) >= capped_matches:
                    break
                self._hit_counter += 1
                hit_id = f"h{self._hit_counter}"
                record = _HitRecord(
                    hit_id=hit_id,
                    item_id=item.item_id,
                    match_start=match.start(),
                    match_end=match.end(),
                    pattern=pattern,
                    search_id=search_id,
                )
                self._hit_records[hit_id] = record
                hits.append(
                    _build_search_hit(
                        record=record,
                        item=item,
                        matched_text=match.group(0),
                        context_words=capped_words,
                        max_snippet_chars=self.limits.max_snippet_chars,
                    )
                )
            if len(hits) >= capped_matches:
                break

        self._record_tool_call(
            "regex_search_context",
            {
                "pattern": pattern,
                "cities": cities,
                "source_kinds": source_kinds,
                "fields": fields,
                "case_sensitive": case_sensitive,
                "context_words": capped_words,
                "max_matches": capped_matches,
            },
            len(hits),
            start,
        )
        self._write_state()
        return hits

    def expand_context_hits(
        self,
        hit_ids: list[str],
        context_words: int | None = None,
    ) -> list[WriterContextSearchHit]:
        """Tool: expand previously returned search hits."""
        start = time.monotonic()
        if not hit_ids:
            raise WriterResearchToolError("HIT_NOT_FOUND", "At least one hit_id is required.")
        if len(set(hit_ids)) != len(hit_ids):
            raise WriterResearchToolError("HIT_NOT_FOUND", "Duplicate hit_ids are not allowed.")
        capped_words = _cap_int(
            context_words,
            self.limits.max_context_words,
            self.limits.max_context_words,
        )
        hits: list[WriterContextSearchHit] = []
        for hit_id in hit_ids:
            record = self._get_hit_record(hit_id)
            item = self._items_by_id[record.item_id]
            matched_text = item.text[record.match_start:record.match_end]
            hits.append(
                _build_search_hit(
                    record=record,
                    item=item,
                    matched_text=matched_text,
                    context_words=capped_words,
                    max_snippet_chars=self.limits.max_snippet_chars,
                )
            )
        self._record_tool_call(
            "expand_context_hits",
            {"hit_ids": hit_ids, "context_words": capped_words},
            len(hits),
            start,
        )
        self._write_state()
        return hits

    def save_context_evidence(
        self,
        hit_ids: list[str],
        reason: str,
    ) -> list[WriterSavedEvidence]:
        """Tool: save useful context hits as writer evidence."""
        start = time.monotonic()
        if not hit_ids:
            raise WriterResearchToolError("HIT_NOT_FOUND", "At least one hit_id is required.")
        saved: list[WriterSavedEvidence] = []
        for hit_id in hit_ids:
            if len(self._saved_evidence) >= self.limits.max_saved_items:
                raise WriterResearchToolError(
                    "SAVE_LIMIT_EXCEEDED",
                    "The writer research saved-evidence budget is exhausted.",
                )
            record = self._get_hit_record(hit_id)
            item = self._items_by_id[record.item_id]
            existing = self._saved_by_item_id.get(item.item_id)
            if existing is not None:
                saved.append(existing)
                continue
            self._saved_counter += 1
            evidence = _build_saved_evidence(
                saved_id=f"ws_{self._saved_counter}",
                ref_id=self._allocate_ref_id(item),
                item=item,
                hit=_build_search_hit(
                    record=record,
                    item=item,
                    matched_text=item.text[record.match_start:record.match_end],
                    context_words=self.limits.max_context_words,
                    max_snippet_chars=self.limits.max_snippet_chars,
                ),
                reason=reason,
            )
            self._saved_evidence.append(evidence)
            self._saved_by_item_id[item.item_id] = evidence
            saved.append(evidence)
        self._record_tool_call(
            "save_context_evidence",
            {"hit_ids": hit_ids, "reason": reason},
            len(saved),
            start,
        )
        self._write_state()
        return saved

    def list_saved_context_evidence(self) -> list[WriterSavedEvidence]:
        """Tool: list evidence saved so far."""
        start = time.monotonic()
        saved = self.saved_evidence()
        self._record_tool_call("list_saved_context_evidence", {}, len(saved), start)
        self._write_state()
        return saved

    def mark_context_evidence_missing(
        self,
        reason: str,
        city_name: str = "",
        field: str = "",
        source_kind: str | None = None,
        searched_patterns: list[str] | None = None,
    ) -> WriterMissingEvidenceRecord:
        """Tool: record a searched-but-missing evidence need."""
        start = time.monotonic()
        resolved_source_kind = _coerce_source_kind(source_kind) if source_kind else None
        self._missing_counter += 1
        record = WriterMissingEvidenceRecord(
            missing_id=f"wm_{self._missing_counter}",
            city_name=city_name.strip(),
            city_key=_normalize_key(city_name),
            source_kind=resolved_source_kind,
            field=field.strip(),
            reason=reason.strip() or "Evidence was not found in writer-visible context.",
            searched_patterns=[
                pattern.strip()
                for pattern in (searched_patterns or [])
                if isinstance(pattern, str) and pattern.strip()
            ],
        )
        self._missing_records.append(record)
        self._record_tool_call(
            "mark_context_evidence_missing",
            {
                "city_name": city_name,
                "field": field,
                "source_kind": source_kind,
                "searched_patterns": searched_patterns,
            },
            1,
            start,
        )
        self._write_state()
        return record

    def saved_evidence_payload(self) -> dict[str, object]:
        """Return the persisted saved-evidence artifact payload."""
        source_kind_counts = Counter(evidence.source_kind for evidence in self._saved_evidence)
        covered_cities = sorted(
            {
                evidence.city_name
                for evidence in self._saved_evidence
                if evidence.city_name
            }
        )
        return {
            "run_id": self.run_id,
            "saved_count": len(self._saved_evidence),
            "covered_cities": covered_cities,
            "source_kind_counts": dict(sorted(source_kind_counts.items())),
            "saved_evidence": [
                evidence.model_dump() for evidence in self._saved_evidence
            ],
            "missing_records": [
                record.model_dump() for record in self._missing_records
            ],
        }

    def workspace_payload(self) -> dict[str, object]:
        """Return the persisted research-workspace artifact payload."""
        return {
            "run_id": self.run_id,
            "context_item_count": len(self.items),
            "context_sources": [
                summary.model_dump() for summary in self.source_summaries()
            ],
            "limits": {
                "max_saved_items": self.limits.max_saved_items,
                "max_regex_searches": self.limits.max_regex_searches,
                "max_matches_per_search": self.limits.max_matches_per_search,
                "default_context_words": self.limits.default_context_words,
                "max_context_words": self.limits.max_context_words,
            },
            "tool_calls": self._tool_calls,
            "saved_count": len(self._saved_evidence),
            "missing_records": [
                record.model_dump() for record in self._missing_records
            ],
        }

    def write_state(self) -> None:
        """Persist current workspace artifacts when an artifact directory exists."""
        self._write_state()

    def _filter_items(
        self,
        *,
        cities: list[str] | None,
        source_kinds: list[str] | None,
        fields: list[str] | None,
    ) -> list[WriterContextItem]:
        """Return context items matching optional tool filters."""
        city_keys = {_normalize_key(city) for city in cities or [] if city.strip()}
        kind_values = {_coerce_source_kind(kind) for kind in source_kinds or []}
        field_values = {
            field.strip().casefold()
            for field in fields or []
            if isinstance(field, str) and field.strip()
        }
        return [
            item
            for item in self.items
            if (not city_keys or item.city_key in city_keys)
            and (not kind_values or item.source_kind in kind_values)
            and (
                not field_values
                or item.field.casefold() in field_values
                or any(field in item.text.casefold() for field in field_values)
            )
        ]

    def _get_hit_record(self, hit_id: str) -> _HitRecord:
        """Return one known hit record or raise a tool error."""
        record = self._hit_records.get(hit_id)
        if record is None:
            raise WriterResearchToolError("HIT_NOT_FOUND", f"Unknown hit_id: {hit_id}")
        return record

    def _allocate_ref_id(self, item: WriterContextItem) -> str:
        """Return existing CCC excerpt ref or allocate a writer ref."""
        if item.source_kind == "ccc_excerpt" and REF_ID_PATTERN.fullmatch(item.ref_id):
            return item.ref_id
        ref_id = f"ref_{self._next_ref_index}"
        self._next_ref_index += 1
        return ref_id

    def _record_tool_call(
        self,
        tool_name: str,
        args: dict[str, object],
        result_count: int,
        start: float,
    ) -> None:
        """Record compact tool-call diagnostics."""
        self._tool_calls.append(
            {
                "tool": tool_name,
                "args": args,
                "result_count": result_count,
                "elapsed_seconds": round(time.monotonic() - start, 6),
            }
        )

    def _write_state(self) -> None:
        """Persist workspace and saved-evidence artifacts when configured."""
        if self.artifact_dir is None:
            return
        write_json(
            self.artifact_dir / "evidence_workspace.json",
            self.workspace_payload(),
            ensure_ascii=False,
        )
        write_json(
            self.artifact_dir / "saved_evidence.json",
            self.saved_evidence_payload(),
            ensure_ascii=False,
        )


def _summarize_sources(items: list[WriterContextItem]) -> list[WriterContextSourceSummary]:
    """Group context items by source kind."""
    by_kind: dict[WriterContextSourceKind, list[WriterContextItem]] = {}
    for item in items:
        by_kind.setdefault(item.source_kind, []).append(item)
    summaries: list[WriterContextSourceSummary] = []
    for source_kind, group in sorted(by_kind.items()):
        summaries.append(
            WriterContextSourceSummary(
                source_kind=source_kind,
                count=len(group),
                cities=sorted({item.city_name for item in group if item.city_name}),
                fields=sorted({item.field for item in group if item.field}),
            )
        )
    return summaries


def _compile_safe_pattern(
    pattern: str,
    case_sensitive: bool,
    limits: WriterResearchLimits,
) -> re.Pattern[str]:
    """Validate and compile the bounded writer regex language."""
    cleaned = pattern.strip()
    if not cleaned:
        raise WriterResearchToolError("REGEX_COMPILE_ERROR", "Regex pattern cannot be empty.")
    if len(cleaned) > limits.max_pattern_chars:
        raise WriterResearchToolError(
            "REGEX_TOO_LONG",
            f"Regex pattern exceeds {limits.max_pattern_chars} characters.",
        )
    if cleaned.count("|") > 30:
        raise WriterResearchToolError("REGEX_UNSAFE", "Regex pattern has too many alternatives.")
    if re.search(r"\\[1-9]", cleaned):
        raise WriterResearchToolError("REGEX_UNSAFE", "Backreferences are not supported.")
    if "(?<=" in cleaned or "(?<!" in cleaned:
        raise WriterResearchToolError("REGEX_UNSAFE", "Lookbehind is not supported.")
    if re.search(r"\((?:\.\*|\.\+|\.\{0,\d+\})\)[+*]", cleaned):
        raise WriterResearchToolError("REGEX_UNSAFE", "Nested unbounded quantifiers are unsafe.")
    flags = re.MULTILINE | re.DOTALL
    if not case_sensitive:
        flags |= re.IGNORECASE
    try:
        return re.compile(cleaned, flags)
    except re.error as exc:
        raise WriterResearchToolError("REGEX_COMPILE_ERROR", str(exc)) from exc


def _build_search_hit(
    *,
    record: _HitRecord,
    item: WriterContextItem,
    matched_text: str,
    context_words: int,
    max_snippet_chars: int,
) -> WriterContextSearchHit:
    """Build a public hit payload from a cached record."""
    snippet = _build_snippet(
        item.text,
        record.match_start,
        record.match_end,
        context_words,
        max_snippet_chars,
    )
    return WriterContextSearchHit(
        search_id=record.search_id,
        hit_id=record.hit_id,
        item_id=item.item_id,
        source_kind=item.source_kind,
        city_name=item.city_name,
        city_key=item.city_key,
        source_id=item.source_id,
        ref_id=item.ref_id,
        field=item.field,
        matched_text=matched_text,
        snippet=snippet,
        line_start=item.line_start,
        line_end=item.line_end,
    )


def _build_saved_evidence(
    *,
    saved_id: str,
    ref_id: str,
    item: WriterContextItem,
    hit: WriterContextSearchHit,
    reason: str,
) -> WriterSavedEvidence:
    """Convert one search hit into saved writer evidence."""
    metadata = dict(item.metadata)
    metadata["hit_id"] = hit.hit_id
    metadata["matched_text"] = hit.matched_text
    return WriterSavedEvidence(
        saved_id=saved_id,
        ref_id=ref_id,
        item_id=item.item_id,
        source_kind=item.source_kind,
        city_name=item.city_name,
        city_key=item.city_key,
        source_id=item.source_id,
        field=item.field,
        quote=item.quote or hit.snippet,
        text=hit.snippet,
        reason=reason.strip(),
        line_start=item.line_start,
        line_end=item.line_end,
        metadata=metadata,
    )


def _build_snippet(
    text: str,
    match_start: int,
    match_end: int,
    context_words: int,
    max_chars: int,
) -> str:
    """Return a word-bounded snippet around one match."""
    spans = list(re.finditer(r"\S+", text))
    if not spans:
        return text[:max_chars].strip()
    match_word_indexes = [
        index
        for index, span in enumerate(spans)
        if span.end() >= match_start and span.start() <= match_end
    ]
    if not match_word_indexes:
        return text[max(0, match_start - max_chars // 2):match_end + max_chars // 2].strip()
    first = max(match_word_indexes[0] - context_words, 0)
    last = min(match_word_indexes[-1] + context_words + 1, len(spans))
    snippet = text[spans[first].start():spans[last - 1].end()].strip()
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max(max_chars - 1, 0)].rstrip() + "..."


def _next_ref_index(items: list[WriterContextItem]) -> int:
    """Return the next available numeric ref suffix after baseline refs."""
    max_index = 0
    for item in items:
        match = REF_ID_PATTERN.fullmatch(item.ref_id)
        if match:
            max_index = max(max_index, int(match.group(0).split("_", 1)[1]))
    return max_index + 1


def _coerce_source_kind(value: str | None) -> WriterContextSourceKind:
    """Validate a source-kind filter."""
    candidate = (value or "").strip()
    if candidate not in _SOURCE_KIND_VALUES:
        raise WriterResearchToolError(
            "SOURCE_KIND_NOT_FOUND",
            f"Unknown writer context source kind: {value}",
        )
    return cast(WriterContextSourceKind, candidate)


def _cap_int(value: int | None, default: int, limit: int) -> int:
    """Apply a positive integer cap."""
    if value is None:
        return max(1, default)
    return max(1, min(value, limit))


def _normalize_key(value: str) -> str:
    """Normalize simple city/filter keys for matching."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


__all__ = [
    "WriterResearchLimits",
    "WriterResearchSession",
    "WriterResearchToolError",
    "build_writer_research_limits",
]
