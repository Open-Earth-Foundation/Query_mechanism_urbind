"""Candidate normalization, record ids, and duplicate merge helpers."""

from __future__ import annotations

import re
from hashlib import sha1
from pathlib import Path

from backend.modules.initiative_extractor.models import (
    InitiativeDocumentSegment,
    InitiativeExtractionCandidate,
    InitiativeExtractionRecord,
    InitiativeRawSegmentResult,
    InitiativeReviewItem,
    InitiativeSemanticDedupeGroup,
    InitiativeSourceRef,
    JsonValue,
)
from backend.modules.initiative_extractor.output_parser import (
    CITY_OVERRIDDEN_FLAG,
    _clean_source_quote,
)
from backend.modules.initiative_extractor.segmentation import (
    detect_source_quality_flags,
)
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import AppConfig

SOURCE_QUOTE_FLAGS = {"source_quote_missing", "source_quote_not_found"}
LOCAL_CODE_PATTERN = re.compile(
    r"\b([A-Z]{1,6}(?:[-.][A-Z0-9]{1,6})?[-.]\d+(?:[.-]\d+)*[A-Z]?)\b"
)


def _default_source_ref(segment: InitiativeDocumentSegment) -> InitiativeSourceRef:
    """Build the canonical source ref from segment metadata."""
    return InitiativeSourceRef(
        source_document=segment.source_document,
        segment_id=segment.segment_id,
        start_line=segment.start_line,
        end_line=segment.end_line,
        section_heading=segment.heading_path,
    )


def _infer_document_local_code(
    candidate: InitiativeExtractionCandidate,
) -> str | None:
    """Infer a source-local action code from the initiative name or quote when possible."""
    if candidate.document_local_code:
        return candidate.document_local_code

    for value in (candidate.source_quote, candidate.initiative.initiative_name):
        if not value:
            continue
        match = LOCAL_CODE_PATTERN.search(value)
        if match:
            return match.group(1)
    return None


def _normalize_candidate(
    candidate: InitiativeExtractionCandidate,
    segment: InitiativeDocumentSegment,
) -> InitiativeExtractionCandidate:
    """Assign segment metadata and validate the quote-only citation."""
    source_refs = [_default_source_ref(segment)]
    source_quote = _clean_source_quote(candidate.source_quote)
    quote_flags: list[str] = []
    initiative = candidate.initiative
    if source_quote is None:
        quote_flags.append("source_quote_missing")
    elif source_quote not in segment.content:
        source_quote = None
        quote_flags.append("source_quote_not_found")
    if initiative.city != segment.city_name:
        initiative = initiative.model_copy(update={"city": segment.city_name})
        quote_flags.append(CITY_OVERRIDDEN_FLAG)
    flags = list(
        dict.fromkeys(
            [
                *candidate.data_quality_flags,
                *quote_flags,
                *detect_source_quality_flags(segment.content),
            ]
        )
    )
    return candidate.model_copy(
        update={
            "initiative": initiative,
            "document_local_code": _infer_document_local_code(candidate),
            "source_quote": source_quote,
            "source_refs": source_refs,
            "data_quality_flags": flags,
        },
        deep=True,
    )


def _normalize_title(value: str) -> str:
    """Normalize initiative titles for deduplication."""
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _record_id_for(
    candidate: InitiativeExtractionCandidate,
    source_document: str,
    segment_id: str,
    candidate_index: int,
) -> str:
    """Build a deterministic record id for one candidate."""
    city_key = normalize_city_key(candidate.initiative.city) or "unknown_city"
    doc_slug = normalize_city_key(Path(source_document).stem) or "document"
    occurrence_hash = sha1(  # noqa: S324
        f"{source_document}|{segment_id}|{candidate_index}".encode("utf-8")
    ).hexdigest()[:8]
    local_code = candidate.document_local_code
    if local_code:
        local_part = normalize_city_key(local_code) or local_code.casefold()
        return f"{city_key}:{doc_slug}:{local_part}:{occurrence_hash}"
    title_hash = sha1(  # noqa: S324
        f"{city_key}|{doc_slug}|{_normalize_title(candidate.initiative.initiative_name)}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{city_key}:{doc_slug}:title_{title_hash}:{occurrence_hash}"


def _dedupe_key(
    candidate: InitiativeExtractionCandidate, source_document: str
) -> tuple[str, str, str]:
    """Build a dedupe key using local code when available, otherwise title."""
    city_key = normalize_city_key(candidate.initiative.city)
    doc_key = normalize_city_key(Path(source_document).stem)
    if candidate.document_local_code:
        return (city_key, doc_key, candidate.document_local_code.casefold().strip())
    return (city_key, doc_key, _normalize_title(candidate.initiative.initiative_name))


def _candidate_source_document(
    candidate: InitiativeExtractionCandidate,
    fallback: str,
) -> str:
    """Return the best source document available for one candidate."""
    if candidate.source_refs:
        return candidate.source_refs[0].source_document
    return fallback


def _extend_prior_initiatives(
    prior_initiatives: list[InitiativeExtractionCandidate],
    raw_results: list[InitiativeRawSegmentResult],
) -> None:
    """Add newly extracted initiatives to the rolling canonical history."""
    seen_keys = {
        _dedupe_key(candidate, _candidate_source_document(candidate, ""))
        for candidate in prior_initiatives
    }
    for raw in raw_results:
        if raw.status != "success":
            continue
        for candidate in raw.initiatives:
            key = _dedupe_key(candidate, raw.source_document)
            if key in seen_keys:
                continue
            prior_initiatives.append(candidate)
            seen_keys.add(key)


def _merge_dicts(
    base: dict[str, JsonValue],
    extra: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Merge numeric dictionaries while preserving existing values."""
    merged = dict(base)
    for key, value in extra.items():
        merged.setdefault(key, value)
    return merged


def _source_quote_score(value: str | None) -> int:
    """Score a quote by useful word count, capped to avoid favoring huge excerpts."""
    if not value:
        return 0
    return min(len(value.split()), 80)


def _choose_source_quote(base: str | None, extra: str | None) -> str | None:
    """Choose the clearest available source quote during duplicate merges."""
    if _source_quote_score(extra) > _source_quote_score(base):
        return extra
    return base


def _merge_record(
    base: InitiativeExtractionRecord,
    extra: InitiativeExtractionCandidate,
) -> InitiativeExtractionRecord:
    """Merge duplicate candidate metadata into an existing record."""
    initiative = base.initiative.model_copy(deep=True)
    for field_name in (
        "general_description",
        "objective_text",
        "implementation_text",
        "planned_outputs_text",
        "delivery_text",
        "funding_text",
        "timeline_text",
    ):
        if not getattr(initiative, field_name) and getattr(
            extra.initiative, field_name
        ):
            setattr(initiative, field_name, getattr(extra.initiative, field_name))
    initiative.numbers.current = _merge_dicts(
        initiative.numbers.current,
        extra.initiative.numbers.current,
    )
    initiative.numbers.planned = _merge_dicts(
        initiative.numbers.planned,
        extra.initiative.numbers.planned,
    )
    return base.model_copy(
        update={
            "initiative": initiative,
            "source_quote": _choose_source_quote(base.source_quote, extra.source_quote),
            "source_refs": [*base.source_refs, *extra.source_refs],
            "document_local_code": base.document_local_code
            or extra.document_local_code,
            "data_quality_flags": list(
                dict.fromkeys([*base.data_quality_flags, *extra.data_quality_flags])
            ),
            "number_context": _merge_dicts(base.number_context, extra.number_context),
            "number_deferred": _merge_dicts(
                base.number_deferred, extra.number_deferred
            ),
            "number_uncertain": _merge_dicts(
                base.number_uncertain, extra.number_uncertain
            ),
            "extraction_notes": list(
                dict.fromkeys([*base.extraction_notes, *extra.extraction_notes])
            ),
        },
        deep=True,
    )


def _apply_semantic_dedupe_groups(
    records: list[InitiativeExtractionRecord],
    groups: list[InitiativeSemanticDedupeGroup],
    config: AppConfig,
) -> tuple[list[InitiativeExtractionRecord], list[InitiativeReviewItem]]:
    """Merge records using accepted semantic duplicate groups."""
    records_by_id = {record.record_id: record for record in records}
    parent = {record.record_id: record.record_id for record in records}
    threshold = config.initiative_extractor.semantic_dedupe_confidence_threshold
    review_items: list[InitiativeReviewItem] = []

    def find(record_id: str) -> str:
        while parent[record_id] != record_id:
            parent[record_id] = parent[parent[record_id]]
            record_id = parent[record_id]
        return record_id

    for group in groups:
        canonical_id = group.canonical_record_id
        if group.confidence < threshold:
            continue
        if canonical_id not in records_by_id:
            review_items.append(
                InitiativeReviewItem(
                    review_type="semantic_dedupe_invalid_record_id",
                    message="Semantic dedupe returned an unknown canonical record id.",
                    record_id=canonical_id,
                    details={
                        "confidence": group.confidence,
                        "rationale": group.rationale,
                    },
                )
            )
            continue
        canonical_root = find(canonical_id)
        for duplicate_id in group.duplicate_record_ids:
            if duplicate_id == canonical_id:
                continue
            if duplicate_id not in records_by_id:
                review_items.append(
                    InitiativeReviewItem(
                        review_type="semantic_dedupe_invalid_record_id",
                        message="Semantic dedupe returned an unknown duplicate record id.",
                        record_id=canonical_id,
                        details={
                            "duplicate_record_id": duplicate_id,
                            "confidence": group.confidence,
                            "rationale": group.rationale,
                        },
                    )
                )
                continue
            parent[find(duplicate_id)] = canonical_root
            review_items.append(
                InitiativeReviewItem(
                    review_type="semantic_duplicate_merged",
                    severity="info",
                    message="Semantic dedupe merged two records that describe the same initiative.",
                    source_document=records_by_id[duplicate_id].source_document,
                    record_id=canonical_id,
                    details={
                        "duplicate_record_id": duplicate_id,
                        "confidence": group.confidence,
                        "rationale": group.rationale,
                    },
                )
            )

    grouped_ids: dict[str, list[str]] = {}
    for record_id in records_by_id:
        grouped_ids.setdefault(find(record_id), []).append(record_id)

    merged_records: list[InitiativeExtractionRecord] = []
    for record in records:
        root_id = find(record.record_id)
        if record.record_id != root_id:
            continue
        merged_record = record
        for duplicate_id in grouped_ids[root_id]:
            if duplicate_id == root_id:
                continue
            merged_record = _merge_record(merged_record, records_by_id[duplicate_id])
        merged_records.append(merged_record)
    return merged_records, review_items


def _build_candidate_records(
    raw_results: list[InitiativeRawSegmentResult],
) -> list[InitiativeExtractionRecord]:
    """Convert raw candidates into stable initiative records without merging."""
    records: list[InitiativeExtractionRecord] = []
    for raw in raw_results:
        for candidate_index, candidate in enumerate(raw.initiatives, start=1):
            records.append(
                InitiativeExtractionRecord(
                    initiative=candidate.initiative,
                    document_local_code=candidate.document_local_code,
                    source_quote=candidate.source_quote,
                    source_refs=candidate.source_refs,
                    data_quality_flags=candidate.data_quality_flags,
                    number_context=candidate.number_context,
                    number_deferred=candidate.number_deferred,
                    number_uncertain=candidate.number_uncertain,
                    extraction_notes=candidate.extraction_notes,
                    record_id=_record_id_for(
                        candidate,
                        raw.source_document,
                        raw.segment_id,
                        candidate_index,
                    ),
                    source_document=raw.source_document,
                )
            )
    return records


def _content_has_meta_text(record: InitiativeExtractionRecord) -> bool:
    """Detect extraction-process prose that should not appear in content fields."""
    values = [
        record.initiative.general_description,
        record.initiative.objective_text,
        record.initiative.implementation_text,
        record.initiative.planned_outputs_text,
        record.initiative.delivery_text,
        record.initiative.funding_text,
        record.initiative.timeline_text,
    ]
    text = " ".join(value or "" for value in values).casefold()
    return "extracted source segment" in text or "not present in the extracted" in text
