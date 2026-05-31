"""Writer-visible context indexing and saved-evidence conversion."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from backend.api.services.source_chunks import load_source_chunks
from backend.modules.orchestrator.utils.references import is_valid_ref_id
from backend.modules.writer.models import (
    WriterContextItem,
    WriterContextSourceKind,
    WriterMissingEvidenceRecord,
    WriterSavedEvidence,
)
from backend.modules.writer.utils.markdown_helpers import (
    city_display_name,
    city_key,
    extract_markdown_bundle,
    extract_markdown_excerpts,
)
from backend.utils.config import AppConfig


_ENRICHMENT_CONTEXT_KEYS: dict[str, WriterContextSourceKind] = {
    "external_evidence": "external_markdown_claim",
    "external_resolutions": "external_markdown_resolution",
    "external_no_evidence": "external_no_evidence",
    "web_findings": "web_finding",
    "assumptions": "assumption",
    "non_estimable": "non_estimable",
    "enriched_fields": "enriched_field",
    "freshness_results": "freshness_result",
}


@dataclass(frozen=True)
class WriterContextIndex:
    """Searchable writer context plus soft-failure records."""

    items: list[WriterContextItem]
    missing_records: list[WriterMissingEvidenceRecord]


def build_writer_context_index(
    *,
    context_bundle: dict[str, object],
    run_dir: Path | None,
    markdown_dir: Path,
    config: AppConfig,
    use_source_chunks: bool,
) -> WriterContextIndex:
    """Build searchable items from the exact writer-safe context bundle."""
    items = _build_excerpt_items(context_bundle)
    missing_records: list[WriterMissingEvidenceRecord] = []
    if use_source_chunks and run_dir is not None:
        chunk_items, chunk_missing = _build_source_chunk_items(
            context_bundle=context_bundle,
            run_dir=run_dir,
            markdown_dir=markdown_dir,
            config=config,
        )
        items.extend(chunk_items)
        missing_records.extend(chunk_missing)
    items.extend(_build_enrichment_items(context_bundle))
    return WriterContextIndex(items=items, missing_records=missing_records)


def apply_saved_evidence_to_context(
    *,
    context_bundle: dict[str, object],
    saved_evidence: list[WriterSavedEvidence],
) -> dict[str, object]:
    """Return a writer context where saved evidence is citation-compatible."""
    updated_context = dict(context_bundle)
    markdown_bundle = dict(extract_markdown_bundle(updated_context))
    excerpts = [dict(excerpt) for excerpt in extract_markdown_excerpts(markdown_bundle)]
    ref_to_excerpt = {
        str(excerpt.get("ref_id", "")).strip(): excerpt
        for excerpt in excerpts
        if is_valid_ref_id(str(excerpt.get("ref_id", "")).strip())
    }

    for evidence in saved_evidence:
        if evidence.source_kind == "ccc_excerpt" and evidence.ref_id in ref_to_excerpt:
            _mark_existing_excerpt_saved(ref_to_excerpt[evidence.ref_id], evidence)
            continue
        if evidence.ref_id in ref_to_excerpt:
            continue
        excerpt = _saved_evidence_to_excerpt(evidence)
        excerpts.append(excerpt)
        ref_to_excerpt[evidence.ref_id] = excerpt

    markdown_bundle["excerpts"] = excerpts
    markdown_bundle["excerpt_count"] = len(excerpts)
    updated_context["markdown"] = markdown_bundle
    updated_context["writer_saved_evidence"] = _build_context_saved_summary(saved_evidence)
    return updated_context


def build_writer_references_payload(
    *,
    run_id: str,
    saved_evidence: list[WriterSavedEvidence],
) -> dict[str, object]:
    """Build the writer-level reference artifact for saved curator evidence."""
    references: list[dict[str, object]] = []
    for index, evidence in enumerate(saved_evidence):
        references.append(
            {
                "ref_id": evidence.ref_id,
                "excerpt_index": index,
                "city_name": evidence.city_name,
                "quote": evidence.quote,
                "partial_answer": evidence.text,
                "source_chunk_ids": _coerce_string_list(
                    evidence.metadata.get("source_chunk_ids")
                ),
                "source_kind": evidence.source_kind,
                "source_id": evidence.source_id,
                "field": evidence.field,
                "line_start": evidence.line_start,
                "line_end": evidence.line_end,
                "writer_saved_id": evidence.saved_id,
                "reason": evidence.reason,
            }
        )
    return {
        "run_id": run_id,
        "reference_count": len(references),
        "references": references,
    }


def _build_excerpt_items(context_bundle: dict[str, object]) -> list[WriterContextItem]:
    """Build context items from accepted CCC excerpts."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    items: list[WriterContextItem] = []
    for index, excerpt in enumerate(extract_markdown_excerpts(markdown_bundle), start=1):
        city_name = city_display_name(str(excerpt.get("city_name", "")))
        ref_id = str(excerpt.get("ref_id", "")).strip()
        source_chunk_ids = _coerce_string_list(excerpt.get("source_chunk_ids"))
        quote = str(excerpt.get("quote", "")).strip()
        partial_answer = str(excerpt.get("partial_answer", "")).strip()
        text = _join_text(
            [
                f"Ref: {ref_id}" if ref_id else "",
                f"City: {city_name}" if city_name else "",
                partial_answer,
                quote,
            ]
        )
        if not text:
            continue
        items.append(
            WriterContextItem(
                item_id=f"ccc_excerpt:{ref_id or index}",
                source_kind="ccc_excerpt",
                city_name=city_name,
                city_key=city_key(city_name),
                source_id=", ".join(source_chunk_ids),
                ref_id=ref_id,
                text=text,
                quote=quote,
                metadata={
                    "source_chunk_ids": source_chunk_ids,
                    "excerpt_index": index - 1,
                },
            )
        )
    return items


def _build_source_chunk_items(
    *,
    context_bundle: dict[str, object],
    run_dir: Path,
    markdown_dir: Path,
    config: AppConfig,
) -> tuple[list[WriterContextItem], list[WriterMissingEvidenceRecord]]:
    """Resolve CCC source chunks and convert them to searchable items."""
    chunk_ids = _collect_source_chunk_ids(context_bundle)
    if not chunk_ids:
        return [], []

    try:
        chunks = load_source_chunks(run_dir, markdown_dir, config, chunk_ids)
        missing_ids: list[str] = []
    except Exception:
        chunks = []
        missing_ids = []
        for chunk_id in chunk_ids:
            try:
                chunks.extend(load_source_chunks(run_dir, markdown_dir, config, [chunk_id]))
            except Exception:
                missing_ids.append(chunk_id)

    items: list[WriterContextItem] = []
    for chunk in chunks:
        city_name = city_display_name(chunk.city_name or "")
        text = _join_text(
            [
                f"Chunk: {chunk.chunk_id}",
                f"City: {city_name}" if city_name else "",
                chunk.heading_path or "",
                chunk.content,
            ]
        )
        items.append(
            WriterContextItem(
                item_id=f"ccc_source_chunk:{chunk.chunk_id}",
                source_kind="ccc_source_chunk",
                city_name=city_name,
                city_key=city_key(city_name),
                source_id=chunk.chunk_id,
                text=text,
                quote=chunk.content,
                metadata={
                    "source_chunk_ids": [chunk.chunk_id],
                    "source_path": chunk.source_path,
                    "heading_path": chunk.heading_path,
                    "block_type": chunk.block_type,
                },
            )
        )

    missing_records = [
        WriterMissingEvidenceRecord(
            missing_id=f"missing_chunk_{index}",
            source_kind="ccc_source_chunk",
            reason=f"CCC source chunk `{chunk_id}` could not be resolved for writer search.",
            searched_patterns=[chunk_id],
        )
        for index, chunk_id in enumerate(missing_ids, start=1)
    ]
    return items, missing_records


def _collect_source_chunk_ids(context_bundle: dict[str, object]) -> list[str]:
    """Collect unique source chunk ids from markdown excerpts."""
    ids: list[str] = []
    seen: set[str] = set()
    for excerpt in extract_markdown_excerpts(extract_markdown_bundle(context_bundle)):
        for chunk_id in _coerce_string_list(excerpt.get("source_chunk_ids")):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            ids.append(chunk_id)
    return ids


def _build_enrichment_items(context_bundle: dict[str, object]) -> list[WriterContextItem]:
    """Build context items from writer-safe enrichment records."""
    enrichment = context_bundle.get("enrichment")
    if not isinstance(enrichment, dict):
        return []

    items: list[WriterContextItem] = []
    for key, source_kind in _ENRICHMENT_CONTEXT_KEYS.items():
        records = enrichment.get(key)
        if not isinstance(records, list):
            continue
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            item = _enrichment_record_to_item(key, source_kind, index, record)
            if item is not None:
                items.append(item)
    return items


def _enrichment_record_to_item(
    key: str,
    source_kind: WriterContextSourceKind,
    index: int,
    record: dict[str, object],
) -> WriterContextItem | None:
    """Convert one enrichment record into a searchable item."""
    city_name = city_display_name(_read_string(record.get("city")))
    field = _read_string(record.get("field")) or _read_string(record.get("field_name"))
    source_id = (
        _read_string(record.get("source_id"))
        or _read_string(record.get("source_url"))
        or _read_string(record.get("reference_data"))
    )
    quote = _read_string(record.get("quote")) or _read_string(record.get("rationale"))
    text = _record_text(record)
    if not text:
        return None
    return WriterContextItem(
        item_id=f"{source_kind}:{index}",
        source_kind=source_kind,
        city_name=city_name,
        city_key=city_key(city_name),
        source_id=source_id,
        field=field,
        text=f"{key}: {text}",
        quote=quote,
        line_start=_read_int(record.get("line_start")),
        line_end=_read_int(record.get("line_end")),
        metadata={"record": record, "enrichment_key": key},
    )


def _mark_existing_excerpt_saved(
    excerpt: dict[str, object],
    evidence: WriterSavedEvidence,
) -> None:
    """Annotate a baseline CCC excerpt as selected by the curator."""
    saved_ids = _coerce_string_list(excerpt.get("writer_saved_ids"))
    if evidence.saved_id not in saved_ids:
        saved_ids.append(evidence.saved_id)
    excerpt["writer_saved_ids"] = saved_ids
    excerpt["writer_saved_id"] = saved_ids[0]
    excerpt["source_kind"] = excerpt.get("source_kind") or "ccc_excerpt"


def _saved_evidence_to_excerpt(evidence: WriterSavedEvidence) -> dict[str, object]:
    """Convert saved non-excerpt evidence into a markdown excerpt record."""
    source_chunk_ids = _coerce_string_list(evidence.metadata.get("source_chunk_ids"))
    return {
        "ref_id": evidence.ref_id,
        "city_name": evidence.city_name,
        "city_key": evidence.city_key,
        "quote": evidence.quote or evidence.text,
        "partial_answer": evidence.text,
        "source_chunk_ids": source_chunk_ids,
        "source_kind": evidence.source_kind,
        "source_id": evidence.source_id,
        "field": evidence.field,
        "line_start": evidence.line_start,
        "line_end": evidence.line_end,
        "writer_saved_id": evidence.saved_id,
        "writer_saved_reason": evidence.reason,
    }


def _build_context_saved_summary(
    saved_evidence: list[WriterSavedEvidence],
) -> dict[str, object]:
    """Build a compact summary for writer-context export and diagnostics."""
    source_kind_counts = Counter(evidence.source_kind for evidence in saved_evidence)
    city_names = sorted(
        {
            evidence.city_name
            for evidence in saved_evidence
            if evidence.city_name
        }
    )
    return {
        "saved_count": len(saved_evidence),
        "covered_cities": city_names,
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "saved_ids": [evidence.saved_id for evidence in saved_evidence],
    }


def _coerce_string_list(value: object) -> list[str]:
    """Return a compact string list."""
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _read_string(value: object) -> str:
    """Return a stripped string or an empty value."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _read_int(value: object) -> int | None:
    """Return an integer value when present."""
    if isinstance(value, int):
        return value
    return None


def _record_text(record: dict[str, object]) -> str:
    """Render one enrichment record into a compact searchable text block."""
    parts: list[str] = []
    for key, value in record.items():
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}: {_stringify(value)}")
    return "\n".join(parts)


def _stringify(value: object) -> str:
    """Render nested values for search indexing without custom JSON handling."""
    if isinstance(value, dict):
        return "; ".join(f"{key}={_stringify(item)}" for key, item in value.items())
    if isinstance(value, list):
        return ", ".join(_stringify(item) for item in value)
    return str(value)


def _join_text(parts: list[str]) -> str:
    """Join non-empty text parts with stable spacing."""
    return "\n".join(part.strip() for part in parts if part and part.strip())


__all__ = [
    "WriterContextIndex",
    "apply_saved_evidence_to_context",
    "build_writer_context_index",
    "build_writer_references_payload",
]
