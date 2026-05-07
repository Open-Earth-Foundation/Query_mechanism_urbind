"""Artifact writers for initiative extraction run outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from backend.modules.initiative_extractor.models import (
    InitiativeDocumentSegment,
    InitiativeExtraction,
    InitiativeExtractionRecord,
    InitiativeRawSegmentResult,
    InitiativeReviewItem,
    InitiativeSemanticDedupeGroup,
)
from backend.utils.config import AppConfig
from backend.utils.json_io import write_json


def _write_jsonl(path: Path, rows: list[object]) -> None:
    """Write JSONL rows with stable UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        if hasattr(row, "model_dump"):
            payload = row.model_dump(mode="json")
        else:
            payload = row
        lines.append(json.dumps(payload, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _canonical_rows(
    records: list[InitiativeExtractionRecord],
) -> list[InitiativeExtraction]:
    """Return v1 canonical initiative objects for public extraction artifacts."""
    return [record.initiative for record in records]


def _write_run_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    documents: list[Path],
    segments: list[InitiativeDocumentSegment],
    raw_results: list[InitiativeRawSegmentResult],
    candidate_records: list[InitiativeExtractionRecord],
    records: list[InitiativeExtractionRecord],
    semantic_groups: list[InitiativeSemanticDedupeGroup],
    review_items: list[InitiativeReviewItem],
    config: AppConfig,
) -> None:
    """Persist all initiative extraction artifacts for a run."""
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "documents": [str(path) for path in documents],
        "model": config.initiative_extractor.model,
        "max_segment_tokens": config.initiative_extractor.max_segment_tokens,
        "segment_overlap_lines": config.initiative_extractor.segment_overlap_lines,
        "prior_initiatives_max_tokens": config.initiative_extractor.prior_initiatives_max_tokens,
        "action_heavy_initiative_threshold": config.initiative_extractor.action_heavy_initiative_threshold,
        "action_heavy_max_followup_calls": config.initiative_extractor.action_heavy_max_followup_calls,
        "semantic_dedupe_enabled": config.initiative_extractor.semantic_dedupe_enabled,
    }
    summary = {
        "run_id": run_id,
        "documents_count": len(documents),
        "segments_count": len(segments),
        "raw_initiatives_count": sum(len(result.initiatives) for result in raw_results),
        "candidate_records_count": len(candidate_records),
        "deduped_initiatives_count": len(records),
        "semantic_duplicate_groups_count": len(semantic_groups),
        "semantic_merged_duplicates_count": max(
            len(candidate_records) - len(records), 0
        ),
        "action_heavy_segments_count": sum(
            1 for result in raw_results if result.action_heavy
        ),
        "action_heavy_followup_iterations_count": sum(
            max(result.extraction_iterations - 1, 0) for result in raw_results
        ),
        "review_items_count": len(review_items),
    }
    write_json(
        run_dir / "00_source" / "source_manifest.json", manifest, ensure_ascii=False
    )
    _write_jsonl(run_dir / "01_segments" / "segments.jsonl", segments)
    _write_jsonl(
        run_dir / "02_raw_extractions" / "raw_segment_extractions.jsonl", raw_results
    )
    _write_jsonl(
        run_dir / "03_deduped" / "candidate_initiatives.jsonl",
        _canonical_rows(candidate_records),
    )
    _write_jsonl(run_dir / "03_deduped" / "candidate_records.jsonl", candidate_records)
    _write_jsonl(
        run_dir / "03_deduped" / "semantic_duplicate_groups.jsonl", semantic_groups
    )
    _write_jsonl(run_dir / "03_deduped" / "initiatives.jsonl", _canonical_rows(records))
    _write_jsonl(run_dir / "03_deduped" / "initiative_records.jsonl", records)
    _write_jsonl(run_dir / "04_review" / "review_items.jsonl", review_items)
    write_json(run_dir / "summary.json", summary, ensure_ascii=False)
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                "# Initiative Extraction Run",
                "",
                "This folder contains artifact-first initiative extraction output.",
                "",
                "- `00_source/source_manifest.json`: source documents and run settings.",
                "- `01_segments/segments.jsonl`: ordered line-aware document segments.",
                "- `02_raw_extractions/raw_segment_extractions.jsonl`: per-segment model output.",
                "- `03_deduped/candidate_initiatives.jsonl`: canonical v1 initiatives before semantic dedupe.",
                (
                    "- `03_deduped/candidate_records.jsonl`: pipeline records before "
                    "semantic dedupe, with generated ids and source quotes."
                ),
                "- `03_deduped/semantic_duplicate_groups.jsonl`: semantic duplicate groups proposed by the LLM.",
                "- `03_deduped/initiatives.jsonl`: final canonical v1 initiative objects.",
                (
                    "- `03_deduped/initiative_records.jsonl`: final pipeline records "
                    "with generated ids and quote-only source citations for TEF mapping."
                ),
                "- `04_review/review_items.jsonl`: coverage and quality review items.",
                "- `summary.json`: run counts.",
                "",
                "No TEF classification or database writes are performed in this step.",
            ]
        ),
        encoding="utf-8",
    )
