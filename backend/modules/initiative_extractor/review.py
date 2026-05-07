"""Review item builders for initiative extraction quality checks."""

from __future__ import annotations

from backend.modules.initiative_extractor.models import (
    InitiativeDocumentSegment,
    InitiativeExtractionRecord,
    InitiativeRawSegmentResult,
    InitiativeReviewItem,
)
from backend.modules.initiative_extractor.records import (
    SOURCE_QUOTE_FLAGS,
    _content_has_meta_text,
)
from backend.utils.config import AppConfig


def _build_review_items(
    *,
    segments: list[InitiativeDocumentSegment],
    raw_results: list[InitiativeRawSegmentResult],
    records: list[InitiativeExtractionRecord],
    duplicate_reviews: list[InitiativeReviewItem],
    config: AppConfig,
) -> list[InitiativeReviewItem]:
    """Build coverage and quality review items for one extraction run."""
    review_items = list(duplicate_reviews)
    result_by_segment = {result.segment_id: result for result in raw_results}
    for segment in segments:
        result = result_by_segment.get(segment.segment_id)
        if result is None:
            continue
        if result.status == "error":
            review_items.append(
                InitiativeReviewItem(
                    review_type="segment_extraction_failed",
                    severity="error",
                    message="Segment failed initiative extraction.",
                    source_document=segment.source_document,
                    segment_id=segment.segment_id,
                )
            )
        if result.action_heavy:
            review_items.append(
                InitiativeReviewItem(
                    review_type="action_heavy_segment",
                    severity="info",
                    message="Segment returned more than the configured action-heavy initiative threshold.",
                    source_document=segment.source_document,
                    segment_id=segment.segment_id,
                    details={
                        "extracted_count": len(result.initiatives),
                        "threshold": config.initiative_extractor.action_heavy_initiative_threshold,
                        "extraction_iterations": result.extraction_iterations,
                        "extraction_complete": result.extraction_complete,
                        "stop_reason": result.stop_reason,
                    },
                )
            )
        for flag in result.segment_data_quality_flags:
            review_type = (
                "action_heavy_extraction_flag"
                if flag.startswith("action_heavy_")
                else "source_quality_flag"
            )
            message = (
                f"Segment has action-heavy extraction flag: {flag}"
                if review_type == "action_heavy_extraction_flag"
                else f"Segment has source quality flag: {flag}"
            )
            review_items.append(
                InitiativeReviewItem(
                    review_type=review_type,
                    severity="info",
                    message=message,
                    source_document=segment.source_document,
                    segment_id=segment.segment_id,
                    details={"flag": flag},
                )
            )

    for record in records:
        if _content_has_meta_text(record):
            review_items.append(
                InitiativeReviewItem(
                    review_type="content_contains_extraction_meta_text",
                    message="Content field contains extraction-process prose.",
                    source_document=record.source_document,
                    record_id=record.record_id,
                    document_local_code=record.document_local_code,
                )
            )
        for flag in record.data_quality_flags:
            if flag in SOURCE_QUOTE_FLAGS:
                review_items.append(
                    InitiativeReviewItem(
                        review_type="source_quote_missing_or_invalid",
                        message="Initiative source quote is missing or was not found in the source segment.",
                        source_document=record.source_document,
                        record_id=record.record_id,
                        document_local_code=record.document_local_code,
                        details={"flag": flag},
                    )
                )
                continue
            review_items.append(
                InitiativeReviewItem(
                    review_type="initiative_quality_flag",
                    severity="info",
                    message=f"Initiative has source quality flag: {flag}",
                    source_document=record.source_document,
                    record_id=record.record_id,
                    document_local_code=record.document_local_code,
                    details={"flag": flag},
                )
            )

    return review_items
