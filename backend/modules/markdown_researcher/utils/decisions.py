"""Decision validation helpers for markdown researcher batches."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.modules.markdown_researcher.models import MarkdownExcerpt


@dataclass
class DecisionValidationResult:
    """Validation outcome for one markdown batch decision partition."""

    is_valid: bool
    accepted_ids: list[str] = field(default_factory=list)
    rejected_ids: list[str] = field(default_factory=list)
    overlap_ids: list[str] = field(default_factory=list)
    unknown_accepted_ids: list[str] = field(default_factory=list)
    unknown_rejected_ids: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)
    unknown_excerpt_source_ids: list[str] = field(default_factory=list)
    violation_codes: list[str] = field(default_factory=list)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return unique non-empty values while preserving input order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def validate_batch_decisions(
    input_chunk_ids: list[str],
    accepted_chunk_ids: list[str],
    rejected_chunk_ids: list[str],
    excerpts: list[MarkdownExcerpt],
) -> DecisionValidationResult:
    """Validate strict accepted/rejected partition invariants for one batch."""
    input_ids = _dedupe_preserve_order(input_chunk_ids)
    accepted_ids = _dedupe_preserve_order(accepted_chunk_ids)
    rejected_ids = _dedupe_preserve_order(rejected_chunk_ids)
    input_set = set(input_ids)
    accepted_set = set(accepted_ids)
    rejected_set = set(rejected_ids)

    overlap_ids = [chunk_id for chunk_id in accepted_ids if chunk_id in rejected_set]
    unknown_accepted_ids = [chunk_id for chunk_id in accepted_ids if chunk_id not in input_set]
    unknown_rejected_ids = [chunk_id for chunk_id in rejected_ids if chunk_id not in input_set]
    decided_set = accepted_set | rejected_set
    missing_ids = [chunk_id for chunk_id in input_ids if chunk_id not in decided_set]

    unknown_excerpt_source_ids: list[str] = []
    seen_excerpt_ids: set[str] = set()
    for excerpt in excerpts:
        for source_chunk_id in excerpt.source_chunk_ids:
            normalized = source_chunk_id.strip()
            if (
                normalized
                and normalized not in accepted_set
                and normalized not in seen_excerpt_ids
            ):
                seen_excerpt_ids.add(normalized)
                unknown_excerpt_source_ids.append(normalized)

    violation_codes: list[str] = []
    if overlap_ids:
        violation_codes.append("overlap")
    if unknown_accepted_ids:
        violation_codes.append("unknown_accepted_ids")
    if unknown_rejected_ids:
        violation_codes.append("unknown_rejected_ids")
    if missing_ids:
        violation_codes.append("missing_decisions")
    if unknown_excerpt_source_ids:
        violation_codes.append("unknown_excerpt_source_ids")

    return DecisionValidationResult(
        is_valid=not violation_codes,
        accepted_ids=accepted_ids,
        rejected_ids=rejected_ids,
        overlap_ids=overlap_ids,
        unknown_accepted_ids=unknown_accepted_ids,
        unknown_rejected_ids=unknown_rejected_ids,
        missing_ids=missing_ids,
        unknown_excerpt_source_ids=unknown_excerpt_source_ids,
        violation_codes=violation_codes,
    )


__all__ = ["DecisionValidationResult", "validate_batch_decisions"]
