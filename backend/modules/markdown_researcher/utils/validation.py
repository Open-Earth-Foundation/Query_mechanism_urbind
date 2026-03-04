"""Validation helpers for markdown researcher excerpt outputs."""

from __future__ import annotations

from backend.modules.markdown_researcher.models import MarkdownExcerpt, ThrownExcerpt


def _normalize_city(value: str) -> str:
    return " ".join(value.strip().replace("_", " ").replace("-", " ").split()).casefold()


def _contains_disallowed_whitespace(value: str) -> bool:
    return "\n" in value or "\r" in value or "\t" in value


def partition_batch_excerpts(
    excerpts: list[MarkdownExcerpt],
    *,
    expected_city_name: str,
    batch_index: int,
    valid_chunk_ids: set[str],
) -> tuple[list[MarkdownExcerpt], list[ThrownExcerpt]]:
    """Split excerpts into valid and rejected groups with machine-readable reasons."""
    accepted: list[MarkdownExcerpt] = []
    rejected: list[ThrownExcerpt] = []
    expected_city_normalized = _normalize_city(expected_city_name)

    for excerpt in excerpts:
        reason_codes: list[str] = []
        quote = excerpt.quote.strip()
        partial_answer = excerpt.partial_answer.strip()
        city_name = excerpt.city_name.strip()
        source_chunk_ids = [chunk_id.strip() for chunk_id in excerpt.source_chunk_ids if chunk_id.strip()]

        if not quote:
            reason_codes.append("empty_quote")
        if not partial_answer:
            reason_codes.append("empty_partial_answer")
        if not city_name:
            reason_codes.append("empty_city_name")
        if _contains_disallowed_whitespace(excerpt.quote):
            reason_codes.append("quote_contains_newline_or_tab")
        if _contains_disallowed_whitespace(excerpt.partial_answer):
            reason_codes.append("partial_answer_contains_newline_or_tab")

        if _normalize_city(city_name) != expected_city_normalized:
            reason_codes.append("city_name_mismatch")

        if not source_chunk_ids:
            reason_codes.append("missing_source_chunk_ids")

        invalid_source_chunk_ids = sorted(
            {
                source_chunk_id
                for source_chunk_id in source_chunk_ids
                if source_chunk_id not in valid_chunk_ids
            }
        )
        if invalid_source_chunk_ids:
            reason_codes.append("unknown_source_chunk_ids")

        if reason_codes:
            rejected.append(
                ThrownExcerpt(
                    quote=excerpt.quote,
                    city_name=excerpt.city_name,
                    partial_answer=excerpt.partial_answer,
                    source_chunk_ids=excerpt.source_chunk_ids,
                    rejection_stage="batch_validation",
                    reason_codes=reason_codes,
                    batch_index=batch_index,
                    expected_city_name=expected_city_name,
                    invalid_source_chunk_ids=invalid_source_chunk_ids,
                )
            )
            continue

        accepted.append(
            MarkdownExcerpt(
                quote=quote,
                city_name=city_name,
                partial_answer=partial_answer,
                source_chunk_ids=source_chunk_ids,
            )
        )

    return accepted, rejected


__all__ = ["partition_batch_excerpts"]
