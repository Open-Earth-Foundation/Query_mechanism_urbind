from backend.modules.markdown_researcher.models import MarkdownExcerpt
from backend.modules.markdown_researcher.utils.decisions import validate_batch_decisions


def test_validate_batch_decisions_accepts_complete_partition() -> None:
    excerpts = [
        MarkdownExcerpt(
            quote="q",
            city_name="Munich",
            partial_answer="a",
            source_chunk_ids=["chunk-1"],
        )
    ]
    result = validate_batch_decisions(
        input_chunk_ids=["chunk-1", "chunk-2"],
        accepted_chunk_ids=["chunk-1"],
        rejected_chunk_ids=["chunk-2"],
        excerpts=excerpts,
    )

    assert result.is_valid is True
    assert result.violation_codes == []


def test_validate_batch_decisions_rejects_overlap() -> None:
    result = validate_batch_decisions(
        input_chunk_ids=["chunk-1", "chunk-2"],
        accepted_chunk_ids=["chunk-1"],
        rejected_chunk_ids=["chunk-1", "chunk-2"],
        excerpts=[],
    )

    assert result.is_valid is False
    assert "overlap" in result.violation_codes


def test_validate_batch_decisions_rejects_unknown_ids() -> None:
    result = validate_batch_decisions(
        input_chunk_ids=["chunk-1"],
        accepted_chunk_ids=["chunk-unknown"],
        rejected_chunk_ids=["chunk-1"],
        excerpts=[],
    )

    assert result.is_valid is False
    assert "unknown_accepted_ids" in result.violation_codes


def test_validate_batch_decisions_rejects_missing_partition() -> None:
    result = validate_batch_decisions(
        input_chunk_ids=["chunk-1", "chunk-2"],
        accepted_chunk_ids=["chunk-1"],
        rejected_chunk_ids=[],
        excerpts=[],
    )

    assert result.is_valid is False
    assert "missing_decisions" in result.violation_codes


def test_validate_batch_decisions_rejects_excerpt_source_not_accepted() -> None:
    excerpts = [
        MarkdownExcerpt(
            quote="q",
            city_name="Munich",
            partial_answer="a",
            source_chunk_ids=["chunk-2"],
        )
    ]
    result = validate_batch_decisions(
        input_chunk_ids=["chunk-1", "chunk-2"],
        accepted_chunk_ids=["chunk-1"],
        rejected_chunk_ids=["chunk-2"],
        excerpts=excerpts,
    )

    assert result.is_valid is False
    assert "unknown_excerpt_source_ids" in result.violation_codes
