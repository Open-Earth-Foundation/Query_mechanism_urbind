from backend.modules.markdown_researcher.utils.output import coerce_markdown_result


def test_coerce_markdown_result_accepts_partial_answer_field() -> None:
    raw = {
        "status": "success",
        "excerpts": [
            {
                "quote": "Munich has deployed 43 existing public chargers as of 2024.",
                "city_name": "Munich",
                "partial_answer": "Munich has deployed 43 existing public chargers as of 2024.",
                "source_chunk_ids": ["chunk_abc123"],
            }
        ],
        "error": None,
    }

    parsed = coerce_markdown_result(raw)

    assert parsed is not None
    assert parsed.status == "success"
    assert len(parsed.excerpts) == 1
    assert (
        parsed.excerpts[0].partial_answer
        == "Munich has deployed 43 existing public chargers as of 2024."
    )
    assert parsed.excerpts[0].source_chunk_ids == ["chunk_abc123"]


def test_coerce_markdown_result_rejects_legacy_answer_field() -> None:
    raw = {
        "status": "success",
        "excerpts": [
            {
                "quote": "Munich has deployed 43 existing public chargers as of 2024.",
                "city_name": "Munich",
                "answer": "Munich has deployed 43 existing public chargers as of 2024.",
            }
        ],
        "error": None,
    }

    parsed = coerce_markdown_result(raw)

    assert parsed is None
