from app.modules.markdown_researcher.utils.output import coerce_markdown_result


def test_coerce_markdown_result_accepts_partial_answer_field() -> None:
    raw = {
        "status": "success",
        "excerpts": [
            {
                "snippet": "S",
                "city_name": "Munich",
                "partial_answer": "Munich has 43 chargers.",
                "relevant": "yes",
            }
        ],
        "error": None,
    }

    parsed = coerce_markdown_result(raw)

    assert parsed is not None
    assert parsed.status == "success"
    assert len(parsed.excerpts) == 1
    assert parsed.excerpts[0].partial_answer == "Munich has 43 chargers."


def test_coerce_markdown_result_accepts_legacy_answer_field() -> None:
    raw = {
        "status": "success",
        "excerpts": [
            {
                "snippet": "S",
                "city_name": "Munich",
                "answer": "The answer is: Munich has 43 chargers.",
                "relevant": "yes",
            }
        ],
        "error": None,
    }

    parsed = coerce_markdown_result(raw)

    assert parsed is not None
    assert parsed.status == "success"
    assert len(parsed.excerpts) == 1
    assert parsed.excerpts[0].partial_answer == "Munich has 43 chargers."
