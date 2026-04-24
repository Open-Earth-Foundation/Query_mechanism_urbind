from typing import Any

from backend.scripts import apply_polish_remaining_extraction_fixes as extraction_fix_v2


ExtractionRecord = dict[str, Any]


def _record(record_id: str, city: str, title: str) -> ExtractionRecord:
    """Build the minimal extraction-record shape used by the rewrite logic."""
    return {
        "record_id": record_id,
        "initiative": {
            "city": city,
            "initiative_name": title,
        },
    }


def _count_titles(records: list[ExtractionRecord], city: str, title: str) -> int:
    """Count exact city-title matches in a list of extraction records."""
    return sum(
        1
        for record in records
        if record["initiative"]["city"] == city
        and record["initiative"]["initiative_name"] == title
    )


def test_rewrite_records_adds_verified_remaining_initiatives_without_duplication() -> None:
    """The second-pass repair should add the verified gaps and keep existing records singular."""
    source_records = [
        _record("source:wroclaw:change_the_stove", "Wroclaw", "Change the stove"),
        _record(
            "source:wroclaw:participatory_budget",
            "Wroclaw",
            "Wroclaw Participatory Budget",
        ),
        _record(
            "source:lodz:neest",
            "Lodz",
            "NEEST - NetZero Emission and Environmentally Sustainable Territories",
        ),
    ]

    corrected_records, manifest = extraction_fix_v2.rewrite_records(source_records)

    corrected_titles = {
        (record["initiative"]["city"], record["initiative"]["initiative_name"])
        for record in corrected_records
    }

    assert len(manifest["added_record_ids"]) == 62
    assert len(corrected_records) == len(source_records) + len(manifest["added_record_ids"])

    assert (
        "Krakow",
        "Programme for the development of renewable energy sources in the Municipality of Krakow",
    ) in corrected_titles
    assert (
        "Warszawa",
        "Green Vision for Warsaw - Green City and Climate Action Plan",
    ) in corrected_titles
    assert (
        "Wroclaw",
        "Sustainable Energy and Climate Action Plan (SECAP)",
    ) in corrected_titles

    assert _count_titles(corrected_records, "Wroclaw", "Change the stove") == 1
    assert _count_titles(corrected_records, "Wroclaw", "Wroclaw Participatory Budget") == 1
    assert (
        _count_titles(
            corrected_records,
            "Lodz",
            "NEEST - NetZero Emission and Environmentally Sustainable Territories",
        )
        == 1
    )
