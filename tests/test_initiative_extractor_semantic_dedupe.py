from pathlib import Path

from backend.modules.initiative_extractor import agent as extractor_agent
from backend.modules.initiative_extractor.models import (
    InitiativeDocumentSegment,
    InitiativeExtraction,
    InitiativeExtractionCandidate,
    InitiativeExtractionRecord,
)


def _candidate(
    *,
    name: str,
    source_quote: str,
    city: str = "Krakow",
) -> InitiativeExtractionCandidate:
    """Build a minimal initiative candidate for dedupe-oriented tests."""
    return InitiativeExtractionCandidate(
        initiative=InitiativeExtraction(
            city=city,
            initiative_name=name,
            general_description="Test initiative.",
            objective_text="Reduce emissions.",
            implementation_text="Implement the action.",
            planned_outputs_text="One deliverable.",
            delivery_text="City delivery.",
            funding_text=None,
            timeline_text=None,
            numbers={"current": {}, "planned": {}},
        ),
        source_quote=source_quote,
    )


def test_normalize_candidate_infers_document_local_code_from_source_quote() -> None:
    """Source-local action codes should be inferred when the model omits them."""
    segment = InitiativeDocumentSegment(
        segment_id="krakow:krakow:seg0001:1-3",
        city_name="Krakow",
        source_document="Krakow.md",
        source_path="documents/Krakow.md",
        start_line=1,
        end_line=3,
        heading_path="Transport",
        content="TR-2 - Project for fast, collision-free rail transport in Krakow (Premetro)",
        token_count=15,
    )
    candidate = _candidate(
        name="Project for fast, collision-free rail transport in Krakow (Premetro)",
        source_quote="TR-2 - Project for fast, collision-free rail transport in Krakow (Premetro)",
    )

    normalized = extractor_agent._normalize_candidate(candidate, segment)

    assert normalized.document_local_code == "TR-2"


def test_semantic_dedupe_payload_includes_local_code_and_quote() -> None:
    """Semantic dedupe payload should include local-code and quote evidence."""
    record = InitiativeExtractionRecord(
        initiative=InitiativeExtraction(
            city="Krakow",
            initiative_name="Modernisation of road lighting in Krakow",
            general_description="Lighting modernization.",
            objective_text="Reduce emissions.",
            implementation_text="Replace luminaires.",
            planned_outputs_text="One program.",
            delivery_text="City delivery.",
            funding_text=None,
            timeline_text=None,
            numbers={"current": {}, "planned": {}},
        ),
        document_local_code="E-11",
        source_quote="E-11 - Project entitled Modernisation of road lighting in Krakow",
        record_id="krakow:krakow:e-11",
        source_document="Krakow.md",
    )

    payload = extractor_agent._semantic_dedupe_payload([record])

    assert payload["records"] == [
        {
            "record_id": "krakow:krakow:e-11",
            "document_local_code": "E-11",
            "source_quote": "E-11 - Project entitled Modernisation of road lighting in Krakow",
            **record.initiative.model_dump(mode="json"),
        }
    ]


def test_semantic_dedupe_prompt_contract_matches_runtime_payload() -> None:
    """Semantic dedupe prompt should mention every runtime payload field."""
    prompt = Path("backend/prompts/initiative_semantic_dedupe_system.md").read_text(
        encoding="utf-8"
    )
    record = InitiativeExtractionRecord(
        initiative=InitiativeExtraction(
            city="Krakow",
            initiative_name="Modernisation of road lighting in Krakow",
            general_description="Lighting modernization.",
            objective_text="Reduce emissions.",
            implementation_text="Replace luminaires.",
            planned_outputs_text="One program.",
            delivery_text="City delivery.",
            funding_text=None,
            timeline_text=None,
            numbers={"current": {}, "planned": {}},
        ),
        document_local_code="E-11",
        source_quote="E-11 - Project entitled Modernisation of road lighting in Krakow",
        record_id="krakow:krakow:e-11",
        source_document="Krakow.md",
    )

    payload = extractor_agent._semantic_dedupe_payload([record])

    for field_name in payload["records"][0]:
        assert f"`{field_name}`" in prompt
    assert "matching `document_local_code` values" in prompt
    assert "Use `source_quote` to distinguish umbrella summaries" in prompt
