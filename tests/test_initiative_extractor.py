import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.modules.initiative_extractor import agent as extractor_agent
from backend.modules.initiative_extractor.models import (
    InitiativeDocumentSegment,
    InitiativeExtraction,
    InitiativeExtractionCandidate,
    InitiativeExtractionRecord,
    InitiativeRawSegmentResult,
    InitiativeSemanticDedupeGroup,
    InitiativeSegmentExtraction,
    InitiativeSegmentStop,
    InitiativeSourceRef,
)
from backend.modules.initiative_extractor.segmentation import build_document_segments
from backend.utils.llm_serialization import parse_llm_serialized
from tests.support import build_test_app_config


class _FakeRunResult:
    """Minimal fake Agents result for initiative extraction tests."""

    def __init__(self, final_output: object, raw_responses: list[dict[str, object]] | None = None) -> None:
        self.final_output = final_output
        self.raw_responses = raw_responses or []


def _candidate(
    *,
    code: str,
    city: str = "Krakow",
    name: str | None = None,
    description: str | None = None,
    source_quote: str | None = None,
    source_document: str = "Krakow.md",
    segment_id: str = "seg1",
) -> InitiativeExtractionCandidate:
    """Build a valid initiative candidate for tests."""
    initiative_name = name or f"Initiative {code}"
    return InitiativeExtractionCandidate(
        initiative=InitiativeExtraction(
            city=city,
            initiative_name=initiative_name,
            general_description=description or f"Description for {code}.",
            objective_text="Reduce emissions.",
            implementation_text="Implement infrastructure.",
            planned_outputs_text="One planned output.",
            delivery_text="City delivery.",
            funding_text="PLN 1,000.",
            timeline_text="2024-2030.",
            numbers={"current": {}, "planned": {"cost_pln": 1000}},
        ),
        document_local_code=code,
        source_quote=source_quote or initiative_name,
        source_refs=[
            InitiativeSourceRef(
                source_document=source_document,
                segment_id=segment_id,
                start_line=1,
                end_line=8,
            )
        ],
    )


def test_initiative_schema_rejects_tef_fields() -> None:
    """Ensure first-pass extraction cannot include TEF classification fields."""
    with pytest.raises(ValidationError):
        InitiativeExtraction.model_validate(
            {
                "city": "Krakow",
                "initiative_name": "Heat pumps",
                "general_description": "Install heat pumps.",
                "numbers": {"current": {}, "planned": {}},
                "tef_id": "district_heating_heat_pumps",
            }
        )


def test_candidate_metadata_ignores_extra_fields() -> None:
    """Wrapper metadata can ignore model helper fields without polluting output."""
    candidate = InitiativeExtractionCandidate.model_validate(
        {
            "initiative": {
                "city": "Krakow",
                "initiative_name": "Heat pumps",
                "general_description": "Install heat pumps.",
                "numbers": {"current": {}, "planned": {}},
            },
            "document_local_code": "BIC-7",
            "source_quote": "Install heat pumps.",
            "tef_id": "should_be_ignored",
        }
    )

    assert candidate.document_local_code == "BIC-7"
    assert candidate.source_quote == "Install heat pumps."
    assert "tef_id" not in candidate.model_dump(mode="json")
    assert "source_refs" not in candidate.model_dump(mode="json")


def test_segmenter_preserves_line_ranges_and_table_rows(tmp_path: Path) -> None:
    """Segmenting should keep ordered line ranges and table rows intact."""
    source = tmp_path / "Krakow.md"
    source.write_text(
        "\n".join(
            [
                "# Climate Contract",
                "",
                "| B-2.2: Outline of individual activities - BIC-1 |",
                "| Description of operation | Name of the action | Heat pumps |",
                "| Receipts and costs | Total costs | PLN 7 000 000 |",
                "",
                "More explanatory text " * 40,
            ]
        ),
        encoding="utf-8",
    )
    config = build_test_app_config(
        initiative_extractor_overrides={
            "max_segment_tokens": 120,
            "segment_overlap_lines": 0,
        }
    )

    segments = build_document_segments(source, config.initiative_extractor)

    assert segments
    assert segments[0].start_line == 1
    assert segments[0].end_line >= 5
    assert "| Receipts and costs | Total costs | PLN 7 000 000 |" in segments[0].content
    assert [segment.start_line for segment in segments] == sorted(
        segment.start_line for segment in segments
    )


def test_prompt_contract_matches_runtime_payload_and_schema() -> None:
    """Prompt input/output fields should match the runtime segment and model contract."""
    segment = InitiativeDocumentSegment(
        segment_id="krakow:krakow:seg0001:1-3",
        city_name="Krakow",
        source_document="Krakow.md",
        source_path="documents/Krakow.md",
        start_line=1,
        end_line=3,
        heading_path="Transport",
        content="B-2.2: Outline of individual activities - TR-1",
        token_count=10,
    )
    prompt = Path("backend/prompts/initiative_extractor_system.md").read_text(encoding="utf-8")
    config = build_test_app_config()
    payload = extractor_agent._segment_payload(segment, [], config)

    for field_name in payload:
        assert f"`{field_name}`" in prompt
    for field_name in InitiativeSegmentExtraction.model_fields:
        assert f"`{field_name}`" in prompt
    for field_name in InitiativeSegmentStop.model_fields:
        assert f"`{field_name}`" in prompt
    for field_name in InitiativeExtraction.model_fields:
        if field_name == "city":
            continue
        assert f"`{field_name}`" in prompt
    assert "`document_local_code`" in prompt
    assert "`source_quote`" in prompt
    assert "pipeline assigns `city` programmatically from input `city_name`" in prompt
    assert "Do not create `source_refs`" in prompt
    assert "Return only `source_quote` as citation text" in prompt
    assert "Do not create TEF fields" in prompt
    assert "Each `InitiativeExtractionCandidate`" in prompt
    assert "`record_id`" in prompt
    assert "formal city initiatives" in prompt
    assert "workshop ideas" in prompt
    assert "legislative amendment proposals" in prompt
    assert "scope or activity labels" in prompt
    assert "umbrella strategy, roadmap, contract, or action-plan documents" in prompt
    assert "`document_local_code` belongs only on the outer `InitiativeExtractionCandidate`" in prompt
    assert "stop_initiative_extraction" in prompt


def test_segment_payload_includes_canonical_prior_only() -> None:
    """Prior initiative context should exclude artifact metadata and traces."""
    segment = InitiativeDocumentSegment(
        segment_id="krakow:krakow:seg0002:4-8",
        city_name="Krakow",
        source_document="Krakow.md",
        source_path="documents/Krakow.md",
        start_line=4,
        end_line=8,
        heading_path="Transport",
        content="Another action",
        token_count=3,
    )
    config = build_test_app_config()
    prior = [_candidate(code="BIC-1", segment_id="seg1")]

    payload = extractor_agent._segment_payload(segment, prior, config)

    history = payload["already_extracted_initiatives"]
    assert isinstance(history, list)
    assert len(history) == 1
    assert set(history[0]) == set(InitiativeExtraction.model_fields)
    assert "source_refs" not in history[0]
    assert "extraction_notes" not in history[0]
    assert "document_local_code" not in history[0]


def test_run_segment_once_reads_function_call_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extractor should read structured tool arguments when final_output is a string."""
    segment = InitiativeDocumentSegment(
        segment_id="krakow:krakow:seg0001:1-3",
        city_name="Krakow",
        source_document="Krakow.md",
        source_path="documents/Krakow.md",
        start_line=1,
        end_line=3,
        heading_path="Transport",
        content="B-2.2: Outline of individual activities - TR-1",
        token_count=10,
    )
    tool_payload = {
        "result": InitiativeSegmentExtraction(
            initiatives=[_candidate(code="TR-1", segment_id=segment.segment_id).initiative]
        ).model_dump(mode="json")
    }

    monkeypatch.setattr(extractor_agent, "_get_thread_agent", lambda *_args: object())
    monkeypatch.setattr(
        extractor_agent,
        "run_agent_sync",
        lambda *_args, **_kwargs: _FakeRunResult(
            "initiatives=[InitiativeExtraction(...)]",
            raw_responses=[
                {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "submit_initiative_extractions",
                            "arguments": json.dumps(tool_payload),
                        }
                    ]
                }
            ],
        ),
    )

    result = extractor_agent._run_segment_once(
        segment,
        build_test_app_config(),
        api_key="test",
        log_llm_payload=False,
        prior_initiatives=[],
        extraction_mode="initial",
        already_extracted_scope="run",
    )

    assert result.status == "success"
    assert len(result.initiatives) == 1
    assert result.initiatives[0].initiative.initiative_name == "Initiative TR-1"


def test_tool_payload_normalization_recovers_common_shape_errors() -> None:
    """Malformed-but-recoverable model output should not drop a whole segment."""
    payload = {
        "initiatives": [
            {
                "initiative": {
                    "initiative_name": "Dense segment initiative",
                    "general_description": "The city describes an initiative in a dense segment.",
                    "document_local_code": "BIC-99",
                    "source_refs": [
                        {
                            "source_document": "Krakow.md",
                            "segment_id": "seg1",
                            "start_line": 1,
                            "end_line": 2,
                        }
                    ],
                    "number_uncertain": ["possible value needs checking"],
                    "numbers": {"current": [], "planned": {"cost_pln": 1000}},
                },
                "number_context": [],
                "number_deferred": None,
                "number_uncertain": [],
            }
        ],
        "segment_data_quality_flags": [],
        "segment_notes": [],
        "error": None,
    }

    result = extractor_agent._coerce_segment_output_payload(
        "submit_initiative_extractions",
        payload,
        city_name="Krakow",
    )

    assert isinstance(result, InitiativeSegmentExtraction)
    candidate = result.initiatives[0]
    assert candidate.initiative.city == "Krakow"
    assert candidate.initiative.initiative_name == "Dense segment initiative"
    assert candidate.initiative.numbers.current == {}


def test_run_segment_once_overwrites_model_city_with_segment_city(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extractor should use segment metadata as the city and source-ref truth."""
    quote = "Developing business models for the development of EV charging stations"
    segment = InitiativeDocumentSegment(
        segment_id="rzeszow:rzeszow:seg0001:1-3",
        city_name="Rzeszow",
        source_document="Rzeszow.md",
        source_path="documents/Rzeszow.md",
        start_line=1,
        end_line=3,
        heading_path="Transport",
        content=quote,
        token_count=10,
    )

    monkeypatch.setattr(extractor_agent, "_get_thread_agent", lambda *_args: object())
    monkeypatch.setattr(
        extractor_agent,
        "run_agent_sync",
        lambda *_args, **_kwargs: _FakeRunResult(
            InitiativeSegmentExtraction(
                initiatives=[
                    _candidate(
                        code="MOB-1",
                        city="Rzesow",
                        name=quote,
                        source_quote=quote,
                        source_document="Wrong.md",
                        segment_id="wrong:seg",
                    )
                ]
            )
        ),
    )

    result = extractor_agent._run_segment_once(
        segment,
        build_test_app_config(),
        api_key="test",
        log_llm_payload=False,
        prior_initiatives=[],
        extraction_mode="initial",
        already_extracted_scope="run",
    )

    candidate = result.initiatives[0]
    assert candidate.initiative.city == "Rzeszow"
    assert len(candidate.source_refs) == 1
    source_ref = candidate.source_refs[0]
    assert source_ref.source_document == "Rzeszow.md"
    assert source_ref.segment_id == "rzeszow:rzeszow:seg0001:1-3"
    assert source_ref.start_line == 1
    assert source_ref.end_line == 3
    assert source_ref.section_heading == "Transport"
    assert "city_overridden_from_segment" in candidate.data_quality_flags


def test_run_segment_once_keeps_valid_quote_only_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extractor should keep a quote when it appears exactly in the segment."""
    quote = "Name of the action | Heat pumps"
    segment = InitiativeDocumentSegment(
        segment_id="krakow:krakow:seg0001:1-3",
        city_name="Krakow",
        source_document="Krakow.md",
        source_path="documents/Krakow.md",
        start_line=1,
        end_line=3,
        heading_path="Transport",
        content=f"| Description of operation | {quote} |",
        token_count=10,
    )

    monkeypatch.setattr(extractor_agent, "_get_thread_agent", lambda *_args: object())
    monkeypatch.setattr(
        extractor_agent,
        "run_agent_sync",
        lambda *_args, **_kwargs: _FakeRunResult(
            InitiativeSegmentExtraction(
                initiatives=[_candidate(code="TR-1", source_quote=quote)]
            )
        ),
    )

    result = extractor_agent._run_segment_once(
        segment,
        build_test_app_config(),
        api_key="test",
        log_llm_payload=False,
        prior_initiatives=[],
        extraction_mode="initial",
        already_extracted_scope="run",
    )

    assert result.initiatives[0].source_quote == quote
    assert "source_quote_missing" not in result.initiatives[0].data_quality_flags
    assert "source_quote_not_found" not in result.initiatives[0].data_quality_flags


def test_run_segment_once_drops_missing_quote_and_flags_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extractor should null invalid quotes before artifact writing."""
    segment = InitiativeDocumentSegment(
        segment_id="krakow:krakow:seg0001:1-3",
        city_name="Krakow",
        source_document="Krakow.md",
        source_path="documents/Krakow.md",
        start_line=1,
        end_line=3,
        heading_path="Transport",
        content="Only source-backed quotes are valid.",
        token_count=10,
    )

    monkeypatch.setattr(extractor_agent, "_get_thread_agent", lambda *_args: object())
    monkeypatch.setattr(
        extractor_agent,
        "run_agent_sync",
        lambda *_args, **_kwargs: _FakeRunResult(
            InitiativeSegmentExtraction(
                initiatives=[_candidate(code="TR-1", source_quote="Missing quote")]
            )
        ),
    )

    result = extractor_agent._run_segment_once(
        segment,
        build_test_app_config(),
        api_key="test",
        log_llm_payload=False,
        prior_initiatives=[],
        extraction_mode="initial",
        already_extracted_scope="run",
    )
    records = extractor_agent._build_candidate_records([result])
    review_items = extractor_agent._build_review_items(
        segments=[segment],
        raw_results=[result],
        records=records,
        duplicate_reviews=[],
        config=build_test_app_config(),
    )

    assert records[0].source_quote is None
    assert "source_quote_not_found" in records[0].data_quality_flags
    assert any(item.review_type == "source_quote_missing_or_invalid" for item in review_items)


def test_candidate_records_keep_repeated_local_codes() -> None:
    """Repeated local initiative codes should stay separate before semantic dedupe."""
    raw_results = [
        InitiativeRawSegmentResult(
            segment_id="seg1",
            source_document="Krakow.md",
            status="success",
            initiatives=[_candidate(code="BIC-1", segment_id="seg1")],
        ),
        InitiativeRawSegmentResult(
            segment_id="seg2",
            source_document="Krakow.md",
            status="success",
            initiatives=[_candidate(code="BIC-1", segment_id="seg2")],
        ),
    ]

    records = extractor_agent._build_candidate_records(raw_results)

    assert len(records) == 2
    assert {record.document_local_code for record in records} == {"BIC-1"}
    assert len({record.record_id for record in records}) == 2
    assert all(len(record.source_refs) == 1 for record in records)


def test_semantic_merge_keeps_clearest_source_quote() -> None:
    """Semantic duplicate merges should preserve the more informative quote."""
    raw_results = [
        InitiativeRawSegmentResult(
            segment_id="seg1",
            source_document="Krakow.md",
            status="success",
            initiatives=[_candidate(code="BIC-1", source_quote="Heat pumps")],
        ),
        InitiativeRawSegmentResult(
            segment_id="seg2",
            source_document="Krakow.md",
            status="success",
            initiatives=[
                _candidate(
                    code="BIC-1",
                    source_quote="Implementation of a local energy programme based on heat pumps",
                )
            ],
        ),
    ]

    candidate_records = extractor_agent._build_candidate_records(raw_results)
    records, _review_items = extractor_agent._apply_semantic_dedupe_groups(
        candidate_records,
        [
            InitiativeSemanticDedupeGroup(
                canonical_record_id=candidate_records[0].record_id,
                duplicate_record_ids=[candidate_records[1].record_id],
                confidence=0.9,
                rationale="Both rows describe the same BIC-1 initiative.",
            )
        ],
        build_test_app_config(),
    )

    assert records[0].source_quote == (
        "Implementation of a local energy programme based on heat pumps"
    )


def test_semantic_dedupe_merges_different_names() -> None:
    """Semantic dedupe should merge records that represent the same initiative."""
    base = InitiativeExtractionRecord(
        **_candidate(
            code="",
            name="Install public EV charging points",
            description="The city will install public EV charging points.",
        ).model_dump(mode="json"),
        record_id="krakow:krakow:title_a",
        source_document="Krakow.md",
    )
    duplicate = InitiativeExtractionRecord(
        **_candidate(
            code="",
            name="Expansion of charging stations",
            description="The city expands publicly accessible electric vehicle charging infrastructure.",
        ).model_dump(mode="json"),
        record_id="krakow:krakow:title_b",
        source_document="Krakow.md",
    )
    config = build_test_app_config(
        initiative_extractor_overrides={"semantic_dedupe_confidence_threshold": 0.78}
    )

    records, review_items = extractor_agent._apply_semantic_dedupe_groups(
        [base, duplicate],
        [
            InitiativeSemanticDedupeGroup(
                canonical_record_id=base.record_id,
                duplicate_record_ids=[duplicate.record_id],
                confidence=0.9,
                rationale="Both records describe the same EV charging infrastructure expansion.",
            )
        ],
        config,
    )

    assert len(records) == 1
    assert records[0].record_id == base.record_id
    assert {item.review_type for item in review_items} == {"semantic_duplicate_merged"}


def test_semantic_dedupe_payload_uses_canonical_fields_only() -> None:
    """Semantic dedupe input should not include source refs or extraction traces."""
    record = InitiativeExtractionRecord(
        **_candidate(code="BIC-1").model_dump(mode="json"),
        record_id="krakow:krakow:bic-1",
        source_document="Krakow.md",
    )

    payload = extractor_agent._semantic_dedupe_payload([record])
    item = payload["records"][0]

    assert set(item) == {
        "record_id",
        "document_local_code",
        "source_quote",
        *InitiativeExtraction.model_fields,
    }
    assert "source_refs" not in item
    assert "extraction_notes" not in item


def test_semantic_dedupe_prompt_contract_matches_payload_and_schema() -> None:
    """Semantic dedupe prompt should match runtime input and output contracts."""
    record = InitiativeExtractionRecord(
        **_candidate(code="BIC-1").model_dump(mode="json"),
        record_id="krakow:krakow:bic-1",
        source_document="Krakow.md",
    )
    payload = extractor_agent._semantic_dedupe_payload([record])
    prompt = Path("backend/prompts/initiative_semantic_dedupe_system.md").read_text(
        encoding="utf-8"
    )

    for field_name in payload:
        assert f"`{field_name}`" in prompt
    assert "`duplicate_groups`" in prompt
    assert "`canonical_record_id`" in prompt
    assert "`duplicate_record_ids`" in prompt
    assert "Do not create TEF fields" in prompt


def test_semantic_dedupe_tool_schema_matches_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic dedupe tool should expose prompt fields without a result wrapper."""
    monkeypatch.setattr(
        "backend.services.agents.build_openrouter_model",
        lambda *_args, **_kwargs: "test-model",
    )

    agent = extractor_agent.build_initiative_semantic_dedupe_agent(
        build_test_app_config(),
        api_key="test",
    )
    schema = agent.tools[0].params_json_schema

    assert set(schema["properties"]) == {"duplicate_groups", "review_notes"}
    assert "result" not in schema["properties"]


def test_process_segment_loops_action_heavy_segment_until_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense segments should loop on the same segment until the model stops."""
    segment = InitiativeDocumentSegment(
        segment_id="krakow:krakow:seg0001:1-50",
        city_name="Krakow",
        source_document="Krakow.md",
        source_path="documents/Krakow.md",
        start_line=1,
        end_line=50,
        heading_path="Dense actions",
        content="Many initiatives in one segment.",
        token_count=20,
    )
    first_result = InitiativeRawSegmentResult(
        segment_id=segment.segment_id,
        source_document=segment.source_document,
        status="success",
        initiatives=[_candidate(code=f"BIC-{index}") for index in range(1, 5)],
    )
    stop_result = InitiativeRawSegmentResult(
        segment_id=segment.segment_id,
        source_document=segment.source_document,
        status="success",
        extraction_complete=True,
        stop_reason="No additional initiatives remain.",
    )
    calls: list[dict[str, object]] = []

    def _fake_run_segment_with_retries(*_args: object, **kwargs: object) -> InitiativeRawSegmentResult:
        calls.append(kwargs)
        return first_result if len(calls) == 1 else stop_result

    monkeypatch.setattr(extractor_agent, "_run_segment_with_retries", _fake_run_segment_with_retries)

    result = extractor_agent._process_segment(
        segment,
        build_test_app_config(),
        api_key="test",
        log_llm_payload=False,
        run_id="test_run",
        prior_initiatives=[],
    )

    assert result.action_heavy is True
    assert result.extraction_iterations == 2
    assert result.extraction_complete is True
    assert result.stop_reason == "No additional initiatives remain."
    assert len(result.initiatives) == 4
    assert calls[0]["extraction_mode"] == "initial"
    assert calls[0]["already_extracted_scope"] == "run"
    assert calls[1]["extraction_mode"] == "dense_followup"
    assert calls[1]["already_extracted_scope"] == "current_segment"
    assert calls[1]["prior_initiatives"] == first_result.initiatives


def test_action_heavy_followup_uses_only_current_segment_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense follow-up should not include the rolling run prior from earlier segments."""
    segment = InitiativeDocumentSegment(
        segment_id="krakow:krakow:seg0001:1-50",
        city_name="Krakow",
        source_document="Krakow.md",
        source_path="documents/Krakow.md",
        start_line=1,
        end_line=50,
        heading_path="Dense actions",
        content="Many initiatives in one segment.",
        token_count=20,
    )
    global_prior = [_candidate(code="GLOBAL")]
    first_candidates = [_candidate(code=f"BIC-{index}") for index in range(1, 5)]
    captured_prior: list[list[InitiativeExtractionCandidate]] = []

    def _fake_run_segment_with_retries(*_args: object, **kwargs: object) -> InitiativeRawSegmentResult:
        captured_prior.append(kwargs["prior_initiatives"])
        if len(captured_prior) == 1:
            return InitiativeRawSegmentResult(
                segment_id=segment.segment_id,
                source_document=segment.source_document,
                status="success",
                initiatives=first_candidates,
            )
        return InitiativeRawSegmentResult(
            segment_id=segment.segment_id,
            source_document=segment.source_document,
            status="success",
            extraction_complete=True,
        )

    monkeypatch.setattr(extractor_agent, "_run_segment_with_retries", _fake_run_segment_with_retries)

    extractor_agent._process_segment(
        segment,
        build_test_app_config(),
        api_key="test",
        log_llm_payload=False,
        run_id="test_run",
        prior_initiatives=global_prior,
    )

    assert captured_prior[0] == global_prior
    assert captured_prior[1] == first_candidates


def test_extraction_pipeline_writes_artifacts_with_fake_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline should write staged JSONL artifacts from fake structured outputs."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    source = docs_dir / "Krakow.md"
    source.write_text(
        "\n".join(
            [
                "B-2.2: Outline of individual activities - BIC-1",
                "| Description of operation | Name of the action | Retrofit homes |",
                "| Estimated greenhouse gas emission reductions | 1 255 tCO2e |",
                "B-2.2: Outline of individual activities - BIC-2",
                "| Description of operation | Name of the action | Retrofit schools |",
                "| Estimated greenhouse gas emission reductions | 554 tCO2e |",
            ]
        ),
        encoding="utf-8",
    )
    config = build_test_app_config(
        initiative_extractor_overrides={
            "max_segment_tokens": 500,
            "segment_overlap_lines": 0,
            "max_workers": 1,
            "semantic_dedupe_enabled": False,
        }
    )

    monkeypatch.setattr(extractor_agent, "build_initiative_extractor_agent", lambda *_args: object())

    def _fake_run_agent_sync(_agent: object, input_data: str, **_kwargs: object) -> _FakeRunResult:
        payload = parse_llm_serialized(input_data)
        segment_id = payload["segment_id"]
        return _FakeRunResult(
            InitiativeSegmentExtraction(
                initiatives=[
                    _candidate(
                        code="BIC-1",
                        segment_id=segment_id,
                        source_quote="Retrofit homes",
                    ),
                    _candidate(
                        code="BIC-2",
                        segment_id=segment_id,
                        source_quote="Retrofit schools",
                    ),
                ],
            )
        )

    monkeypatch.setattr(extractor_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extractor_agent.extract_initiatives(
        markdown_path=docs_dir,
        config=config,
        api_key="test",
        output_root=tmp_path / "output",
        run_id="test_run",
        selected_cities=["Krakow"],
    )

    run_dir = Path(result.output_dir)
    initiatives_path = run_dir / "03_deduped" / "initiatives.jsonl"
    initiative_records_path = run_dir / "03_deduped" / "initiative_records.jsonl"
    candidate_records_path = run_dir / "03_deduped" / "candidate_records.jsonl"
    assert initiatives_path.exists()
    assert initiative_records_path.exists()
    assert candidate_records_path.exists()
    assert result.deduped_initiatives_count == 2
    assert (run_dir / "01_segments" / "segments.jsonl").exists()
    assert (run_dir / "02_raw_extractions" / "raw_segment_extractions.jsonl").exists()
    assert (run_dir / "04_review" / "review_items.jsonl").exists()
    rows = [
        json.loads(line)
        for line in initiatives_path.read_text(encoding="utf-8").splitlines()
    ]
    record_rows = [
        json.loads(line)
        for line in initiative_records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert set(rows[0]) == set(InitiativeExtraction.model_fields)
    assert "initiative" not in rows[0]
    assert "record_id" not in rows[0]
    assert "source_quote" not in rows[0]
    assert "source_refs" not in rows[0]
    assert "record_id" in record_rows[0]
    assert "initiative" in record_rows[0]
    assert "source_quote" in record_rows[0]
    assert not {
        "source_refs",
        "source_path",
        "segment_id",
        "start_line",
        "end_line",
        "source_ref_id",
    } & set(record_rows[0])
    assert "tef_id" not in json.dumps(rows)


def test_coverage_audit_flags_action_heavy_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage audit should flag segments that triggered dense follow-up extraction."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Krakow.md").write_text(
        "\n".join(
            [
                "B-2.2: Outline of individual activities - BIC-1",
                "| Description of operation | Name of the action | Retrofit homes |",
                "| Total and unit costs | PLN 1 000 000 |",
            ]
        ),
        encoding="utf-8",
    )
    config = build_test_app_config(
        initiative_extractor_overrides={
            "max_segment_tokens": 500,
            "segment_overlap_lines": 0,
            "max_workers": 1,
            "semantic_dedupe_enabled": False,
        }
    )

    monkeypatch.setattr(extractor_agent, "build_initiative_extractor_agent", lambda *_args: object())
    outputs = [
        InitiativeSegmentExtraction(
            initiatives=[_candidate(code=f"BIC-{index}").initiative for index in range(1, 5)]
        ),
        InitiativeSegmentStop(reason="No additional initiatives remain."),
    ]

    def _fake_run_agent_sync(_agent: object, _input_data: str, **_kwargs: object) -> _FakeRunResult:
        return _FakeRunResult(outputs.pop(0))

    monkeypatch.setattr(extractor_agent, "run_agent_sync", _fake_run_agent_sync)

    result = extractor_agent.extract_initiatives(
        markdown_path=docs_dir,
        config=config,
        api_key="test",
        output_root=tmp_path / "output",
        run_id="coverage",
        selected_cities=["Krakow"],
    )

    review_rows = [
        json.loads(line)
        for line in (Path(result.output_dir) / "04_review" / "review_items.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    review_types = {row["review_type"] for row in review_rows}
    assert "action_heavy_segment" in review_types
