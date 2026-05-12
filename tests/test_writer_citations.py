import logging
import json
from pathlib import Path
import re

import pytest

from backend.modules.writer import agent as writer_agent
from backend.modules.writer.models import WriterOutput
from backend.modules.writer.utils.markdown_helpers import (
    extract_cited_ref_ids,
    normalize_reference_citations,
)
from backend.modules.writer.utils.multi_pass import build_writer_context_bundle
from backend.services.run_logger import RunLogger
from backend.utils.config import AppConfig
from backend.utils.llm_serialization import parse_llm_serialized
from backend.utils.paths import create_run_paths
from tests.support import build_test_app_config


class _FakeRunResult:
    def __init__(self, final_output: WriterOutput) -> None:
        self.final_output = final_output


def _extract_coverage_payloads(records: list[logging.LogRecord]) -> list[dict[str, object]]:
    """Parse WRITER_CITATION_COVERAGE payloads from captured logs."""
    payloads: list[dict[str, object]] = []
    for record in records:
        message = record.message
        if not message.startswith("WRITER_CITATION_COVERAGE "):
            continue
        payload_raw = message.split("WRITER_CITATION_COVERAGE ", 1)[1].strip()
        payload = json.loads(payload_raw)
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _build_test_config(tmp_path: Path) -> AppConfig:
    """Build the writer test config with the required agent sections."""
    return build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
    )


def test_normalize_reference_citations_canonicalizes_compact_tokens() -> None:
    content = "Munich [ref1] Berlin [ ref2 ] Keep [note] Invalid [ref_0]"

    normalized = normalize_reference_citations(content)

    assert normalized == "Munich [ref_1] Berlin [ref_2] Keep [note] Invalid [ref_0]"
    assert extract_cited_ref_ids(normalized) == {"ref_1", "ref_2"}


def test_normalize_reference_citations_logs_warning_for_rewrites(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="backend.modules.writer.utils.markdown_helpers")

    normalized = normalize_reference_citations("Munich [ref1] Berlin [ref2]")

    assert normalized == "Munich [ref_1] Berlin [ref_2]"
    assert any(
        "Normalized compact reference citations" in record.message for record in caplog.records
    )


def test_writer_logs_warning_when_citations_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 1,
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich has 43 charging points as of 2024.",
                    "partial_answer": "Munich has 43 charging points as of 2024.",
                }
            ],
        }
    }

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        writer_agent,
        "run_agent_sync",
        lambda *_args, **_kwargs: _FakeRunResult(
            WriterOutput(content="## Answer\n\nMunich has 43 charging points as of 2024.")
        ),
    )

    caplog.set_level(logging.WARNING, logger=writer_agent.__name__)

    writer_agent.write_markdown(
        question="What charging points are documented?",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
    )

    messages = [record.message for record in caplog.records]
    assert any("contains no [ref_n] citations" in message for message in messages)


def test_writer_logs_warning_when_unknown_ref_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 1,
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Leipzig",
                    "quote": "Leipzig plans charging expansion.",
                    "partial_answer": "Leipzig plans charging expansion.",
                }
            ],
        }
    }

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        writer_agent,
        "run_agent_sync",
        lambda *_args, **_kwargs: _FakeRunResult(
            WriterOutput(content="Leipzig plans charging expansion. [ref_99]")
        ),
    )

    caplog.set_level(logging.WARNING, logger=writer_agent.__name__)

    writer_agent.write_markdown(
        question="What is Leipzig planning?",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
    )

    messages = [record.message for record in caplog.records]
    assert any("unknown reference ids: ref_99" in message for message in messages)


def test_writer_retries_when_city_citation_coverage_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich charging evidence.",
                    "partial_answer": "Munich charging evidence.",
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Berlin",
                    "quote": "Berlin charging evidence.",
                    "partial_answer": "Berlin charging evidence.",
                },
            ],
        }
    }

    captured_inputs: list[dict[str, object]] = []
    responses = [
        WriterOutput(content="Munich update [ref_1]"),
        WriterOutput(content="Munich update [ref_1]\nBerlin update [ref_2]"),
    ]

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(
        _agent: object,
        input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        captured_inputs.append(parse_llm_serialized(input_text))
        output = responses.pop(0)
        return _FakeRunResult(output)

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)
    caplog.set_level(logging.INFO, logger=writer_agent.__name__)

    output = writer_agent.write_markdown(
        question="Summarize city charging evidence.",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-writer-retry",
    )

    assert len(captured_inputs) == 2
    assert "reconsideration" not in captured_inputs[0]
    assert isinstance(captured_inputs[1].get("reconsideration"), dict)
    assert "Berlin update [ref_2]" in output.content
    assert "## Cities considered" in output.content
    assert "- Munich" in output.content
    assert "- Berlin" in output.content
    coverage_payloads = _extract_coverage_payloads(caplog.records)
    assert any(
        payload.get("status") == "retrying" and payload.get("coverage_ratio") == "1/2"
        for payload in coverage_payloads
    )
    assert any(
        payload.get("status") == "confirmed" and payload.get("coverage_ratio") == "2/2"
        for payload in coverage_payloads
    )


def test_writer_normalizes_compact_reference_tokens_before_coverage_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich charging evidence.",
                    "partial_answer": "Munich charging evidence.",
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Berlin",
                    "quote": "Berlin charging evidence.",
                    "partial_answer": "Berlin charging evidence.",
                },
            ],
        }
    }

    captured_inputs: list[dict[str, object]] = []
    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(
        _agent: object,
        input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        captured_inputs.append(parse_llm_serialized(input_text))
        return _FakeRunResult(WriterOutput(content="Munich update [ref1]\nBerlin update [ref2]"))

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)
    caplog.set_level(logging.INFO)

    output = writer_agent.write_markdown(
        question="Summarize city charging evidence.",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-writer-compact-refs",
    )

    assert len(captured_inputs) == 1
    assert "[ref_1]" in output.content
    assert "[ref_2]" in output.content
    assert "[ref1]" not in output.content
    assert "[ref2]" not in output.content
    coverage_payloads = _extract_coverage_payloads(caplog.records)
    assert any(
        payload.get("status") == "confirmed" and payload.get("coverage_ratio") == "2/2"
        for payload in coverage_payloads
    )


def test_writer_treats_city_name_aliases_as_one_city_for_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Coverage should not split the same city across hyphen and underscore aliases."""
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 1,
            "selected_city_names": ["Vitoria-Gasteiz", "Vitoria_Gasteiz"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Vitoria_Gasteiz",
                    "city_key": "vitoria-gasteiz",
                    "quote": "Vitoria evidence.",
                    "partial_answer": "Vitoria evidence.",
                }
            ],
        }
    }

    captured_inputs: list[dict[str, object]] = []
    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(
        _agent: object,
        input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        captured_inputs.append(parse_llm_serialized(input_text))
        return _FakeRunResult(WriterOutput(content="Vitoria summary [ref_1]"))

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)
    caplog.set_level(logging.INFO, logger=writer_agent.__name__)

    output = writer_agent.write_markdown(
        question="Summarize Vitoria evidence.",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-vitoria-alias",
    )

    assert len(captured_inputs) == 1
    assert "reconsideration" not in captured_inputs[0]
    assert "Vitoria summary [ref_1]" in output.content
    coverage_payloads = _extract_coverage_payloads(caplog.records)
    assert any(
        payload.get("status") == "confirmed" and payload.get("coverage_ratio") == "1/1"
        for payload in coverage_payloads
    )


def test_writer_appends_no_evidence_section_for_selected_city_without_excerpts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 1,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich evidence.",
                    "partial_answer": "Munich evidence.",
                }
            ],
        }
    }

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        writer_agent,
        "run_agent_sync",
        lambda *_args, **_kwargs: _FakeRunResult(
            WriterOutput(content="Munich evidence summary [ref_1]")
        ),
    )

    output = writer_agent.write_markdown(
        question="Summarize selected cities.",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
    )

    assert "## Cities with no important evidence found" in output.content
    assert "- Berlin: no important evidence was found in the provided excerpts." in output.content
    assert "## Cities considered" in output.content
    assert "- Munich" in output.content
    assert "- Berlin" in output.content


def test_writer_does_not_retry_for_layout_when_city_coverage_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "analysis_mode": "aggregate",
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich evidence.",
                    "partial_answer": "Munich evidence.",
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Berlin",
                    "quote": "Berlin evidence.",
                    "partial_answer": "Berlin evidence.",
                },
            ],
        },
    }

    captured_inputs: list[dict[str, object]] = []
    responses = [
        WriterOutput(
            content=(
                "## What’s distinctive\n"
                "- **Munich:** Needs charging rollout. [ref_1]\n"
                "- **Berlin:** Needs network upgrades. [ref_2]"
            )
        ),
        WriterOutput(
            content=(
                "## Group Synthesis\n"
                "Across Munich and Berlin, shared needs are charging rollout and "
                "network upgrades. [ref_1][ref_2]"
            )
        ),
    ]

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(
        _agent: object,
        input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        captured_inputs.append(parse_llm_serialized(input_text))
        return _FakeRunResult(responses.pop(0))

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)
    caplog.set_level(logging.INFO, logger=writer_agent.__name__)

    output = writer_agent.write_markdown(
        question="What are shared needs and quantities?",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-aggregate-retry",
    )

    assert len(captured_inputs) == 1
    assert "reconsideration" not in captured_inputs[0]
    assert "## What" in output.content
    assert "## Cities considered" in output.content
    coverage_payloads = _extract_coverage_payloads(caplog.records)
    assert any(
        payload.get("status") == "confirmed" and payload.get("coverage_ratio") == "2/2"
        for payload in coverage_payloads
    )


def test_writer_allows_city_by_city_layout_when_question_explicitly_requests_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "analysis_mode": "aggregate",
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich evidence.",
                    "partial_answer": "Munich evidence.",
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Berlin",
                    "quote": "Berlin evidence.",
                    "partial_answer": "Berlin evidence.",
                },
            ],
        },
    }

    call_count = {"count": 0}
    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())

    def _single_city_by_city_output(*_args, **_kwargs) -> _FakeRunResult:
        call_count["count"] += 1
        return _FakeRunResult(
            WriterOutput(
                content=(
                    "## Per city\n"
                    "- **Munich:** Needs charging rollout. [ref_1]\n"
                    "- **Berlin:** Needs network upgrades. [ref_2]"
                )
            )
        )

    monkeypatch.setattr(writer_agent, "run_agent_sync", _single_city_by_city_output)

    output = writer_agent.write_markdown(
        question="Please provide the answer city by city with separate sections.",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
    )

    assert call_count["count"] == 1
    assert "## Per city" in output.content


def test_writer_does_not_retry_for_plain_city_prefixed_lines_when_covered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "analysis_mode": "aggregate",
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Athens", "Istanbul"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Athens",
                    "quote": "Athens evidence.",
                    "partial_answer": "Athens evidence.",
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Istanbul",
                    "quote": "Istanbul evidence.",
                    "partial_answer": "Istanbul evidence.",
                },
            ],
        },
    }

    captured_inputs: list[dict[str, object]] = []
    responses = [
        WriterOutput(
            content=(
                "## Distinctive needs\n"
                "Athens: needs cooling and charging rollout. [ref_1]\n"
                "Istanbul: needs wastewater and transit upgrades. [ref_2]"
            )
        ),
        WriterOutput(
            content=(
                "## Group synthesis\n"
                "Across Athens and Istanbul, shared needs are cooling adaptation, "
                "mobility electrification, and network upgrades. [ref_1][ref_2]"
            )
        ),
    ]

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(
        _agent: object,
        input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        captured_inputs.append(parse_llm_serialized(input_text))
        return _FakeRunResult(responses.pop(0))

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)

    output = writer_agent.write_markdown(
        question="What are shared needs and quantities?",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-aggregate-prefix-retry",
    )

    assert len(captured_inputs) == 1
    assert "reconsideration" not in captured_inputs[0]
    assert "## Distinctive needs" in output.content


def test_writer_replaces_existing_model_footer_with_canonical_footer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich evidence.",
                    "partial_answer": "Munich evidence.",
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Berlin",
                    "quote": "Berlin evidence.",
                    "partial_answer": "Berlin evidence.",
                },
            ],
        }
    }

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        writer_agent,
        "run_agent_sync",
        lambda *_args, **_kwargs: _FakeRunResult(
            WriterOutput(
                content=(
                    "## Summary\n"
                    "Shared needs include charging and heating upgrades. [ref_1][ref_2]\n\n"
                    "Cities considered:\n"
                    "- Munich [ref_1]\n"
                    "- Berlin [ref_2]"
                )
            )
        ),
    )

    output = writer_agent.write_markdown(
        question="Summarize selected cities.",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
    )

    footer_matches = re.findall(
        r"(?im)^\s*(?:##\s*cities considered|cities considered:)\s*$",
        output.content,
    )
    assert len(footer_matches) == 1
    assert "## Cities considered" in output.content
    assert "- Munich [ref_1]" not in output.content
    assert "- Berlin [ref_2]" not in output.content


def test_writer_footer_does_not_mark_external_evidence_city_as_no_evidence() -> None:
    context_bundle: dict[str, object] = {
        "selected_cities": ["Krakow"],
        "markdown": {
            "excerpt_count": 0,
            "selected_city_names": ["Krakow"],
            "excerpts": [],
        },
        "enrichment": {
            "external_evidence": [
                {
                    "city": "Krakow",
                    "field": "transport_climate_investment_2023_eur",
                    "value": 448039290,
                    "source_id": "krakow-investment",
                    "line_start": 341,
                    "line_end": 361,
                }
            ],
        },
    }

    content, *_ = writer_agent._prepare_writer_content(
        content=(
            "## Executive Summary\n"
            "Krakow is covered by external evidence (krakow-investment:L341-L361)."
        ),
        context_bundle=context_bundle,
        selected_city_names=["Krakow"],
    )

    assert "## Cities with no important evidence found" not in content
    assert "## Cities considered" in content
    assert "- Krakow" in content


def test_writer_footer_marks_ccc_only_enrichment_as_no_evidence() -> None:
    context_bundle: dict[str, object] = {
        "selected_cities": ["Krakow"],
        "markdown": {
            "excerpt_count": 0,
            "selected_city_names": ["Krakow"],
            "excerpts": [],
        },
        "enrichment": {
            "enriched_fields": [
                {
                    "city": "Krakow",
                    "field": "secap_local_co2_reduction_2030_target",
                    "status": "resolved",
                    "source": "ccc",
                }
            ],
        },
    }

    content, *_ = writer_agent._prepare_writer_content(
        content="## Executive Summary\nKrakow has only CCC status metadata.",
        context_bundle=context_bundle,
        selected_city_names=["Krakow"],
    )

    assert "## Cities with no important evidence found" in content
    assert "- Krakow: no important evidence was found in the provided excerpts." in content


def test_build_writer_context_bundle_keeps_only_writer_relevant_markdown_fields() -> None:
    context_bundle: dict[str, object] = {
        "research_question": "Refined question",
        "analysis_mode": "aggregate",
        "final": "output/final.md",
        "enrichment": {
            "field_manifest": {
                "query_fields": [{"field": "capex", "scope": "municipal"}],
                "non_estimable_fields": [],
            },
            "gap_manifest": {"city_gaps": []},
            "status": "internal bookkeeping",
        },
        "markdown": {
            "status": "success",
            "analysis_mode": "aggregate",
            "excerpt_count": 2,
            "accepted_chunk_ids": ["chunk-1"],
            "rejected_chunk_ids": ["chunk-2"],
            "unresolved_chunk_ids": ["chunk-3"],
            "batch_failures": [{"city_name": "Munich"}],
            "decision_audit": {"ok": True},
            "error": {"code": "PARTIAL"},
            "selected_city_names": ["Munich", "Berlin"],
            "inspected_city_names": ["Munich", "Berlin"],
            "selected_cities": ["munich", "berlin"],
            "inspected_cities": ["munich", "berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich evidence.",
                    "partial_answer": "Munich evidence.",
                }
            ],
        },
    }

    markdown_payload = context_bundle["markdown"]
    assert isinstance(markdown_payload, dict)
    excerpts = markdown_payload["excerpts"]
    assert isinstance(excerpts, list)
    writer_bundle = build_writer_context_bundle(
        context_bundle=context_bundle,
        excerpts=excerpts,
        city_names=["Munich", "Berlin"],
    )

    assert writer_bundle["research_question"] == "Refined question"
    assert "sql" not in writer_bundle
    assert "final" not in writer_bundle
    enrichment = writer_bundle["enrichment"]
    assert isinstance(enrichment, dict)
    assert enrichment["field_manifest"]["query_fields"][0]["field"] == "capex"
    assert enrichment["gap_manifest"] == {"city_gaps": []}
    assert "status" not in enrichment
    markdown_bundle = writer_bundle["markdown"]
    assert isinstance(markdown_bundle, dict)
    assert markdown_bundle["status"] == "success"
    assert markdown_bundle["excerpt_count"] == 1
    assert markdown_bundle["selected_city_names"] == ["Munich", "Berlin"]
    assert "accepted_chunk_ids" not in markdown_bundle
    assert "rejected_chunk_ids" not in markdown_bundle
    assert "unresolved_chunk_ids" not in markdown_bundle
    assert "batch_failures" not in markdown_bundle
    assert "decision_audit" not in markdown_bundle
    assert "error" not in markdown_bundle


def test_build_writer_context_bundle_filters_enrichment_to_batch_cities() -> None:
    context_bundle: dict[str, object] = {
        "research_question": "Refined question",
        "analysis_mode": "city_by_city",
        "enrichment": {
            "field_manifest": {
                "query_fields": [{"field": "capex", "scope": "municipal"}],
                "non_estimable_fields": [],
            },
            "gap_manifest": {
                "city_gaps": [
                    {"city": "Munich", "blank_fields": ["capex"]},
                    {"city": "Berlin", "blank_fields": ["capex"]},
                ]
            },
            "enriched_fields": [
                {"city": "Munich", "field": "capex"},
                {"city": "Berlin", "field": "capex"},
            ],
            "external_evidence": [
                {"city": "Munich", "field": "capex"},
                {"city": "Berlin", "field": "capex"},
            ],
            "external_resolutions": [
                {"city": "Munich", "field": "capex"},
                {"city": "Berlin", "field": "capex"},
            ],
            "external_no_evidence": [
                {"city": "Munich", "field": "capex"},
                {"city": "Berlin", "field": "capex"},
            ],
            "assumptions": [
                {"city": "Munich", "field_name": "capex"},
                {"city": "Berlin", "field_name": "capex"},
            ],
            "non_estimable": [
                {"city": "Munich", "field_name": "capex"},
                {"city": "Berlin", "field_name": "capex"},
            ],
            "web_findings": [
                {"city": "Munich", "field": "capex"},
                {"city": "Berlin", "field": "capex"},
            ],
            "freshness_results": [
                {"city": "Munich", "field": "capex"},
                {"city": "Berlin", "field": "capex"},
            ],
            "meta": {"total_gaps": 2},
        },
        "markdown": {
            "status": "success",
            "analysis_mode": "city_by_city",
            "excerpt_count": 0,
            "excerpts": [],
        },
    }

    writer_bundle = build_writer_context_bundle(
        context_bundle=context_bundle,
        excerpts=[],
        city_names=["Munich"],
    )

    enrichment = writer_bundle["enrichment"]
    assert isinstance(enrichment, dict)
    assert enrichment["field_manifest"]["query_fields"][0]["field"] == "capex"
    assert enrichment["meta"] == {"total_gaps": 2}
    assert enrichment["gap_manifest"]["city_gaps"] == [
        {"city": "Munich", "blank_fields": ["capex"]}
    ]
    for key in (
        "enriched_fields",
        "external_evidence",
        "external_resolutions",
        "external_no_evidence",
        "assumptions",
        "non_estimable",
        "web_findings",
        "freshness_results",
    ):
        assert [record["city"] for record in enrichment[key]] == ["Munich"]


def test_writer_returns_partial_coverage_metadata_after_retry_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich charging evidence.",
                    "partial_answer": "Munich charging evidence.",
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Berlin",
                    "quote": "Berlin charging evidence.",
                    "partial_answer": "Berlin charging evidence.",
                },
            ],
        }
    }

    responses = [
        WriterOutput(content="Munich update [ref_1]"),
        WriterOutput(content="Munich update [ref_1]"),
    ]

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())

    def _fake_run_agent_sync(
        _agent: object,
        _input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        return _FakeRunResult(responses.pop(0))

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)
    caplog.set_level(logging.WARNING, logger=writer_agent.__name__)

    output = writer_agent.write_markdown(
        question="Summarize city charging evidence.",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-writer-partial",
    )

    assert output.citation_coverage is not None
    assert output.citation_coverage.status == "partial"
    assert output.citation_coverage.coverage_ratio == "1/2"
    assert output.citation_coverage.missing_cities == ["Berlin"]
    coverage_payloads = _extract_coverage_payloads(caplog.records)
    assert any(
        payload.get("status") == "exhausted" and payload.get("coverage_ratio") == "1/2"
        for payload in coverage_payloads
    )


def test_writer_uses_multi_pass_batches_and_combines_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    config.writer.multi_pass_threshold_tokens = 200
    config.writer.multi_pass_chunk_tokens = 200
    question = "Summarize retrofit evidence across selected cities."
    context_bundle: dict[str, object] = {
        "analysis_mode": "aggregate",
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich retrofit evidence. " * 80,
                    "partial_answer": "Munich retrofit evidence. " * 80,
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Berlin",
                    "quote": "Berlin retrofit evidence. " * 80,
                    "partial_answer": "Berlin retrofit evidence. " * 80,
                },
            ],
        },
    }

    paths = create_run_paths(
        config.runs_dir,
        "run-writer-multi-pass",
        config.orchestrator.context_bundle_name,
    )
    run_logger = RunLogger(paths, question)
    captured_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        writer_agent,
        "build_writer_combine_agent",
        lambda *_args, **_kwargs: object(),
    )

    def _fake_run_agent_sync(
        _agent: object,
        input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        payload = parse_llm_serialized(input_text)
        captured_payloads.append(payload)
        if "draft_answers" in payload:
            return _FakeRunResult(
                WriterOutput(
                    content=(
                        "## Group synthesis\n"
                        "Munich scales deep retrofit delivery. [ref_1]\n"
                        "Berlin focuses on existing-building upgrades. [ref_2]"
                    )
                )
            )
        selected_cities = payload.get("selected_cities")
        if selected_cities == ["Munich"]:
            return _FakeRunResult(
                WriterOutput(content="Munich scales deep retrofit delivery. [ref_1]")
            )
        if selected_cities == ["Berlin"]:
            return _FakeRunResult(
                WriterOutput(content="Berlin focuses on existing-building upgrades. [ref_2]")
            )
        raise AssertionError(f"Unexpected payload: {payload}")

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)

    output = writer_agent.write_markdown(
        question=question,
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-writer-multi-pass",
        run_logger=run_logger,
        paths=paths,
    )

    assert len(captured_payloads) == 3
    assert captured_payloads[0]["selected_cities"] == ["Munich"]
    assert captured_payloads[1]["selected_cities"] == ["Berlin"]
    assert "draft_answers" in captured_payloads[2]
    assert output.citation_coverage is not None
    assert output.citation_coverage.status == "confirmed"
    assert output.citation_coverage.coverage_ratio == "2/2"
    assert "## Cities considered" in output.content
    assert "- Munich" in output.content
    assert "- Berlin" in output.content
    assert run_logger.run_log["writer_multi_pass"]["batch_count"] == 2
    assert "writer_multi_pass" in run_logger.run_log["artifacts"]
    artifact_path = Path(run_logger.run_log["artifacts"]["writer_multi_pass"])
    assert artifact_path.exists()


def test_writer_retries_transient_error_while_combining_multi_pass_drafts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Multi-pass combine should retry transient provider failures before surfacing an error."""
    config = _build_test_config(tmp_path)
    config.writer.multi_pass_threshold_tokens = 200
    config.writer.multi_pass_chunk_tokens = 200
    config.retry.backoff_base_seconds = 0.0
    config.retry.backoff_max_seconds = 0.0
    question = "Summarize retrofit evidence across selected cities."
    context_bundle: dict[str, object] = {
        "analysis_mode": "aggregate",
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich retrofit evidence. " * 80,
                    "partial_answer": "Munich retrofit evidence. " * 80,
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Berlin",
                    "quote": "Berlin retrofit evidence. " * 80,
                    "partial_answer": "Berlin retrofit evidence. " * 80,
                },
            ],
        },
    }

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        writer_agent,
        "build_writer_combine_agent",
        lambda *_args, **_kwargs: object(),
    )

    combine_attempts = {"count": 0}

    def _fake_run_agent_sync(
        _agent: object,
        input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        payload = parse_llm_serialized(input_text)
        if "draft_answers" in payload:
            combine_attempts["count"] += 1
            if combine_attempts["count"] == 1:
                raise json.JSONDecodeError("temporary combine failure", "{}", 0)
            return _FakeRunResult(
                WriterOutput(
                    content=(
                        "## Group synthesis\n"
                        "Munich scales deep retrofit delivery. [ref_1]\n"
                        "Berlin focuses on existing-building upgrades. [ref_2]"
                    )
                )
            )
        selected_cities = payload.get("selected_cities")
        if selected_cities == ["Munich"]:
            return _FakeRunResult(
                WriterOutput(content="Munich scales deep retrofit delivery. [ref_1]")
            )
        if selected_cities == ["Berlin"]:
            return _FakeRunResult(
                WriterOutput(content="Berlin focuses on existing-building upgrades. [ref_2]")
            )
        raise AssertionError(f"Unexpected payload: {payload}")

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)
    caplog.set_level(logging.WARNING)

    output = writer_agent.write_markdown(
        question=question,
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-writer-combine-retry",
    )

    assert combine_attempts["count"] == 2
    assert output.citation_coverage is not None
    assert output.citation_coverage.status == "confirmed"
    assert any(
        "RETRY_EVENT operation=writer.combine_llm_call" in record.message
        for record in caplog.records
    )


def test_writer_errors_when_post_batch_payload_still_exceeds_llm_input_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writer should fail closed when one remaining batch still exceeds the LLM input budget."""
    config = _build_test_config(tmp_path)
    config.writer.multi_pass_threshold_tokens = 100
    config.writer.multi_pass_chunk_tokens = 100
    config.writer.max_input_tokens = 120
    question = "Summarize retrofit evidence across selected cities."
    context_bundle: dict[str, object] = {
        "analysis_mode": "aggregate",
        "markdown": {
            "excerpt_count": 1,
            "selected_city_names": ["Munich"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich retrofit evidence. " * 120,
                    "partial_answer": "Munich retrofit evidence. " * 120,
                }
            ],
        },
    }

    build_writer_agent_called = {"value": False}

    def _unexpected_agent(*_args: object, **_kwargs: object) -> object:
        build_writer_agent_called["value"] = True
        return object()

    monkeypatch.setattr(writer_agent, "build_writer_agent", _unexpected_agent)

    with pytest.raises(
        ValueError,
        match="Writer payload remains above the configured LLM input token limit after batching",
    ):
        writer_agent.write_markdown(
            question=question,
            context_bundle=context_bundle,
            config=config,
            api_key="test-key",
            log_llm_payload=False,
            run_id="run-writer-oversize-batch",
        )

    assert build_writer_agent_called["value"] is False


def test_writer_multi_pass_diagnostics_can_be_persisted_without_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    config.writer.multi_pass_threshold_tokens = 200
    config.writer.multi_pass_chunk_tokens = 200
    question = "Summarize retrofit evidence across selected cities."
    context_bundle: dict[str, object] = {
        "analysis_mode": "aggregate",
        "markdown": {
            "excerpt_count": 2,
            "selected_city_names": ["Munich", "Berlin"],
            "excerpts": [
                {
                    "ref_id": "ref_1",
                    "city_name": "Munich",
                    "quote": "Munich retrofit evidence. " * 80,
                    "partial_answer": "Munich retrofit evidence. " * 80,
                },
                {
                    "ref_id": "ref_2",
                    "city_name": "Berlin",
                    "quote": "Berlin retrofit evidence. " * 80,
                    "partial_answer": "Berlin retrofit evidence. " * 80,
                },
            ],
        },
    }

    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        writer_agent,
        "build_writer_combine_agent",
        lambda *_args, **_kwargs: object(),
    )

    def _fake_run_agent_sync(
        _agent: object,
        input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        payload = parse_llm_serialized(input_text)
        if "draft_answers" in payload:
            return _FakeRunResult(
                WriterOutput(
                    content=(
                        "## Group synthesis\n"
                        "Munich scales deep retrofit delivery. [ref_1]\n"
                        "Berlin focuses on existing-building upgrades. [ref_2]"
                    )
                )
            )
        selected_cities = payload.get("selected_cities")
        if selected_cities == ["Munich"]:
            return _FakeRunResult(
                WriterOutput(content="Munich scales deep retrofit delivery. [ref_1]")
            )
        if selected_cities == ["Berlin"]:
            return _FakeRunResult(
                WriterOutput(content="Berlin focuses on existing-building upgrades. [ref_2]")
            )
        raise AssertionError(f"Unexpected payload: {payload}")

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)

    output = writer_agent.write_markdown(
        question=question,
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-writer-multi-pass",
    )

    assert output.citation_coverage is not None
    assert output.citation_coverage.status == "confirmed"
