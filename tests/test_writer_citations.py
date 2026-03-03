import logging
import json
from pathlib import Path
import re

import pytest

from backend.modules.writer import agent as writer_agent
from backend.modules.writer.models import WriterOutput
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)


class _FakeRunResult:
    def __init__(self, final_output: WriterOutput) -> None:
        self.final_output = final_output


class _FakeRunData:
    def __init__(self, raw_responses: list[object]) -> None:
        self.raw_responses = raw_responses


class _FakeMaxTurnsExceeded(Exception):
    def __init__(self, message: str, run_data: object | None = None) -> None:
        super().__init__(message)
        self.run_data = run_data


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
    return AppConfig(
        orchestrator=OrchestratorConfig(
            model="test-model",
            context_bundle_name="context_bundle.json",
        ),
        sql_researcher=SqlResearcherConfig(model="test-model"),
        markdown_researcher=MarkdownResearcherConfig(model="test-model"),
        writer=AgentConfig(model="test-model"),
        runs_dir=tmp_path / "output",
        markdown_dir=tmp_path / "documents",
        enable_sql=False,
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
        captured_inputs.append(json.loads(input_text))
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
        captured_inputs.append(json.loads(input_text))
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
        captured_inputs.append(json.loads(input_text))
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


def test_writer_switches_to_no_tool_fallback_after_max_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 1,
            "selected_city_names": ["Munich"],
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

    primary_agent = object()
    fallback_agent = object()
    captured_runs: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(writer_agent, "MaxTurnsExceeded", _FakeMaxTurnsExceeded)
    monkeypatch.setattr(
        writer_agent,
        "build_writer_agent",
        lambda *_args, **_kwargs: primary_agent,
    )
    monkeypatch.setattr(
        writer_agent,
        "build_writer_no_tool_agent",
        lambda *_args, **_kwargs: fallback_agent,
    )

    def _fake_run_agent_sync(
        agent: object,
        input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        payload = json.loads(input_text)
        captured_runs.append((agent, payload))
        if len(captured_runs) == 1:
            raise _FakeMaxTurnsExceeded(
                "Max turns (10) exceeded",
                run_data=_FakeRunData(
                    raw_responses=[
                        {
                            "output": [
                                {"type": "output_text", "text": "Primary draft summary [ref_1]"}
                            ]
                        }
                    ]
                ),
            )
        return _FakeRunResult(WriterOutput(content="Fallback final answer [ref_1]"))

    monkeypatch.setattr(writer_agent, "run_agent_sync", _fake_run_agent_sync)
    caplog.set_level(logging.WARNING, logger=writer_agent.__name__)

    output = writer_agent.write_markdown(
        question="Summarize selected cities.",
        context_bundle=context_bundle,
        config=config,
        api_key="test-key",
        log_llm_payload=False,
        run_id="run-fallback-success",
    )

    assert len(captured_runs) == 2
    assert captured_runs[0][0] is primary_agent
    assert captured_runs[1][0] is fallback_agent
    reconsideration = captured_runs[1][1].get("reconsideration")
    assert isinstance(reconsideration, dict)
    assert reconsideration.get("previous_answer") == "Primary draft summary [ref_1]"
    assert output.content.startswith("Fallback final answer [ref_1]")
    assert any(
        "WRITER_MAX_TURNS_FALLBACK" in record.message for record in caplog.records
    )


def test_writer_raises_when_no_tool_fallback_also_hits_max_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_test_config(tmp_path)
    context_bundle: dict[str, object] = {
        "markdown": {
            "excerpt_count": 1,
            "selected_city_names": ["Munich"],
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

    monkeypatch.setattr(writer_agent, "MaxTurnsExceeded", _FakeMaxTurnsExceeded)
    monkeypatch.setattr(writer_agent, "build_writer_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        writer_agent,
        "build_writer_no_tool_agent",
        lambda *_args, **_kwargs: object(),
    )

    call_count = {"count": 0}

    def _always_max_turns(
        _agent: object,
        _input_text: str,
        log_llm_payload: bool,
        **_kwargs: object,
    ) -> _FakeRunResult:
        del log_llm_payload
        call_count["count"] += 1
        raise _FakeMaxTurnsExceeded("Max turns (10) exceeded")

    monkeypatch.setattr(writer_agent, "run_agent_sync", _always_max_turns)

    with pytest.raises(_FakeMaxTurnsExceeded):
        writer_agent.write_markdown(
            question="Summarize selected cities.",
            context_bundle=context_bundle,
            config=config,
            api_key="test-key",
            log_llm_payload=False,
            run_id="run-fallback-failed",
        )

    assert call_count["count"] == 2
