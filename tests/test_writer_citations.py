import logging
from pathlib import Path

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
