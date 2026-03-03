from __future__ import annotations

from pathlib import Path

from backend.modules.orchestrator.utils.io import (
    sanitize_final_content,
    write_final_output,
)
from backend.services.run_logger import RunLogger
from backend.utils.paths import create_run_paths


def test_sanitize_final_content_strips_operational_lines() -> None:
    raw_content = "\n".join(
        [
            "## Answer",
            "Observed demand is 120 MW.",
            "Finish reason: completed (write)",
            "Max turns (10) exceeded",
            "agents.exceptions.MaxTurnsExceeded: Max turns (10) exceeded",
            "Traceback (most recent call last):",
        ]
    )

    sanitized = sanitize_final_content(raw_content)

    assert "Finish reason" not in sanitized
    assert "Max turns" not in sanitized
    assert "MaxTurnsExceeded" not in sanitized
    assert "Traceback" not in sanitized
    assert "Observed demand is 120 MW." in sanitized


def test_write_final_output_writes_question_and_clean_answer(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "run-io-clean", "context_bundle.json")
    run_logger = RunLogger(paths, "What is the city demand?")
    content = "\n".join(
        [
            "## Answer",
            "Demand estimate is 80 MW.",
            "Finish reason: completed (write)",
        ]
    )

    write_final_output(
        question="What is the city demand?",
        content=content,
        paths=paths,
        run_logger=run_logger,
        finish_reason="completed (write)",
    )

    rendered = paths.final_output.read_text(encoding="utf-8")
    assert rendered.startswith("# Question\nWhat is the city demand?\n\n")
    assert "Demand estimate is 80 MW." in rendered
    assert "Finish reason" not in rendered
