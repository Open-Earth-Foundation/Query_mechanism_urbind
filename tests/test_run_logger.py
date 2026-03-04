from __future__ import annotations

import json
from pathlib import Path

from backend.services.run_logger import RunLogger
from backend.utils.paths import create_run_paths


def test_run_logger_extracts_error_log_and_registers_artifact(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "run-logger-test", "context_bundle.json")
    logger = RunLogger(paths, "How are cities progressing?")
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.final_output.write_text("# Final\nAnswer", encoding="utf-8")
    run_log_text = "\n".join(
        [
            "2026-01-01 00:00:00 foo.py:10 - INFO - startup",
            "2026-01-01 00:00:01 foo.py:11 - ERROR - markdown failed",
            "2026-01-01 00:00:02 foo.py:12 - WARNING - RETRY_EVENT {\"operation\":\"markdown.batch_extraction\"}",
            "2026-01-01 00:00:03 foo.py:13 - ERROR - RETRY_EXHAUSTED {\"operation\":\"chat.citation_coverage\"}",
            "2026-01-01 00:00:04 foo.py:14 - CRITICAL - fatal provider error",
        ]
    )
    (paths.base_dir / "run.log").write_text(run_log_text, encoding="utf-8")

    logger.finalize("completed", final_output_path=paths.final_output, finish_reason="completed")

    payload = json.loads(paths.run_log.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", {})
    assert isinstance(artifacts, dict)
    assert "error_log" in artifacts

    error_log_path = Path(str(artifacts["error_log"]))
    assert error_log_path.exists()
    error_lines = error_log_path.read_text(encoding="utf-8")
    assert " - ERROR - markdown failed" in error_lines
    assert "RETRY_EXHAUSTED" in error_lines
    assert " - CRITICAL - fatal provider error" in error_lines


def test_run_logger_persists_analysis_mode_in_inputs_and_context(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "run-logger-mode", "context_bundle.json")
    logger = RunLogger(paths, "Compare selected cities")

    logger.update_analysis_mode("city_by_city")

    run_payload = json.loads(paths.run_log.read_text(encoding="utf-8"))
    context_payload = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    assert run_payload["inputs"]["analysis_mode"] == "city_by_city"
    assert context_payload["analysis_mode"] == "city_by_city"


def test_run_logger_parses_plain_text_retry_payloads(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "run-logger-retry-text", "context_bundle.json")
    logger = RunLogger(paths, "Why retries happened?")
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.final_output.write_text("# Final\nAnswer", encoding="utf-8")
    run_log_text = "\n".join(
        [
            "2026-01-01 00:00:00 foo.py:10 - INFO - startup",
            (
                "2026-01-01 00:00:01 foo.py:11 - WARNING - RETRY_EVENT "
                "operation=markdown.batch_extraction run_id=run-logger-retry-text "
                "attempt=1/5 error=true error_type=APIConnectionError "
                "reason='provider HTTP 404' http_status=404 rate_limited=false "
                "next_backoff_seconds=1.000 error_message='404 Not Found' "
                "context='city_name=aachen; batch_index=1'"
            ),
            (
                "2026-01-01 00:00:02 foo.py:12 - ERROR - RETRY_EXHAUSTED "
                "operation=chat.citation_coverage run_id=run-logger-retry-text "
                "attempt=5/5 error=true error_type=RateLimitError "
                "reason='provider rate limit' http_status=429 rate_limited=true "
                "next_backoff_seconds=none error_message='Too many requests' context='none'"
            ),
        ]
    )
    (paths.base_dir / "run.log").write_text(run_log_text, encoding="utf-8")

    logger.finalize("completed", final_output_path=paths.final_output, finish_reason="completed")

    payload = json.loads(paths.run_log.read_text(encoding="utf-8"))
    retry_summary = payload.get("retry_summary")
    assert isinstance(retry_summary, dict)
    assert retry_summary["total_events"] == 2
    assert retry_summary["exhausted_events"] == 1
    assert retry_summary["by_operation"] == {
        "chat.citation_coverage": 1,
        "markdown.batch_extraction": 1,
    }
