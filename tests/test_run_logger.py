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

    payload = json.loads(paths.api_state.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    error_log_path = paths.error_log
    assert error_log_path.exists()
    error_lines = error_log_path.read_text(encoding="utf-8")
    assert " - ERROR - markdown failed" in error_lines
    assert "RETRY_EXHAUSTED" in error_lines
    assert " - CRITICAL - fatal provider error" in error_lines


def test_run_logger_persists_analysis_mode_in_inputs_and_context(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "run-logger-mode", "context_bundle.json")
    logger = RunLogger(paths, "Compare selected cities")

    logger.update_analysis_mode("city_by_city")

    run_payload = json.loads(paths.api_state.read_text(encoding="utf-8"))
    context_payload = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    assert run_payload["inputs"]["analysis_mode"] == "city_by_city"
    assert context_payload["analysis_mode"] == "city_by_city"


def test_run_logger_persists_query_inputs_in_log_context_and_summary(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "run-logger-queries", "context_bundle.json")
    logger = RunLogger(paths, "Original question")
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.final_output.write_text("# Final\nAnswer", encoding="utf-8")

    logger.update_query_inputs(
        original_question="Original question",
        canonical_research_query="Original question",
        retrieval_queries=[
            "Original question",
            "Complementary retrieval query",
        ],
        query_mode="dev",
    )
    logger.finalize("completed", final_output_path=paths.final_output, finish_reason="completed")

    run_payload = json.loads(paths.api_state.read_text(encoding="utf-8"))
    context_payload = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    run_summary = paths.run_summary.read_text(encoding="utf-8")

    assert run_payload["inputs"]["original_question"] == "Original question"
    assert run_payload["inputs"]["canonical_research_query"] == "Original question"
    assert run_payload["inputs"]["query_mode"] == "dev"
    assert run_payload["inputs"]["retrieval_query_1"] == "Original question"
    assert run_payload["inputs"]["retrieval_query_2"] == "Complementary retrieval query"
    assert run_payload["inputs"]["retrieval_query_3"] is None

    assert context_payload["original_question"] == "Original question"
    assert context_payload["research_question"] == "Original question"
    assert context_payload["query_mode"] == "dev"
    assert context_payload["retrieval_queries"] == [
        "Original question",
        "Complementary retrieval query",
    ]

    assert "Original question: Original question" in run_summary
    assert "Query mode: dev" in run_summary
    assert "Primary retrieval query: Original question" in run_summary
    assert "Retrieval query 1: Original question" in run_summary
    assert "Retrieval query 2: Complementary retrieval query" in run_summary
    assert "Retrieval query 3: (none)" in run_summary
    assert "MARKDOWN_FAILURE_SUMMARY\nnone" in run_summary


def test_run_logger_uses_fixed_stage_numbers_in_summary_and_stage_files(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "run-logger-stages", "context_bundle.json")
    logger = RunLogger(paths, "Original question")
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.final_output.write_text("# Final\nAnswer", encoding="utf-8")

    logger.update_query_inputs(
        original_question="Original question",
        canonical_research_query="Original question",
        retrieval_queries=["Original question"],
        query_mode="standard",
    )
    logger.write_input_snapshot_stage()
    logger.write_stage_detail(
        "enrichment",
        {
            "inputs": {},
            "outputs": {},
            "metrics": {"external_evidence_count": 1},
        },
    )
    logger.record_decision({"step": "enrichment", "status": "completed"})
    logger.finalize("completed", final_output_path=paths.final_output, finish_reason="completed")

    summary_events = [
        json.loads(line)
        for line in paths.summary_events.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stage_numbers = {
        event["payload"].get("step"): event.get("stage_number")
        for event in summary_events
        if isinstance(event.get("payload"), dict) and event["payload"].get("step")
    }
    assert stage_numbers["input_snapshot"] == 1
    assert stage_numbers["query_preparation"] == 2
    assert stage_numbers["enrichment"] == 8
    assert stage_numbers["finalize"] == 15
    assert all(event["event_type"] == "stage_completed" for event in summary_events)

    stage_paths = {path.name for path in paths.stages_dir.iterdir() if path.is_file()}
    assert "001_input_snapshot.json" in stage_paths
    assert "002_query_preparation.json" in stage_paths
    assert "008_enrichment.json" in stage_paths
    assert "015_finalize.json" in stage_paths

    enrichment_payload = json.loads(
        (paths.stages_dir / "008_enrichment.json").read_text(encoding="utf-8")
    )
    assert enrichment_payload["decisions"] == [
        {"step": "enrichment", "status": "completed"}
    ]


def test_run_logger_normalizes_selected_city_matching_for_all_city_labels(
    tmp_path: Path,
) -> None:
    paths = create_run_paths(tmp_path, "run-logger-cities", "context_bundle.json")
    logger = RunLogger(paths, "Compare selected cities")

    logger.record_markdown_inputs(
        markdown_dir=tmp_path / "documents",
        selected_cities_planned=["Aachen", "New York", "São-Paulo", "AACHEN"],
        markdown_chunks=[
            {"city_name": "aachen", "city_key": "aachen", "path": "aachen.md"},
            {"city_name": "NEW_YORK", "city_key": "new_york", "path": "new_york.md"},
            {"city_name": "são paulo", "city_key": "são_paulo", "path": "sao_paulo.md"},
        ],
        markdown_source_mode="standard_chunking",
        analysis_mode="aggregate",
    )

    run_payload = json.loads(paths.api_state.read_text(encoding="utf-8"))
    inputs = run_payload["inputs"]
    assert inputs["selected_cities_planned"] == ["aachen", "new_york", "são_paulo"]
    assert inputs["selected_cities_found"] == ["aachen", "new_york", "são_paulo"]

    markdown_inputs_payload = json.loads(
        (paths.stages_dir / "004_markdown_inputs.json").read_text(encoding="utf-8")
    )
    assert markdown_inputs_payload["outputs"]["missing_selected_cities"] == []


def test_run_logger_input_snapshot_records_requested_city_scope(tmp_path: Path) -> None:
    paths = create_run_paths(tmp_path, "run-logger-city-scope", "context_bundle.json")
    logger = RunLogger(paths, "Compare selected cities")

    logger.update_requested_city_scope(["Aachen", "New York", "AACHEN"])
    logger.write_input_snapshot_stage()

    input_snapshot = json.loads(
        (paths.stages_dir / "001_input_snapshot.json").read_text(encoding="utf-8")
    )
    assert input_snapshot["inputs"]["city_scope_mode"] == "selected_cities"
    assert input_snapshot["inputs"]["selected_cities_planned"] == [
        "aachen",
        "new_york",
    ]
    assert "selected_cities_found" not in input_snapshot["inputs"]
    assert "markdown_file_count" not in input_snapshot["inputs"]


def test_run_logger_does_not_refresh_input_snapshot_from_markdown_inputs(
    tmp_path: Path,
) -> None:
    paths = create_run_paths(tmp_path, "run-logger-input-once", "context_bundle.json")
    logger = RunLogger(paths, "Compare selected cities")

    logger.update_requested_city_scope(["Aachen"])
    logger.write_input_snapshot_stage(
        snapshot_summary={"execution": {"resolved_run_id": "run-logger-input-once"}},
        snapshot_artifacts={"execution_snapshot": "stage_files/001_input_snapshot/execution_snapshot.json"},
    )
    logger.record_markdown_inputs(
        markdown_dir=tmp_path / "documents",
        selected_cities_planned=["Aachen"],
        markdown_chunks=[
            {"city_name": "Aachen", "city_key": "aachen", "path": "Aachen.md"},
        ],
        markdown_source_mode="vector_store_retrieval",
        analysis_mode="aggregate",
    )

    input_snapshot = json.loads(
        (paths.stages_dir / "001_input_snapshot.json").read_text(encoding="utf-8")
    )
    assert input_snapshot["snapshot_summary"] == {
        "execution": {"resolved_run_id": "run-logger-input-once"}
    }
    assert input_snapshot["snapshots"] == {
        "execution_snapshot": "stage_files/001_input_snapshot/execution_snapshot.json"
    }
    assert "selected_cities_found" not in input_snapshot["inputs"]
    assert "markdown_chunk_count" not in input_snapshot["inputs"]


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

    payload = json.loads(paths.api_state.read_text(encoding="utf-8"))
    retry_summary = payload.get("retry_summary")
    assert isinstance(retry_summary, dict)
    assert retry_summary["total_events"] == 2
    assert retry_summary["exhausted_events"] == 1
    assert retry_summary["by_operation"] == {
        "chat.citation_coverage": 1,
        "markdown.batch_extraction": 1,
    }


def test_run_logger_writer_citation_coverage_stage_uses_coverage_counts(
    tmp_path: Path,
) -> None:
    paths = create_run_paths(tmp_path, "run-logger-coverage", "context_bundle.json")
    logger = RunLogger(paths, "How complete is the draft?")

    logger.record_writer_citation_coverage(
        {
            "status": "confirmed",
            "attempt": 1,
            "max_attempts": 2,
            "coverage_confirmed": 1,
            "coverage_required": 3,
            "coverage_ratio": "1/3",
            "missing_cities": ["berlin", "munich"],
            "analysis_mode": "aggregate",
        }
    )

    stage_payload = json.loads(
        (paths.stages_dir / "013_writer_citation_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    assert stage_payload["metrics"]["confirmed_city_count"] == 1
    assert stage_payload["metrics"]["required_city_count"] == 3
