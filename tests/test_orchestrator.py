import json
import logging
from pathlib import Path

import pytest

from backend.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.orchestrator.module import (
    _build_retrieval_queries,
    _collect_markdown_decision_artifacts,
    run_pipeline,
)
from backend.modules.orchestrator.utils import build_markdown_city_summary
from backend.modules.writer.models import WriterCitationCoverage, WriterOutput
from backend.utils.config import AppConfig
from backend.utils.logging_config import setup_logger
from backend.utils.paths import RunPaths
from tests.support import build_test_app_config


def _research_question_path(paths: RunPaths) -> Path:
    """Return the canonical research-question artifact path for one run."""
    return (
        paths.base_dir
        / "stage_files"
        / "002_query_preparation"
        / "research_question.json"
    )


def _stage_file_path(paths: RunPaths, folder: str, filename: str) -> Path:
    """Return one stage-file path under the numbered stage folder."""
    return paths.base_dir / "stage_files" / folder / filename


def _build_test_config(*, runs_dir: Path, markdown_dir: Path) -> AppConfig:
    """Build an orchestrator test config with vector retrieval disabled."""
    return build_test_app_config(
        runs_dir=runs_dir,
        markdown_dir=markdown_dir,
        vector_store_overrides={"enabled": False},
    )


def _stub_markdown(
    question: str,
    documents: list[dict[str, str]],
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> MarkdownResearchResult:
    """Return one grounded markdown excerpt for pipeline tests."""
    _ = question, documents, config, api_key
    excerpt = MarkdownExcerpt(
        quote="Munich has deployed 43 existing public chargers as of 2024.",
        city_name="Munich",
        partial_answer="Munich has deployed 43 existing public chargers as of 2024.",
    )
    return MarkdownResearchResult(excerpts=[excerpt])


def _stub_writer(
    question: str,
    context_bundle: dict,
    config: AppConfig,
    api_key: str,
    **_kwargs: dict[str, object],
) -> WriterOutput:
    """Return one deterministic writer output for pipeline tests."""
    _ = question, context_bundle, config, api_key
    return WriterOutput(content="# Answer\n\nStub")


def _reset_root_handlers() -> None:
    """Remove and close all handlers from the root logger."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()


def test_build_retrieval_queries_trims_dedupes_case_insensitively_and_caps() -> None:
    assert _build_retrieval_queries(
        "  Main query  ",
        "main query",
        "  Second query  ",
        "",
        "SECOND QUERY",
        "Third query",
        "Fourth ignored query",
    ) == [
        "Main query",
        "Second query",
        "Third query",
    ]


def test_collect_markdown_decision_artifacts_keeps_rejected_chunks_lightweight() -> None:
    markdown_chunks = [
        {
            "chunk_id": "chunk-1",
            "content": "Accepted chunk content",
            "city_name": "Munich",
            "city_key": "munich",
            "path": "documents/Munich.md",
            "heading_path": "Mobility > Charging",
            "block_type": "paragraph",
            "distance": "0.123",
            "chunk_index": 1,
        },
        {
            "chunk_id": "chunk-2",
            "content": "Rejected chunk content",
            "city_name": "Leipzig",
            "city_key": "leipzig",
            "path": "documents/Leipzig.md",
            "heading_path": "Buildings > Retrofit",
            "block_type": "table",
            "distance": "0.456",
            "chunk_index": 2,
        },
    ]
    markdown_result = MarkdownResearchResult(
        excerpts=[
            MarkdownExcerpt(
                quote="Accepted quote",
                city_name="Munich",
                partial_answer="Accepted answer",
                source_chunk_ids=["chunk-1"],
            )
        ],
        accepted_chunk_ids=["chunk-1"],
        rejected_chunk_ids=["chunk-2"],
    )

    rejected_artifact, audit_artifact = (
        _collect_markdown_decision_artifacts(markdown_chunks, markdown_result)
    )

    assert rejected_artifact["rejected_chunk_ids"] == ["chunk-2"]
    assert rejected_artifact["rejected_by_city"] == {"leipzig": ["chunk-2"]}
    assert "rejected_chunks" not in rejected_artifact
    assert audit_artifact["status"] == "complete"


def test_build_markdown_city_summary_rolls_up_city_coverage() -> None:
    markdown_chunks = [
        {
            "chunk_id": "chunk-1",
            "city_name": "Munich",
            "city_key": "munich",
        },
        {
            "chunk_id": "chunk-2",
            "city_name": "Munich",
            "city_key": "munich",
        },
        {
            "chunk_id": "chunk-3",
            "city_name": "Leipzig",
            "city_key": "leipzig",
        },
    ]
    markdown_bundle = {
        "accepted_chunk_ids": ["chunk-1"],
        "rejected_chunk_ids": ["chunk-2", "chunk-3"],
        "unresolved_chunk_ids": [],
        "excerpts": [
            {
                "city_name": "Munich",
                "city_key": "munich",
                "source_chunk_ids": ["chunk-1"],
            }
        ],
    }
    decision_audit_artifact = {
        "missing_chunk_ids": [],
        "batch_failures": [
            {
                "city_name": "Leipzig",
                "batch_index": 1,
                "reason": "MARKDOWN_BATCH_FAILURE",
                "unresolved_chunk_ids": ["chunk-3"],
            }
        ],
    }
    batches_payload = {
        "batches": [
            {"city_name": "munich", "batch_index": 1},
            {"city_name": "leipzig", "batch_index": 1},
        ]
    }

    summary = build_markdown_city_summary(
        markdown_chunks=markdown_chunks,
        markdown_bundle=markdown_bundle,
        decision_audit_artifact=decision_audit_artifact,
        batches_payload=batches_payload,
    )

    assert summary["cities_with_excerpts"] == ["munich"]
    assert summary["cities_without_excerpts"] == ["leipzig"]
    assert summary["cities_with_failures"] == ["leipzig"]
    assert summary["cities"] == [
        {
            "city_name": "Leipzig",
            "batch_count": 1,
            "chunk_count": 1,
            "accepted_chunk_count": 0,
            "rejected_chunk_count": 1,
            "unresolved_chunk_count": 1,
            "excerpt_count": 0,
            "status": "partial",
            "error": {"reasons": ["MARKDOWN_BATCH_FAILURE"]},
            "city_key": "leipzig",
        },
        {
            "city_name": "Munich",
            "batch_count": 1,
            "chunk_count": 2,
            "accepted_chunk_count": 1,
            "rejected_chunk_count": 1,
            "unresolved_chunk_count": 0,
            "excerpt_count": 1,
            "status": "success",
            "error": None,
            "city_key": "munich",
        },
    ]


def test_build_markdown_city_summary_dedupes_failed_batch_unresolved_chunks() -> None:
    """Avoid double-counting unresolved chunks repeated in batch failures."""
    markdown_chunks = [
        {
            "chunk_id": "chunk-1",
            "city_name": "Leipzig",
            "city_key": "leipzig",
        }
    ]
    markdown_bundle = {
        "accepted_chunk_ids": [],
        "rejected_chunk_ids": ["chunk-1"],
        "unresolved_chunk_ids": ["chunk-1"],
        "excerpts": [],
    }
    decision_audit_artifact = {
        "missing_chunk_ids": [],
        "batch_failures": [
            {
                "city_name": "Leipzig",
                "batch_index": 1,
                "reason": "MARKDOWN_BATCH_FAILURE",
                "unresolved_chunk_ids": ["chunk-1"],
            }
        ],
    }

    summary = build_markdown_city_summary(
        markdown_chunks=markdown_chunks,
        markdown_bundle=markdown_bundle,
        decision_audit_artifact=decision_audit_artifact,
    )

    assert summary["cities"] == [
        {
            "city_name": "Leipzig",
            "batch_count": 0,
            "chunk_count": 1,
            "accepted_chunk_count": 0,
            "rejected_chunk_count": 1,
            "unresolved_chunk_count": 1,
            "excerpt_count": 0,
            "status": "partial",
            "error": {"reasons": ["MARKDOWN_BATCH_FAILURE"]},
            "city_key": "leipzig",
        }
    ]


def test_run_pipeline_creates_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    final_output = paths.final_output.read_text(encoding="utf-8")
    run_log = json.loads(paths.api_state.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed"
    assert "Finish reason:" not in final_output
    assert _stage_file_path(
        paths,
        "007_markdown_context_handoff",
        "context_bundle_after_markdown.json",
    ).exists()
    stage_paths = {path.name for path in paths.stages_dir.iterdir() if path.is_file()}
    assert "007_markdown_context_handoff.json" in stage_paths
    assert _stage_file_path(
        paths,
        "006_markdown_extraction",
        "city_summary.json",
    ).exists()
    markdown_stage = json.loads(
        (paths.stages_dir / "006_markdown_extraction.json").read_text(encoding="utf-8")
    )
    assert markdown_stage["outputs"]["city_summary"] == (
        "stage_files/006_markdown_extraction/city_summary.json"
    )
    assert markdown_stage["outputs"]["cities_with_excerpts"] == ["munich"]
    assert markdown_stage["outputs"]["cities_without_excerpts"] == []
    summary_events = [
        json.loads(line)
        for line in paths.summary_events.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stage_events = [
        event
        for event in summary_events
        if event.get("event_type") == "stage_completed"
    ]
    assert stage_events[0]["payload"]["step"] == "input_snapshot"
    assert stage_events[0]["stage_number"] == 1
    assert stage_events[1]["payload"]["step"] == "query_preparation"
    assert stage_events[1]["stage_number"] == 2


def test_run_pipeline_keeps_city_scope_on_root_context_not_markdown_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
        selected_cities=["Munich"],
    )

    context_bundle = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    markdown_snapshot = json.loads(
        _stage_file_path(
            paths,
            "007_markdown_context_handoff",
            "context_bundle_after_markdown.json",
        ).read_text(encoding="utf-8")
    )

    assert context_bundle["selected_cities"] == ["munich"]
    assert context_bundle["selected_city_names"] == ["Munich"]
    assert context_bundle["inspected_cities"] == ["munich"]
    assert context_bundle["inspected_city_names"] == ["Munich"]
    assert context_bundle["city_scope_mode"] == "selected_cities"
    assert markdown_snapshot["selected_cities"] == ["munich"]
    assert markdown_snapshot["selected_city_names"] == ["Munich"]
    assert markdown_snapshot["inspected_cities"] == ["munich"]
    assert markdown_snapshot["inspected_city_names"] == ["Munich"]
    assert markdown_snapshot["city_scope_mode"] == "selected_cities"
    assert isinstance(markdown_snapshot.get("markdown"), dict)


def test_run_pipeline_writes_enrichment_context_handoff_when_enrichment_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
        vector_store_overrides={"enabled": False},
        enrichment_overrides={"enabled": True},
    )

    def _stub_enrichment(
        question: str,
        context_bundle: dict[str, object],
        **_kwargs: dict[str, object],
    ) -> dict[str, object]:
        del question
        updated = dict(context_bundle)
        updated["enrichment"] = {
            "field_manifest": {"query_fields": [], "non_estimable_fields": []},
            "gap_manifest": {"city_gaps": []},
            "enriched_fields": [],
            "web_findings": [],
            "external_evidence": [],
            "external_resolutions": [],
            "external_no_evidence": [],
            "freshness_results": [],
            "meta": {
                "created_at": "2026-01-01T00:00:00Z",
                "gap_analyst_model": "test-model",
                "total_gaps": 0,
                "estimable_count": 0,
                "non_estimable_count": 0,
                "web_findings_count": 0,
                "external_evidence_count": 0,
                "elapsed_seconds": 0.0,
            },
        }
        updated["assumptions"] = {
            "assumptions": [],
            "non_estimable": [],
            "saturation_warning": None,
            "meta": {"assumption_count": 0},
        }
        return updated

    monkeypatch.setattr(
        "backend.modules.orchestrator.module.run_enrichment_pipeline",
        _stub_enrichment,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )

    context_bundle = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    assert "assumptions" in context_bundle
    assert "assumptions" not in context_bundle["enrichment"]


def test_run_pipeline_persists_partial_writer_output_as_completed_with_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    def _stub_partial_writer(
        question: str,
        context_bundle: dict,
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> WriterOutput:
        del question, context_bundle, config, api_key
        return WriterOutput(
            content="# Answer\n\nPartial",
            citation_coverage=WriterCitationCoverage(
                status="partial",
                attempt=2,
                max_attempts=2,
                coverage_confirmed=1,
                coverage_required=2,
                coverage_ratio="1/2",
                missing_cities=["Berlin"],
                analysis_mode="aggregate",
            ),
        )

    paths = run_pipeline(
        question="What initiatives exist for Munich and Berlin?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_stub_partial_writer,
    )

    assert paths.final_output.exists()
    run_log = json.loads(paths.api_state.read_text(encoding="utf-8"))
    assert run_log["status"] == "completed_with_gaps"
    assert run_log["finish_reason"].startswith(
        "completed_with_gaps (writer partial citation coverage 1/2)"
    )
    assert run_log["writer_citation_coverage"]["missing_cities"] == ["Berlin"]


def test_run_pipeline_refreshes_vector_store_snapshot_after_auto_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector-backed runs refresh snapshot state after full-corpus readiness updates."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
        vector_store_overrides={"enabled": True, "auto_update_on_run": True},
    )

    snapshot_hashes = iter(["before-update", "after-update"])

    def _fake_build_vector_store_snapshot(
        _config: AppConfig,
        *,
        update_stats: object | None = None,
        selected_cities: list[str] | None = None,
    ) -> dict[str, object]:
        auto_update = None
        if update_stats is not None:
            if isinstance(update_stats, dict):
                update_mode = update_stats["update_mode"]
                files_changed = update_stats["files_changed"]
                files_unchanged = update_stats["files_unchanged"]
                files_deleted = update_stats["files_deleted"]
                chunks_created = update_stats["chunks_created"]
                changed_files = update_stats["changed_files"]
                deleted_files = update_stats["deleted_files"]
            else:
                update_mode = update_stats.update_mode
                files_changed = update_stats.files_changed
                files_unchanged = update_stats.files_unchanged
                files_deleted = update_stats.files_deleted
                chunks_created = update_stats.chunks_created
                changed_files = update_stats.changed_files
                deleted_files = update_stats.deleted_files
            was_dry_run = bool(
                update_stats["dry_run"] if isinstance(update_stats, dict) else update_stats.dry_run
            )
            auto_update = {
                "checked": True,
                "ran": update_mode is not None and not was_dry_run,
                "applied": update_mode is not None and not was_dry_run,
                "dry_run": was_dry_run,
                "update_mode": update_mode,
                "trigger": "auto_update_on_run",
                "selected_cities": selected_cities or [],
                "stats": {
                    "files_changed": files_changed,
                    "files_unchanged": files_unchanged,
                    "files_deleted": files_deleted,
                    "chunks_created": chunks_created,
                    "changed_files": changed_files,
                    "deleted_files": deleted_files,
                },
            }
        return {
            "enabled": True,
            "collection_name": "test_chunks",
            "index_manifest_hash": next(snapshot_hashes),
            "manifest_summary": {"chunk_count": 1},
            "auto_update": auto_update,
        }

    monkeypatch.setattr(
        "backend.modules.orchestrator.module.build_vector_store_snapshot",
        _fake_build_vector_store_snapshot,
    )
    readiness_calls: list[dict[str, object]] = []

    class FakeVectorStoreWarmup:
        def ensure_ready_for_run(self, *, config: AppConfig, docs_dir: Path) -> str | None:
            readiness_calls.append({"config": config, "docs_dir": docs_dir})
            return None

        def snapshot(self) -> dict[str, object]:
            return {
                "status": "completed",
                "stats": {
                    "files_indexed": 1,
                    "files_changed": 1,
                    "files_unchanged": 0,
                    "files_deleted": 0,
                    "chunks_created": 1,
                    "table_chunks": 0,
                    "min_tokens": 0,
                    "avg_tokens": 0.0,
                    "max_tokens": 0,
                    "dry_run": False,
                    "update_mode": "incremental_update",
                    "changed_files": [
                        {
                            "source_path": "documents/Munich.md",
                            "status": "modified",
                            "previous_chunk_count": 1,
                            "current_chunk_count": 2,
                            "removed_previous_chunk_count": 1,
                        }
                    ],
                    "deleted_files": [],
                },
            }

    monkeypatch.setattr(
        "backend.modules.orchestrator.module.VectorStoreWarmup",
        FakeVectorStoreWarmup,
    )
    monkeypatch.setattr(
        "backend.modules.orchestrator.module.retrieve_chunks_for_queries",
        lambda **_: ([], {"seed_chunks": []}),
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        selected_cities=["Munich"],
        vector_update_docs_dir=docs_dir,
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )

    vector_snapshot = json.loads(
        _stage_file_path(
            paths,
            "001_input_snapshot",
            "vector_store_snapshot.json",
        ).read_text(encoding="utf-8")
    )
    planned_stages = json.loads(
        _stage_file_path(
            paths,
            "001_input_snapshot",
            "planned_stages.json",
        ).read_text(encoding="utf-8")
    )
    input_snapshot = json.loads((paths.stages_dir / "001_input_snapshot.json").read_text(encoding="utf-8"))

    assert vector_snapshot["index_manifest_hash"] == "after-update"
    assert planned_stages["schema_version"] == "1.0"
    assert any(stage["id"] == "enrichment" for stage in planned_stages["stages"])
    assert (
        input_snapshot["snapshot_summary"]["vector_store"]["index_manifest_hash"]
        == "after-update"
    )
    auto_update = vector_snapshot["auto_update"]
    assert auto_update["ran"] is True
    assert auto_update["update_mode"] == "incremental_update"
    assert auto_update["stats"]["files_changed"] == 1
    assert readiness_calls[0]["docs_dir"] == docs_dir
    assert auto_update["stats"]["changed_files"] == [
        {
            "source_path": "documents/Munich.md",
            "status": "modified",
            "previous_chunk_count": 1,
            "current_chunk_count": 2,
            "removed_previous_chunk_count": 1,
        }
    ]
    assert input_snapshot["snapshot_summary"]["vector_store"]["auto_update"] == auto_update


def test_run_pipeline_blocks_when_full_corpus_vector_store_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector-backed runs fail early when the shared full-corpus index is stale."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
        vector_store_overrides={"enabled": True, "auto_update_on_run": False},
    )

    class FakeVectorStoreWarmup:
        def ensure_ready_for_run(self, *, config: AppConfig, docs_dir: Path) -> str | None:
            del config, docs_dir
            return "Vector store is stale. Run `bash scripts/update_vector_store_maintenance.sh` and retry."

        def snapshot(self) -> dict[str, object]:
            return {
                "status": "stale",
                "stats": {
                    "files_indexed": 2,
                    "files_changed": 1,
                    "files_unchanged": 1,
                    "files_deleted": 0,
                    "chunks_created": 0,
                    "table_chunks": 0,
                    "min_tokens": 0,
                    "avg_tokens": 0.0,
                    "max_tokens": 0,
                    "dry_run": True,
                    "update_mode": "incremental_update",
                    "changed_files": [],
                    "deleted_files": [],
                },
            }

    monkeypatch.setattr(
        "backend.modules.orchestrator.module.VectorStoreWarmup",
        FakeVectorStoreWarmup,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        selected_cities=["Munich"],
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )

    api_state = json.loads(paths.api_state.read_text(encoding="utf-8"))
    error_log = paths.error_log.read_text(encoding="utf-8")
    run_log = (paths.base_dir / "run.log").read_text(encoding="utf-8")

    assert api_state["status"] == "failed"
    assert api_state["finish_reason"] == "markdown_extraction_failed"
    assert "Vector store is stale." in error_log
    assert "Vector store is stale." in run_log


def test_run_pipeline_passes_run_logger_and_paths_to_writer_when_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )
    captured: dict[str, object] = {}

    def _writer_with_runtime_context(
        question: str,
        context_bundle: dict,
        config: AppConfig,
        api_key: str,
        run_logger: object,
        paths: object,
        **_kwargs: dict[str, object],
    ) -> WriterOutput:
        del question, context_bundle, config, api_key
        captured["run_logger"] = run_logger
        captured["paths"] = paths
        return WriterOutput(content="# Answer\n\nStub")

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_writer_with_runtime_context,
    )

    assert paths.final_output.exists()
    assert captured["run_logger"] is not None
    assert captured["paths"] == paths


def test_run_pipeline_detaches_run_log_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _reset_root_handlers()
    setup_logger()

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    try:
        first_paths = run_pipeline(
            question="First question?",
            config=config,
            run_id="run1",
            markdown_func=_stub_markdown,
            writer_func=_stub_writer,
        )
        first_run_log_path = str(first_paths.base_dir / "run.log")
        assert all(
            not isinstance(handler, logging.FileHandler)
            or handler.baseFilename != first_run_log_path
            for handler in logging.getLogger().handlers
        )
        logging.getLogger(__name__).warning("MARKER_AFTER_RUN1")

        second_paths = run_pipeline(
            question="Second question?",
            config=config,
            run_id="run2",
            markdown_func=_stub_markdown,
            writer_func=_stub_writer,
        )
        second_run_log_path = str(second_paths.base_dir / "run.log")
        assert all(
            not isinstance(handler, logging.FileHandler)
            or handler.baseFilename not in {first_run_log_path, second_run_log_path}
            for handler in logging.getLogger().handlers
        )
        logging.getLogger(__name__).warning("MARKER_AFTER_RUN2")

        first_log = (first_paths.base_dir / "run.log").read_text(encoding="utf-8")

        assert "MARKER_AFTER_RUN2" not in first_log
    finally:
        _reset_root_handlers()


def test_run_pipeline_standard_mode_uses_verbatim_question_before_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )
    captured: dict[str, str] = {}

    def _capture_markdown_question(
        question: str,
        documents: list[dict[str, str]],
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> MarkdownResearchResult:
        captured["question"] = question
        return _stub_markdown(question, documents, config, api_key, **_kwargs)

    paths = run_pipeline(
        question="What initiatives exist for Munich as typed?",
        config=config,
        markdown_func=_capture_markdown_question,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    assert captured["question"] == "What initiatives exist for Munich as typed?"
    context_bundle = json.loads(paths.context_bundle.read_text(encoding="utf-8"))
    assert context_bundle["research_question"] == "What initiatives exist for Munich as typed?"
    research_payload = json.loads(_research_question_path(paths).read_text(encoding="utf-8"))
    assert research_payload["retrieval_queries"] == [
        "What initiatives exist for Munich as typed?"
    ]
    assert (
        research_payload["canonical_research_query"]
        == "What initiatives exist for Munich as typed?"
    )
    assert research_payload["query_mode"] == "standard"


def test_run_pipeline_standard_mode_uses_optional_queries_when_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")
    (docs_dir / "Leipzig.md").write_text("# Leipzig\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )
    captured: dict[str, str] = {}

    def _capture_markdown_question(
        question: str,
        documents: list[dict[str, str]],
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> MarkdownResearchResult:
        captured["question"] = question
        return _stub_markdown(question, documents, config, api_key, **_kwargs)

    paths = run_pipeline(
        question="Compare Munich and Leipzig initiatives as written.",
        config=config,
        selected_cities=["Munich", "Leipzig"],
        query_2="implementation milestones",
        query_3="reported budget metrics",
        markdown_func=_capture_markdown_question,
        writer_func=_stub_writer,
    )

    assert captured["question"] == "Compare Munich and Leipzig initiatives as written."
    research_payload = json.loads(_research_question_path(paths).read_text(encoding="utf-8"))
    assert research_payload["query_mode"] == "standard"
    assert research_payload["retrieval_queries"] == [
        "Compare Munich and Leipzig initiatives as written.",
        "implementation milestones",
        "reported budget metrics",
    ]
    assert research_payload["retrieval_query_2"] == "implementation milestones"
    assert research_payload["retrieval_query_3"] == "reported budget metrics"


def test_run_pipeline_dev_mode_uses_direct_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )
    captured: dict[str, str] = {}

    def _capture_markdown_question(
        question: str,
        documents: list[dict[str, str]],
        config: AppConfig,
        api_key: str,
        **_kwargs: dict[str, object],
    ) -> MarkdownResearchResult:
        captured["question"] = question
        return _stub_markdown(question, documents, config, api_key, **_kwargs)

    paths = run_pipeline(
        question="Main direct query",
        config=config,
        query_mode="dev",
        query_2="Second direct query",
        query_3="Third direct query",
        markdown_func=_capture_markdown_question,
        writer_func=_stub_writer,
    )

    assert paths.final_output.exists()
    assert captured["question"] == "Main direct query"
    research_payload = json.loads(_research_question_path(paths).read_text(encoding="utf-8"))
    assert research_payload["query_mode"] == "dev"
    assert research_payload["retrieval_queries"] == [
        "Main direct query",
        "Second direct query",
        "Third direct query",
    ]


def test_run_pipeline_logs_resolved_run_id_when_requested_id_is_suffixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")

    config = _build_test_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
    )

    first_paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        run_id="duplicate-run",
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )
    second_paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        run_id="duplicate-run",
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )

    assert first_paths.base_dir.name == "duplicate-run"
    assert second_paths.base_dir.name == "duplicate-run_01"

    second_run_log = (second_paths.base_dir / "run.log").read_text(encoding="utf-8")
    assert "run_id=duplicate-run_01 query_mode=standard query_count=1" in second_run_log
    assert "run_id=duplicate-run_01 markdown_source_mode=standard_chunking" in second_run_log
    assert "run_id=duplicate-run query_mode=standard query_count=1" not in second_run_log
