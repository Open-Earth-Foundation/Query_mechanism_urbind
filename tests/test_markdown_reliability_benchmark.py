from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from backend.benchmarks.reliability_testing import (
    DEFAULT_MATRIX_CONFIG_PATH,
    load_reliability_matrix,
    run_markdown_reliability_benchmark,
)
from backend.models import ErrorInfo
from backend.modules.markdown_researcher.models import (
    MarkdownBatchFailure,
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.vector_store.models import RetrievedChunk
from backend.scripts.run_markdown_reliability_benchmark import main, parse_args


def _write_matrix(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_fake_retrieval_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            city_name="Aachen",
            raw_text="Aachen chunk",
            source_path="documents/Aachen.md",
            heading_path="H1",
            block_type="paragraph",
            distance=0.1,
            chunk_id="a1",
            metadata={"city_key": "aachen", "chunk_index": 0},
        ),
        RetrievedChunk(
            city_name="Berlin",
            raw_text="Berlin chunk",
            source_path="documents/Berlin.md",
            heading_path="H1",
            block_type="paragraph",
            distance=0.2,
            chunk_id="b1",
            metadata={"city_key": "berlin", "chunk_index": 0},
        ),
    ]


def test_load_reliability_matrix_reads_default_yaml() -> None:
    matrix = load_reliability_matrix(DEFAULT_MATRIX_CONFIG_PATH)

    assert len(matrix.retrieval_queries) == 3
    assert matrix.payload_capture_mode == "failed_only"
    assert len(matrix.models) >= 7
    assert all(
        model.id != "moonshotai/kimi-k2-thinking:nitro" for model in matrix.models
    )
    assert all(model.id != "bytedance-seed/seed-2.0-lite" for model in matrix.models)
    assert all(model.id != "x-ai/grok-4-fast" for model in matrix.models)


def test_parse_args_supports_repeatable_model_filters() -> None:
    args = parse_args(
        ["--model", "model-a", "--model", "model-b", "--city", "Aachen", "--city", "Oslo"]
    )

    assert args.model == ["model-a", "model-b"]
    assert args.city == ["Aachen", "Oslo"]


def test_run_markdown_reliability_benchmark_reuses_retrieval_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.yml"
    _write_matrix(
        matrix_path,
        {
            "question": "Benchmark question",
            "retrieval_queries": ["q1", "q2", "q3"],
            "selected_cities": [],
            "payload_capture_mode": "failed_only",
            "markdown_defaults": {
                "max_turns": 7,
                "batch_max_chunks": 1,
                "max_workers": 2,
                "reasoning_effort": "low",
            },
            "models": [
                {"id": "model-a", "enabled": True},
                {"id": "model-b", "enabled": True, "reasoning_effort": None},
            ],
        },
    )

    retrieval_calls = {"count": 0}
    retrieval_selected_cities: list[list[str]] = []
    markdown_calls: list[dict[str, object]] = []

    def _fake_retrieve(
        *,
        queries: list[str],
        config,
        docs_dir: Path,
        selected_cities: list[str],
        run_id: str,
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        del config, docs_dir, run_id
        retrieval_calls["count"] += 1
        retrieval_selected_cities.append(list(selected_cities))
        return (
            _build_fake_retrieval_chunks(),
            {"queries": list(queries), "retrieved_total_chunks": 2},
        )

    def _fake_markdown(
        *,
        question: str,
        documents: list[dict[str, object]],
        config,
        api_key: str,
        run_id: str,
        log_llm_payload: bool = False,
        batch_payload_mode: str = "off",
        batch_payload_recorder=None,
    ) -> MarkdownResearchResult:
        del question, api_key, run_id, log_llm_payload
        markdown_calls.append(
            {
                "model": config.markdown_researcher.model,
                "reasoning_effort": config.markdown_researcher.reasoning_effort,
                "chunk_ids": [str(document["chunk_id"]) for document in documents],
                "payload_mode": batch_payload_mode,
            }
        )
        if config.markdown_researcher.model == "model-b":
            if batch_payload_recorder is not None:
                batch_payload_recorder(
                    {
                        "city_name": "berlin",
                        "batch_index": 1,
                        "success": False,
                        "failure_reason": "MARKDOWN_MAX_TURNS_EXCEEDED",
                        "attempts": [
                            {
                                "attempt": 1,
                                "outcome": "max_turns_exceeded",
                                "events": [],
                            }
                        ],
                    }
                )
            return MarkdownResearchResult(
                status="success",
                excerpts=[
                    MarkdownExcerpt(
                        quote="Aachen allocated funds.",
                        city_name="Aachen",
                        partial_answer="Aachen allocated funds.",
                        source_chunk_ids=["a1"],
                    )
                ],
                accepted_chunk_ids=["a1"],
                unresolved_chunk_ids=["b1"],
                batch_failures=[
                    MarkdownBatchFailure(
                        city_name="berlin",
                        batch_index=1,
                        reason="MARKDOWN_MAX_TURNS_EXCEEDED",
                        unresolved_chunk_ids=["b1"],
                    )
                ],
                error=ErrorInfo(
                    code="MARKDOWN_PARTIAL_BATCH_FAILURE",
                    message="Some batches failed.",
                ),
            )
        return MarkdownResearchResult(
            status="success",
            excerpts=[],
            accepted_chunk_ids=["a1", "b1"],
            rejected_chunk_ids=[],
        )

    report = run_markdown_reliability_benchmark(
        benchmark_id="bench_1",
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        matrix_config_path=matrix_path,
        selected_cities=["Aachen", "Berlin"],
        api_key_override="test-key",
        retrieve_func=_fake_retrieve,
        markdown_func=_fake_markdown,
    )

    assert retrieval_calls["count"] == 1
    assert retrieval_selected_cities == [["Aachen", "Berlin"]]
    assert len(markdown_calls) == 2
    assert markdown_calls[0]["chunk_ids"] == ["a1", "b1"]
    assert markdown_calls[1]["chunk_ids"] == ["a1", "b1"]
    assert markdown_calls[0]["reasoning_effort"] == "low"
    assert markdown_calls[1]["reasoning_effort"] is None
    assert markdown_calls[0]["payload_mode"] == "failed_only"

    assert report.batch_count == 2
    assert report.retrieved_count == 2
    assert report.selected_cities == ["Aachen", "Berlin"]
    assert report.status == "completed"
    assert len(report.results) == 2
    model_results = {result.model_id: result for result in report.results}
    assert model_results["model-a"].failed_batches == 0
    assert model_results["model-b"].failed_batches == 1
    assert model_results["model-b"].failed_entirely is False
    assert report.summary["models_with_failed_batches"] == 1

    benchmark_root = tmp_path / "output" / "bench_1"
    assert (benchmark_root / "progress.json").exists()
    assert (benchmark_root / "retrieval.json").exists()
    assert (benchmark_root / "batches.json").exists()
    assert (benchmark_root / "benchmark_report.json").exists()
    assert (benchmark_root / "benchmark_report.md").exists()
    assert (benchmark_root / "model-a" / "run.log").exists()
    assert (benchmark_root / "model-a" / "error_log.txt").exists()
    assert (benchmark_root / "model-a" / "model_result.json").exists()
    assert (benchmark_root / "model-a" / "markdown" / "accepted_excerpts.json").exists()
    assert (benchmark_root / "model-b" / "markdown" / "failed_batch_payloads.json").exists()

    progress_payload = json.loads(
        (benchmark_root / "progress.json").read_text(encoding="utf-8")
    )
    assert progress_payload["status"] == "completed"
    assert progress_payload["results_written"] == 2
    assert progress_payload["completed_model_ids"] == ["model-a", "model-b"]

    report_payload = json.loads(
        (benchmark_root / "benchmark_report.json").read_text(encoding="utf-8")
    )
    assert report_payload["status"] == "completed"
    assert len(report_payload["results"]) == 2


def test_run_markdown_reliability_benchmark_can_filter_models(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yml"
    _write_matrix(
        matrix_path,
        {
            "question": "Benchmark question",
            "retrieval_queries": ["q1"],
            "selected_cities": [],
            "payload_capture_mode": "off",
            "markdown_defaults": {
                "max_turns": 5,
                "batch_max_chunks": 1,
                "max_workers": 1,
                "reasoning_effort": None,
            },
            "models": [
                {"id": "model-a", "enabled": True},
                {"id": "model-b", "enabled": True},
            ],
        },
    )

    executed_models: list[str] = []

    def _fake_retrieve(
        *,
        queries: list[str],
        config,
        docs_dir: Path,
        selected_cities: list[str],
        run_id: str,
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        del queries, config, docs_dir, selected_cities, run_id
        return (_build_fake_retrieval_chunks(), {"queries": ["q1"]})

    def _fake_markdown(
        *,
        question: str,
        documents: list[dict[str, object]],
        config,
        api_key: str,
        **kwargs,
    ) -> MarkdownResearchResult:
        del question, documents, api_key, kwargs
        executed_models.append(config.markdown_researcher.model)
        return MarkdownResearchResult(
            status="success",
            accepted_chunk_ids=["a1", "b1"],
            excerpts=[],
        )

    report = run_markdown_reliability_benchmark(
        benchmark_id="bench_filter",
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        matrix_config_path=matrix_path,
        selected_models=["model-b"],
        api_key_override="test-key",
        retrieve_func=_fake_retrieve,
        markdown_func=_fake_markdown,
    )

    assert executed_models == ["model-b"]
    assert [result.model_id for result in report.results] == ["model-b"]


def test_run_markdown_reliability_benchmark_handles_full_model_failure(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.yml"
    _write_matrix(
        matrix_path,
        {
            "question": "Benchmark question",
            "retrieval_queries": ["q1"],
            "selected_cities": [],
            "payload_capture_mode": "off",
            "markdown_defaults": {
                "max_turns": 5,
                "batch_max_chunks": 1,
                "max_workers": 1,
                "reasoning_effort": None,
            },
            "models": [{"id": "model-a", "enabled": True}],
        },
    )

    def _fake_retrieve(
        *,
        queries: list[str],
        config,
        docs_dir: Path,
        selected_cities: list[str],
        run_id: str,
    ) -> tuple[list[RetrievedChunk], dict[str, object]]:
        del queries, config, docs_dir, selected_cities, run_id
        return (_build_fake_retrieval_chunks(), {"queries": ["q1"]})

    def _failing_markdown(**kwargs) -> MarkdownResearchResult:
        del kwargs
        raise RuntimeError("boom")

    report = run_markdown_reliability_benchmark(
        benchmark_id="bench_fail",
        output_dir=tmp_path / "output",
        config_path=Path("llm_config.yaml"),
        matrix_config_path=matrix_path,
        api_key_override="test-key",
        retrieve_func=_fake_retrieve,
        markdown_func=_failing_markdown,
    )

    assert report.results[0].failed_entirely is True
    assert report.results[0].failed_batches == 2
    assert report.results[0].unresolved_total == 2
    assert report.results[0].error_code == "MARKDOWN_BENCHMARK_EXCEPTION"
    assert (tmp_path / "output" / "bench_fail" / "model-a" / "model_result.json").exists()


def test_run_markdown_reliability_script_passes_cli_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):
        captured.update(kwargs)

        class _Report:
            benchmark_id = "bench_script"
            output_dir = str(tmp_path / "output" / "bench_script")

        return _Report()

    monkeypatch.setattr(
        "backend.scripts.run_markdown_reliability_benchmark.run_markdown_reliability_benchmark",
        _fake_runner,
    )
    monkeypatch.setattr(
        "backend.scripts.run_markdown_reliability_benchmark.setup_logger",
        lambda: None,
    )

    main(
        [
            "--benchmark-id",
            "bench_script",
            "--output-dir",
            str(tmp_path / "output"),
            "--config",
            "llm_config.yaml",
            "--matrix-config",
            str(DEFAULT_MATRIX_CONFIG_PATH),
            "--model",
            "model-a",
        ]
    )

    assert captured["benchmark_id"] == "bench_script"
    assert captured["output_dir"] == tmp_path / "output"
    assert captured["config_path"] == Path("llm_config.yaml")
    assert captured["matrix_config_path"] == DEFAULT_MATRIX_CONFIG_PATH
    assert captured["selected_models"] == ["model-a"]
    assert captured["selected_cities"] is None


def test_run_markdown_reliability_script_passes_city_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):
        captured.update(kwargs)

        class _Report:
            benchmark_id = "bench_script"
            output_dir = str(tmp_path / "output" / "bench_script")

        return _Report()

    monkeypatch.setattr(
        "backend.scripts.run_markdown_reliability_benchmark.run_markdown_reliability_benchmark",
        _fake_runner,
    )
    monkeypatch.setattr(
        "backend.scripts.run_markdown_reliability_benchmark.setup_logger",
        lambda: None,
    )

    main(
        [
            "--benchmark-id",
            "bench_script",
            "--output-dir",
            str(tmp_path / "output"),
            "--city",
            "Aachen",
            "--city",
            "Oslo",
        ]
    )

    assert captured["selected_cities"] == ["Aachen", "Oslo"]
