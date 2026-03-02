from __future__ import annotations

from pathlib import Path

import pytest

from backend.benchmarks.runner import (
    BenchmarkMarkdownConfig,
    BenchmarkModeConfig,
    _collect_llm_issue_counts,
    run_retrieval_strategy_benchmark,
)
from backend.scripts.run_retrieval_benchmark import _build_markdown_configs


def test_build_markdown_configs_parses_and_deduplicates() -> None:
    configs = _build_markdown_configs(["16:8", "32:4", "16:8", "32:8"])
    assert [cfg.name for cfg in configs] == ["b16_w8", "b32_w4", "b32_w8"]
    assert [cfg.batch_max_chunks for cfg in configs] == [16, 32, 32]
    assert [cfg.max_workers for cfg in configs] == [8, 4, 8]


@pytest.mark.parametrize(
    "raw_option",
    ["", "16", "16:x", "x:8", "0:8", "16:0"],
)
def test_build_markdown_configs_rejects_invalid_option(raw_option: str) -> None:
    with pytest.raises(ValueError):
        _build_markdown_configs([raw_option])


def test_collect_llm_issue_counts_tracks_rate_limits_and_failures(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    run_log.write_text(
        "\n".join(
            [
                "foo - WARNING - OpenAI HTTP error response: POST https://x -> 429 Too Many Requests",
                "foo - WARNING - OpenAI HTTP error response: POST https://x -> 503 Service Unavailable",
                'foo - INFO - HTTP Request: POST https://x "HTTP/1.1 429 Too Many Requests"',
                (
                    "foo - ERROR - RETRY_EXHAUSTED "
                    '{"error_type":"RateLimitError","error_message":"Too many requests"}'
                ),
                (
                    "foo - ERROR - RETRY_EXHAUSTED "
                    '{"error_type":"APIConnectionError","error_message":"Connection error"}'
                ),
                (
                    "foo - ERROR - RETRY_EXHAUSTED "
                    '{"error_type":"RetryableBadOutput","error_message":"output_none"}'
                ),
                "foo - WARNING - AGENT_MAX_TURNS_DIAGNOSTICS {}",
            ]
        ),
        encoding="utf-8",
    )

    counts = _collect_llm_issue_counts(run_log)
    assert counts["rate_limit_count"] == 3
    assert counts["not_working_count"] == 4
    assert counts["http_error_count"] == 2
    assert counts["retry_exhausted_count"] == 3
    assert counts["max_turns_count"] == 1
    assert counts["bad_output_count"] == 1


def test_collect_llm_issue_counts_uses_max_turns_fallback(tmp_path: Path) -> None:
    run_log = tmp_path / "run.log"
    run_log.write_text(
        "foo - WARNING - Markdown Aachen batch 1 hit max turns limit.",
        encoding="utf-8",
    )

    counts = _collect_llm_issue_counts(run_log)
    assert counts["max_turns_count"] == 1
    assert counts["not_working_count"] == 1


def test_benchmark_continues_when_run_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "backend.benchmarks.runner._load_questions",
        lambda _path: ["Test question"],
    )
    monkeypatch.setattr(
        "backend.benchmarks.runner._load_env_overrides",
        lambda _files: {},
    )
    monkeypatch.setattr(
        "backend.benchmarks.runner._run_mode_question",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("429 Too Many Requests")),
    )
    monkeypatch.setattr(
        "backend.benchmarks.runner._run_benchmark_judging",
        lambda **_kwargs: ([], {}, {}),
    )

    report = run_retrieval_strategy_benchmark(
        benchmark_id="test_benchmark",
        output_dir=tmp_path / "output",
        config_path=tmp_path / "llm_config.yaml",
        docs_dir=tmp_path / "docs",
        questions_file=tmp_path / "questions.txt",
        selected_cities=[],
        repetitions=1,
        mode_configs=[BenchmarkModeConfig(name="standard_chunking", env_files=[])],
        markdown_configs=[
            BenchmarkMarkdownConfig(name="b16_w8", batch_max_chunks=16, max_workers=8)
        ],
        use_query_overrides=False,
        query_overrides_path=None,
        log_llm_payload=False,
    )

    assert len(report.results) == 1
    row = report.results[0]
    assert row.success is False
    assert row.llm_rate_limit_count == 1
    assert row.llm_not_working_count == 0
    assert row.llm_issue_total == 1
    assert report.mode_config_summary["standard_chunking | b16_w8"]["runs_failed"] == 1.0
