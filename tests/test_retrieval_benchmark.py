from __future__ import annotations

from pathlib import Path

import pytest

from backend.benchmarks.runner import _collect_llm_issue_counts
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
