from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.utils.config import MarkdownResearcherConfig, load_config


def _base_config_lines() -> list[str]:
    """Return the minimal valid YAML lines shared by config tests."""
    return [
        "orchestrator:",
        "  model: test-model",
        "markdown_researcher:",
        "  model: test-model",
        "  chunk_overlap_tokens: 2000",
        "  batch_max_chunks: 32",
        "  max_workers: 8",
        "  request_backoff_base_seconds: 0.5",
        "  request_backoff_max_seconds: 2.0",
        "writer:",
        "  model: test-model",
        "chat:",
        "  model: openai/gpt-5.4-mini",
        "  provider_timeout_seconds: 60.0",
        "  followup_router_max_excerpts_per_source: 50",
        "assumptions_reviewer:",
        "  model: openai/gpt-5.4-mini",
        "benchmark_fact_judge:",
        "  model: openai/gpt-5.4-mini",
        "retry:",
        "  backoff_base_seconds: 1.0",
        "  backoff_max_seconds: 30.0",
    ]


def _write_config(tmp_path: Path, extra_lines: list[str] | None = None) -> Path:
    """Write one temporary llm_config.yaml for load_config tests."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join([*_base_config_lines(), *(extra_lines or [])]),
        encoding="utf-8",
    )
    return config_path


def test_load_config_ignores_removed_vector_store_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Removed vector-store env vars do not affect markdown-only config loading."""
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("VECTOR_STORE_ENABLED", "true")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "custom-chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "old-collection")

    config = load_config(config_path)

    assert config.runs_dir == Path("output")
    assert config.markdown_dir == Path("documents")


def test_markdown_researcher_config_applies_safe_runtime_defaults() -> None:
    """Markdown researcher direct construction preserves safe defaults."""
    config = MarkdownResearcherConfig(
        model="test-model",
        chunk_overlap_tokens=2000,
        batch_max_chunks=32,
    )

    assert config.max_workers == 2
    assert config.request_backoff_base_seconds == 2.0
    assert config.request_backoff_max_seconds == 10.0


def test_load_config_ignores_removed_vector_store_yaml_section(tmp_path: Path) -> None:
    """Legacy vector-store YAML is ignored rather than reintroduced into AppConfig."""
    config_path = _write_config(
        tmp_path,
        [
            "vector_store:",
            "  embedding_model: custom-embedding-model",
            "  retrieval_max_distance: 0.75",
            "  retrieval_max_chunks_per_city_query: 42",
        ],
    )

    config = load_config(config_path)

    assert config.markdown_researcher.model == "test-model"
    assert config.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_load_config_reads_markdown_reasoning_effort_from_yaml(tmp_path: Path) -> None:
    """Markdown reasoning effort is loaded when configured."""
    config_path = _write_config(
        tmp_path,
        [
            "markdown_researcher:",
            "  model: openai/gpt-5.4-mini",
            "  reasoning_effort: none",
            "  chunk_overlap_tokens: 2000",
            "  batch_max_chunks: 32",
            "  max_workers: 8",
            "  request_backoff_base_seconds: 0.5",
            "  request_backoff_max_seconds: 2.0",
        ],
    )

    config = load_config(config_path)

    assert config.markdown_researcher.reasoning_effort == "none"


def test_load_config_reads_markdown_strict_decision_audit_from_yaml(
    tmp_path: Path,
) -> None:
    """Markdown strict decision-audit flag is loaded when configured."""
    config_path = _write_config(
        tmp_path,
        [
            "markdown_researcher:",
            "  model: test-model",
            "  chunk_overlap_tokens: 2000",
            "  batch_max_chunks: 32",
            "  max_workers: 8",
            "  request_backoff_base_seconds: 0.5",
            "  request_backoff_max_seconds: 2.0",
            "  strict_decision_audit: true",
        ],
    )

    config = load_config(config_path)

    assert config.markdown_researcher.strict_decision_audit is True


def test_load_config_reads_agent_reasoning_effort_for_gpt_modules(tmp_path: Path) -> None:
    """Agent-level reasoning effort is loaded for GPT-backed modules."""
    config_path = _write_config(
        tmp_path,
        [
            "orchestrator:",
            "  model: openai/gpt-5.4-mini",
            "  reasoning_effort: high",
            "markdown_researcher:",
            "  model: openai/gpt-5.4-mini",
            "  chunk_overlap_tokens: 2000",
            "  batch_max_chunks: 32",
            "  max_workers: 8",
            "  request_backoff_base_seconds: 0.5",
            "  request_backoff_max_seconds: 2.0",
            "writer:",
            "  model: openai/gpt-5.4-mini",
            "  reasoning_effort: high",
            "chat:",
            "  model: openai/gpt-5.4-mini",
            "  reasoning_effort: high",
            "  provider_timeout_seconds: 60.0",
            "  followup_router_max_excerpts_per_source: 50",
            "assumptions_reviewer:",
            "  model: openai/gpt-5.4-mini",
            "  reasoning_effort: high",
        ],
    )

    config = load_config(config_path)

    assert config.orchestrator.reasoning_effort == "high"
    assert config.writer.reasoning_effort == "high"
    assert config.chat.reasoning_effort == "high"
    assert config.assumptions_reviewer.reasoning_effort == "high"


def test_load_config_rejects_invalid_markdown_reasoning_effort(tmp_path: Path) -> None:
    """Invalid markdown reasoning effort values are rejected."""
    config_path = _write_config(
        tmp_path,
        [
            "markdown_researcher:",
            "  model: openai/gpt-5.4-mini",
            "  reasoning_effort: ultra",
            "  chunk_overlap_tokens: 2000",
            "  batch_max_chunks: 32",
            "  max_workers: 8",
            "  request_backoff_base_seconds: 0.5",
            "  request_backoff_max_seconds: 2.0",
        ],
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_load_config_reads_required_chat_defaults_from_yaml(tmp_path: Path) -> None:
    """Chat settings come from YAML instead of hidden model defaults."""
    config = load_config(_write_config(tmp_path))

    assert config.chat.max_history_messages == 12
    assert not config.chat.followup_search_enabled
    assert config.chat.max_auto_followup_bundles == 3
    assert config.chat.provider_timeout_seconds == 60.0
    assert config.chat.followup_router_max_history_messages == 6
    assert config.chat.followup_router_max_excerpts_per_source == 50
    assert config.retry.backoff_base_seconds == 1.0
    assert config.retry.backoff_max_seconds == 30.0


def test_load_config_applies_chat_and_assumptions_defaults_when_sections_missing(
    tmp_path: Path,
) -> None:
    """Missing chat and assumptions sections fall back to safe model defaults."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "markdown_researcher:",
                "  model: test-model",
                "  chunk_overlap_tokens: 2000",
                "  batch_max_chunks: 32",
                "  max_workers: 8",
                "  request_backoff_base_seconds: 0.5",
                "  request_backoff_max_seconds: 2.0",
                "writer:",
                "  model: test-model",
                "retry:",
                "  backoff_base_seconds: 1.0",
                "  backoff_max_seconds: 30.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.chat.model == "openai/gpt-5.4-mini"
    assert config.chat.provider_timeout_seconds == 60.0
    assert config.chat.followup_router_max_history_messages == 6
    assert config.chat.followup_router_max_excerpts_per_source == 50
    assert config.assumptions_reviewer.model == "openai/gpt-5.4-mini"
    assert config.benchmark_fact_judge.model == "openai/gpt-5.4-mini"
    assert config.benchmark_fact_judge.max_output_tokens == 600


def test_load_config_applies_retry_defaults_when_section_missing(tmp_path: Path) -> None:
    """Missing retry config falls back to RetryConfig defaults."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "markdown_researcher:",
                "  model: test-model",
                "  chunk_overlap_tokens: 2000",
                "  batch_max_chunks: 32",
                "  max_workers: 8",
                "  request_backoff_base_seconds: 0.5",
                "  request_backoff_max_seconds: 2.0",
                "writer:",
                "  model: test-model",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.retry.max_attempts == 5
    assert config.retry.backoff_base_seconds == 1.0
    assert config.retry.backoff_max_seconds == 30.0


def test_load_config_reads_central_retry_settings_from_yaml(tmp_path: Path) -> None:
    """Retry settings can be overridden via top-level retry config."""
    config_path = _write_config(
        tmp_path,
        [
            "retry:",
            "  max_attempts: 7",
            "  backoff_base_seconds: 0.25",
            "  backoff_max_seconds: 3.5",
        ],
    )

    config = load_config(config_path)

    assert config.retry.max_attempts == 7
    assert config.retry.backoff_base_seconds == 0.25
    assert config.retry.backoff_max_seconds == 3.5
