from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.utils.config import MarkdownResearcherConfig, load_config


def _write_minimal_config(tmp_path: Path) -> Path:
    """Write a minimal valid config file for load_config tests."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "sql_researcher:",
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
                "  model: openai/gpt-5.2",
                "  provider_timeout_seconds: 60.0",
                "  followup_router_max_excerpts_per_source: 50",
                "assumptions_reviewer:",
                "  model: openai/gpt-5.2",
                "retry:",
                "  backoff_base_seconds: 1.0",
                "  backoff_max_seconds: 30.0",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_config_ignores_removed_vector_store_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Vector-store tuning env vars are ignored in favor of llm_config.yaml values."""
    config_path = _write_minimal_config(tmp_path)
    monkeypatch.setenv("EMBEDDING_CHUNK_TOKENS", "abc")
    monkeypatch.setenv("VECTOR_STORE_RETRIEVAL_MAX_DISTANCE", "not-a-float")
    monkeypatch.setenv("EMBEDDING_MAX_INPUT_TOKENS", "7000")

    config = load_config(config_path)

    assert config.vector_store.embedding_chunk_tokens == 800
    assert config.vector_store.retrieval_max_distance == 1.0
    assert config.vector_store.embedding_max_input_tokens == 8000


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


def test_load_config_applies_chroma_persist_path_env_and_derives_manifest_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CHROMA_PERSIST_PATH env override updates both store root and default manifest path."""
    config_path = _write_minimal_config(tmp_path)
    chroma_path = tmp_path / "custom-chroma"
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(chroma_path))

    config = load_config(config_path)

    assert config.vector_store.chroma_persist_path == chroma_path
    assert config.vector_store.index_manifest_path == chroma_path / "index_manifest.json"


def test_load_config_reads_vector_store_settings_from_yaml(
    tmp_path: Path,
) -> None:
    """Vector-store retrieval and embedding knobs are loaded from llm_config.yaml."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "sql_researcher:",
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
                "  model: openai/gpt-5.2",
                "  provider_timeout_seconds: 60.0",
                "  followup_router_max_excerpts_per_source: 50",
                "assumptions_reviewer:",
                "  model: openai/gpt-5.2",
                "retry:",
                "  backoff_base_seconds: 1.0",
                "  backoff_max_seconds: 30.0",
                "vector_store:",
                "  embedding_model: custom-embedding-model",
                "  retrieval_max_distance: 0.75",
                "  retrieval_max_chunks_per_city_query: 42",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.vector_store.embedding_model == "custom-embedding-model"
    assert config.vector_store.retrieval_max_distance == 0.75
    assert config.vector_store.retrieval_max_chunks_per_city_query == 42


def test_load_config_reads_markdown_reasoning_effort_from_yaml(
    tmp_path: Path,
) -> None:
    """Markdown reasoning effort is loaded when configured."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "sql_researcher:",
                "  model: test-model",
                "markdown_researcher:",
                "  model: x-ai/grok-4.1-fast",
                "  reasoning_effort: none",
                "  chunk_overlap_tokens: 2000",
                "  batch_max_chunks: 32",
                "  max_workers: 8",
                "  request_backoff_base_seconds: 0.5",
                "  request_backoff_max_seconds: 2.0",
                "writer:",
                "  model: test-model",
                "chat:",
                "  model: openai/gpt-5.2",
                "  provider_timeout_seconds: 60.0",
                "  followup_router_max_excerpts_per_source: 50",
                "assumptions_reviewer:",
                "  model: openai/gpt-5.2",
                "retry:",
                "  backoff_base_seconds: 1.0",
                "  backoff_max_seconds: 30.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.markdown_researcher.reasoning_effort == "none"


def test_load_config_reads_markdown_strict_decision_audit_from_yaml(
    tmp_path: Path,
) -> None:
    """Markdown strict decision-audit flag is loaded when configured."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "sql_researcher:",
                "  model: test-model",
                "markdown_researcher:",
                "  model: test-model",
                "  strict_decision_audit: true",
                "writer:",
                "  model: test-model",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.markdown_researcher.strict_decision_audit is True


def test_load_config_reads_agent_reasoning_effort_for_gpt_modules(
    tmp_path: Path,
) -> None:
    """Agent-level reasoning effort is loaded for non-markdown GPT modules."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: openai/gpt-5.2",
                "  reasoning_effort: high",
                "sql_researcher:",
                "  model: openai/gpt-5.2",
                "  reasoning_effort: high",
                "markdown_researcher:",
                "  model: x-ai/grok-4.1-fast",
                "  chunk_overlap_tokens: 2000",
                "  batch_max_chunks: 32",
                "  max_workers: 8",
                "  request_backoff_base_seconds: 0.5",
                "  request_backoff_max_seconds: 2.0",
                "writer:",
                "  model: openai/gpt-5.2",
                "  reasoning_effort: high",
                "chat:",
                "  model: openai/gpt-5.2",
                "  reasoning_effort: high",
                "  provider_timeout_seconds: 60.0",
                "  followup_router_max_excerpts_per_source: 50",
                "assumptions_reviewer:",
                "  model: openai/gpt-5.2",
                "  reasoning_effort: high",
                "retry:",
                "  backoff_base_seconds: 1.0",
                "  backoff_max_seconds: 30.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.orchestrator.reasoning_effort == "high"
    assert config.sql_researcher.reasoning_effort == "high"
    assert config.writer.reasoning_effort == "high"
    assert config.chat.reasoning_effort == "high"
    assert config.assumptions_reviewer.reasoning_effort == "high"


def test_load_config_rejects_invalid_markdown_reasoning_effort(
    tmp_path: Path,
) -> None:
    """Invalid markdown reasoning effort values are rejected."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "sql_researcher:",
                "  model: test-model",
                "markdown_researcher:",
                "  model: x-ai/grok-4.1-fast",
                "  reasoning_effort: ultra",
                "  chunk_overlap_tokens: 2000",
                "  batch_max_chunks: 32",
                "  max_workers: 8",
                "  request_backoff_base_seconds: 0.5",
                "  request_backoff_max_seconds: 2.0",
                "writer:",
                "  model: test-model",
                "chat:",
                "  model: openai/gpt-5.2",
                "  provider_timeout_seconds: 60.0",
                "  followup_router_max_excerpts_per_source: 50",
                "assumptions_reviewer:",
                "  model: openai/gpt-5.2",
                "retry:",
                "  backoff_base_seconds: 1.0",
                "  backoff_max_seconds: 30.0",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_load_config_reads_required_chat_defaults_from_yaml(tmp_path: Path) -> None:
    """Chat settings come from YAML instead of hidden model defaults."""
    config_path = _write_minimal_config(tmp_path)

    config = load_config(config_path)

    assert config.chat.max_history_messages == 12
    assert not config.chat.followup_search_enabled
    assert config.chat.max_auto_followup_bundles == 3
    assert config.chat.provider_timeout_seconds == 60.0
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
                "sql_researcher:",
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

    assert config.chat.model == "openai/gpt-5.2"
    assert config.chat.provider_timeout_seconds == 60.0
    assert config.chat.followup_router_max_excerpts_per_source == 50
    assert config.assumptions_reviewer.model == "openai/gpt-5.2"


def test_load_config_applies_retry_defaults_when_section_missing(
    tmp_path: Path,
) -> None:
    """Missing retry config falls back to RetryConfig defaults."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "sql_researcher:",
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
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "sql_researcher:",
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
                "  model: openai/gpt-5.2",
                "  provider_timeout_seconds: 60.0",
                "  followup_router_max_excerpts_per_source: 50",
                "assumptions_reviewer:",
                "  model: openai/gpt-5.2",
                "retry:",
                "  max_attempts: 7",
                "  backoff_base_seconds: 0.25",
                "  backoff_max_seconds: 3.5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.retry.max_attempts == 7
    assert config.retry.backoff_base_seconds == 0.25
    assert config.retry.backoff_max_seconds == 3.5
