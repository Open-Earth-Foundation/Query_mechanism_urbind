from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.utils.config import (
    MarkdownResearcherConfig,
    load_config,
    resolve_path_relative_to_config,
)


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
        "benchmark_number_extractor:",
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
    """Vector-store tuning env vars are ignored in favor of llm_config.yaml values."""
    config_path = _write_config(tmp_path)
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
    config_path = _write_config(tmp_path)
    chroma_path = tmp_path / "custom-chroma"
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(chroma_path))

    config = load_config(config_path)

    assert config.vector_store.chroma_persist_path == chroma_path
    assert config.vector_store.index_manifest_path == chroma_path / "index_manifest.json"


def test_load_config_resolves_relative_runtime_paths_against_config_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Relative YAML runtime paths should anchor to the config file directory."""
    monkeypatch.setenv("RUNS_DIR", "")
    monkeypatch.setenv("MARKDOWN_DIR", "")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", "")
    monkeypatch.setenv("EXTERNAL_SOURCE_DIR", "")
    config_dir = tmp_path / "nested" / "config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator:",
                "  model: test-model",
                "markdown_researcher:",
                "  model: test-model",
                "  chunk_overlap_tokens: 2000",
                "  batch_max_chunks: 32",
                "writer:",
                "  model: test-model",
                "runs_dir: output-data",
                "markdown_dir: docs-data",
                "enrichment:",
                "  model: openai/gpt-5.4-mini",
                "  external_source_dir: docs-data/source_library",
                "vector_store:",
                "  chroma_persist_path: local-chroma",
                "  index_manifest_path: local-chroma/index_manifest.json",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.runs_dir == (config_dir / "output-data").resolve()
    assert config.markdown_dir == (config_dir / "docs-data").resolve()
    assert config.enrichment.external_source_dir == (
        config_dir / "docs-data" / "source_library"
    ).resolve()
    assert config.vector_store.chroma_persist_path == (config_dir / "local-chroma").resolve()
    assert config.vector_store.index_manifest_path == (
        config_dir / "local-chroma" / "index_manifest.json"
    ).resolve()


def test_load_config_resolves_relative_markdown_env_override_against_config_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Relative MARKDOWN_DIR env overrides should anchor to the config file directory."""
    config_dir = tmp_path / "nested" / "config"
    config_dir.mkdir(parents=True)
    config_path = _write_config(config_dir)
    monkeypatch.setenv("MARKDOWN_DIR", "docs-env")

    config = load_config(config_path)

    assert config.markdown_dir == (config_dir / "docs-env").resolve()


def test_resolve_path_relative_to_config_anchors_relative_paths(
    tmp_path: Path,
) -> None:
    """CLI path overrides should resolve relative to the config file directory."""
    config_dir = tmp_path / "nested" / "config"
    config_dir.mkdir(parents=True)
    config_path = _write_config(config_dir)

    resolved = resolve_path_relative_to_config(config_path, Path("documents"))

    assert resolved == (config_dir / "documents").resolve()


def test_load_config_reads_vector_store_settings_from_yaml(tmp_path: Path) -> None:
    """Vector-store retrieval and embedding knobs are loaded from llm_config.yaml."""
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

    assert config.vector_store.embedding_model == "custom-embedding-model"
    assert config.vector_store.retrieval_max_distance == 0.75
    assert config.vector_store.retrieval_max_chunks_per_city_query == 42


def test_load_config_applies_vector_store_auto_update_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """VECTOR_STORE_AUTO_UPDATE_ON_RUN overrides the YAML default."""
    config_path = _write_config(
        tmp_path,
        [
            "vector_store:",
            "  auto_update_on_run: false",
        ],
    )
    monkeypatch.setenv("VECTOR_STORE_AUTO_UPDATE_ON_RUN", "true")

    config = load_config(config_path)

    assert config.vector_store.auto_update_on_run is True


def test_load_config_applies_vector_store_update_mode_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VECTOR_STORE_UPDATE_MODE selects local or Kubernetes updater orchestration."""
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "orchestrator: {model: openai/gpt-5.4-mini}",
                "markdown_researcher: {model: openai/gpt-5.4-mini}",
                "writer: {model: openai/gpt-5.4-mini}",
                "vector_store:",
                "  update_mode: local_process",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VECTOR_STORE_UPDATE_MODE", "kubernetes_job")

    config = load_config(config_path)

    assert config.vector_store.update_mode == "kubernetes_job"


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
    assert config.benchmark_number_extractor.model == "openai/gpt-5.4-mini"
    assert config.benchmark_number_extractor.max_output_tokens == 900


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


def test_load_config_reads_benchmark_number_extractor_from_yaml(tmp_path: Path) -> None:
    """Benchmark number extractor settings are loaded from llm_config.yaml."""
    config_path = _write_config(
        tmp_path,
        [
            "benchmark_number_extractor:",
            "  model: openai/gpt-5.4-mini",
            "  max_output_tokens: 750",
            "  reasoning_effort: medium",
        ],
    )

    config = load_config(config_path)

    assert config.benchmark_number_extractor.model == "openai/gpt-5.4-mini"
    assert config.benchmark_number_extractor.max_output_tokens == 750
    assert config.benchmark_number_extractor.reasoning_effort == "medium"


def test_load_config_applies_mlflow_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MLflow observability is optional and disabled by default."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    for key in (
        "MLFLOW_ENABLED",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_ARTIFACT_PATH",
        "MLFLOW_TRACE_MODE",
        "MLFLOW_FAIL_ON_ERROR",
    ):
        monkeypatch.delenv(key, raising=False)

    config = load_config(_write_config(tmp_path))

    assert config.mlflow.enabled is False
    assert config.mlflow.tracking_uri is None
    assert config.mlflow.experiment_name == "URBIND"
    assert config.mlflow.artifact_path == "run_artifacts"
    assert config.mlflow.trace_mode == "consolidated"
    assert config.mlflow.fail_on_error is False


def test_load_config_applies_mlflow_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All MLFLOW_* env vars override YAML/default settings."""
    config_path = _write_config(tmp_path)
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "Test Experiment")
    monkeypatch.setenv("MLFLOW_ARTIFACT_PATH", "full_run")
    monkeypatch.setenv("MLFLOW_TRACE_MODE", "consolidated")
    monkeypatch.setenv("MLFLOW_FAIL_ON_ERROR", "true")

    config = load_config(config_path)

    assert config.mlflow.enabled is True
    assert config.mlflow.tracking_uri == "file:///tmp/mlruns"
    assert config.mlflow.experiment_name == "Test Experiment"
    assert config.mlflow.artifact_path == "full_run"
    assert config.mlflow.trace_mode == "consolidated"
    assert config.mlflow.fail_on_error is True
