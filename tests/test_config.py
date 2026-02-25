from pathlib import Path

import pytest

from backend.utils.config import load_config


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
                "writer:",
                "  model: test-model",
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
                "writer:",
                "  model: test-model",
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
