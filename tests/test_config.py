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


def test_load_config_raises_clear_error_for_invalid_int_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid integer env values raise errors that include the env var name."""
    config_path = _write_minimal_config(tmp_path)
    monkeypatch.setenv("EMBEDDING_CHUNK_TOKENS", "abc")

    with pytest.raises(ValueError, match="EMBEDDING_CHUNK_TOKENS"):
        load_config(config_path)


def test_load_config_raises_clear_error_for_invalid_float_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid float env values raise errors that include the env var name."""
    config_path = _write_minimal_config(tmp_path)
    monkeypatch.setenv("VECTOR_STORE_RETRIEVAL_MAX_DISTANCE", "not-a-float")

    with pytest.raises(ValueError, match="VECTOR_STORE_RETRIEVAL_MAX_DISTANCE"):
        load_config(config_path)
