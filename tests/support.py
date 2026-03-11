"""Shared configuration builders for tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from backend.utils.config import AppConfig, VectorStoreConfig

ModelT = TypeVar("ModelT", bound=BaseModel)
TEST_CONFIG_PATH = Path(__file__).resolve().parents[1] / "llm_config.yaml"


@lru_cache(maxsize=1)
def load_repo_test_config() -> AppConfig:
    """Load the repository llm_config.yaml once for deterministic test defaults."""
    raw = yaml.safe_load(TEST_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)


def _apply_overrides(model: ModelT, overrides: dict[str, object] | None) -> ModelT:
    """Return a copied model updated with explicit test overrides."""
    if not overrides:
        return model
    return model.model_copy(update=overrides)


def build_test_app_config(
    *,
    runs_dir: Path = Path("output"),
    markdown_dir: Path = Path("documents"),
    source_db_path: Path = Path("data/source.db"),
    enable_sql: bool | None = None,
    vector_store: VectorStoreConfig | None = None,
    vector_store_overrides: dict[str, object] | None = None,
    orchestrator_overrides: dict[str, object] | None = None,
    sql_researcher_overrides: dict[str, object] | None = None,
    markdown_researcher_overrides: dict[str, object] | None = None,
    writer_overrides: dict[str, object] | None = None,
    chat_overrides: dict[str, object] | None = None,
    assumptions_reviewer_overrides: dict[str, object] | None = None,
    retry_overrides: dict[str, object] | None = None,
) -> AppConfig:
    """Build a test AppConfig seeded from the repository llm_config.yaml."""
    config = load_repo_test_config().model_copy(deep=True)
    config.orchestrator = _apply_overrides(config.orchestrator, orchestrator_overrides)
    config.sql_researcher = _apply_overrides(config.sql_researcher, sql_researcher_overrides)
    config.markdown_researcher = _apply_overrides(
        config.markdown_researcher,
        markdown_researcher_overrides,
    )
    config.writer = _apply_overrides(config.writer, writer_overrides)
    config.chat = _apply_overrides(config.chat, chat_overrides)
    config.assumptions_reviewer = _apply_overrides(
        config.assumptions_reviewer,
        assumptions_reviewer_overrides,
    )
    config.retry = _apply_overrides(config.retry, retry_overrides)
    config.vector_store = _apply_overrides(config.vector_store, vector_store_overrides)
    if vector_store is not None:
        config.vector_store = vector_store
    config.runs_dir = runs_dir
    config.markdown_dir = markdown_dir
    config.source_db_path = source_db_path
    if enable_sql is not None:
        config.enable_sql = enable_sql
    return config
