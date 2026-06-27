"""Shared configuration builders for tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from backend.utils.config import AppConfig, EnrichmentConfig, VectorStoreConfig

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
    vector_store: VectorStoreConfig | None = None,
    vector_store_overrides: dict[str, object] | None = None,
    orchestrator_overrides: dict[str, object] | None = None,
    markdown_researcher_overrides: dict[str, object] | None = None,
    initiative_extractor_overrides: dict[str, object] | None = None,
    tef_mapper_overrides: dict[str, object] | None = None,
    writer_overrides: dict[str, object] | None = None,
    chat_overrides: dict[str, object] | None = None,
    assumptions_reviewer_overrides: dict[str, object] | None = None,
    enrichment_overrides: dict[str, object] | None = None,
    retry_overrides: dict[str, object] | None = None,
    mlflow_overrides: dict[str, object] | None = None,
) -> AppConfig:
    """Build a test AppConfig seeded from the repository llm_config.yaml."""
    test_chroma_path = runs_dir / "_test_chroma"
    config = load_repo_test_config().model_copy(deep=True)
    config.orchestrator = _apply_overrides(config.orchestrator, orchestrator_overrides)
    config.markdown_researcher = _apply_overrides(
        config.markdown_researcher,
        markdown_researcher_overrides,
    )
    config.initiative_extractor = _apply_overrides(
        config.initiative_extractor,
        initiative_extractor_overrides,
    )
    config.tef_mapper = _apply_overrides(config.tef_mapper, tef_mapper_overrides)
    if (
        not tef_mapper_overrides
        or "numeric_unit_classifier_enabled" not in tef_mapper_overrides
    ):
        config.tef_mapper = config.tef_mapper.model_copy(
            update={"numeric_unit_classifier_enabled": False}
        )
    config.writer = _apply_overrides(config.writer, writer_overrides)
    config.chat = _apply_overrides(config.chat, chat_overrides)
    config.assumptions_reviewer = _apply_overrides(
        config.assumptions_reviewer,
        assumptions_reviewer_overrides,
    )
    # Default the new split-flow / tier-1-first flags off in tests so existing
    # test mocks (built for the legacy single-pass flow) keep working.  Tests
    # exercising the new flow can opt in via enrichment_overrides.
    config.enrichment = config.enrichment.model_copy(
        update={"use_split_gap_flow": False, "tier1_first_search": False}
    )
    config.enrichment = _apply_overrides(config.enrichment, enrichment_overrides)
    if (
        not enrichment_overrides
        or "external_source_search_enabled" not in enrichment_overrides
    ):
        config.enrichment = config.enrichment.model_copy(
            update={"external_source_search_enabled": False}
        )
    config.retry = _apply_overrides(config.retry, retry_overrides)
    config.mlflow = _apply_overrides(config.mlflow, mlflow_overrides)
    config.vector_store = config.vector_store.model_copy(
        update={
            "enabled": False,
            "auto_update_on_run": False,
            "chroma_persist_path": test_chroma_path,
            "index_manifest_path": test_chroma_path / "index_manifest.json",
        }
    )
    config.vector_store = _apply_overrides(config.vector_store, vector_store_overrides)
    if vector_store is not None:
        config.vector_store = vector_store
    config.runs_dir = runs_dir
    config.markdown_dir = markdown_dir
    return config
