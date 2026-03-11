"""Shared configuration builders for tests."""

from __future__ import annotations

from pathlib import Path

from backend.utils.config import (
    AgentConfig,
    AppConfig,
    AssumptionsReviewerConfig,
    ChatConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    RetryConfig,
    SqlResearcherConfig,
    VectorStoreConfig,
)


def build_markdown_researcher_config(
    *,
    model: str = "test-model",
    overrides: dict[str, object] | None = None,
) -> MarkdownResearcherConfig:
    """Build a markdown researcher config with the required test defaults."""
    data: dict[str, object] = {
        "model": model,
        "max_chunk_tokens": 40_000,
        "chunk_overlap_tokens": 2000,
        "batch_max_chunks": 32,
        "max_workers": 8,
        "request_backoff_base_seconds": 0.5,
        "request_backoff_max_seconds": 2.0,
    }
    if overrides:
        data.update(overrides)
    return MarkdownResearcherConfig(**data)


def build_chat_config(
    *,
    model: str = "openai/gpt-5.2",
    overrides: dict[str, object] | None = None,
) -> ChatConfig:
    """Build a chat config with the required test defaults."""
    data: dict[str, object] = {
        "model": model,
        "provider_timeout_seconds": 60.0,
        "followup_router_max_excerpts_per_source": 50,
    }
    if overrides:
        data.update(overrides)
    return ChatConfig(**data)


def build_retry_config(
    *,
    overrides: dict[str, object] | None = None,
) -> RetryConfig:
    """Build the shared retry config used across tests."""
    data: dict[str, object] = {
        "backoff_base_seconds": 1.0,
        "backoff_max_seconds": 30.0,
    }
    if overrides:
        data.update(overrides)
    return RetryConfig(**data)


def build_test_app_config(
    *,
    orchestrator_model: str = "test-model",
    sql_researcher_model: str = "test-model",
    markdown_researcher_model: str = "test-model",
    writer_model: str = "test-model",
    chat_model: str = "openai/gpt-5.2",
    assumptions_reviewer_model: str = "test-model",
    runs_dir: Path = Path("output"),
    markdown_dir: Path = Path("documents"),
    source_db_path: Path = Path("data/source.db"),
    enable_sql: bool = False,
    vector_store: VectorStoreConfig | None = None,
    orchestrator_overrides: dict[str, object] | None = None,
    sql_researcher_overrides: dict[str, object] | None = None,
    markdown_researcher_overrides: dict[str, object] | None = None,
    writer_overrides: dict[str, object] | None = None,
    chat_overrides: dict[str, object] | None = None,
    assumptions_reviewer_overrides: dict[str, object] | None = None,
    retry_overrides: dict[str, object] | None = None,
) -> AppConfig:
    """Build an AppConfig with the current required sections for tests."""
    orchestrator_data: dict[str, object] = {
        "model": orchestrator_model,
        "context_bundle_name": "context_bundle.json",
    }
    if orchestrator_overrides:
        orchestrator_data.update(orchestrator_overrides)

    sql_researcher_data: dict[str, object] = {"model": sql_researcher_model}
    if sql_researcher_overrides:
        sql_researcher_data.update(sql_researcher_overrides)

    writer_data: dict[str, object] = {"model": writer_model}
    if writer_overrides:
        writer_data.update(writer_overrides)

    assumptions_reviewer_data: dict[str, object] = {
        "model": assumptions_reviewer_model,
    }
    if assumptions_reviewer_overrides:
        assumptions_reviewer_data.update(assumptions_reviewer_overrides)

    app_config_data: dict[str, object] = {
        "orchestrator": OrchestratorConfig(**orchestrator_data),
        "sql_researcher": SqlResearcherConfig(**sql_researcher_data),
        "markdown_researcher": build_markdown_researcher_config(
            model=markdown_researcher_model,
            overrides=markdown_researcher_overrides,
        ),
        "writer": AgentConfig(**writer_data),
        "chat": build_chat_config(model=chat_model, overrides=chat_overrides),
        "assumptions_reviewer": AssumptionsReviewerConfig(
            **assumptions_reviewer_data
        ),
        "retry": build_retry_config(overrides=retry_overrides),
        "runs_dir": runs_dir,
        "markdown_dir": markdown_dir,
        "source_db_path": source_db_path,
        "enable_sql": enable_sql,
    }
    if vector_store is not None:
        app_config_data["vector_store"] = vector_store

    return AppConfig(**app_config_data)
