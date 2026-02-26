from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel):
    model: str
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    context_window_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None
    input_token_reserve: int = 2000


class OrchestratorConfig(AgentConfig):
    context_bundle_name: str = "context_bundle.json"


class SqlResearcherConfig(AgentConfig):
    max_result_tokens: int = 100000
    max_rows: int = 10000
    pre_orchestrator_rounds: int = 2


class MarkdownResearcherConfig(AgentConfig):
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    max_files: int = 200
    max_file_bytes: int = 5_000_000
    max_chunk_tokens: Optional[int] = None
    chunk_overlap_tokens: int = 200
    batch_max_chunks: int = 4
    batch_max_input_tokens: Optional[int] = None
    batch_overhead_tokens: int = 600
    max_workers: int = 2
    max_retries: int = 2
    retry_base_seconds: float = 0.8
    retry_max_seconds: float = 6.0
    request_backoff_base_seconds: float = 2.0
    request_backoff_max_seconds: float = 10.0


class ChatConfig(AgentConfig):
    max_history_messages: int = 24


class AssumptionsReviewerConfig(AgentConfig):
    """Configuration for two-pass missing-data discovery."""


class VectorStoreConfig(BaseModel):
    enabled: bool = False
    chroma_persist_path: Path = Field(default_factory=lambda: Path(".chroma"))
    chroma_collection_name: str = "markdown_chunks"
    embedding_model: str = "text-embedding-3-large"
    embedding_max_input_tokens: int | None = 8000
    embedding_batch_size: int = 100
    embedding_max_retries: int = 3
    embedding_retry_base_seconds: float = 0.8
    embedding_retry_max_seconds: float = 8.0
    embedding_chunk_tokens: int = 800
    embedding_chunk_overlap_tokens: int = 80
    table_row_group_max_rows: int = 25
    retrieval_max_distance: float | None = 1.0
    retrieval_fallback_min_chunks_per_city_query: int = 20
    retrieval_max_chunks_per_city_query: int = 60
    retrieval_max_chunks_per_city: int | None = 300
    context_window_chunks: int = 0
    table_context_window_chunks: int = 1
    auto_update_on_run: bool = False
    index_manifest_path: Path = Field(
        default_factory=lambda: Path(".chroma/index_manifest.json")
    )


class AppConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    orchestrator: OrchestratorConfig
    sql_researcher: SqlResearcherConfig
    markdown_researcher: MarkdownResearcherConfig
    writer: AgentConfig
    chat: ChatConfig = Field(default_factory=lambda: ChatConfig(model="openai/gpt-5.2"))
    assumptions_reviewer: AssumptionsReviewerConfig = Field(
        default_factory=lambda: AssumptionsReviewerConfig(model="openai/gpt-5.2")
    )
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    runs_dir: Path = Field(default_factory=lambda: Path("output"))
    source_db_path: Path = Field(default_factory=lambda: Path("data/source.db"))
    source_db_url: str | None = None
    markdown_dir: Path = Field(default_factory=lambda: Path("documents"))
    enable_sql: bool = False

def _parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    load_dotenv()
    path = config_path or Path("llm_config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)

    runs_dir = os.getenv("RUNS_DIR")
    source_db_path = os.getenv("SOURCE_DB_PATH")
    markdown_dir = os.getenv("MARKDOWN_DIR")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")
    database_url = os.getenv("DATABASE_URL")
    enable_sql = os.getenv("ENABLE_SQL")
    vector_store_enabled = os.getenv("VECTOR_STORE_ENABLED")
    chroma_persist_path = os.getenv("CHROMA_PERSIST_PATH")
    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME")

    if runs_dir:
        config.runs_dir = Path(runs_dir)
    if source_db_path:
        config.source_db_path = Path(source_db_path)
    if markdown_dir:
        config.markdown_dir = Path(markdown_dir)
    if openrouter_base_url:
        config.openrouter_base_url = openrouter_base_url
    if database_url:
        config.source_db_url = database_url
    if enable_sql is not None:
        parsed = _parse_env_bool(enable_sql)
        if parsed is not None:
            config.enable_sql = parsed
    if vector_store_enabled is not None:
        parsed = _parse_env_bool(vector_store_enabled)
        if parsed is not None:
            config.vector_store.enabled = parsed
    if chroma_persist_path:
        manifest_default = Path(".chroma/index_manifest.json")
        config.vector_store.chroma_persist_path = Path(chroma_persist_path)
        if config.vector_store.index_manifest_path == manifest_default:
            config.vector_store.index_manifest_path = (
                config.vector_store.chroma_persist_path / "index_manifest.json"
            )
    if chroma_collection_name:
        config.vector_store.chroma_collection_name = chroma_collection_name

    return config


def get_openrouter_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in the environment.")
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
    if not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    # Disable OpenAI Agents tracing when using OpenRouter to avoid authentication errors
    # OpenRouter keys are not recognized by OpenAI's tracing endpoint
    if not os.getenv("OPENAI_AGENTS_DISABLE_TRACING"):
        os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
    return api_key


def get_database_url() -> str:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError("DATABASE_URL is not set in the environment.")
    return database_url


__all__ = [
    "AgentConfig",
    "OrchestratorConfig",
    "SqlResearcherConfig",
    "MarkdownResearcherConfig",
    "ChatConfig",
    "AssumptionsReviewerConfig",
    "VectorStoreConfig",
    "AppConfig",
    "load_config",
    "get_openrouter_api_key",
    "get_database_url",
]
