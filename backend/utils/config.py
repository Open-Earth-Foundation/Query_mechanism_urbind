from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel):
    model: str
    temperature: Optional[float] = None
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
    embedding_chunk_tokens: int = 800
    embedding_chunk_overlap_tokens: int = 80
    table_row_group_max_rows: int = 25
    retrieval_max_distance: float | None = None
    retrieval_fallback_min_chunks_per_city_query: int = 20
    retrieval_max_chunks_per_city_query: int = 100
    retrieval_max_chunks_per_city: int | None = None
    context_window_chunks: int = 1
    table_context_window_chunks: int = 2
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
    embedding_model = os.getenv("EMBEDDING_MODEL")
    embedding_chunk_tokens = os.getenv("EMBEDDING_CHUNK_TOKENS")
    embedding_chunk_overlap_tokens = os.getenv("EMBEDDING_CHUNK_OVERLAP_TOKENS")
    table_row_group_max_rows = os.getenv("TABLE_ROW_GROUP_MAX_ROWS")
    markdown_batch_max_chunks = os.getenv("MARKDOWN_BATCH_MAX_CHUNKS")
    markdown_batch_max_input_tokens = os.getenv("MARKDOWN_BATCH_MAX_INPUT_TOKENS")
    markdown_batch_overhead_tokens = os.getenv("MARKDOWN_BATCH_OVERHEAD_TOKENS")
    vector_store_retrieval_max_distance = os.getenv(
        "VECTOR_STORE_RETRIEVAL_MAX_DISTANCE"
    )
    vector_store_retrieval_fallback_min_chunks_per_city_query = os.getenv(
        "VECTOR_STORE_RETRIEVAL_FALLBACK_MIN_CHUNKS_PER_CITY_QUERY"
    )
    vector_store_retrieval_max_chunks_per_city_query = os.getenv(
        "VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY_QUERY"
    )
    vector_store_retrieval_max_chunks_per_city = os.getenv(
        "VECTOR_STORE_RETRIEVAL_MAX_CHUNKS_PER_CITY"
    )
    vector_store_context_window_chunks = os.getenv("VECTOR_STORE_CONTEXT_WINDOW_CHUNKS")
    vector_store_table_context_window_chunks = os.getenv(
        "VECTOR_STORE_TABLE_CONTEXT_WINDOW_CHUNKS"
    )
    vector_store_auto_update_on_run = os.getenv("VECTOR_STORE_AUTO_UPDATE_ON_RUN")
    index_manifest_path = os.getenv("INDEX_MANIFEST_PATH")

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
        config.vector_store.chroma_persist_path = Path(chroma_persist_path)
    if chroma_collection_name:
        config.vector_store.chroma_collection_name = chroma_collection_name
    if embedding_model:
        config.vector_store.embedding_model = embedding_model
    if embedding_chunk_tokens:
        config.vector_store.embedding_chunk_tokens = int(embedding_chunk_tokens)
    if embedding_chunk_overlap_tokens:
        config.vector_store.embedding_chunk_overlap_tokens = int(
            embedding_chunk_overlap_tokens
        )
    if table_row_group_max_rows:
        config.vector_store.table_row_group_max_rows = int(table_row_group_max_rows)
    if markdown_batch_max_chunks:
        config.markdown_researcher.batch_max_chunks = int(markdown_batch_max_chunks)
    if markdown_batch_max_input_tokens:
        config.markdown_researcher.batch_max_input_tokens = int(
            markdown_batch_max_input_tokens
        )
    if markdown_batch_overhead_tokens:
        config.markdown_researcher.batch_overhead_tokens = int(
            markdown_batch_overhead_tokens
        )
    if vector_store_retrieval_max_distance:
        config.vector_store.retrieval_max_distance = float(
            vector_store_retrieval_max_distance
        )
    if vector_store_retrieval_fallback_min_chunks_per_city_query:
        value = int(vector_store_retrieval_fallback_min_chunks_per_city_query)
        config.vector_store.retrieval_fallback_min_chunks_per_city_query = value
    if vector_store_retrieval_max_chunks_per_city_query:
        config.vector_store.retrieval_max_chunks_per_city_query = int(
            vector_store_retrieval_max_chunks_per_city_query
        )
    if vector_store_retrieval_max_chunks_per_city:
        config.vector_store.retrieval_max_chunks_per_city = int(
            vector_store_retrieval_max_chunks_per_city
        )
    if vector_store_context_window_chunks:
        config.vector_store.context_window_chunks = int(vector_store_context_window_chunks)
    if vector_store_table_context_window_chunks:
        config.vector_store.table_context_window_chunks = int(
            vector_store_table_context_window_chunks
        )
    if vector_store_auto_update_on_run is not None:
        parsed = _parse_env_bool(vector_store_auto_update_on_run)
        if parsed is not None:
            config.vector_store.auto_update_on_run = parsed
    if index_manifest_path:
        config.vector_store.index_manifest_path = Path(index_manifest_path)

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
