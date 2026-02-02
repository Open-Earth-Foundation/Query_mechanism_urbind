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
    max_workers: int = 2
    max_retries: int = 2
    retry_base_seconds: float = 0.8
    retry_max_seconds: float = 6.0


class AppConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    orchestrator: OrchestratorConfig
    sql_researcher: SqlResearcherConfig
    markdown_researcher: MarkdownResearcherConfig
    writer: AgentConfig
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    runs_dir: Path = Field(default_factory=lambda: Path("output"))
    source_db_path: Path = Field(default_factory=lambda: Path("data/source.db"))
    source_db_url: str | None = None
    markdown_dir: Path = Field(default_factory=lambda: Path("documents"))


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
    "AppConfig",
    "load_config",
    "get_openrouter_api_key",
    "get_database_url",
]
