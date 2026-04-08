from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class AgentConfig(BaseModel):
    model: str
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    context_window_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None
    input_token_reserve: int = 2000
    max_turns: int = 10
    reasoning_effort: ReasoningEffort | None = None


class OrchestratorConfig(AgentConfig):
    context_bundle_name: str = "context_bundle.json"


class MarkdownResearcherConfig(AgentConfig):
    max_files: int = 200
    max_file_bytes: int = 5_000_000
    max_chunk_tokens: Optional[int] = None
    chunk_overlap_tokens: int = 2000
    batch_max_chunks: int = 32
    batch_max_input_tokens: Optional[int] = None
    batch_overhead_tokens: int = 600
    max_workers: int = 2
    request_backoff_base_seconds: float = 2.0
    request_backoff_max_seconds: float = 10.0
    strict_decision_audit: bool = False


class ChatConfig(AgentConfig):
    max_history_messages: int = 12
    max_context_total_tokens: int = 220_000
    min_prompt_token_cap: int = 20_000
    provider_timeout_seconds: float = 60.0
    prompt_token_buffer: int = 2_000
    multi_pass_threshold_tokens: int = 200_000
    multi_pass_chunk_tokens: int = 150_000
    followup_search_enabled: bool = False
    max_auto_followup_bundles: int = 3
    followup_router_max_history_messages: int = 6
    followup_router_max_excerpts_per_source: int = 50


class WriterConfig(AgentConfig):
    """Configuration for the writer agent."""

    max_coverage_attempts: int = 2


class AssumptionsReviewerConfig(AgentConfig):
    """Configuration for two-pass missing-data discovery."""


class BenchmarkFactJudgeConfig(BaseModel):
    """LLM-as-judge settings for gold recall fact-presence checks."""

    model: str
    temperature: float = 0.0
    max_output_tokens: int = 600
    reasoning_effort: ReasoningEffort | None = "high"

class RetryConfig(BaseModel):
    """Shared retry policy for LLM and retrieval operations.

    Defaults keep ``AppConfig.retry`` optional when the YAML omits a retry block.
    """

    max_attempts: int = 5
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 30.0


class AppConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    orchestrator: OrchestratorConfig
    markdown_researcher: MarkdownResearcherConfig
    writer: WriterConfig
    chat: ChatConfig = Field(
        default_factory=lambda: ChatConfig(model="openai/gpt-5.4-mini")
    )
    assumptions_reviewer: AssumptionsReviewerConfig = Field(
        default_factory=lambda: AssumptionsReviewerConfig(model="openai/gpt-5.4-mini")
    )
    benchmark_fact_judge: BenchmarkFactJudgeConfig = Field(
        default_factory=lambda: BenchmarkFactJudgeConfig(model="openai/gpt-5.4-mini")
    )
    retry: RetryConfig = Field(default_factory=RetryConfig)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    runs_dir: Path = Field(default_factory=lambda: Path("output"))
    markdown_dir: Path = Field(default_factory=lambda: Path("documents"))

    @field_validator("writer", mode="before")
    @classmethod
    def _coerce_writer_config(cls, value: object) -> object:
        """Accept generic agent configs and coerce them into WriterConfig."""
        if isinstance(value, AgentConfig):
            return value.model_dump()
        return value


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load config from YAML and apply supported environment overrides."""
    load_dotenv()
    path = config_path or Path("llm_config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)

    runs_dir = os.getenv("RUNS_DIR")
    markdown_dir = os.getenv("MARKDOWN_DIR")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")

    if runs_dir:
        config.runs_dir = Path(runs_dir)
    if markdown_dir:
        config.markdown_dir = Path(markdown_dir)
    if openrouter_base_url:
        config.openrouter_base_url = openrouter_base_url

    return config


def load_cached_config(
    config_path: Path | None = None,
    *,
    cache_owner: object | None = None,
    loader: Callable[[Path | None], AppConfig] | None = None,
) -> AppConfig:
    """Load config once per file mtime and return deep copies from an optional cache."""
    path = Path(config_path or "llm_config.yaml")
    load = loader or load_config
    if cache_owner is None:
        return load(path)

    current_mtime_ns = path.stat().st_mtime_ns
    cached_config = getattr(cache_owner, "_cached_app_config", None)
    cached_path = getattr(cache_owner, "_cached_app_config_path", None)
    cached_mtime_ns = getattr(cache_owner, "_cached_app_config_mtime_ns", None)
    if (
        isinstance(cached_config, AppConfig)
        and isinstance(cached_path, Path)
        and cached_path == path
        and cached_mtime_ns == current_mtime_ns
    ):
        return cached_config.model_copy(deep=True)

    config = load(path)
    current_mtime_ns = path.stat().st_mtime_ns
    setattr(cache_owner, "_cached_app_config", config)
    setattr(cache_owner, "_cached_app_config_path", path)
    setattr(cache_owner, "_cached_app_config_mtime_ns", current_mtime_ns)
    return config.model_copy(deep=True)


def resolve_openrouter_api_key(
    api_key_override: str | None = None,
    *,
    allow_missing: bool = False,
) -> str:
    """Resolve an OpenRouter API key from override or environment."""
    load_dotenv()
    if isinstance(api_key_override, str):
        cleaned_override = api_key_override.strip()
        if cleaned_override:
            return cleaned_override

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        if allow_missing:
            return "missing-openrouter-api-key"
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


def get_openrouter_api_key() -> str:
    """Return the configured OpenRouter API key and mirror it to OpenAI vars."""
    return resolve_openrouter_api_key()


__all__ = [
    "AgentConfig",
    "OrchestratorConfig",
    "MarkdownResearcherConfig",
    "ChatConfig",
    "WriterConfig",
    "AssumptionsReviewerConfig",
    "BenchmarkFactJudgeConfig",
    "RetryConfig",
    "AppConfig",
    "load_config",
    "load_cached_config",
    "resolve_openrouter_api_key",
    "get_openrouter_api_key",
]
