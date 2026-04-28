"""Pydantic models and loader for the data sources manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SourceKind = Literal[
    "markdown_corpus",
    "structured_lookup",
    "vector_collection",
    "web_allowlist",
]
SourceProvider = Literal["github", "local"]

DEFAULT_MANIFEST_PATH = Path("backend/data/sources_manifest.yaml")


class IngestionInputs(BaseModel):
    paths: list[str]
    formats: list[str]
    excludes: list[str] = Field(default_factory=list)


class IngestionOutput(BaseModel):
    """Kind-specific output spec.

    Different ingestion kinds populate different fields here.  We keep one
    flat shape and allow extras so handlers can read what they need without
    forcing every kind to declare every field.
    """

    model_config = ConfigDict(extra="allow")

    path: str | None = None
    naming: str | None = None
    chroma_persist_path: str | None = None
    collection: str | None = None


class CoverageConfig(BaseModel):
    """What an ingestion covers — used by runtime modules to decide
    whether the source is relevant to a given query.

    Each kind picks the fields that make sense for it.  Extras are allowed
    so we can extend coverage shapes without a schema migration.
    """

    model_config = ConfigDict(extra="allow")

    cities: list[str] | None = None
    country: str | None = None
    countries: list[str] | None = None
    fields: list[str] | None = None
    scope: list[str] | None = None


class IngestionConfig(BaseModel):
    id: str
    kind: SourceKind
    inputs: IngestionInputs
    output: IngestionOutput
    handler: str | None = None
    converter: str | None = None
    embedder: str | None = None
    consumed_by: str | None = None
    coverage: CoverageConfig | None = None


class SourceConfig(BaseModel):
    id: str
    name: str
    provider: SourceProvider
    repo: str | None = None
    pinned_commit: str | None = None
    license: str | None = None
    ingestions: list[IngestionConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_provider_fields(self) -> SourceConfig:
        if self.provider == "github" and not self.repo:
            raise ValueError(f"source {self.id!r}: provider=github requires `repo`")
        return self


class Manifest(BaseModel):
    version: int = 1
    sources: list[SourceConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Manifest:
        source_ids: set[str] = set()
        ingestion_ids: set[str] = set()
        for source in self.sources:
            if source.id in source_ids:
                raise ValueError(f"duplicate source id: {source.id!r}")
            source_ids.add(source.id)
            for ingestion in source.ingestions:
                if ingestion.id in ingestion_ids:
                    raise ValueError(f"duplicate ingestion id: {ingestion.id!r}")
                ingestion_ids.add(ingestion.id)
        return self

    def get_source(self, source_id: str) -> SourceConfig | None:
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def get_ingestion(
        self, ingestion_id: str
    ) -> tuple[SourceConfig, IngestionConfig] | None:
        for source in self.sources:
            for ingestion in source.ingestions:
                if ingestion.id == ingestion_id:
                    return source, ingestion
        return None

    def iter_ingestions(self) -> list[tuple[SourceConfig, IngestionConfig]]:
        return [
            (source, ingestion)
            for source in self.sources
            for ingestion in source.ingestions
        ]


def load_manifest(path: Path | None = None) -> Manifest:
    """Load and validate the data sources manifest from yaml."""
    manifest_path = path or DEFAULT_MANIFEST_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return Manifest.model_validate(raw)


__all__ = [
    "CoverageConfig",
    "DEFAULT_MANIFEST_PATH",
    "IngestionConfig",
    "IngestionInputs",
    "IngestionOutput",
    "Manifest",
    "SourceConfig",
    "SourceKind",
    "SourceProvider",
    "load_manifest",
]
