from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


BlockType = Literal["paragraph", "table", "list", "code"]


class EmbeddingProvider(Protocol):
    """Interface for pluggable embedding providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""


@dataclass(frozen=True)
class MdBlock:
    block_type: BlockType
    text: str
    heading_path: list[str]
    start_line: int | None = None
    end_line: int | None = None
    table_id: str | None = None
    row_group_index: int | None = None
    table_title: str | None = None


@dataclass(frozen=True)
class PackedChunk:
    raw_text: str
    embedding_text: str
    block_type: BlockType
    heading_path: str
    token_count: int
    chunk_index: int
    start_line: int | None = None
    end_line: int | None = None
    table_id: str | None = None
    row_group_index: int | None = None
    table_title: str | None = None


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    document: str
    metadata: dict[str, str | int | float | bool | None]
    embedding: list[float] | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    city_name: str
    raw_text: str
    source_path: str
    heading_path: str
    block_type: str
    distance: float
    chunk_id: str
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)
