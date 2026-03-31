from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


BlockType = Literal["paragraph", "table", "list", "code"]
RetrievalOrigin = Literal["seed", "neighbor"]
RetrievalSelectionMode = Literal[
    "distance_qualified",
    "fallback_top_up",
    "neighbor_context",
]


class EmbeddingProvider(Protocol):
    """Interface for pluggable embedding providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a list of texts.

        Returns one entry per input text. An entry is ``None`` when the text
        permanently failed to embed after all retries (e.g. oversized input
        that the provider silently rejects). Callers must handle ``None``
        entries explicitly.
        """


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
class RetrievedChunkProvenance:
    """Explain how one chunk entered the retrieval output.

    We keep this explicit so retrieval artifacts can surface current vector DB
    behavior: whether a chunk was a direct distance-qualified hit, a fallback
    top-up, or a neighbor added only for local context.
    """

    origin: RetrievalOrigin = "seed"
    selection_mode: RetrievalSelectionMode = "distance_qualified"
    seed_rank: int | None = None
    seed_query_ids: list[str] = field(default_factory=list)
    expanded_from_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedChunk:
    city_name: str
    raw_text: str
    source_path: str
    heading_path: str
    block_type: str
    distance: float
    chunk_id: str
    chunk_index: int | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    provenance: RetrievedChunkProvenance = field(
        default_factory=RetrievedChunkProvenance
    )
