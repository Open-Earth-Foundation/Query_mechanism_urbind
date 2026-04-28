"""Handler: convert PDFs to markdown, block-chunk, embed, write to a Chroma collection.

Output target is set via ``ingestion.output``:
- ``chroma_persist_path``: path to a Chroma persistent directory (e.g. ``.chroma/benchmarks/``)
- ``collection``: collection name within that directory

The handler reuses the chunking + embedding stack from
``backend.modules.vector_store`` but writes to a *separate* persist
path + collection from the existing CCC markdown index, so there is
no coupling.

Each chunk's metadata carries ``source_id`` (manifest source id) and
``ingestion_id`` so the runtime can attribute retrieved chunks back to
the manifest.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.modules.sources.handlers import IngestionContext, register
from backend.modules.sources.handlers.pdf_to_markdown import (
    _convert_pdf,
    _hash_bytes,
    _iter_pdfs,
    _slugify,
)
from backend.modules.sources.state import IngestionState
from backend.modules.vector_store.chroma_store import ChromaStore
from backend.modules.vector_store.chunk_packer import pack_blocks
from backend.modules.vector_store.manifest import build_chunk_id, compute_content_hash
from backend.modules.vector_store.markdown_blocks import parse_markdown_blocks
from backend.modules.vector_store.models import EmbeddingProvider, IndexedChunk

logger = logging.getLogger(__name__)


_HANDLER_NAME = "ingest.pdf_to_vector"
_DEFAULT_CHUNK_TOKENS = 800
_DEFAULT_CHUNK_OVERLAP = 80
_DEFAULT_TABLE_ROW_GROUP_MAX_ROWS = 25
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def _build_chunks_for_pdf(
    *,
    upstream_path: Path,
    pdf_bytes: bytes,
    convert_fn: Callable[[bytes], str],
    upstream_root: Path,
    source_id: str,
    ingestion_id: str,
    chunk_tokens: int,
    chunk_overlap: int,
    table_row_group_max_rows: int,
) -> list[IndexedChunk]:
    """Convert one PDF to markdown and produce IndexedChunk objects."""
    markdown = convert_fn(pdf_bytes)
    blocks = parse_markdown_blocks(markdown)
    packed = pack_blocks(
        blocks=blocks,
        max_tokens=chunk_tokens,
        overlap_tokens=chunk_overlap,
        table_row_group_max_rows=table_row_group_max_rows,
    )

    rel_path = upstream_path.relative_to(upstream_root).as_posix()
    tier = rel_path.split("/", 1)[0] if "/" in rel_path else ""
    file_hash = _hash_bytes(pdf_bytes)
    file_stem = _slugify(upstream_path.stem)

    chunks: list[IndexedChunk] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for chunk in packed:
        content_hash = compute_content_hash(chunk.raw_text)
        chunk_id = build_chunk_id(
            source_path=rel_path,
            chunk_index=chunk.chunk_index,
            content_hash=content_hash,
        )
        metadata: dict[str, Any] = {
            "source_id": source_id,
            "ingestion_id": ingestion_id,
            "source_path": rel_path,
            "doc_slug": file_stem,
            "tier": tier,
            "block_type": chunk.block_type,
            "heading_path": chunk.heading_path,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "content_hash": content_hash,
            "file_hash": file_hash,
            "raw_text": chunk.raw_text,
            "created_at": timestamp,
            "updated_at": timestamp,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "table_id": chunk.table_id,
            "row_group_index": chunk.row_group_index,
            "table_title": chunk.table_title,
            "chunk_id": chunk_id,
        }
        chunks.append(
            IndexedChunk(
                chunk_id=chunk_id,
                document=chunk.embedding_text,
                metadata=metadata,
            )
        )
    return chunks


def _attach_embeddings(
    chunks: list[IndexedChunk],
    provider: EmbeddingProvider,
) -> list[IndexedChunk]:
    """Embed chunk documents and return new IndexedChunks with embeddings attached."""
    if not chunks:
        return []
    texts = [chunk.document for chunk in chunks]
    embeddings = provider.embed_texts(texts)
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"pdf_to_vector: embedding response size mismatch "
            f"chunks={len(chunks)} embeddings={len(embeddings)}"
        )
    out: list[IndexedChunk] = []
    failures = 0
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        if embedding is None:
            failures += 1
            continue
        out.append(
            IndexedChunk(
                chunk_id=chunk.chunk_id,
                document=chunk.document,
                metadata=chunk.metadata,
                embedding=embedding,
            )
        )
    if failures:
        raise RuntimeError(
            f"pdf_to_vector: {failures} chunk(s) failed to embed; aborting to avoid partial index."
        )
    return out


def _make_default_provider(model: str) -> EmbeddingProvider:
    """Construct the default embedding provider lazily."""
    from backend.modules.vector_store.indexer import OpenAIEmbeddingProvider

    return OpenAIEmbeddingProvider(model=model)


def run_pdf_to_vector(
    context: IngestionContext,
    *,
    convert_fn: Callable[[bytes], str] | None = None,
    embedder: EmbeddingProvider | None = None,
) -> IngestionState:
    """Convert upstream PDFs into chunks and write them to a Chroma collection."""
    output = context.ingestion.output
    inputs = context.ingestion.inputs

    persist_path_raw = output.chroma_persist_path
    if not persist_path_raw:
        raise ValueError(
            f"ingestion {context.ingestion.id!r}: pdf_to_vector requires output.chroma_persist_path"
        )
    if not output.collection:
        raise ValueError(
            f"ingestion {context.ingestion.id!r}: pdf_to_vector requires output.collection"
        )

    persist_path = (context.project_root / persist_path_raw).resolve()
    persist_path.mkdir(parents=True, exist_ok=True)

    convert = convert_fn or _convert_pdf
    embedding_model = context.ingestion.embedder or _DEFAULT_EMBEDDING_MODEL
    provider = embedder or _make_default_provider(embedding_model)

    upstream_root = context.upstream_root
    input_roots = [upstream_root / p for p in inputs.paths]
    exclude_paths = [upstream_root / p for p in inputs.excludes]
    pdf_files = _iter_pdfs(input_roots, exclude_paths)
    logger.info("pdf_to_vector: found %d PDFs", len(pdf_files))

    all_chunks: list[IndexedChunk] = []
    file_records: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        pdf_bytes = pdf_path.read_bytes()
        chunks = _build_chunks_for_pdf(
            upstream_path=pdf_path,
            pdf_bytes=pdf_bytes,
            convert_fn=convert,
            upstream_root=upstream_root,
            source_id=context.source.id,
            ingestion_id=context.ingestion.id,
            chunk_tokens=_DEFAULT_CHUNK_TOKENS,
            chunk_overlap=_DEFAULT_CHUNK_OVERLAP,
            table_row_group_max_rows=_DEFAULT_TABLE_ROW_GROUP_MAX_ROWS,
        )
        all_chunks.extend(chunks)
        rel_input = pdf_path.relative_to(upstream_root).as_posix()
        file_records.append(
            {
                "input": rel_input,
                "input_sha": _hash_bytes(pdf_bytes),
                "chunk_count": len(chunks),
            }
        )
        logger.info("pdf_to_vector: %s -> %d chunks", rel_input, len(chunks))

    embedded = _attach_embeddings(all_chunks, provider)
    logger.info("pdf_to_vector: embedded %d chunks", len(embedded))

    store = ChromaStore(persist_path=persist_path, collection_name=output.collection)
    store.reset_collection()
    if embedded:
        store.upsert(embedded)
    logger.info(
        "pdf_to_vector: wrote collection=%s persist_path=%s",
        output.collection,
        persist_path,
    )

    return IngestionState(
        ingestion_id=context.ingestion.id,
        source_id=context.source.id,
        last_ingested_at=datetime.now(timezone.utc).isoformat(),
        source_commit=context.resolved_commit,
        embedder=embedding_model,
        collection=output.collection,
        chroma_persist_path=str(persist_path.relative_to(context.project_root))
        if persist_path.is_relative_to(context.project_root)
        else str(persist_path),
        file_count=len(file_records),
        chunk_count=len(embedded),
        files=file_records,
    )


@register(_HANDLER_NAME)
def _entrypoint(context: IngestionContext) -> IngestionState:
    return run_pdf_to_vector(context)


__all__ = ["run_pdf_to_vector"]
