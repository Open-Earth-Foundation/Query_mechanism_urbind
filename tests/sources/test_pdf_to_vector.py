"""Tests for the pdf_to_vector ingestion handler and benchmark retriever.

Uses stub ``convert_fn`` and ``embedder`` so the tests don't depend on
pymupdf4llm or OpenAI embeddings.  The Chroma store itself runs locally
against a tmp persist path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.modules.sources.handlers import IngestionContext
from backend.modules.sources.handlers.pdf_to_vector import run_pdf_to_vector
from backend.modules.sources.manifest import (
    IngestionConfig,
    IngestionInputs,
    IngestionOutput,
    SourceConfig,
)
from backend.modules.vector_store.benchmark_retriever import retrieve_benchmark_excerpts
from backend.modules.vector_store.models import EmbeddingProvider


class _StubEmbedder:
    """Returns a deterministic 8-dim embedding based on text length + char codes."""

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        out: list[list[float] | None] = []
        for text in texts:
            base = [float(ord(c) % 8) for c in text[:8].ljust(8, "x")]
            length = float(len(text))
            out.append([length] + base[:7])
        return out


def _make_pdf_layout(tmp_path: Path) -> Path:
    root = tmp_path / "upstream"
    (root / "tier-2-national-datasets").mkdir(parents=True)
    (root / "tier-3-european-databases").mkdir(parents=True)
    (root / "tier-2-national-datasets" / "kba.pdf").write_bytes(b"%PDF-kba")
    (root / "tier-2-national-datasets" / "bnetza.xlsx").write_bytes(b"binary")
    (root / "tier-3-european-databases" / "icct.pdf").write_bytes(b"%PDF-icct")
    return root


def _make_context(
    *,
    upstream_root: Path,
    project_root: Path,
    persist_subdir: str = ".chroma/benchmarks/",
    excludes: list[str] | None = None,
) -> IngestionContext:
    source = SourceConfig(
        id="urbind_additional",
        name="URBIND Additional Documentation",
        provider="github",
        repo="Open-Earth-Foundation/urbind-additional-documentation",
        pinned_commit="abcdef",
    )
    ingestion = IngestionConfig(
        id="urbind_benchmarks",
        kind="vector_collection",
        inputs=IngestionInputs(
            paths=["tier-2-national-datasets/", "tier-3-european-databases/"],
            formats=["pdf"],
            excludes=excludes or ["tier-2-national-datasets/bnetza.xlsx"],
        ),
        output=IngestionOutput(
            chroma_persist_path=persist_subdir,
            collection="urbind_benchmarks_test",
        ),
        handler="ingest.pdf_to_vector",
        embedder="stub-model",
        consumed_by="vector_store.benchmark_retriever",
    )
    return IngestionContext(
        source=source,
        ingestion=ingestion,
        upstream_root=upstream_root,
        project_root=project_root,
        resolved_commit="abcdef",
    )


def _stub_convert_fn(b: bytes) -> str:
    """Produce small deterministic markdown that exercises blocks + tables."""
    tag = b.decode("utf-8", errors="ignore").split("-", 1)[-1]
    return (
        f"# Heading {tag}\n\n"
        f"This is the {tag} document body about charging infrastructure costs in Germany.\n\n"
        "| Year | Stations |\n|------|----------|\n| 2024 | 100 |\n| 2025 | 150 |\n"
    )


def test_pdf_to_vector_writes_separate_collection(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    upstream = _make_pdf_layout(tmp_path)
    ctx = _make_context(upstream_root=upstream, project_root=project_root)

    state = run_pdf_to_vector(
        ctx,
        convert_fn=_stub_convert_fn,
        embedder=_StubEmbedder(),
    )

    dump = state.model_dump()
    assert dump["file_count"] == 2  # pdf only; xlsx excluded
    assert dump["chunk_count"] >= 2
    assert dump["collection"] == "urbind_benchmarks_test"
    assert "benchmarks" in dump["chroma_persist_path"]

    # The Chroma persist path lives in the project_root, isolated from any
    # other Chroma collection.
    persist_path = (project_root / ".chroma" / "benchmarks").resolve()
    assert persist_path.exists()


def test_pdf_to_vector_metadata_carries_source_attribution(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    upstream = _make_pdf_layout(tmp_path)
    ctx = _make_context(upstream_root=upstream, project_root=project_root)

    run_pdf_to_vector(
        ctx,
        convert_fn=_stub_convert_fn,
        embedder=_StubEmbedder(),
    )

    persist_path = (project_root / ".chroma" / "benchmarks").resolve()

    def _stub_embed(queries: list[str]) -> dict[str, list[float]]:
        embedder = _StubEmbedder()
        embeddings = embedder.embed_texts(queries)
        return {q: e for q, e in zip(queries, embeddings, strict=True) if e is not None}

    excerpts = retrieve_benchmark_excerpts(
        ["charging infrastructure cost"],
        k=5,
        persist_path=persist_path,
        collection_name="urbind_benchmarks_test",
        embed_fn=_stub_embed,
    )
    assert excerpts, "expected at least one excerpt to be returned"
    for excerpt in excerpts:
        assert excerpt.source_id == "urbind_additional"
        assert excerpt.ingestion_id == "urbind_benchmarks"
        assert excerpt.tier in {"tier-2-national-datasets", "tier-3-european-databases"}
        assert excerpt.source_path.endswith(".pdf")
        assert excerpt.raw_text


def test_pdf_to_vector_requires_collection_and_persist_path(tmp_path: Path) -> None:
    upstream = _make_pdf_layout(tmp_path)
    ctx = _make_context(upstream_root=upstream, project_root=tmp_path / "project")
    ctx.ingestion.output.collection = None
    with pytest.raises(ValueError, match="output.collection"):
        run_pdf_to_vector(ctx, convert_fn=_stub_convert_fn, embedder=_StubEmbedder())


def test_pdf_to_vector_missing_persist_raises(tmp_path: Path) -> None:
    upstream = _make_pdf_layout(tmp_path)
    ctx = _make_context(upstream_root=upstream, project_root=tmp_path / "project")
    ctx.ingestion.output.chroma_persist_path = None
    with pytest.raises(ValueError, match="output.chroma_persist_path"):
        run_pdf_to_vector(ctx, convert_fn=_stub_convert_fn, embedder=_StubEmbedder())


def test_handler_is_registered() -> None:
    from backend.modules.sources.handlers import REGISTRY

    assert "ingest.pdf_to_vector" in REGISTRY


def test_benchmark_retriever_missing_persist_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        retrieve_benchmark_excerpts(
            ["x"],
            persist_path=tmp_path / "nope",
            embed_fn=lambda qs: {q: [0.0] * 8 for q in qs},
        )


def test_benchmark_retriever_empty_queries_returns_empty(tmp_path: Path) -> None:
    excerpts = retrieve_benchmark_excerpts(
        [""],
        persist_path=tmp_path,
        embed_fn=lambda qs: {q: [0.0] * 8 for q in qs},
    )
    assert excerpts == []
