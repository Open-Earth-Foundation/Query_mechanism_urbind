import pytest

from backend.modules.vector_store.models import RetrievedChunk
from backend.modules.vector_store import retriever as retriever_module
from backend.modules.vector_store.retriever import (
    as_markdown_documents,
    build_retrieval_artifact,
    retrieve_chunks_for_queries,
)
from backend.modules.vector_store.manifest import save_manifest
from backend.utils.config import AppConfig
from tests.support import build_test_app_config


def test_as_markdown_documents_maps_required_fields() -> None:
    chunks = [
        RetrievedChunk(
            city_name="Munich",
            raw_text="## Initiative\nEvidence block",
            source_path="documents/Munich.md",
            heading_path="Mobility > Charging",
            block_type="table",
            distance=0.231234,
            chunk_id="munich-1",
            metadata={"city_key": "munich"},
        )
    ]

    documents = as_markdown_documents(chunks)

    assert documents == [
        {
            "path": "documents/Munich.md",
            "city_name": "Munich",
            "city_key": "munich",
            "content": "## Initiative\nEvidence block",
            "chunk_id": "munich-1",
            "distance": "0.231234",
            "heading_path": "Mobility > Charging",
            "block_type": "table",
            "chunk_index": None,
        }
    ]


def test_build_retrieval_artifact_includes_chunk_text() -> None:
    chunk = RetrievedChunk(
        city_name="Munich",
        raw_text="## Initiative\nEvidence block",
        source_path="documents/Munich.md",
        heading_path="Mobility > Charging",
        block_type="table",
        distance=0.231234,
        chunk_id="munich-1",
        chunk_index=3,
        metadata={"city_key": "munich"},
    )

    artifact = build_retrieval_artifact(
        queries=["original question"],
        selected_cities=["Munich"],
        final_chunks=[chunk],
        retrieval_meta={},
    )

    assert artifact["chunks"] == [
        {
            "chunk_id": "munich-1",
            "chunk_text": "## Initiative\nEvidence block",
            "city_name": "Munich",
            "city_key": "munich",
            "source_path": "documents/Munich.md",
            "heading_path": "Mobility > Charging",
            "block_type": "table",
            "distance": 0.231234,
            "chunk_index": 3,
            "provenance": {
                "origin": "seed",
                "selection_mode": "distance_qualified",
                "seed_rank": None,
                "seed_query_ids": [],
                "expanded_from_chunk_ids": [],
            },
        }
    ]


def _build_test_config() -> AppConfig:
    """Build the retriever test config with current required sections."""
    config = build_test_app_config()
    config.vector_store.retrieval_fallback_min_chunks_per_city_query = 2
    config.vector_store.retrieval_max_chunks_per_city_query = 3
    config.vector_store.retrieval_max_distance = 0.2
    config.vector_store.context_window_chunks = 1
    config.vector_store.table_context_window_chunks = 2
    return config


def test_retrieve_chunks_for_queries_applies_distance_floor_and_neighbor_expansion(
    monkeypatch,
    tmp_path,
) -> None:
    config = _build_test_config()
    config.vector_store.index_manifest_path = tmp_path / "index_manifest.json"
    save_manifest(
        config.vector_store.index_manifest_path,
        {"files": {"documents/Munich.md": {"file_hash": "h1", "chunk_ids": ["chunk-9", "chunk-10", "chunk-11", "chunk-12"]}}},
    )

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings
            assert where == {"city_key": "munich"}
            return {
                "ids": [["chunk-10", "chunk-11", "chunk-12"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "primary chunk 10",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 10,
                    },
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "primary chunk 11",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "table",
                        "chunk_index": 11,
                    },
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "too far chunk 12",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 12,
                    },
                ]],
                "distances": [[0.1, 0.15, 0.5]],
            }

        def get(self, where, limit):
            assert limit >= 1
            clauses = where.get("$and", [])
            requested_indices: list[int] = []
            for clause in clauses:
                if not isinstance(clause, dict) or "chunk_index" not in clause:
                    continue
                chunk_index_clause = clause["chunk_index"]
                if isinstance(chunk_index_clause, dict):
                    values = chunk_index_clause.get("$in", [])
                    if isinstance(values, list):
                        requested_indices = [
                            value for value in values if isinstance(value, int)
                        ]
                break
            ids: list[str] = []
            metadatas: list[dict[str, object]] = []
            if 9 in requested_indices:
                ids.append("chunk-9")
                metadatas.append(
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "neighbor chunk 9",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 9,
                    }
                )
            if 11 in requested_indices:
                ids.append("chunk-11")
                metadatas.append(
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "neighbor chunk 11",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "table",
                        "chunk_index": 11,
                    }
                )
            return {"ids": ids, "metadatas": metadatas}

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )
    monkeypatch.setattr(
        retriever_module,
        "ChromaStore",
        lambda persist_path, collection_name, distance_metric="l2": _FakeStore(),  # noqa: ARG005
    )

    chunks, meta = retrieve_chunks_for_queries(
        queries=["original question", "keyword query", "metrics query"],
        config=config,
        docs_dir=tmp_path / "documents",
        selected_cities=["Munich"],
    )

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert "chunk-10" in chunk_ids
    assert "chunk-11" in chunk_ids
    assert "chunk-9" in chunk_ids  # pulled in by neighbor expansion
    assert "chunk-12" not in chunk_ids  # filtered out by distance threshold
    assert meta["min_chunks_per_city"] == 2
    assert meta["seed_retrieved_total_chunks"] == 2
    assert meta["neighbor_expanded_total_chunks"] == 1

    seed_chunks = {chunk.chunk_id: chunk for chunk in meta["seed_chunks"]}
    assert seed_chunks["chunk-10"].provenance.origin == "seed"
    assert seed_chunks["chunk-10"].provenance.selection_mode == "distance_qualified"
    assert seed_chunks["chunk-10"].provenance.seed_rank == 1
    assert seed_chunks["chunk-11"].provenance.seed_rank == 2

    chunk_index = {chunk.chunk_id: chunk for chunk in chunks}
    assert chunk_index["chunk-9"].provenance.origin == "neighbor"
    assert chunk_index["chunk-9"].provenance.selection_mode == "neighbor_context"
    assert chunk_index["chunk-9"].provenance.expanded_from_chunk_ids == [
        "chunk-10",
        "chunk-11",
    ]


def test_retrieve_chunks_for_queries_falls_back_to_top_n_when_no_chunks_pass_distance(
    monkeypatch,
    tmp_path,
) -> None:
    config = _build_test_config()
    config.vector_store.index_manifest_path = tmp_path / "index_manifest.json"
    save_manifest(
        config.vector_store.index_manifest_path,
        {"files": {"documents/Munich.md": {"file_hash": "h1", "chunk_ids": ["chunk-1", "chunk-2"]}}},
    )
    config.vector_store.retrieval_max_distance = 0.01

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings
            assert n_results == 3
            assert where == {"city_key": "munich"}
            return {
                "ids": [["chunk-1", "chunk-2"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "fallback chunk 1",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 1,
                    },
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "fallback chunk 2",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 2,
                    },
                ]],
                "distances": [[0.5, 0.6]],
            }

        def get(self, where, limit):
            del where, limit
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )
    monkeypatch.setattr(
        retriever_module,
        "ChromaStore",
        lambda persist_path, collection_name, distance_metric="l2": _FakeStore(),  # noqa: ARG005
    )

    chunks, _meta = retrieve_chunks_for_queries(
        queries=["original question"],
        config=config,
        docs_dir=tmp_path / "documents",
        selected_cities=["Munich"],
    )

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-2"]


def test_retrieve_chunks_for_queries_tops_up_when_too_few_chunks_pass_distance(
    monkeypatch,
    tmp_path,
) -> None:
    config = _build_test_config()
    config.vector_store.index_manifest_path = tmp_path / "index_manifest.json"
    save_manifest(
        config.vector_store.index_manifest_path,
        {"files": {"documents/Munich.md": {"file_hash": "h1", "chunk_ids": ["chunk-10", "chunk-11", "chunk-12"]}}},
    )
    config.vector_store.retrieval_max_distance = 0.2

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings
            assert n_results == 3
            assert where == {"city_key": "munich"}
            return {
                "ids": [["chunk-10", "chunk-11", "chunk-12"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "pass chunk 10",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 10,
                    },
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "top-up chunk 11",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 11,
                    },
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "top-up chunk 12",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 12,
                    },
                ]],
                "distances": [[0.1, 0.5, 0.6]],
            }

        def get(self, where, limit):
            del where, limit
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )
    monkeypatch.setattr(
        retriever_module,
        "ChromaStore",
        lambda persist_path, collection_name, distance_metric="l2": _FakeStore(),  # noqa: ARG005
    )

    chunks, _meta = retrieve_chunks_for_queries(
        queries=["original question"],
        config=config,
        docs_dir=tmp_path / "documents",
        selected_cities=["Munich"],
    )

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-10", "chunk-11"]
    chunk_index = {chunk.chunk_id: chunk for chunk in chunks}
    assert chunk_index["chunk-10"].provenance.selection_mode == "distance_qualified"
    assert chunk_index["chunk-10"].provenance.seed_rank == 1
    assert chunk_index["chunk-11"].provenance.selection_mode == "fallback_top_up"
    assert chunk_index["chunk-11"].provenance.seed_rank == 2


def test_retrieve_chunks_for_queries_uses_manifest_cities_when_not_selected(
    monkeypatch,
    tmp_path,
) -> None:
    config = _build_test_config()
    config.vector_store.index_manifest_path = tmp_path / "index_manifest.json"
    save_manifest(
        config.vector_store.index_manifest_path,
        {
            "files": {
                "documents/Munich.md": {"file_hash": "h1", "chunk_ids": ["chunk-1"]},
            }
        },
    )

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings, n_results
            assert where == {"city_key": "munich"}
            return {
                "ids": [["chunk-1"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "manifest city chunk",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 1,
                    },
                ]],
                "distances": [[0.1]],
            }

        def get(self, where, limit):
            del where, limit
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )
    monkeypatch.setattr(
        retriever_module,
        "ChromaStore",
        lambda persist_path, collection_name, distance_metric="l2": _FakeStore(),  # noqa: ARG005
    )

    chunks, meta = retrieve_chunks_for_queries(
        queries=["original question"],
        config=config,
        docs_dir=tmp_path / "documents",
        selected_cities=None,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1"]
    assert meta["cities"] == ["munich"]


def test_retrieve_chunks_for_queries_dedupes_manifest_city_aliases(
    monkeypatch,
    tmp_path,
) -> None:
    """Manifest entries that differ only by separator style should map to one city."""
    config = _build_test_config()
    config.vector_store.index_manifest_path = tmp_path / "index_manifest.json"
    save_manifest(
        config.vector_store.index_manifest_path,
        {
            "files": {
                "documents/vitoria-gasteiz.md": {"file_hash": "h1", "chunk_ids": ["chunk-1"]},
                "documents/Vitoria_Gasteiz.md": {"file_hash": "h2", "chunk_ids": ["chunk-2"]},
            }
        },
    )

    query_calls: list[dict[str, object]] = []

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings, n_results
            query_calls.append(where)
            assert where == {"city_key": "vitoria_gasteiz"}
            return {
                "ids": [["chunk-1"]],
                "metadatas": [[
                    {
                        "city_name": "Vitoria_Gasteiz",
                        "city_key": "vitoria-gasteiz",
                        "raw_text": "alias chunk",
                        "source_path": "documents/Vitoria_Gasteiz.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 1,
                    },
                ]],
                "distances": [[0.1]],
            }

        def get(self, where, limit):
            del where, limit
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )
    monkeypatch.setattr(
        retriever_module,
        "ChromaStore",
        lambda persist_path, collection_name, distance_metric="l2": _FakeStore(),  # noqa: ARG005
    )

    chunks, meta = retrieve_chunks_for_queries(
        queries=["original question"],
        config=config,
        docs_dir=tmp_path / "documents",
        selected_cities=None,
    )

    assert len(query_calls) == 1
    assert meta["cities"] == ["vitoria_gasteiz"]
    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1"]
    assert as_markdown_documents(chunks)[0]["city_key"] == "vitoria_gasteiz"


def test_retrieve_chunks_for_queries_fails_fast_when_manifest_missing(
    monkeypatch,
    tmp_path,
) -> None:
    config = _build_test_config()
    config.vector_store.index_manifest_path = tmp_path / "missing_manifest.json"

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )

    with pytest.raises(FileNotFoundError):
        retrieve_chunks_for_queries(
            queries=["original question"],
            config=config,
            docs_dir=tmp_path / "documents",
            selected_cities=None,
        )


def test_retrieve_chunks_for_queries_fails_when_selected_city_not_indexed(
    monkeypatch,
    tmp_path,
) -> None:
    config = _build_test_config()
    config.vector_store.index_manifest_path = tmp_path / "index_manifest.json"
    save_manifest(
        config.vector_store.index_manifest_path,
        {
            "files": {
                "documents/Munich.md": {"file_hash": "h1", "chunk_ids": ["chunk-1"]},
            }
        },
    )

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )

    with pytest.raises(ValueError, match="not indexed"):
        retrieve_chunks_for_queries(
            queries=["original question"],
            config=config,
            docs_dir=tmp_path / "documents",
            selected_cities=["Berlin"],
        )


def test_retrieve_chunks_for_queries_retries_query_by_embedding_until_success(
    monkeypatch,
    tmp_path,
) -> None:
    config = _build_test_config()
    config.retry.max_attempts = 5
    config.vector_store.context_window_chunks = 0
    config.vector_store.table_context_window_chunks = 0
    config.vector_store.index_manifest_path = tmp_path / "index_manifest.json"
    save_manifest(
        config.vector_store.index_manifest_path,
        {"files": {"documents/Munich.md": {"file_hash": "h1", "chunk_ids": ["chunk-1"]}}},
    )

    call_counts = {"query": 0}

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings, n_results
            call_counts["query"] += 1
            assert where == {"city_key": "munich"}
            if call_counts["query"] < 3:
                raise RuntimeError("temporary query failure")
            return {
                "ids": [["chunk-1"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "retried query chunk",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 1,
                    }
                ]],
                "distances": [[0.1]],
            }

        def get(self, where, limit):
            del where, limit
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )
    monkeypatch.setattr(
        retriever_module,
        "ChromaStore",
        lambda persist_path, collection_name, distance_metric="l2": _FakeStore(),  # noqa: ARG005
    )

    chunks, _meta = retrieve_chunks_for_queries(
        queries=["original question"],
        config=config,
        docs_dir=tmp_path / "documents",
        selected_cities=["Munich"],
    )

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1"]
    assert call_counts["query"] == 3


def test_retrieve_chunks_for_queries_retries_neighbor_get_and_raises_after_max_attempts(
    monkeypatch,
    tmp_path,
) -> None:
    config = _build_test_config()
    config.retry.max_attempts = 5
    config.vector_store.context_window_chunks = 1
    config.vector_store.table_context_window_chunks = 1
    config.vector_store.index_manifest_path = tmp_path / "index_manifest.json"
    save_manifest(
        config.vector_store.index_manifest_path,
        {"files": {"documents/Munich.md": {"file_hash": "h1", "chunk_ids": ["chunk-1"]}}},
    )

    call_counts = {"get": 0}

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings, n_results
            assert where == {"city_key": "munich"}
            return {
                "ids": [["chunk-1"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "seed chunk",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 1,
                    }
                ]],
                "distances": [[0.1]],
            }

        def get(self, where, limit):
            del where, limit
            call_counts["get"] += 1
            raise RuntimeError("temporary get failure")

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )
    monkeypatch.setattr(
        retriever_module,
        "ChromaStore",
        lambda persist_path, collection_name, distance_metric="l2": _FakeStore(),  # noqa: ARG005
    )

    with pytest.raises(RuntimeError, match="temporary get failure"):
        retrieve_chunks_for_queries(
            queries=["original question"],
            config=config,
            docs_dir=tmp_path / "documents",
            selected_cities=["Munich"],
        )
    assert call_counts["get"] == config.retry.max_attempts


def test_retrieve_chunks_for_queries_merges_seed_query_ids_and_prefers_qualified_mode(
    monkeypatch,
    tmp_path,
) -> None:
    config = _build_test_config()
    config.vector_store.context_window_chunks = 0
    config.vector_store.table_context_window_chunks = 0
    config.vector_store.index_manifest_path = tmp_path / "index_manifest.json"
    save_manifest(
        config.vector_store.index_manifest_path,
        {"files": {"documents/Munich.md": {"file_hash": "h1", "chunk_ids": ["chunk-1", "chunk-2", "chunk-3"]}}},
    )

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings, n_results
            assert where == {"city_key": "munich"}
            if not hasattr(self, "call_count"):
                self.call_count = 0
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "ids": [["chunk-1", "chunk-2"]],
                    "metadatas": [[
                        {
                            "city_name": "Munich",
                            "city_key": "munich",
                            "raw_text": "fallback chunk 1",
                            "source_path": "documents/Munich.md",
                            "heading_path": "H1",
                            "block_type": "paragraph",
                            "chunk_index": 1,
                        },
                        {
                            "city_name": "Munich",
                            "city_key": "munich",
                            "raw_text": "fallback chunk 2",
                            "source_path": "documents/Munich.md",
                            "heading_path": "H1",
                            "block_type": "paragraph",
                            "chunk_index": 2,
                        },
                    ]],
                    "distances": [[0.5, 0.6]],
                }
            return {
                "ids": [["chunk-1", "chunk-3"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "qualified chunk 1",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 1,
                    },
                    {
                        "city_name": "Munich",
                        "city_key": "munich",
                        "raw_text": "qualified chunk 3",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 3,
                    },
                ]],
                "distances": [[0.05, 0.06]],
            }

        def get(self, where, limit):
            del where, limit
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(
        retriever_module,
        "_embed_queries",
        lambda queries, config: {query: [0.01, 0.02] for query in queries},  # noqa: ARG005
    )
    monkeypatch.setattr(
        retriever_module,
        "ChromaStore",
        lambda persist_path, collection_name, distance_metric="l2": _FakeStore(),  # noqa: ARG005
    )

    chunks, meta = retrieve_chunks_for_queries(
        queries=["original question", "metrics query"],
        config=config,
        docs_dir=tmp_path / "documents",
        selected_cities=["Munich"],
    )

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1", "chunk-3", "chunk-2"]
    seed_chunks = {chunk.chunk_id: chunk for chunk in meta["seed_chunks"]}
    assert seed_chunks["chunk-1"].provenance.seed_query_ids == ["q1", "q2"]
    assert seed_chunks["chunk-1"].provenance.selection_mode == "distance_qualified"
    assert seed_chunks["chunk-1"].provenance.seed_rank == 1
    assert seed_chunks["chunk-3"].provenance.seed_rank == 2
    assert seed_chunks["chunk-2"].provenance.selection_mode == "fallback_top_up"

