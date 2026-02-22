import pytest

from backend.modules.vector_store.models import RetrievedChunk
from backend.modules.vector_store import retriever as retriever_module
from backend.modules.vector_store.retriever import (
    as_markdown_documents,
    retrieve_chunks_for_queries,
)
from backend.modules.vector_store.manifest import save_manifest
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
)


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


def _build_test_config() -> AppConfig:
    config = AppConfig(
        orchestrator=OrchestratorConfig(model="test", context_bundle_name="context_bundle.json"),
        sql_researcher=SqlResearcherConfig(model="test"),
        markdown_researcher=MarkdownResearcherConfig(model="test"),
        writer=AgentConfig(model="test"),
    )
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
        lambda persist_path, collection_name: _FakeStore(),  # noqa: ARG005
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
        lambda persist_path, collection_name: _FakeStore(),  # noqa: ARG005
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
        lambda persist_path, collection_name: _FakeStore(),  # noqa: ARG005
    )

    chunks, _meta = retrieve_chunks_for_queries(
        queries=["original question"],
        config=config,
        docs_dir=tmp_path / "documents",
        selected_cities=["Munich"],
    )

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-10", "chunk-11"]


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
        lambda persist_path, collection_name: _FakeStore(),  # noqa: ARG005
    )

    chunks, meta = retrieve_chunks_for_queries(
        queries=["original question"],
        config=config,
        docs_dir=tmp_path / "documents",
        selected_cities=None,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-1"]
    assert meta["cities"] == ["munich"]


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
