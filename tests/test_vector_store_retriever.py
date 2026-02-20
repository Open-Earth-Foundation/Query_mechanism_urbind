from backend.modules.vector_store.models import RetrievedChunk
from backend.modules.vector_store import retriever as retriever_module
from backend.modules.vector_store.retriever import (
    as_markdown_documents,
    retrieve_chunks_for_queries,
)
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
            "content": "## Initiative\nEvidence block",
            "chunk_id": "munich-1",
            "distance": "0.231234",
            "heading_path": "Mobility > Charging",
            "block_type": "table",
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

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings
            assert where == {"city_name": "Munich"}
            return {
                "ids": [["chunk-10", "chunk-11", "chunk-12"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "raw_text": "primary chunk 10",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 10,
                    },
                    {
                        "city_name": "Munich",
                        "raw_text": "primary chunk 11",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "table",
                        "chunk_index": 11,
                    },
                    {
                        "city_name": "Munich",
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
            del limit
            clauses = where.get("$and", [])
            chunk_index = None
            for clause in clauses:
                if isinstance(clause, dict) and "chunk_index" in clause:
                    chunk_index = clause["chunk_index"]
                    break
            if chunk_index == 9:
                return {
                    "ids": ["chunk-9"],
                    "metadatas": [
                        {
                            "city_name": "Munich",
                            "raw_text": "neighbor chunk 9",
                            "source_path": "documents/Munich.md",
                            "heading_path": "H1",
                            "block_type": "paragraph",
                            "chunk_index": 9,
                        }
                    ],
                }
            if chunk_index == 11:
                return {
                    "ids": ["chunk-11"],
                    "metadatas": [
                        {
                            "city_name": "Munich",
                            "raw_text": "neighbor chunk 11",
                            "source_path": "documents/Munich.md",
                            "heading_path": "H1",
                            "block_type": "table",
                            "chunk_index": 11,
                        }
                    ],
                }
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(
        retriever_module,
        "_embed_query_text",
        lambda query, config: [0.01, 0.02],  # noqa: ARG005
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
    config.vector_store.retrieval_max_distance = 0.01

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings
            assert n_results == 3
            assert where == {"city_name": "Munich"}
            return {
                "ids": [["chunk-1", "chunk-2"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "raw_text": "fallback chunk 1",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 1,
                    },
                    {
                        "city_name": "Munich",
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
        "_embed_query_text",
        lambda query, config: [0.01, 0.02],  # noqa: ARG005
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
    config.vector_store.retrieval_max_distance = 0.2

    class _FakeStore:
        def query_by_embedding(self, query_embeddings, n_results, where):
            del query_embeddings
            assert n_results == 3
            assert where == {"city_name": "Munich"}
            return {
                "ids": [["chunk-10", "chunk-11", "chunk-12"]],
                "metadatas": [[
                    {
                        "city_name": "Munich",
                        "raw_text": "pass chunk 10",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 10,
                    },
                    {
                        "city_name": "Munich",
                        "raw_text": "top-up chunk 11",
                        "source_path": "documents/Munich.md",
                        "heading_path": "H1",
                        "block_type": "paragraph",
                        "chunk_index": 11,
                    },
                    {
                        "city_name": "Munich",
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
        "_embed_query_text",
        lambda query, config: [0.01, 0.02],  # noqa: ARG005
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
