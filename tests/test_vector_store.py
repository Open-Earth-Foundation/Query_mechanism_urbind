from __future__ import annotations

from pathlib import Path

import pytest

from backend.modules.vector_store import chroma_store as chroma_store_module
from backend.modules.vector_store.chroma_store import ChromaStore
from backend.modules.vector_store.indexer import update_markdown_index
from backend.modules.vector_store.manifest import load_manifest
from backend.modules.vector_store.manifest import build_chunk_id
from backend.modules.vector_store.markdown_blocks import parse_markdown_blocks
from backend.modules.vector_store.chunk_packer import pack_blocks
from backend.modules.vector_store.table_utils import split_markdown_table_by_row_groups
from backend.utils.config import (
    AgentConfig,
    AppConfig,
    MarkdownResearcherConfig,
    OrchestratorConfig,
    SqlResearcherConfig,
    VectorStoreConfig,
)


def _build_config(tmp_path: Path) -> AppConfig:
    """Build minimal config object for vector store tests."""
    return AppConfig(
        orchestrator=OrchestratorConfig(model="test"),
        sql_researcher=SqlResearcherConfig(model="test"),
        markdown_researcher=MarkdownResearcherConfig(model="test"),
        writer=AgentConfig(model="test"),
        vector_store=VectorStoreConfig(
            enabled=True,
            chroma_persist_path=tmp_path / ".chroma",
            chroma_collection_name="test_chunks",
            embedding_model="test-embedding",
            embedding_chunk_tokens=80,
            embedding_chunk_overlap_tokens=8,
            table_row_group_max_rows=2,
            index_manifest_path=tmp_path / ".chroma" / "index_manifest.json",
        ),
    )


def test_parse_markdown_blocks_tracks_heading_path() -> None:
    """Parser preserves heading stacks in child blocks."""
    text = "\n".join(
        [
            "# City",
            "",
            "Overview paragraph.",
            "",
            "## Finance",
            "",
            "Budget paragraph.",
        ]
    )
    blocks = parse_markdown_blocks(text)
    assert blocks
    finance_blocks = [block for block in blocks if "Budget paragraph." in block.text]
    assert finance_blocks
    assert finance_blocks[0].heading_path == ["City", "Finance"]


def test_parse_markdown_blocks_detects_ccc_style_table() -> None:
    """Parser detects markdown tables with separator rows."""
    text = "\n".join(
        [
            "# City",
            "",
            "| Indicator | 2023 | 2024 |",
            "| --- | ---: | ---: |",
            "| Emissions | 120 | 110 |",
            "| Budget | 20 | 23 |",
        ]
    )
    blocks = parse_markdown_blocks(text)
    table_blocks = [block for block in blocks if block.block_type == "table"]
    assert len(table_blocks) == 1
    assert "Indicator" in table_blocks[0].text


def test_parse_markdown_blocks_merges_table_caption_line() -> None:
    """Single-line 'Table ...' caption is merged into following table block."""
    text = "\n".join(
        [
            "# City",
            "",
            "Table AP- 6 Current laws, directives and strategies at federal level",
            "",
            "| Type | Name |",
            "| --- | --- |",
            "| Law | Energy Act |",
        ]
    )
    blocks = parse_markdown_blocks(text)
    table_blocks = [block for block in blocks if block.block_type == "table"]
    assert len(table_blocks) == 1
    assert table_blocks[0].text.startswith(
        "Table AP- 6 Current laws, directives and strategies at federal level"
    )
    assert table_blocks[0].table_title == (
        "Table AP- 6 Current laws, directives and strategies at federal level"
    )
    assert table_blocks[0].start_line == 3
    caption_paragraphs = [
        block
        for block in blocks
        if block.block_type == "paragraph"
        and "Table AP- 6 Current laws" in block.text
    ]
    assert not caption_paragraphs


def test_table_row_group_split_repeats_headers() -> None:
    """Table row-group splitting repeats header and separator rows."""
    table = "\n".join(
        [
            "| A | B |",
            "| --- | --- |",
            "| r1 | x |",
            "| r2 | y |",
            "| r3 | z |",
        ]
    )
    groups = split_markdown_table_by_row_groups(table, max_rows_per_group=1)
    assert len(groups) == 3
    for group in groups:
        lines = group.splitlines()
        assert lines[0] == "| A | B |"
        assert lines[1] == "| --- | --- |"


def test_build_chunk_id_is_deterministic() -> None:
    """Chunk id generation is stable for same inputs."""
    chunk_id_1 = build_chunk_id("documents/Munich.md", 2, "abc123")
    chunk_id_2 = build_chunk_id("documents/Munich.md", 2, "abc123")
    chunk_id_3 = build_chunk_id("documents/Munich.md", 3, "abc123")
    assert chunk_id_1 == chunk_id_2
    assert chunk_id_1 != chunk_id_3


def test_manifest_update_skips_unchanged_and_updates_changed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Incremental updater skips unchanged files and processes changed files."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True)
    file_path = docs_dir / "Munich.md"
    file_path.write_text("# Munich\n\nInitial content.", encoding="utf-8")
    config = _build_config(tmp_path)

    class FakeStore:
        upsert_calls: list[int] = []
        delete_calls: list[int] = []

        def __init__(self, persist_path: Path, collection_name: str) -> None:
            self.persist_path = persist_path
            self.collection_name = collection_name

        def delete(self, ids: list[str]) -> None:
            self.delete_calls.append(len(ids))

        def upsert(self, chunks) -> None:
            self.upsert_calls.append(len(chunks))

    class FakeEmbeddingProvider:
        def __init__(self, model: str, base_url: str | None = None) -> None:
            self.model = model
            self.base_url = base_url

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("backend.modules.vector_store.indexer.ChromaStore", FakeStore)
    monkeypatch.setattr(
        "backend.modules.vector_store.indexer.OpenAIEmbeddingProvider",
        FakeEmbeddingProvider,
    )

    first_stats = update_markdown_index(config=config, docs_dir=docs_dir, dry_run=False)
    assert first_stats.files_changed == 1
    assert FakeStore.upsert_calls

    second_stats = update_markdown_index(config=config, docs_dir=docs_dir, dry_run=False)
    assert second_stats.files_changed == 0
    assert second_stats.files_unchanged == 1

    file_path.write_text("# Munich\n\nUpdated content.", encoding="utf-8")
    third_stats = update_markdown_index(config=config, docs_dir=docs_dir, dry_run=False)
    assert third_stats.files_changed == 1
    assert third_stats.chunks_created > 0


def test_pack_blocks_keeps_tables_as_table_chunks() -> None:
    """Table blocks remain standalone table chunks when near paragraphs."""
    text = "\n".join(
        [
            "# Section",
            "",
            "Intro paragraph.",
            "",
            "Table AP- 1 Demo table",
            "",
            "| A | B |",
            "| --- | --- |",
            "| r1 | x |",
            "| r2 | y |",
            "",
            "After table paragraph.",
        ]
    )
    blocks = parse_markdown_blocks(text)
    chunks = pack_blocks(blocks=blocks, max_tokens=200, overlap_tokens=20)
    table_chunks = [chunk for chunk in chunks if chunk.block_type == "table"]
    assert table_chunks
    assert all("| --- | --- |" in chunk.raw_text for chunk in table_chunks)
    assert all(chunk.table_title is not None for chunk in table_chunks)


def test_reset_collection_ignores_collection_not_found_error(monkeypatch) -> None:
    """Reset ignores missing-collection deletion errors and still recreates it."""

    class FakeCollectionNotFoundError(Exception):
        pass

    class FakeClient:
        def __init__(self) -> None:
            self.recreated = False

        def delete_collection(self, name: str) -> None:
            del name
            raise FakeCollectionNotFoundError("Collection not found")

        def get_or_create_collection(self, name: str) -> dict[str, str]:
            del name
            self.recreated = True
            return {}

    monkeypatch.setattr(
        chroma_store_module,
        "COLLECTION_NOT_FOUND_ERROR_TYPES",
        (FakeCollectionNotFoundError,),
    )
    store = ChromaStore.__new__(ChromaStore)
    store._client = FakeClient()
    store._collection_name = "test"

    store.reset_collection()

    assert store._client.recreated is True


def test_reset_collection_reraises_unexpected_delete_errors(monkeypatch) -> None:
    """Reset propagates delete failures that are not missing-collection errors."""

    class FakeClient:
        def delete_collection(self, name: str) -> None:
            del name
            raise RuntimeError("Permission denied")

        def get_or_create_collection(self, name: str) -> dict[str, str]:
            del name
            return {}

    monkeypatch.setattr(chroma_store_module, "COLLECTION_NOT_FOUND_ERROR_TYPES", ())
    store = ChromaStore.__new__(ChromaStore)
    store._client = FakeClient()
    store._collection_name = "test"

    with pytest.raises(RuntimeError, match="Permission denied"):
        store.reset_collection()


def test_update_markdown_index_filtered_city_does_not_delete_other_manifest_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtered update only deletes removed files within the selected city scope."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True)
    munich_path = docs_dir / "Munich.md"
    berlin_path = docs_dir / "Berlin.md"
    munich_path.write_text("# Munich\n\nCurrent content.", encoding="utf-8")
    config = _build_config(tmp_path)

    # Seed manifest with one selected-city entry and one unrelated-city entry.
    config.vector_store.index_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.vector_store.index_manifest_path.write_text(
        """
{
  "files": {
    "%s": {"file_hash": "old-munich", "chunk_ids": ["old-m-1"]},
    "%s": {"file_hash": "old-berlin", "chunk_ids": ["old-b-1"]}
  }
}
"""
        % (munich_path.as_posix(), berlin_path.as_posix()),
        encoding="utf-8",
    )

    class FakeStore:
        delete_calls: list[list[str]] = []
        upsert_calls: list[int] = []

        def __init__(self, persist_path: Path, collection_name: str) -> None:
            self.persist_path = persist_path
            self.collection_name = collection_name

        def delete(self, ids: list[str]) -> None:
            self.delete_calls.append(ids)

        def upsert(self, chunks) -> None:
            self.upsert_calls.append(len(chunks))

    class FakeEmbeddingProvider:
        def __init__(self, model: str, base_url: str | None = None) -> None:
            self.model = model
            self.base_url = base_url

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("backend.modules.vector_store.indexer.ChromaStore", FakeStore)
    monkeypatch.setattr(
        "backend.modules.vector_store.indexer.OpenAIEmbeddingProvider",
        FakeEmbeddingProvider,
    )

    stats = update_markdown_index(
        config=config,
        docs_dir=docs_dir,
        selected_cities=["Munich"],
        dry_run=False,
    )

    assert stats.files_deleted == 0
    assert ["old-b-1"] not in FakeStore.delete_calls
    manifest = load_manifest(config.vector_store.index_manifest_path)
    files = manifest.get("files", {})
    assert berlin_path.as_posix() in files
