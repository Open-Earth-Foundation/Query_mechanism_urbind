from pathlib import Path

from backend.modules.markdown_researcher.services import (
    build_city_batches,
    load_markdown_documents,
    split_documents_by_city,
)
from backend.utils.config import MarkdownResearcherConfig


def test_load_markdown_documents_filters_selected_cities(tmp_path: Path) -> None:
    (tmp_path / "Munich.md").write_text("# Munich\n\nText", encoding="utf-8")
    (tmp_path / "Leipzig.md").write_text("# Leipzig\n\nText", encoding="utf-8")
    config = MarkdownResearcherConfig(model="test")

    docs = load_markdown_documents(
        tmp_path,
        config,
        selected_cities=["Munich"],
    )

    assert docs
    assert all(doc["city_name"] == "Munich" for doc in docs)
    assert all(doc["city_key"] == "munich" for doc in docs)


def test_load_markdown_documents_city_filter_is_case_insensitive(
    tmp_path: Path,
) -> None:
    (tmp_path / "Munich.md").write_text("# Munich\n\nText", encoding="utf-8")
    config = MarkdownResearcherConfig(model="test")

    docs = load_markdown_documents(
        tmp_path,
        config,
        selected_cities=["munich"],
    )

    assert docs
    assert all(doc["city_name"] == "Munich" for doc in docs)
    assert all(doc["city_key"] == "munich" for doc in docs)


def test_load_markdown_documents_adds_stable_chunk_ids(tmp_path: Path) -> None:
    (tmp_path / "Munich.md").write_text("# Munich\n\nAlpha\n\nBeta", encoding="utf-8")
    config = MarkdownResearcherConfig(model="test")

    first_docs = load_markdown_documents(tmp_path, config, selected_cities=["Munich"])
    second_docs = load_markdown_documents(tmp_path, config, selected_cities=["Munich"])

    assert first_docs
    assert all(doc.get("chunk_id") for doc in first_docs)
    assert [doc["chunk_id"] for doc in first_docs] == [
        doc["chunk_id"] for doc in second_docs
    ]


def test_build_city_batches_respects_city_and_limits() -> None:
    documents = [
        {"city_name": "Leipzig", "city_key": "leipzig", "chunk_id": "l1", "content": "aa"},
        {"city_name": "Leipzig", "city_key": "leipzig", "chunk_id": "l2", "content": "bbb"},
        {"city_name": "Leipzig", "city_key": "leipzig", "chunk_id": "l3", "content": "cccc"},
        {"city_name": "Munich", "city_key": "munich", "chunk_id": "m1", "content": "aa"},
        {"city_name": "Munich", "city_key": "munich", "chunk_id": "m2", "content": "bbb"},
    ]
    by_city = split_documents_by_city(documents)

    batches = build_city_batches(
        documents_by_city=by_city,
        max_batch_input_tokens=5,
        max_batch_chunks=2,
        token_counter=len,
    )

    for city_key, _batch_index, batch in batches:
        assert all(str(doc.get("city_key", "")).strip().lower() == city_key for doc in batch)
        assert len(batch) <= 2
        assert sum(len(str(doc["content"])) for doc in batch) <= 5

    seen_ids = [str(doc["chunk_id"]) for _city, _idx, batch in batches for doc in batch]
    assert seen_ids == ["l1", "l2", "l3", "m1", "m2"]


def test_build_city_batches_keeps_oversized_chunk_singleton() -> None:
    documents = [
        {"city_name": "Leipzig", "city_key": "leipzig", "chunk_id": "l1", "content": "small"},
        {"city_name": "Leipzig", "city_key": "leipzig", "chunk_id": "l2", "content": "oversized"},
        {"city_name": "Leipzig", "city_key": "leipzig", "chunk_id": "l3", "content": "tiny"},
    ]
    by_city = split_documents_by_city(documents)

    batches = build_city_batches(
        documents_by_city=by_city,
        max_batch_input_tokens=6,
        max_batch_chunks=4,
        token_counter=len,
    )

    batch_ids = [[str(doc["chunk_id"]) for doc in batch] for _city, _idx, batch in batches]
    assert batch_ids == [["l1"], ["l2"], ["l3"]]
