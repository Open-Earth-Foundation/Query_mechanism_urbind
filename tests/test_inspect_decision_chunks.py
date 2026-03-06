from pathlib import Path

from backend.scripts.inspect_decision_chunks import (
    _collect_rows,
    _default_output_path,
    _decision_ids,
    _normalize_retrieval_index,
    _passes_city_filter,
)


def test_decision_ids_dedupes_and_skips_empty() -> None:
    payload = {
        "accepted_chunk_ids": ["chunk-1", "chunk-1", "", " chunk-2 "],
    }

    result = _decision_ids(payload, "accepted")

    assert result == ["chunk-1", "chunk-2"]


def test_normalize_retrieval_index_maps_chunk_ids() -> None:
    payload = {
        "chunks": [
            {"chunk_id": "chunk-1", "city_key": "aachen"},
            {"chunk_id": "chunk-2", "city_key": "mannheim"},
        ]
    }

    index = _normalize_retrieval_index(payload)

    assert set(index.keys()) == {"chunk-1", "chunk-2"}
    assert index["chunk-1"]["city_key"] == "aachen"


def test_passes_city_filter_supports_city_key_and_name() -> None:
    item = {"city_key": "aachen", "city_name": "Aachen"}

    assert _passes_city_filter(item, "AACHEN") is True
    assert _passes_city_filter(item, "mannheim") is False


def test_default_output_path_targets_markdown_report_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260306_1034"

    output_path = _default_output_path(run_dir)

    assert output_path == run_dir / "markdown" / "decision_chunks_report.md"


def test_collect_rows_without_limit_returns_all_rows() -> None:
    class _FakeStore:
        def get(self, ids: list[str], limit: int) -> dict[str, object]:
            return {"metadatas": [{"raw_text": "content"}], "documents": ["content"]}

    retrieval_index = {
        "chunk-1": {
            "city_name": "Aachen",
            "city_key": "aachen",
            "distance": 0.1,
            "block_type": "paragraph",
            "source_path": "documents/Aachen.md",
            "heading_path": "Heading A",
        },
        "chunk-2": {
            "city_name": "Aachen",
            "city_key": "aachen",
            "distance": 0.2,
            "block_type": "table",
            "source_path": "documents/Aachen.md",
            "heading_path": "Heading B",
        },
    }

    rows, missing = _collect_rows(
        decision="rejected",
        chunk_ids=["chunk-1", "chunk-2"],
        retrieval_index=retrieval_index,
        store=_FakeStore(),  # type: ignore[arg-type]
        city_filter=None,
        show_content=False,
        max_content_chars=100,
        limit=None,
    )

    assert len(rows) == 2
    assert missing == 0
