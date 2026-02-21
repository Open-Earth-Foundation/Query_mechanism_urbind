from __future__ import annotations

from pathlib import Path

from backend.modules.vector_store.benchmarking import run_chunking_benchmark


def test_run_chunking_benchmark_reports_final_and_individual_metrics(
    tmp_path: Path,
) -> None:
    """Benchmark returns final score and inspectable individual metrics."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "CityA.md").write_text(
        "\n".join(
            [
                "# CityA",
                "",
                "Table AP- 1 Building stock",
                "",
                "| Type | Count |",
                "| --- | --- |",
                "| A | 10 |",
            ]
        ),
        encoding="utf-8",
    )
    (docs_dir / "CityB.md").write_text(
        "\n".join(
            [
                "# CityB",
                "",
                "Paragraph one.",
                "",
                "## Section",
                "",
                "Paragraph two.",
            ]
        ),
        encoding="utf-8",
    )

    result = run_chunking_benchmark(
        docs_dir=docs_dir,
        chunk_tokens=200,
        overlap_tokens=20,
        table_row_group_max_rows=10,
        sample_size=2,
        seed=42,
    )

    assert "final_accuracy_score" in result.metrics
    assert "caption_linkage_rate" in result.metrics
    assert "table_header_valid_rate" in result.metrics
    assert "table_detection_rate" in result.metrics
    assert "heading_alignment_rate" in result.metrics
    assert "token_budget_compliance_rate" in result.metrics
    assert result.sample_size == 2
    assert len(result.per_file) == 2


def test_run_chunking_benchmark_caption_linkage_detects_table_captions(
    tmp_path: Path,
) -> None:
    """Caption linkage metric reflects caption attachment behavior."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "CityA.md").write_text(
        "\n".join(
            [
                "# CityA",
                "",
                "Table AP- 6 Current laws",
                "",
                "| Type | Name |",
                "| --- | --- |",
                "| Law | Act |",
            ]
        ),
        encoding="utf-8",
    )

    result = run_chunking_benchmark(
        docs_dir=docs_dir,
        chunk_tokens=200,
        overlap_tokens=20,
        table_row_group_max_rows=10,
        sample_size=1,
        seed=1,
    )

    assert result.counts["source_captioned_tables"] == 1
    assert result.counts["table_chunks_with_caption"] >= 1
    assert result.metrics["caption_linkage_rate"] == 1.0
