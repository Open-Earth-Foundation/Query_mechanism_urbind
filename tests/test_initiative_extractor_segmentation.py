from pathlib import Path

from backend.modules.initiative_extractor.segmentation import build_document_segments
from tests.support import build_test_app_config


def test_oversized_block_is_split_without_breaking_line_boundaries(tmp_path: Path) -> None:
    """Large blocks should split into multiple bounded segments at source-line boundaries."""
    source = tmp_path / "Krakow.md"
    source.write_text(
        "\n".join(
            [
                "# Climate Contract",
                *[f"| Action row {index} | Detailed description {'x' * 120} |" for index in range(1, 81)],
            ]
        ),
        encoding="utf-8",
    )
    config = build_test_app_config(
        initiative_extractor_overrides={
            "max_segment_tokens": 200,
            "segment_overlap_lines": 0,
        }
    )

    segments = build_document_segments(source, config.initiative_extractor)

    assert len(segments) > 1
    assert all(segment.token_count <= config.initiative_extractor.max_segment_tokens for segment in segments)
    assert all(
        line.startswith("| Action row ") or line.startswith("# Climate Contract")
        for segment in segments
        for line in segment.content.splitlines()
    )
    assert segments[0].start_line == 1
    assert segments[-1].end_line == 81
