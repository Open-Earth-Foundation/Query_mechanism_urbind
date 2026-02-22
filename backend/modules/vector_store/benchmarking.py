from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from backend.modules.vector_store.chunk_packer import pack_blocks
from backend.modules.vector_store.markdown_blocks import parse_markdown_blocks

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_CAPTION_RE = re.compile(r"^\s*table\s+", re.IGNORECASE)


@dataclass(frozen=True)
class ChunkingBenchmarkResult:
    """Structured chunking benchmark output."""

    seed: int
    sample_size: int
    total_docs_available: int
    sampled_docs: list[str]
    counts: dict[str, int]
    metrics: dict[str, float]
    per_file: list[dict[str, str | int | float]]


def _is_table_separator(line: str) -> bool:
    """Return True when line resembles markdown table separator."""
    stripped = line.strip()
    if "|" not in stripped:
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell or "") is not None for cell in cells)


def _looks_like_table_start(lines: list[str], index: int) -> bool:
    """Return True when line and next line start markdown table."""
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and _is_table_separator(lines[index + 1])


def _detect_source_tables(lines: list[str]) -> tuple[int, int]:
    """Count source tables and source tables with explicit caption lines."""
    table_count = 0
    captioned_table_count = 0
    for idx in range(len(lines) - 1):
        if not _looks_like_table_start(lines, idx):
            continue
        table_count += 1
        cursor = idx - 1
        while cursor >= 0 and not lines[cursor].strip():
            cursor -= 1
        if cursor >= 0 and _TABLE_CAPTION_RE.match(lines[cursor].strip()):
            captioned_table_count += 1
    return table_count, captioned_table_count


def _has_valid_table_header(raw_text: str) -> bool:
    """Check whether chunk raw text contains a valid markdown table header+separator."""
    lines = [line for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return False
    if _TABLE_CAPTION_RE.match(lines[0].strip()) and len(lines) >= 3:
        lines = lines[1:]
    if len(lines) < 2:
        return False
    return "|" in lines[0] and _is_table_separator(lines[1])


def _expected_heading_path(lines: list[str], line_no: int | None) -> str:
    """Compute heading path active at a given source line."""
    if line_no is None or line_no <= 0:
        return ""
    stack: list[tuple[int, str]] = []
    max_idx = min(line_no - 1, len(lines))
    for idx in range(max_idx):
        match = _HEADING_RE.match(lines[idx].strip())
        if match is None:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return " > ".join([title for _, title in stack])


def _safe_rate(numerator: int, denominator: int) -> float:
    """Compute rate and handle zero denominator."""
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def run_chunking_benchmark(
    docs_dir: Path,
    chunk_tokens: int,
    overlap_tokens: int,
    table_row_group_max_rows: int,
    sample_size: int,
    seed: int,
) -> ChunkingBenchmarkResult:
    """Run corpus chunking benchmark on a deterministic random subset."""
    files = sorted(docs_dir.rglob("*.md"))
    rng = random.Random(seed)
    chosen = files if sample_size <= 0 or sample_size >= len(files) else rng.sample(files, sample_size)
    chosen = sorted(chosen)

    total_source_tables = 0
    total_source_captioned_tables = 0
    total_detected_tables = 0
    total_caption_attached = 0
    total_valid_table_headers = 0
    total_chunks = 0
    total_token_compliant = 0
    total_heading_aligned = 0
    total_heading_checked = 0
    per_file: list[dict[str, str | int | float]] = []

    for path in chosen:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        source_tables, source_captioned = _detect_source_tables(lines)
        blocks = parse_markdown_blocks(text)
        chunks = pack_blocks(
            blocks=blocks,
            max_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            table_row_group_max_rows=table_row_group_max_rows,
        )

        table_chunks = [chunk for chunk in chunks if chunk.block_type == "table"]
        caption_attached = [chunk for chunk in table_chunks if chunk.table_title]
        valid_table_headers = [chunk for chunk in table_chunks if _has_valid_table_header(chunk.raw_text)]

        heading_checked = 0
        heading_aligned = 0
        for chunk in chunks:
            expected = _expected_heading_path(lines, chunk.start_line)
            if not expected:
                continue
            heading_checked += 1
            if chunk.heading_path == expected:
                heading_aligned += 1

        token_compliant = [chunk for chunk in chunks if chunk.token_count <= chunk_tokens]
        per_file.append(
            {
                "source_path": path.as_posix(),
                "source_tables": source_tables,
                "source_captioned_tables": source_captioned,
                "chunk_count": len(chunks),
                "table_chunk_count": len(table_chunks),
                "caption_attached_count": len(caption_attached),
                "valid_table_header_count": len(valid_table_headers),
                "heading_alignment_rate": _safe_rate(heading_aligned, heading_checked),
            }
        )

        total_source_tables += source_tables
        total_source_captioned_tables += source_captioned
        total_detected_tables += len(table_chunks)
        total_caption_attached += len(caption_attached)
        total_valid_table_headers += len(valid_table_headers)
        total_chunks += len(chunks)
        total_token_compliant += len(token_compliant)
        total_heading_aligned += heading_aligned
        total_heading_checked += heading_checked

    table_detection_rate = min(_safe_rate(total_detected_tables, total_source_tables), 1.0)
    caption_linkage_rate = min(
        _safe_rate(total_caption_attached, total_source_captioned_tables),
        1.0,
    )
    table_header_valid_rate = _safe_rate(total_valid_table_headers, max(total_detected_tables, 1))
    token_budget_compliance_rate = _safe_rate(total_token_compliant, max(total_chunks, 1))
    heading_alignment_rate = _safe_rate(total_heading_aligned, max(total_heading_checked, 1))

    final_score = (
        0.35 * caption_linkage_rate
        + 0.25 * table_header_valid_rate
        + 0.20 * table_detection_rate
        + 0.10 * heading_alignment_rate
        + 0.10 * token_budget_compliance_rate
    )

    return ChunkingBenchmarkResult(
        seed=seed,
        sample_size=len(chosen),
        total_docs_available=len(files),
        sampled_docs=[path.as_posix() for path in chosen],
        counts={
            "source_tables": total_source_tables,
            "source_captioned_tables": total_source_captioned_tables,
            "detected_table_chunks": total_detected_tables,
            "table_chunks_with_caption": total_caption_attached,
            "table_chunks_with_valid_header": total_valid_table_headers,
            "total_chunks": total_chunks,
            "token_compliant_chunks": total_token_compliant,
            "heading_checked_chunks": total_heading_checked,
            "heading_aligned_chunks": total_heading_aligned,
        },
        metrics={
            "caption_linkage_rate": caption_linkage_rate,
            "table_header_valid_rate": table_header_valid_rate,
            "table_detection_rate": table_detection_rate,
            "heading_alignment_rate": heading_alignment_rate,
            "token_budget_compliance_rate": token_budget_compliance_rate,
            "final_accuracy_score": final_score,
        },
        per_file=per_file,
    )


__all__ = ["ChunkingBenchmarkResult", "run_chunking_benchmark"]
