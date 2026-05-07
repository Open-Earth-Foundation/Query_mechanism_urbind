from __future__ import annotations

import re
from pathlib import Path

from backend.modules.initiative_extractor.models import InitiativeDocumentSegment
from backend.utils.city_normalization import normalize_city_key
from backend.utils.config import InitiativeExtractorConfig
from backend.utils.tokenization import count_tokens


def _project_relative_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    project_root = Path(__file__).resolve().parents[3]
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _slug(value: str) -> str:
    """Normalize a value for stable segment identifiers."""
    normalized = normalize_city_key(value)
    return normalized or re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _line_blocks(lines: list[str]) -> list[tuple[int, int, list[str]]]:
    """Group lines into blocks without splitting individual source lines."""
    blocks: list[tuple[int, int, list[str]]] = []
    current: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal current, start_line
        if current:
            blocks.append((start_line, end_line, current))
            current = []

    for index, line in enumerate(lines, start=1):
        is_heading = line.lstrip().startswith("#")
        if is_heading:
            flush(index - 1)
        if not current:
            start_line = index
        current.append(line)
        if not line.strip():
            flush(index)

    flush(len(lines))
    return blocks


def _heading_for_line(lines: list[str], line_number: int) -> str | None:
    """Return the nearest preceding markdown or initiative heading."""
    headings: list[str] = []
    for line in lines[:line_number]:
        stripped = line.strip()
        if stripped.startswith("#"):
            headings.append(stripped.lstrip("#").strip())
            continue
    if not headings:
        return None
    return " > ".join(headings[-3:])


def _split_oversized_block(lines: list[str], max_tokens: int) -> list[list[str]]:
    """Split one oversized line block into token-bounded line chunks."""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = max(count_tokens(line), 1)
        if current and current_tokens + line_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += line_tokens

    if current:
        chunks.append(current)
    return chunks


def _make_segment(
    *,
    city_name: str,
    source_path: str,
    source_document: str,
    start_line: int,
    lines: list[str],
    index: int,
    all_lines: list[str],
    parent_segment_id: str | None = None,
) -> InitiativeDocumentSegment:
    """Create one segment model from source lines."""
    end_line = start_line + len(lines) - 1
    prefix = f"{_slug(city_name)}:{_slug(Path(source_document).stem)}:seg{index:04d}"
    segment_id = f"{prefix}:{start_line}-{end_line}"
    return InitiativeDocumentSegment(
        segment_id=segment_id,
        city_name=city_name,
        source_document=source_document,
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        heading_path=_heading_for_line(all_lines, start_line),
        content="\n".join(lines),
        token_count=count_tokens("\n".join(lines)),
        parent_segment_id=parent_segment_id,
    )


def build_document_segments(
    path: Path,
    config: InitiativeExtractorConfig,
    *,
    city_name: str | None = None,
) -> list[InitiativeDocumentSegment]:
    """Split a markdown document into ordered line-aware extraction segments."""
    resolved_city = city_name or path.stem
    source_path = _project_relative_path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = _line_blocks(lines)
    segments: list[InitiativeDocumentSegment] = []
    current_lines: list[str] = []
    current_start = 1
    current_tokens = 0
    max_tokens = max(config.max_segment_tokens, 1)
    overlap_lines = max(config.segment_overlap_lines, 0)

    def flush() -> None:
        nonlocal current_lines, current_start, current_tokens
        if not current_lines:
            return
        segment = _make_segment(
            city_name=resolved_city,
            source_path=source_path,
            source_document=path.name,
            start_line=current_start,
            lines=current_lines,
            index=len(segments) + 1,
            all_lines=lines,
        )
        segments.append(segment)
        if overlap_lines:
            current_lines = current_lines[-overlap_lines:]
            current_start = segment.end_line - len(current_lines) + 1
            current_tokens = count_tokens("\n".join(current_lines))
        else:
            current_lines = []
            current_tokens = 0

    for block_start, _block_end, block_lines in blocks:
        sliced_blocks = _split_oversized_block(block_lines, max_tokens)
        slice_start = block_start
        for sliced_lines in sliced_blocks:
            block_tokens = count_tokens("\n".join(sliced_lines))
            if current_lines and current_tokens + block_tokens > max_tokens:
                flush()
            if not current_lines:
                current_start = slice_start
            current_lines.extend(sliced_lines)
            current_tokens += block_tokens
            slice_start += len(sliced_lines)

    flush()
    return segments


def detect_source_quality_flags(content: str) -> list[str]:
    """Detect visible source-conversion issues that should be reviewed."""
    flags: list[str] = []
    if re.search(r"[ĂĹÂ�]|â[€“€ťś‚]", content):
        flags.append("encoding_or_ocr_artifacts")
    if re.search(r"\n(?:#|O|\d{1,2})\n", content):
        flags.append("page_artifacts_inside_section")
    if "will be completed in future" in content.casefold():
        flags.append("source_contains_deferred_values")
    return flags


__all__ = [
    "build_document_segments",
    "detect_source_quality_flags",
]
