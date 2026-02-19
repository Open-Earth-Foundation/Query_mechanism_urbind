from __future__ import annotations

import re

from backend.modules.vector_store.models import MdBlock

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+")


def _is_table_separator(line: str) -> bool:
    """Return True when line looks like a markdown table separator row."""
    stripped = line.strip()
    if "|" not in stripped:
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell or "") is not None for cell in cells)


def _is_heading(line: str) -> bool:
    """Return True when line is an ATX heading."""
    return _HEADING_RE.match(line.strip()) is not None


def _is_list_line(line: str) -> bool:
    """Return True when line starts a markdown list item."""
    return _LIST_RE.match(line) is not None


def _looks_like_table_start(lines: list[str], index: int) -> bool:
    """Return True when current+next lines start a markdown table."""
    if index + 1 >= len(lines):
        return False
    header = lines[index].strip()
    separator = lines[index + 1].strip()
    return "|" in header and _is_table_separator(separator)


def _heading_path_from_stack(heading_stack: list[tuple[int, str]]) -> list[str]:
    """Convert heading stack tuples into plain heading names."""
    return [heading for _, heading in heading_stack]


def _looks_like_table_caption(line: str) -> bool:
    """Return True when line resembles a table title/caption."""
    normalized = line.strip().casefold()
    return normalized.startswith("table ")


def parse_markdown_blocks(text: str) -> list[MdBlock]:
    """Parse markdown into table-aware blocks with heading paths."""
    lines = text.splitlines()
    blocks: list[MdBlock] = []
    heading_stack: list[tuple[int, str]] = []
    i = 0

    while i < len(lines):
        current = lines[i]
        stripped = current.strip()
        line_no = i + 1

        if not stripped:
            i += 1
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match is not None:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            blocks.append(
                MdBlock(
                    block_type="paragraph",
                    text=current,
                    heading_path=_heading_path_from_stack(heading_stack),
                    start_line=line_no,
                    end_line=line_no,
                )
            )
            i += 1
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence_marker = stripped[:3]
            start = i
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence_marker):
                i += 1
            if i < len(lines):
                i += 1
            code_text = "\n".join(lines[start:i]).strip()
            blocks.append(
                MdBlock(
                    block_type="code",
                    text=code_text,
                    heading_path=_heading_path_from_stack(heading_stack),
                    start_line=start + 1,
                    end_line=i,
                )
            )
            continue

        if _looks_like_table_start(lines, i):
            start = i
            caption_text: str | None = None
            caption_line: int | None = None
            if blocks:
                previous_block = blocks[-1]
                if (
                    previous_block.block_type == "paragraph"
                    and len(previous_block.text.splitlines()) == 1
                    and _looks_like_table_caption(previous_block.text)
                ):
                    previous_end = previous_block.end_line or 0
                    between_lines = lines[previous_end:start]
                    if all(not line.strip() for line in between_lines):
                        caption_text = previous_block.text
                        caption_line = previous_block.start_line
                        blocks.pop()
            i += 2
            while i < len(lines) and "|" in lines[i]:
                i += 1
            table_text = "\n".join(lines[start:i]).strip()
            if caption_text:
                table_text = f"{caption_text}\n\n{table_text}"
            blocks.append(
                MdBlock(
                    block_type="table",
                    text=table_text,
                    heading_path=_heading_path_from_stack(heading_stack),
                    start_line=caption_line if caption_line is not None else start + 1,
                    end_line=i,
                    table_title=caption_text,
                )
            )
            continue

        if _is_list_line(current):
            start = i
            i += 1
            while i < len(lines) and (not lines[i].strip() or _is_list_line(lines[i])):
                if _is_heading(lines[i]) or _looks_like_table_start(lines, i):
                    break
                i += 1
            list_text = "\n".join(lines[start:i]).strip()
            blocks.append(
                MdBlock(
                    block_type="list",
                    text=list_text,
                    heading_path=_heading_path_from_stack(heading_stack),
                    start_line=start + 1,
                    end_line=i,
                )
            )
            continue

        start = i
        i += 1
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                if i < len(lines) and not lines[i].strip():
                    continue
                break
            if _is_heading(lines[i]) or _looks_like_table_start(lines, i):
                break
            if lines[i].strip().startswith(("```", "~~~")) or _is_list_line(lines[i]):
                break
            i += 1
        paragraph_text = "\n".join(lines[start:i]).strip()
        if paragraph_text:
            blocks.append(
                MdBlock(
                    block_type="paragraph",
                    text=paragraph_text,
                    heading_path=_heading_path_from_stack(heading_stack),
                    start_line=start + 1,
                    end_line=i,
                )
            )

    return blocks


__all__ = ["parse_markdown_blocks"]
