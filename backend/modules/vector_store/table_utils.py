from __future__ import annotations


def parse_markdown_table(table_text: str) -> tuple[str | None, str, str, list[str]]:
    """Return optional caption, header row, separator row, and body rows."""
    lines = [line for line in table_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None, table_text.strip(), "", []

    caption: str | None = None
    header_idx = 0
    if "|" not in lines[0] and len(lines) >= 3 and "|" in lines[1]:
        caption = lines[0]
        header_idx = 1

    if header_idx + 1 >= len(lines):
        return caption, lines[header_idx], "", []
    header = lines[header_idx]
    separator = lines[header_idx + 1]
    rows = lines[header_idx + 2 :]
    return caption, header, separator, rows


def split_markdown_table_by_row_groups(
    table_text: str,
    max_rows_per_group: int,
) -> list[str]:
    """Split table into markdown row groups while repeating header rows."""
    caption, header, separator, rows = parse_markdown_table(table_text)
    if not separator or max_rows_per_group <= 0:
        return [table_text.strip()]
    if not rows:
        lines = [header, separator]
        if caption:
            lines.insert(0, caption)
        return ["\n".join(lines).strip()]

    grouped: list[str] = []
    for start in range(0, len(rows), max_rows_per_group):
        group_rows = rows[start : start + max_rows_per_group]
        lines = [header, separator, *group_rows]
        if caption:
            lines.insert(0, caption)
        grouped.append("\n".join(lines).strip())
    return grouped


def summarize_table_for_embedding(
    raw_table: str,
    heading_path: list[str],
    table_title: str | None = None,
    max_preview_rows: int = 5,
) -> str:
    """Create a deterministic table summary for embedding text."""
    caption, header, _, rows = parse_markdown_table(raw_table)
    columns = [cell.strip() for cell in header.strip("|").split("|")]
    title_parts: list[str] = []
    if heading_path:
        title_parts.append(" > ".join(heading_path))
    if table_title:
        title_parts.append(table_title)
    elif caption:
        title_parts.append(caption)
    title = " | ".join(title_parts) if title_parts else "Table"
    preview = "\n".join(rows[:max(0, max_preview_rows)]).strip()
    row_count = len(rows)
    parts = [
        f"Table context: {title}",
        f"Columns: {', '.join(columns)}",
        f"Row count: {row_count}",
    ]
    if preview:
        parts.append("Preview rows:")
        parts.append(preview)
    return "\n".join(parts).strip()


__all__ = [
    "parse_markdown_table",
    "split_markdown_table_by_row_groups",
    "summarize_table_for_embedding",
]
