from __future__ import annotations

import hashlib

from backend.modules.vector_store.models import MdBlock, PackedChunk
from backend.modules.vector_store.table_utils import (
    parse_markdown_table,
    split_markdown_table_by_row_groups,
    summarize_table_for_embedding,
)
from backend.utils.tokenization import chunk_text, count_tokens


def _build_table_id(block: MdBlock) -> str:
    """Build deterministic table id from block content and position."""
    payload = f"{' > '.join(block.heading_path)}::{block.start_line}:{block.end_line}:{block.text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"table_{digest[:12]}"


def _split_large_table_block(
    block: MdBlock,
    max_tokens: int,
    max_rows_per_group: int,
) -> list[MdBlock]:
    """Split oversized table block into row groups respecting token budget."""
    row_groups = split_markdown_table_by_row_groups(block.text, max_rows_per_group)
    table_id = _build_table_id(block)
    split_blocks: list[MdBlock] = []

    for group_index, group in enumerate(row_groups):
        if count_tokens(group) <= max_tokens:
            split_blocks.append(
                MdBlock(
                    block_type="table",
                    text=group,
                    heading_path=block.heading_path,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    table_id=table_id,
                    row_group_index=group_index,
                    table_title=block.table_title,
                )
            )
            continue

        lines = group.splitlines()
        if len(lines) <= 3:
            split_blocks.append(
                MdBlock(
                    block_type="table",
                    text=group,
                    heading_path=block.heading_path,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    table_id=table_id,
                    row_group_index=group_index,
                    table_title=block.table_title,
                )
            )
            continue

        caption, header, separator, rows = parse_markdown_table(group)
        if not separator:
            split_blocks.append(
                MdBlock(
                    block_type="table",
                    text=group,
                    heading_path=block.heading_path,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    table_id=table_id,
                    row_group_index=group_index,
                    table_title=block.table_title,
                )
            )
            continue

        prefix_lines = [header, separator]
        if caption:
            prefix_lines.insert(0, caption)
        current_rows: list[str] = []
        local_group_index = 0
        for row in rows:
            candidate = "\n".join([*prefix_lines, *current_rows, row]).strip()
            if current_rows and count_tokens(candidate) > max_tokens:
                split_blocks.append(
                    MdBlock(
                        block_type="table",
                        text="\n".join([*prefix_lines, *current_rows]).strip(),
                        heading_path=block.heading_path,
                        start_line=block.start_line,
                        end_line=block.end_line,
                        table_id=table_id,
                        row_group_index=(group_index * 1000) + local_group_index,
                        table_title=block.table_title,
                    )
                )
                local_group_index += 1
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_rows:
            split_blocks.append(
                MdBlock(
                    block_type="table",
                    text="\n".join([*prefix_lines, *current_rows]).strip(),
                    heading_path=block.heading_path,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    table_id=table_id,
                    row_group_index=(group_index * 1000) + local_group_index,
                    table_title=block.table_title,
                )
            )

    return split_blocks


def _split_large_non_table_block(block: MdBlock, max_tokens: int) -> list[MdBlock]:
    """Split one oversized non-table block into token-bounded blocks."""
    split_blocks: list[MdBlock] = []
    for chunk in chunk_text(block.text, max_tokens=max_tokens, overlap_tokens=0):
        if not chunk.strip():
            continue
        split_blocks.append(
            MdBlock(
                block_type=block.block_type,
                text=chunk,
                heading_path=block.heading_path,
                start_line=block.start_line,
                end_line=block.end_line,
            )
        )
    return split_blocks or [block]


def _normalize_blocks_for_budget(
    blocks: list[MdBlock],
    max_tokens: int,
    max_rows_per_group: int,
) -> list[MdBlock]:
    """Replace oversized blocks with budget-compliant split blocks."""
    normalized: list[MdBlock] = []
    for block in blocks:
        block_tokens = count_tokens(block.text)
        if block.block_type == "table" and block_tokens > max_tokens:
            normalized.extend(
                _split_large_table_block(block, max_tokens, max_rows_per_group)
            )
            continue
        if block.block_type != "table" and block_tokens > max_tokens:
            normalized.extend(_split_large_non_table_block(block, max_tokens))
            continue
        normalized.append(block)
    return normalized


def _join_block_texts(blocks: list[MdBlock]) -> str:
    """Join block texts with stable blank-line separators."""
    return "\n\n".join(block.text.strip() for block in blocks if block.text.strip()).strip()


def pack_blocks(
    blocks: list[MdBlock],
    max_tokens: int,
    overlap_tokens: int,
    table_row_group_max_rows: int = 25,
) -> list[PackedChunk]:
    """Greedily pack parsed markdown blocks into token-limited chunks."""
    if max_tokens <= 0:
        max_tokens = 800
    normalized_blocks = _normalize_blocks_for_budget(
        blocks, max_tokens=max_tokens, max_rows_per_group=table_row_group_max_rows
    )
    if not normalized_blocks:
        return []

    chunks: list[PackedChunk] = []
    current_blocks: list[MdBlock] = []

    def flush_chunk(chunk_blocks: list[MdBlock]) -> None:
        if not chunk_blocks:
            return
        raw_text = _join_block_texts(chunk_blocks)
        if not raw_text:
            return
        block_types = {block.block_type for block in chunk_blocks}
        chunk_type = chunk_blocks[0].block_type if len(block_types) == 1 else "paragraph"
        heading_values: list[str] = []
        for block in chunk_blocks:
            for heading in block.heading_path:
                if heading not in heading_values:
                    heading_values.append(heading)
        heading_path = " > ".join(heading_values)
        first_block = chunk_blocks[0]
        embedding_text = raw_text
        table_id = None
        row_group_index = None
        table_title = None
        if chunk_type == "table":
            table_title = first_block.table_title
            max_preview = 5
            while True:
                embedding_text = summarize_table_for_embedding(
                    raw_table=raw_text,
                    heading_path=first_block.heading_path,
                    table_title=table_title,
                    max_preview_rows=max_preview,
                )
                if count_tokens(embedding_text) <= max_tokens or max_preview == 0:
                    break
                max_preview -= 1
            table_id = first_block.table_id
            row_group_index = first_block.row_group_index
        chunks.append(
            PackedChunk(
                raw_text=raw_text,
                embedding_text=embedding_text,
                block_type=chunk_type,
                heading_path=heading_path,
                token_count=count_tokens(raw_text),
                chunk_index=len(chunks),
                start_line=first_block.start_line,
                end_line=chunk_blocks[-1].end_line,
                table_id=table_id,
                row_group_index=row_group_index,
                table_title=table_title,
            )
        )

    def collect_overlap_blocks(previous_blocks: list[MdBlock]) -> list[MdBlock]:
        """Collect trailing blocks up to the configured overlap token budget."""
        if overlap_tokens <= 0:
            return []
        overlap_blocks: list[MdBlock] = []
        overlap_budget = 0
        for previous_block in reversed(previous_blocks):
            block_tokens = count_tokens(previous_block.text)
            if overlap_blocks and overlap_budget + block_tokens > overlap_tokens:
                break
            overlap_blocks.insert(0, previous_block)
            overlap_budget += block_tokens
        return overlap_blocks

    for block in normalized_blocks:
        if not block.text.strip():
            continue
        if current_blocks:
            current_has_table = any(
                existing.block_type == "table" for existing in current_blocks
            )
            incoming_is_table = block.block_type == "table"
            if current_has_table != incoming_is_table:
                flush_chunk(current_blocks)
                # Keep table boundaries strict; do not overlap across table/non-table transitions.
                current_blocks = []

        if block.block_type == "table":
            if current_blocks:
                flush_chunk(current_blocks)
                current_blocks = []
            flush_chunk([block])
            continue

        candidate_blocks = [*current_blocks, block]
        candidate_text = _join_block_texts(candidate_blocks)
        if not candidate_text:
            continue
        if count_tokens(candidate_text) > max_tokens:
            if current_blocks:
                previous_blocks = current_blocks
                flush_chunk(previous_blocks)
                current_blocks = collect_overlap_blocks(previous_blocks)
                while current_blocks and count_tokens(
                    _join_block_texts([*current_blocks, block])
                ) > max_tokens:
                    current_blocks.pop(0)
                candidate_blocks = [*current_blocks, block]
                candidate_text = _join_block_texts(candidate_blocks)
            if candidate_text and count_tokens(candidate_text) > max_tokens:
                for split_block in _split_large_non_table_block(block, max_tokens):
                    flush_chunk([split_block])
                current_blocks = []
                continue
        current_blocks.append(block)

    flush_chunk(current_blocks)
    return chunks


__all__ = ["pack_blocks"]
