from __future__ import annotations

import tiktoken


def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text))


def chunk_text(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    if max_tokens <= 0:
        return [text]
    encoding = get_encoding()
    tokens = encoding.encode(text)
    if not tokens:
        return [""]

    step = max(max_tokens - overlap_tokens, 1)
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
        if start + max_tokens >= len(tokens):
            break
    return chunks


def get_max_input_tokens(
    context_window_tokens: int | None,
    max_output_tokens: int | None,
    input_token_reserve: int,
    max_input_tokens: int | None,
) -> int | None:
    if max_input_tokens is not None:
        return max_input_tokens
    if context_window_tokens is None:
        return None
    reserve = max(input_token_reserve, 0)
    output_tokens = max(max_output_tokens or 0, 0)
    available = context_window_tokens - reserve - output_tokens
    return max(available, 0)


__all__ = ["count_tokens", "chunk_text", "get_max_input_tokens"]
